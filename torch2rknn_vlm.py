# conda activate rknn_vlm && python torch2rknn_vlm.py

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from tqdm import tqdm
from PIL import Image
from rknn.api import RKNN
from rkllm.api import RKLLM
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration, SmolVLMForConditionalGeneration
from torchvision.transforms.functional import InterpolationMode

MODEL_CKPT_MAP = {
    'Qwen2.5-VL': 'Qwen/Qwen2.5-VL-3B-Instruct',
    'InternVL3_5-4B': 'OpenGVLab/InternVL3_5-4B',
    'InternVL3-1B': 'OpenGVLab/InternVL3-1B',
    'SmolVLM-256M': 'HuggingFaceTB/SmolVLM-256M-Instruct',
    'SmolVLM-500M': 'HuggingFaceTB/SmolVLM-500M-Instruct'
}

assert (MODEL_NAME := 'InternVL3-1B') in list(MODEL_CKPT_MAP.keys())
# There is references for minicpm, smolvlm, 
# check https://github.com/airockchip/rknn-llm/blob/main/examples/multimodal_model_demo/export/export_vision.py
MODEL_CKPT_PATH = MODEL_CKPT_MAP[MODEL_NAME]
ONNX_PATH = os.path.join(os.getcwd(), f"models/{MODEL_NAME}/{MODEL_NAME}_vision.onnx")
RKNN_PATH = os.path.join(os.getcwd(), f"models/{MODEL_NAME}/{MODEL_NAME}_vision.rknn")
RKLLM_PATH = os.path.join(os.getcwd(), f"models/{MODEL_NAME}/{MODEL_NAME}_llm.rkllm")
PLATFORM = 'rk3588'
CALIBARION_DATASET_PATH = os.path.join(os.getcwd(), f"data/inputs_{MODEL_NAME}.json")
DO_QUANT = True
QUANT_DTYPE = 'w8a8'
NUM_NPU_CORE = 3
MAX_CONTEXT = 16384 #16384

DEVICE = 'cuda' # sometimes using cuda as device cause error, notImplementedError for some ops

os.makedirs(os.path.dirname(ONNX_PATH), exist_ok=True)

if MODEL_NAME == 'Qwen2.5-VL':
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_CKPT_PATH,
        torch_dtype=torch.float16, # 注意此处的数据类型，由于 rknn 目前仅支持 float32 ，因此需要指定；若是在加载权重时限制了数据类型，需要自行修改config.json中的 "use_flash_attn" 参数为 false
        low_cpu_mem_usage=True, _attn_implementation="eager",
        trust_remote_code=True).eval().to(DEVICE)
    
    MEAN = [[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255]]
    STD = [[0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255]]

    N, C, H, W = [1, 3, 224, 224]

elif MODEL_NAME in ['InternVL3_5-4B', 'InternVL3-1B']:
    model = AutoModel.from_pretrained(
        MODEL_CKPT_PATH,
        torch_dtype=torch.float32, # try lower this when export multiple weights file instead of just one onnx
        low_cpu_mem_usage=True,
        trust_remote_code=True).eval().to(DEVICE)
        
    MEAN = [[0.485 * 255, 0.456 * 255, 0.406 * 255]]
    STD = [[0.229 * 255, 0.224 * 255, 0.225 * 255]]

    N, C, H, W = [1, 3, 448, 448]

elif MODEL_NAME in ['SmolVLM-256M', 'SmolVLM-500M']:

    class smolvlm_embeddings(torch.nn.Module):
        def __init__(self, embed):
            super().__init__()
            self.embed = embed

        def forward(self, pixel_values: torch.FloatTensor, patch_attention_mask: torch.BoolTensor) -> torch.Tensor:
            batch_size, _, max_im_h, max_im_w = pixel_values.shape

            patch_embeds = self.embed.patch_embedding(pixel_values)
            embeddings = patch_embeds.flatten(2).transpose(1, 2)

            max_nb_patches_h, max_nb_patches_w = max_im_h // self.embed.patch_size, max_im_w // self.embed.patch_size
            boundaries = torch.arange(
                1 / self.embed.num_patches_per_side, 1.0, 1 / self.embed.num_patches_per_side, device=pixel_values.device
            )
            position_ids = torch.full(
                size=(batch_size, max_nb_patches_h * max_nb_patches_w), fill_value=0, device=pixel_values.device
            )

            for batch_idx, p_attn_mask in enumerate(patch_attention_mask):
                nb_patches_h = p_attn_mask[:, 0].sum()
                nb_patches_w = p_attn_mask[0].sum()

                h_indices = torch.arange(nb_patches_h, device=position_ids.device, dtype=position_ids.dtype)
                w_indices = torch.arange(nb_patches_w, device=position_ids.device, dtype=position_ids.dtype)

                fractional_coords_h = h_indices / nb_patches_h * (1 - 1e-6)
                fractional_coords_w = w_indices / nb_patches_w * (1 - 1e-6)

                bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
                bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)

                pos_ids = (bucket_coords_h[:, None] * self.embed.num_patches_per_side + bucket_coords_w).flatten()
                position_ids[batch_idx][p_attn_mask.view(-1)] = pos_ids

            position_ids = position_ids.to(torch.long)
            embeddings = embeddings + self.embed.position_embedding(position_ids)
            return embeddings

    class smolvlm_vision(torch.nn.Module):
        def __init__(self, vlm):
            super().__init__()
            self.vlm = vlm
            self.txtm = vlm.model.text_model
            self.vpm = vlm.model.vision_model
            self.connector = vlm.model.connector
            self.vpm.embeddings = smolvlm_embeddings(self.vpm.embeddings)
            
        def forward(self, pixel_values):
            image_hidden_states = self.vpm(pixel_values).last_hidden_state
            image_hidden_states = self.connector(image_hidden_states)
            return image_hidden_states

    model = SmolVLMForConditionalGeneration.from_pretrained( # if this work, test AutoModel
        MODEL_CKPT_PATH,
        torch_dtype=torch.bfloat16,
        _attn_implementation="eager"
    ).to(DEVICE)
    model = smolvlm_vision(model).to(torch.float32).eval()

    MEAN = [[0.5 * 255, 0.5 * 255, 0.5 * 255]]
    STD = [[0.5 * 255, 0.5 * 255, 0.5 * 255]]

    N, C, H, W = [1, 3, 512, 512]

# -------------------------------------------------------------------------------------------------------
# Convert Vision part of VLM to ONNX

if os.path.exists(ONNX_PATH):
    print(f'{ONNX_PATH} exist, skipping: Convert Vision part of VLM to ONNX')
else:
    try:
        if MODEL_NAME == 'Qwen2.5-VL':
            
            class PatchedQwen2_5Vision(nn.Module):

                def __init__(self, vlm):
                    super().__init__()
                    self.vlm = vlm
                    self.vlm.rot_pos_emb = self.rot_pos_emb
                    self.vlm.get_window_index = self.get_window_index

                    self.spatial_merge_size = self.vlm.spatial_merge_size
                    self.spatial_merge_unit = self.vlm.spatial_merge_unit

                def rot_pos_emb(self, grid_thw):
                    pos_ids = []

                    assert torch.equal(grid_thw, torch.tensor([[N // 2 + N % 2, H//14, W//14]], dtype=torch.int64))
                    #  if this assertion failed, you will have to change t, h, w, max_grid_size accordingly
                    # /home/paiworker1/miniconda3/envs/rknn_vlm/lib/python3.10/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py for reference
                    t = torch.tensor(N // 2 + N % 2, dtype=torch.int64)
                    h, w = torch.tensor(H//14, dtype=torch.int64), torch.tensor(W//14, dtype=torch.int64)

                    max_grid_size = max(h, w)

                    hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
                    hpos_ids = hpos_ids.reshape(
                        h // self.spatial_merge_size,
                        self.spatial_merge_size,
                        w // self.spatial_merge_size,
                        self.spatial_merge_size,
                    )
                    hpos_ids = hpos_ids.permute(0, 2, 1, 3)
                    hpos_ids = hpos_ids.flatten()

                    wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
                    wpos_ids = wpos_ids.reshape(
                        h // self.spatial_merge_size,
                        self.spatial_merge_size,
                        w // self.spatial_merge_size,
                        self.spatial_merge_size,
                    )
                    wpos_ids = wpos_ids.permute(0, 2, 1, 3)
                    wpos_ids = wpos_ids.flatten()
                    pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
                    pos_ids = torch.cat(pos_ids, dim=0)
                    rotary_pos_emb_full = self.vlm.rotary_pos_emb(max_grid_size)
                    rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
                    return rotary_pos_emb
                    
                def get_window_index(self, grid_thw):
                    window_index: list = []
                    cu_window_seqlens: list = [0]
                    window_index_id = 0
                    vit_merger_window_size = self.vlm.window_size // self.spatial_merge_size // self.vlm.patch_size

                    assert torch.equal(grid_thw, torch.tensor([[N // 2 + N % 2, H//14, W//14]], dtype=torch.int64))
                    grid_t = torch.tensor(N // 2 + N % 2, dtype=torch.int64)
                    grid_h, grid_w = torch.tensor(H//14, dtype=torch.int64), torch.tensor(W//14, dtype=torch.int64)
                    
                    llm_grid_h, llm_grid_w = (
                        grid_h // self.spatial_merge_size,
                        grid_w // self.spatial_merge_size,
                    )
                    index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
                    pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
                    pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
                    num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
                    num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
                    index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
                    index_padded = index_padded.reshape(
                        grid_t,
                        num_windows_h,
                        vit_merger_window_size,
                        num_windows_w,
                        vit_merger_window_size,
                    )
                    index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                        grid_t,
                        num_windows_h * num_windows_w,
                        vit_merger_window_size,
                        vit_merger_window_size,
                    )
                    seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
                    index_padded = index_padded.reshape(-1)
                    index_new = index_padded[index_padded != -100]
                    window_index.append(index_new + window_index_id)
                    cu_seqlens_tmp = seqlens.cumsum(0) * self.vlm.spatial_merge_unit + cu_window_seqlens[-1]
                    cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
                    window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
                    window_index = torch.cat(window_index, dim=0)

                    return window_index, cu_window_seqlens
                
                def forward(self, hidden_states, grid_thw):
                    hidden_states = self.vlm.patch_embed(hidden_states)
                    rotary_pos_emb = self.rot_pos_emb(grid_thw)
                    window_index, cu_window_seqlens = self.get_window_index(grid_thw)
                    cu_window_seqlens = torch.tensor(
                        cu_window_seqlens,
                        device=hidden_states.device,
                        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
                    )
                    cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

                    seq_len, _ = hidden_states.size()
                    hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
                    hidden_states = hidden_states[window_index, :, :]
                    hidden_states = hidden_states.reshape(seq_len, -1)
                    rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
                    rotary_pos_emb = rotary_pos_emb[window_index, :, :]
                    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
                    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
                    position_embeddings = (emb.cos(), emb.sin())

                    grid_t = torch.tensor([1], dtype=torch.int64)
                    grid_h, grid_w = torch.tensor([16], dtype=torch.int64), torch.tensor([16], dtype=torch.int64)

                    cu_seqlens = torch.repeat_interleave(grid_h * grid_w, grid_t).cumsum(
                        dim=0,
                        # Select dtype based on the following factors:
                        #  - FA2 requires that cu_seqlens_q must have dtype int32
                        #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
                        # See https://github.com/huggingface/transformers/pull/34852 for more information
                        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
                    )
                    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

                    for layer_num, blk in enumerate(self.vlm.blocks):
                        if layer_num in self.vlm.fullatt_block_indexes:
                            cu_seqlens_now = cu_seqlens
                        else:
                            cu_seqlens_now = cu_window_seqlens
                        if self.vlm.gradient_checkpointing and self.vlm.training:
                            hidden_states = self.vlm._gradient_checkpointing_func(
                                blk.__call__, hidden_states, cu_seqlens_now, None, position_embeddings
                            )
                        else:
                            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens_now, position_embeddings=position_embeddings)

                    hidden_states = self.vlm.merger(hidden_states)
                    reverse_indices = torch.argsort(window_index)
                    hidden_states = hidden_states[reverse_indices, :]

                    return hidden_states

            class qwen2_5_vl_3b_vision(nn.Module):
                def __init__(self, vlm, batch_size):
                    super().__init__()
                    self.merge_size = 2
                    self.temporal_patch_size = 2
                    self.patch_size = 14
                    self.channel = C
                    self.vpm = PatchedQwen2_5Vision(vlm.visual)
                    self.batch_size = batch_size

                def forward(self, pixel_value, grid_thw):
                    if self.batch_size == 1:
                        patches = pixel_value.repeat(self.temporal_patch_size, 1, 1, 1)
                    elif self.batch_size % self.temporal_patch_size == 1:
                        repeat_image = pixel_value[-1:, ...].repeat(2, 1, 1, 1)
                        patches = torch.cat((pixel_value, repeat_image), dim=0)
                    else:
                        patches = pixel_value
                    #grid_t, grid_h, grid_w = grid_thw[0][0], grid_thw[0][1], grid_thw[0][2]
                    grid_t, grid_h, grid_w = N // 2 + N % 2, H//14, W//14
                    patches = patches.reshape(grid_t, self.temporal_patch_size, self.channel, 
                                            grid_h//self.merge_size, self.merge_size, self.patch_size, grid_w//self.merge_size, self.merge_size, self.patch_size)
                    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
                    flatten_patches = patches.reshape(grid_t * grid_h * grid_w, self.channel * self.temporal_patch_size * self.patch_size * self.patch_size)
                    
                    return self.vpm(flatten_patches, grid_thw)

            pixel_values = torch.randn(N, C, H, W) #, device=model.device, dtype=torch.float32)
            grid_thw = torch.tensor([[N // 2 + N % 2, H//14, W//14]], dtype=torch.int64)
            model = qwen2_5_vl_3b_vision(model, N)
            out = model(pixel_values, grid_thw)

            torch.onnx.export(model, 
                        (pixel_values, grid_thw), 
                        ONNX_PATH,
                        input_names=['pixel', 'grid_thw'],
                        #dynamic_axes={'pixel': {2: 'height', 3: 'width'}},
                        opset_version=15)

        elif MODEL_NAME in ['InternVL3_5-4B', 'InternVL3-1B']:

            pixel_values = torch.randn(N, C, H, W, device=model.device, dtype=torch.float32)
            model.forward = model.extract_feature
            model = model.to(torch.float32).eval()
            try:
                torch.onnx.export(model, pixel_values, ONNX_PATH)
            except torch.OutOfMemoryError:
                model = model.cpu()
                pixel_values = pixel_values.cpu()
                torch.onnx.export(model, pixel_values, ONNX_PATH)

        elif MODEL_NAME in ['SmolVLM-256M', 'SmolVLM-500M']:

            pixel_values = torch.randn(N, C, H ,W, device=DEVICE, dtype=torch.float32)        
            out = model(pixel_values)
            torch.onnx.export(model, 
                        pixel_values, 
                        ONNX_PATH,
                        input_names=['pixel'],
                        )
                
        print(f'Export vlm_vision_onnx to {ONNX_PATH}')
    except torch.OutOfMemoryError as e:
        print(e, 'try changing device to cpu')

# -------------------------------------------------------------------------------------------------------
# Convert Vision part of VLM to RKNN

if os.path.exists(RKNN_PATH):
    print(f'{RKNN_PATH} exist, skipping: Convert Vision part of VLM to RKNN')
else:
    rknn = RKNN(verbose=False)
    rknn.config(target_platform=PLATFORM, mean_values=MEAN, std_values=STD)
    rknn.load_onnx(ONNX_PATH)
    rknn.build(do_quantization=False, dataset=None)
    rknn.export_rknn(RKNN_PATH)

    print(f'Export vlm_vision_rknn to {RKNN_PATH}')

# -------------------------------------------------------------------------------------------------------
# Before Converting LLM part of VLM to RKNN, prepare dataset for calibarion

MEAN = MEAN[0]
STD = STD[0]

generate_calib = True
if os.path.exists(CALIBARION_DATASET_PATH):

    datasets = json.load(open("data/datasets.json", 'r'))

    try:
        existing_data = json.load(open(CALIBARION_DATASET_PATH, 'r'))
        if len(existing_data) == len(datasets):
            print(f"Calibration dataset already exists and matches size ({len(datasets)} samples). Skipping regeneration.")
            generate_calib = False
        else:
            print(f"Calibration dataset size mismatch ({len(existing_data)} vs {len(datasets)}). Regenerating...")
    except Exception as e:
        print(f"Failed to load existing calibration file ({e}). Regenerating...")

if generate_calib:
    if MODEL_NAME == 'Qwen2.5-VL':
        # use '/home/paiworker1/KX/.Workspace/.Tools/rknn-llm-main/examples/multimodal_model_demo/data/make_input_embeds_for_quantize.py' for references
        raise NotImplementedError

    elif MODEL_NAME in ['InternVL3_5-4B', 'InternVL3-1B']:

        def build_transform(input_size):
            transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD)
            ])
            return transform

        def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
            best_ratio_diff = float('inf')
            best_ratio = (1, 1)
            area = width * height
            for ratio in target_ratios:
                target_aspect_ratio = ratio[0] / ratio[1]
                ratio_diff = abs(aspect_ratio - target_aspect_ratio)
                if ratio_diff < best_ratio_diff:
                    best_ratio_diff = ratio_diff
                    best_ratio = ratio
                elif ratio_diff == best_ratio_diff:
                    if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                        best_ratio = ratio
            return best_ratio

        def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
            orig_width, orig_height = image.size
            aspect_ratio = orig_width / orig_height

            # calculate the existing image aspect ratio
            target_ratios = set(
                (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
                i * j <= max_num and i * j >= min_num)
            target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

            # find the closest aspect ratio to the target
            target_aspect_ratio = find_closest_aspect_ratio(
                aspect_ratio, target_ratios, orig_width, orig_height, image_size)

            # calculate the target width and height
            target_width = image_size * target_aspect_ratio[0]
            target_height = image_size * target_aspect_ratio[1]
            blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

            # resize the image
            resized_img = image.resize((target_width, target_height))
            processed_images = []
            for i in range(blocks):
                box = (
                    (i % (target_width // image_size)) * image_size,
                    (i // (target_width // image_size)) * image_size,
                    ((i % (target_width // image_size)) + 1) * image_size,
                    ((i // (target_width // image_size)) + 1) * image_size
                )
                # split the image
                split_img = resized_img.crop(box)
                processed_images.append(split_img)
            assert len(processed_images) == blocks
            if use_thumbnail and len(processed_images) != 1:
                thumbnail_img = image.resize((image_size, image_size))
                processed_images.append(thumbnail_img)
            return processed_images

        def load_image(image_file, input_size=448, max_num=12):
            image = Image.open(image_file).convert('RGB')
            transform = build_transform(input_size=input_size)
            images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
            pixel_values = [transform(image) for image in images]
            pixel_values = torch.stack(pixel_values)
            return pixel_values

        tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT_PATH, trust_remote_code=True, use_fast=False)
        assert H == W
        datasets = json.load(open("data/datasets.json", 'r'))

        with open(CALIBARION_DATASET_PATH, 'w') as json_file:
            json_file.write('[\n')
            first = True

            for i, data in enumerate(tqdm(datasets)):

                input_question = f"Image-1: <image>\n{data['input']}"
                input_ids = tokenizer(input_question, return_tensors="pt").input_ids.to(DEVICE)
                inputs_embeds = model.get_input_embeddings()(input_ids)
                pixel_values = load_image(os.path.join(data["image_path"], data["image"]), input_size=H, max_num=1).to(torch.float32).to(DEVICE)
                with torch.no_grad():
                    image_embeds = model.extract_feature(pixel_values)
                image_mask = input_ids == tokenizer.convert_tokens_to_ids("<image>")
                inputs_embeds[image_mask] = image_embeds
                inputs_embeds = inputs_embeds.cpu().detach().numpy()

                input_dict = {
                    "input_embed": inputs_embeds.tolist(),
                    "target": data["target"]
                }

                if not first:
                    json_file.write(',\n')
                else:
                    first = False

                json.dump(input_dict, json_file)

            json_file.write('\n]')
    
    elif MODEL_NAME in ['SmolVLM-256M', 'SmolVLM-500M']:

        processor = AutoProcessor.from_pretrained(MODEL_CKPT_PATH)
        datasets = json.load(open("data/datasets.json", 'r'))

        def move_inputs_to_device(inputs, device):
            """Move all tensors in processor output to the specified device."""
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    inputs[k] = v.to(device)
            return inputs

        def get_inputs_embeds(model, processor, data, device):
            """Generate combined text + image embeddings with fallback to CPU on OOM."""
            image = Image.open(os.path.join(data["image_path"], data["image"]))

            conversation = [

                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a helpful vision-language assistant that can understand and describe images accurately."
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": data["input"]},
                    ],
                }
            ]

            text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

            # Prepare input tensors
            inputs = processor(
                text=[text_prompt],
                images=[image],
                padding=True,
                return_tensors="pt"
            )

            try:
                # Move model + inputs to device
                model = model.to(device)
                inputs = move_inputs_to_device(inputs, device)

                # Text embeddings
                inputs_embeds = model.txtm.embed_tokens(inputs["input_ids"])

                # Image embeddings
                pixel_values = inputs["pixel_values"].type(model.vpm.dtype)
                image_mask = inputs["input_ids"] == model.vlm.config.image_token_id
                image_embeds = model.vlm.get_image_features(pixel_values).to(inputs_embeds.device)

            except torch.OutOfMemoryError:
                print("⚠️  GPU OOM detected, falling back to CPU...")
                torch.cuda.empty_cache()
                model = model.cpu()
                inputs = move_inputs_to_device(inputs, "cpu")

                inputs_embeds = model.txtm.embed_tokens(inputs["input_ids"])
                pixel_values = inputs["pixel_values"].type(model.vpm.dtype)
                image_mask = inputs["input_ids"] == model.vlm.config.image_token_id
                image_embeds = model.vlm.get_image_features(pixel_values).to(inputs_embeds.device)

            # Combine image + text embeddings
            inputs_embeds[image_mask] = image_embeds.view(-1, image_embeds.shape[-1])
            return inputs_embeds.cpu().detach().numpy()


        with open(CALIBARION_DATASET_PATH, 'w') as json_file:
            json_file.write('[\n')

            for i, data in enumerate(tqdm(datasets)):

                input_dict = {
                    "input_embed": get_inputs_embeds(model, processor, data, DEVICE).tolist(),
                    "target": data["target"]
                }

                if i > 0:
                    json_file.write(',\n')

                json.dump(input_dict, json_file)

            json_file.write('\n]')

# -------------------------------------------------------------------------------------------------------
# Convert LLM part of VLM to RKNN

if os.path.exists(RKLLM_PATH):
    print(f'{RKLLM_PATH} exist, skipping: Convert LLM part of VLM to RKNN')
else:
    llm = RKLLM()

    ret = llm.load_huggingface(model=snapshot_download(repo_id=MODEL_CKPT_PATH), device='cuda')
    assert ret == 0, f'ret = {ret}, Load model failed!'

    qparams = None
    ret = llm.build(
        do_quantization=DO_QUANT, 
        optimization_level=0, 
        quantized_dtype=QUANT_DTYPE,
        quantized_algorithm='normal', 
        target_platform=PLATFORM, 
        num_npu_core=NUM_NPU_CORE, 
        extra_qparams=qparams, 
        dataset=CALIBARION_DATASET_PATH,
        max_context=MAX_CONTEXT
    )
    assert ret == 0, f'ret = {ret}, Build model failed!'

    ret = llm.export_rkllm(RKLLM_PATH)
    assert ret == 0, f'ret = {ret}, Export model failed!'
