from open_clip.factory import create_model_from_pretrained, get_model_config, get_tokenizer
import argparse
import torch
import timm

parser = argparse.ArgumentParser(description="Push to Hugging Face Hub")
parser.add_argument(
    "--model", default='ViT-B-16' ,type=str, help="Name of the model to use.",
)
parser.add_argument(
    "--pretrained", default='/lpai/open_clip-main/src/logs/clip-vitb16-cc12m-epochs25/checkpoints/epoch_25.pt' ,type=str,
    help="Use a pretrained CLIP model weights with the specified tag or file path.",
)
parser.add_argument(
    "--repo-id", type=str,
    help="Destination HF Hub repo-id ie 'organization/model_id'.",
)
parser.add_argument(
    "--precision", type=str, default='amp',
)
parser.add_argument(
    '--image-mean', type=float, nargs='+', default=None, metavar='MEAN',
    help='Override default image mean value of dataset')
parser.add_argument(
    '--image-std', type=float, nargs='+', default=None, metavar='STD',
    help='Override default image std deviation of of dataset')
parser.add_argument(
    '--image-interpolation',
    default=None, type=str, choices=['bicubic', 'bilinear', 'random'],
    help="image resize interpolation"
)
parser.add_argument(
    '--image-resize-mode',
    default=None, type=str, choices=['shortest', 'longest', 'squash'],
    help="image resize mode during inference"
)
parser.add_argument(
    "--hf-tokenizer-self",
    default=False,
    action="store_true",
    help="make hf_tokenizer_name point in uploaded config point to itself"
)
args = parser.parse_args()

print(f'Saving model {args.model} with pretrained weights {args.pretrained} to Hugging Face Hub at {args.repo_id}')



model_name = args.model
pretrained = args.pretrained
precision = args.precision
image_mean = args.image_mean
image_std = args.image_std
image_interpolation = args.image_interpolation
image_resize_mode = args.image_resize_mode


model, preprocess_eval = create_model_from_pretrained(
    model_name,
    pretrained=pretrained,
    precision=precision,
    image_mean=image_mean,
    image_std=image_std,
    image_interpolation=image_interpolation,
    image_resize_mode=image_resize_mode
)

visual_keyword = 'transformer.resblocks'

visual_model_clip = model.visual
inputs = torch.randn(2, 3, 224, 224)
txt_inputs = torch.randn(2, 77)
outputs = model(inputs, txt_inputs)


# rename CLIP pre-trained keys
state_dict = model.visual.state_dict()
linear_keyword = 'head'
new_state_dict = {}
for k in list(state_dict.keys()):
    # retain only base_encoder up to before the embedding layer
    if 'class_embedding' in k:
        new_k = 'vision_model.embeddings.' + 'class_embedding'
        new_state_dict[new_k] = state_dict[k]
    if 'positional_embedding' in k:
        new_k = 'vision_model.embeddings.' + 'position_embedding.weight'
        new_state_dict[new_k] = state_dict[k]
    if 'conv1.weight' in k :
        new_k = 'vision_model.embeddings.' + 'patch_embedding.weight'
        new_state_dict[new_k] = state_dict[k]
    if 'ln_pre' in k:
        new_k = 'vision_model.' + 'pre_layrnorm.' + k[len('ln_pre'):]
        new_state_dict[new_k] = state_dict[k]
    if 'resblocks' in k:
        old_k = k.replace('transformer.resblocks', 'vision_model.encoder.layers')
        if 'ln_1' in k:
            new_k = old_k.replace('ln_1', 'layer_norm1')
            new_state_dict[new_k] = state_dict[k]
        if 'attn.in_proj' in k:
            new_k = old_k.replace('attn.in_proj_', 'self_attn.k_proj.')
            new_state_dict[new_k] = state_dict[k]
            new_k = old_k.replace('attn.in_proj_', 'self_attn.v_proj.')
            new_state_dict[new_k] = state_dict[k]
            new_k = old_k.replace('attn.in_proj_', 'self_attn.q_proj.')
            new_state_dict[new_k] = state_dict[k]
        if 'attn.out_proj' in k:
            new_k = old_k.replace('attn.out_proj', 'self_attn.out_proj')
            new_state_dict[new_k] = state_dict[k]
        if 'ln_2' in k:
            new_k = old_k.replace('ln_2', 'layer_norm2')
            new_state_dict[new_k] = state_dict[k]
        if 'c_fc' in k:
            new_k = old_k.replace('c_fc', 'fc1')
            new_state_dict[new_k] = state_dict[k]
        if 'c_proj' in k:
            new_k = old_k.replace('c_proj', 'fc2')
            new_state_dict[new_k] = state_dict[k]


# create model
# blocks.0.mlp.fc2.weight -> transformer.resblocks.0.mlp.c_proj.weight
# c_fc -> fc1  c_proj -> fc2 
# transformer.resblocks.0.attn.in_proj_weight
# vision_model.encoder.layers.0.self_attn.k_proj.weight

# state_dict = torch.load(pretrained)

# model_config = get_model_config(model_name)

# state_dict['state_dict']['module.ln_final.bias'] == model.cpu().state_dict()['ln_final.bias']


from transformers import CLIPModel, CLIPImageProcessor, CLIPVisionModel
path = "/lpai/volumes/so-volume-ga/models/clip-vit-base-patch16"
image_processor = CLIPImageProcessor.from_pretrained(path)
vision_tower = CLIPVisionModel.from_pretrained(path)
clip_model = CLIPModel.from_pretrained(path)

msg = vision_tower.load_state_dict(new_state_dict, strict=False)

print(msg)