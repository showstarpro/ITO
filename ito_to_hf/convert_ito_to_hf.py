import dataclasses
import os
import re
import sys
import torch

import open_clip
from open_clip.model import CLIPVisionCfg, CLIPTextCfg
from transformers import CLIPVisionConfig, CLIPTextConfig, CLIPConfig

from modeling_clip_native import NativeOpenCLIPModel


VISION_CONFIG_MAP = {
    "layers": "num_hidden_layers",
    "width": "hidden_size",
    "patch_size": "patch_size",
    "image_size": "image_size",
}

TEXT_CONFIG_MAP = {
    "layers": "num_hidden_layers",
    "width": "hidden_size",
    "heads": "num_attention_heads",
    "context_length": "max_position_embeddings",
    "vocab_size": "vocab_size",
}

STATE_DICT_PATTERNS = [
    # Vision
    (r"visual\.class_embedding", "vision_model.embeddings.class_embedding"),
    (r"visual\.positional_embedding", "vision_model.embeddings.position_embedding.weight"),
    (r"visual\.conv1\.(\w+)", "vision_model.embeddings.patch_embedding.{0}"),
    (r"visual\.ln_pre\.(\w+)", "vision_model.pre_layrnorm.{0}"),
    (r"visual\.ln_post\.(\w+)", "vision_model.post_layernorm.{0}"),
    (
        r"visual\.transformer\.resblocks\.(\d+)\.ln_1\.(\w+)",
        "vision_model.encoder.layers.{0}.layer_norm1.{1}",
    ),
    (
        r"visual\.transformer\.resblocks\.(\d+)\.ln_2\.(\w+)",
        "vision_model.encoder.layers.{0}.layer_norm2.{1}",
    ),
    (
        r"visual\.transformer\.resblocks\.(\d+)\.attn\.out_proj\.(\w+)",
        "vision_model.encoder.layers.{0}.self_attn.out_proj.{1}",
    ),
    (
        r"visual\.transformer\.resblocks\.(\d+)\.mlp\.c_fc\.(\w+)",
        "vision_model.encoder.layers.{0}.mlp.fc1.{1}",
    ),
    (
        r"visual\.transformer\.resblocks\.(\d+)\.mlp\.c_proj\.(\w+)",
        "vision_model.encoder.layers.{0}.mlp.fc2.{1}",
    ),
    # Text
    (r"token_embedding\.weight", "text_model.embeddings.token_embedding.weight"),
    (r"positional_embedding", "text_model.embeddings.position_embedding.weight"),
    (r"ln_final\.(\w+)", "text_model.final_layer_norm.{0}"),
    (
        r"transformer\.resblocks\.(\d+)\.ln_1\.(\w+)",
        "text_model.encoder.layers.{0}.layer_norm1.{1}",
    ),
    (
        r"transformer\.resblocks\.(\d+)\.ln_2\.(\w+)",
        "text_model.encoder.layers.{0}.layer_norm2.{1}",
    ),
    (
        r"transformer\.resblocks\.(\d+)\.attn\.out_proj\.(\w+)",
        "text_model.encoder.layers.{0}.self_attn.out_proj.{1}",
    ),
    (
        r"transformer\.resblocks\.(\d+)\.mlp\.c_fc\.(\w+)",
        "text_model.encoder.layers.{0}.mlp.fc1.{1}",
    ),
    (
        r"transformer\.resblocks\.(\d+)\.mlp\.c_proj\.(\w+)",
        "text_model.encoder.layers.{0}.mlp.fc2.{1}",
    ),
]


def convert_vision_config(config: CLIPVisionCfg):
    config = dataclasses.asdict(config)
    new_config = {
        "hidden_act": "quick_gelu",  # OpenCLIP 原生使用 quick_gelu
    }
    for key, value in config.items():
        if key in VISION_CONFIG_MAP:
            new_config[VISION_CONFIG_MAP[key]] = value
        elif key == "head_width":
            new_config["num_attention_heads"] = config["width"] // value
        elif key == "mlp_ratio":
            new_config["intermediate_size"] = int(config["width"] * value)
        elif not key.startswith("timm") and value:
            print(f"WARNING: Unknown vision key '{key}'.")

    return CLIPVisionConfig(**new_config)


def convert_text_config(config: CLIPTextCfg):
    config = dataclasses.asdict(config)
    new_config = {
        "hidden_act": "quick_gelu",
    }
    for key, value in config.items():
        if key in TEXT_CONFIG_MAP:
            new_config[TEXT_CONFIG_MAP[key]] = value
        elif key == "embed_dim":
            # embed_dim 用于投影层
            pass
        elif value and key not in ["hf_model_name", "hf_tokenizer_name", "hf_model_pretrained", "proj", "pooler_type", "embed_dim", "output_tokens"]:
            print(f"WARNING: Unknown text key '{key}'.")
    
    # 计算 intermediate_size (通常是 hidden_size * 4)
    if "intermediate_size" not in new_config and "width" in config:
        new_config["intermediate_size"] = config["width"] * 4

    return CLIPTextConfig(**new_config)


def remove_module_prefix(state_dict):
    """移除 module. 前缀（DDP/DataParallel 训练产生的）"""
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[new_key] = v
    return new_state_dict


def convert_state_dict(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        found = False
        
        # 跳过自定义模块
        if k.startswith("vision_pos") or k.startswith("sentence_transformer") or k.startswith("nrom"):
            print(f"[IGNORED] {k}")
            continue
        
        # 特殊处理：视觉注意力块
        if match := re.match(r"visual\.transformer\.resblocks\.(\d+)\.attn\.in_proj_(\w+)", k):
            # chunk weights into three
            chunks = v.chunk(3, dim=0)
            for proj_name, proj_v in zip(["q_proj", "k_proj", "v_proj"], chunks):
                new_k = f"vision_model.encoder.layers.{match.group(1)}.self_attn.{proj_name}.{match.group(2)}"
                print(k, "--->", new_k)
                new_state_dict[new_k] = proj_v
                found = True
        
        # 特殊处理：文本注意力块
        elif match := re.match(r"transformer\.resblocks\.(\d+)\.attn\.in_proj_(\w+)", k):
            chunks = v.chunk(3, dim=0)
            for proj_name, proj_v in zip(["q_proj", "k_proj", "v_proj"], chunks):
                new_k = f"text_model.encoder.layers.{match.group(1)}.self_attn.{proj_name}.{match.group(2)}"
                print(k, "--->", new_k)
                new_state_dict[new_k] = proj_v
                found = True
        
        # 转置视觉投影
        elif k == "visual.proj":
            new_k = "visual_projection.weight"
            print(k, "--->", new_k)
            new_state_dict[new_k] = v.T
            found = True
        
        # 转置文本投影
        elif k == "text_projection":
            new_k = "text_projection.weight"
            print(k, "--->", new_k)
            new_state_dict[new_k] = v.T
            found = True
        
        # logit_scale
        elif k == "logit_scale":
            new_k = "logit_scale"
            print(k, "--->", new_k)
            new_state_dict[new_k] = v
            found = True
        
        # 其他模式匹配
        else:
            for pattern, replacement in STATE_DICT_PATTERNS:
                if match := re.match(pattern, k):
                    new_k = replacement.format(*match.groups())
                    print(k, "--->", new_k)
                    new_state_dict[new_k] = v
                    found = True
                    break
        
        if not found:
            print(f"[UNMATCHED] {k}")

    return new_state_dict


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python convert_to_hf.py <model_name> <pretrained_path> [output_dir]")
        print("示例: python convert_to_hf.py ViT-B-16 /path/to/checkpoint.pt ./output")
        sys.exit(1)

    model_name = sys.argv[1]
    pretrained = sys.argv[2]
    output_base_dir = sys.argv[3] if len(sys.argv) > 3 else "models"  # 默认为 "models"

    print("=" * 60)
    print("步骤 1: 手动加载检查点...")
    print("=" * 60)
    checkpoint = torch.load(pretrained, map_location='cpu', weights_only=False)
    
    if 'state_dict' in checkpoint:
        full_state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        full_state_dict = checkpoint['model']
    else:
        full_state_dict = checkpoint
    
    print(f"检查点包含 {len(full_state_dict)} 个参数")
    
    # 移除 module. 前缀
    print("\n" + "=" * 60)
    print("步骤 2: 移除 module. 前缀...")
    print("=" * 60)
    full_state_dict = remove_module_prefix(full_state_dict)
    print(f"处理后的键示例: {list(full_state_dict.keys())[:3]}")
    
    print("\n" + "=" * 60)
    print("步骤 3: 创建标准 OpenCLIP 模型...")
    print("=" * 60)
    openclip_config = open_clip.get_model_config(model_name)
    openclip_model = open_clip.create_model(model_name, pretrained=False)
    
    print("\n" + "=" * 60)
    print("步骤 4: 过滤并加载匹配的权重...")
    print("=" * 60)
    model_keys = set(openclip_model.state_dict().keys())
    filtered_state_dict = {k: v for k, v in full_state_dict.items() if k in model_keys}
    
    print(f"标准模型需要: {len(model_keys)} 个参数")
    print(f"从检查点匹配: {len(filtered_state_dict)} 个参数")
    print(f"忽略的自定义参数: {len(full_state_dict) - len(filtered_state_dict)} 个")
    
    ignored_keys = set(full_state_dict.keys()) - model_keys
    if ignored_keys:
        print("\n被忽略的自定义模块:")
        custom_modules = {}
        for k in sorted(ignored_keys):
            prefix = k.split('.')[0]
            custom_modules[prefix] = custom_modules.get(prefix, 0) + 1
        for prefix, count in custom_modules.items():
            print(f"  - {prefix}: {count} 个参数")
    
    missing_keys, unexpected_keys = openclip_model.load_state_dict(filtered_state_dict, strict=False)
    
    if missing_keys:
        print(f"\n警告: {len(missing_keys)} 个键未找到（这是正常的，因为模型是随机初始化的）")
    
    print("\n" + "=" * 60)
    print("步骤 5: 创建 Hugging Face 配置...")
    print("=" * 60)
    
    # 转换配置
    vision_config = convert_vision_config(CLIPVisionCfg(**openclip_config["vision_cfg"]))
    text_config = convert_text_config(CLIPTextCfg(**openclip_config["text_cfg"]))
    
    # 直接创建 CLIPConfig
    config = CLIPConfig(
        vision_config=vision_config.to_dict(),
        text_config=text_config.to_dict(),
        projection_dim=openclip_config["embed_dim"],
    )
    
    print(f"✓ 视觉编码器: {vision_config.num_hidden_layers} 层")
    print(f"✓ 文本编码器: {text_config.num_hidden_layers} 层")
    print(f"✓ 投影维度: {openclip_config['embed_dim']}")

    print("\n" + "=" * 60)
    print("步骤 6: 转换权重...")
    print("=" * 60)
    state_dict = convert_state_dict(openclip_model.state_dict())

    print("\n" + "=" * 60)
    print("步骤 7: 创建 Hugging Face 模型...")
    print("=" * 60)
    model, loading_info = NativeOpenCLIPModel.from_pretrained(
        None, config=config, state_dict=state_dict, output_loading_info=True
    )
    print(loading_info)

    print("\n" + "=" * 60)
    print("步骤 8: 保存模型和处理器...")
    print("=" * 60)
    out_path = os.path.join(output_base_dir, f"ITO-{model_name}-data1b-10ep")
    os.makedirs(out_path, exist_ok=True)
    model.save_pretrained(out_path)
    
    # 保存图像处理器配置
    from transformers import CLIPImageProcessor
    image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
    image_processor.save_pretrained(out_path)
    
    print(f"✓ 模型已保存到: {out_path}")
    print(f"✓ 图像处理器已保存到: {out_path}")
    print("\n" + "=" * 60)
    print("转换完成！")
    print("=" * 60)
    
    # 保存转换信息
    info_file = os.path.join(out_path, "conversion_info.txt")
    with open(info_file, 'w') as f:
        f.write(f"原始检查点: {pretrained}\n")
        f.write(f"模型名称: {model_name}\n")
        f.write(f"总参数: {len(full_state_dict)}\n")
        f.write(f"转换参数: {len(filtered_state_dict)}\n")
        f.write(f"忽略参数: {len(ignored_keys)}\n")
        f.write(f"\n被忽略的自定义模块统计:\n")
        for prefix, count in custom_modules.items():
            f.write(f"  - {prefix}: {count} 个参数\n")
    
    print(f"✓ 转换信息已保存到: {info_file}")