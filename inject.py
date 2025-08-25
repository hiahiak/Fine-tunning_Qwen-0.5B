
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from LoRALinear import LoRALinearLayer

with open('config.json','r') as f:
    config = json.load(f)

# 配置参数
MODEL_NAME = "./model"
LORA_RANK = config["lora_rank"]
LORA_ALPHA = config["lora_alpha"]
TARGET_MODULES = config["target_modules"]
TARGET_LAYERS = config["target_layers"]

# 加载分词器 模型
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,device_map="auto")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# 模型注入逻辑
def inject(model, rank, alpha, target_modules, target_layers):
    print("\nStarting LoRA injection...")
    injected_count = 0
    # Qwen 的正确路径是 model.layers
    for name, module in model.named_modules():
        # 名字示例: model.layers.23.self_attn.q_proj
        if any(target in name for target in target_modules):
            try:
                name_parts = name.split('.')
                # 确认路径结构符合预期
                if name_parts[0] == 'model' and name_parts[1] == 'layers' and name_parts[3] == 'self_attn':
                    layer_index = int(name_parts[2])
                else:
                    continue # 结构不匹配，跳过
            except (ValueError, IndexError):
                continue
            
            if layer_index in target_layers:
                parent_name = '.'.join(name_parts[:-1])
                parent_module = model.get_submodule(parent_name)
                orig_linear_name = name_parts[-1]
                orig_linear = getattr(parent_module, orig_linear_name)
                
                lora_linear = LoRALinearLayer(orig_linear, rank, alpha)
                setattr(parent_module, orig_linear_name, lora_linear)
                
                print(f"  -> Successfully injected LoRA into: {name}")
                injected_count += 1
    print(f"Injection finished. Total modules injected: {injected_count}")

inject(model,LORA_RANK,LORA_ALPHA,TARGET_MODULES,TARGET_LAYERS)

# 冻结原模型中参数
def freeze_params(model):
    total_params = 0
    trained_params = 0
    for name,param in model.named_parameters():
        total_params += param.numel()
        if "lora_" in name:
            trained_params += param.numel()
            param.requires_grad = True
        else:
            param.requires_grad = False
    print(f"total params:{total_params},trained params:{trained_params}")

freeze_params(model)
