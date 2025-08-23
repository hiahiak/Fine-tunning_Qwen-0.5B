
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
    for name, _ in model.named_modules():
        if any(target in name for target in target_modules):
            layer_index = int(name.split('.')[2])
            if layer_index not in target_layers:
                continue
            #定位到父模块
            parent_name = '.'.join(name.split('.')[:-1])
            parent_module = model.get_submodule(parent_name)
            #原始线性层
            orig_linear_name = name.split('.')[-1]
            orig_linear = getattr(parent_module,orig_linear_name)
            #修改替换
            lora_linear = LoRALinearLayer(orig_linear,rank,alpha)
            setattr(parent_module,orig_linear_name,lora_linear)
            print(f"Injecting into module: {name}")

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
