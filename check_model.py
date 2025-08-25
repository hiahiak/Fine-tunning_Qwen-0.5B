# check_model.py
from transformers import AutoModelForCausalLM
import json

with open('config.json','r') as f:
    config = json.load(f)

# 模型路径
MODEL_NAME = "./model" 
# 如果你之前下载的模型有问题，可以尝试直接从Hugging Face加载
# MODEL_NAME = config["model_name"] 

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
print(model)