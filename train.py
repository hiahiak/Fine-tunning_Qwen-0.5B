from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.optim as optim
import inject
from data import qa_data
import json
import matplotlib.pyplot as plt
import os
import random
#from torch.optim.lr_scheduler import CosineAnnealingLR

# --- 1. 配置与初始化 ---
with open("config.json", "r") as f:
    config = json.load(f)

LR = config["lr"]
EPOCH = config["epoch"]
BATCH_SIZE = 4

os.makedirs("output", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
inject.model.to(device)

class QAdataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]

        # 1. 【核心策略】构建最简单的“问：... 答：...”格式
        if example.get("input") and example["input"].strip() != "":
            # 如果有上下文，格式为 "上下文\n问：问题\n答：答案"
            prompt_text = f"{example['input']}\n问：{example['instruction']}\n答："
        else:
            # 如果没有上下文，格式为 "问：问题\n答：答案"
            prompt_text = f"问：{example['instruction']}\n答："
        
        answer_text = example['output'] + self.tokenizer.eos_token
        
        # (后续的数据处理逻辑与上一版完全相同)
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False).input_ids
        answer_ids = self.tokenizer(answer_text, add_special_tokens=False).input_ids
        
        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
        
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
            
        pad_len = self.max_length - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * pad_len
        labels += [-100] * pad_len
        
        input_ids = torch.tensor(input_ids)
        labels = torch.tensor(labels)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

# 分为训练集、测试集 （0.8，0.2）
full_dataset = QAdataset(qa_data, inject.tokenizer)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
print(f"Dataset split: {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")

# --- 3. 训练过程  ---
train_param = filter(lambda p: p.requires_grad, inject.model.parameters())
optimizer = optim.AdamW(train_param, lr=LR,weight_decay=0.01)

train_losses, val_losses = [], []
best_val_loss, best_epoch = float('inf'), 0

print("--- Starting Training ---")
for epoch in range(EPOCH):
    inject.model.train()
    total_train_loss = 0
    for i,batch in enumerate(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        output = inject.model(**batch)
        loss = output.loss
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item() #gpu张量->cpu数字
    avg_train_loss = total_train_loss / len(train_loader) # average per batch
    train_losses.append(avg_train_loss)

    inject.model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            output = inject.model(**batch)
            loss = output.loss
            total_val_loss += loss.item()
    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    print(f"Epoch {epoch+1}/{EPOCH} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_epoch = epoch + 1
        if os.path.exists("output/best_model.pth"):
            os.remove("output/best_model.pth")
            print("已删除旧文件")
        print(f"  -> New best model found! Saving to output/best_model.pth")
        torch.save(inject.model.state_dict(), "output/best_model.pth")

print(f"--- Training Finished ---\nBest model from epoch {best_epoch} saved.")

# --- 4. 绘制损失曲线 ---
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.axvline(x=best_epoch-1, color='r', linestyle='--', label=f'Best Epoch ({best_epoch})')
plt.title("Training and Validation Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("output/loss_curve.png")
plt.show()

# --- 5. 使用聊天模板进行推理 ---
print("\n--- Loading best model for inference ---")
inject.model.load_state_dict(torch.load("output/best_model.pth", weights_only=True))
inject.model.eval()

# 准备符合聊天模板的输入
question = "hiahia是谁"
prompt = f"问：{question}\n答："

prompt_data = inject.tokenizer(prompt, return_tensors="pt").to(device)

print(f"\n[Diagnostic Info] Prompt sent to model:\n{prompt}")
print("\n--- Generating Response ---")
with torch.no_grad():
    output_ids = inject.model.generate(
        input_ids=prompt_data.input_ids,
        attention_mask=prompt_data.attention_mask,
        max_new_tokens=100,
        do_sample=False,
        num_beams=3, #使用束搜索
        repetition_penalty=1.1
    )

# 从模型输出中，只解码新生成的部分
input_length = prompt_data.input_ids.shape[1]
generated_ids = output_ids[0][input_length:]
result = inject.tokenizer.decode(generated_ids, skip_special_tokens=True)

print(f"\n\n--- Final Answer ---\n")
print(result)


