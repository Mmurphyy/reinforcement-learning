# DPO (Direct Preference Optimization) 训练代码
# 基于人类偏好数据的强化学习训练

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
from typing import List, Dict, Tuple
import os

# 1. 数据格式转换 - 将原始数据转换为DPO训练格式
def convert_to_dpo_format(raw_data: List[Dict]) -> List[Dict]:
    """
    将原始数据转换为DPO训练格式
    DPO格式: {"prompt": str, "chosen": str, "rejected": str}
    """
    dpo_data = []
    
    for item in raw_data:
        query = item["query"]
        base_response = item["base_response"]
        expanded_answer = item["expanded_answer"]
        
        # 将expanded_answer作为chosen（更好的回答）
        # 将base_response作为rejected（较差的回答）
        dpo_item = {
            "prompt": query,
            "chosen": expanded_answer,
            "rejected": base_response
        }
        dpo_data.append(dpo_item)
    
    return dpo_data

# 2. DPO数据集类
class DPODataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 编码prompt
        prompt_tokens = self.tokenizer(
            item["prompt"], 
            truncation=True, 
            max_length=self.max_length//2,
            padding="max_length",
            return_tensors="pt"
        )
        
        # 编码chosen和rejected回答
        chosen_tokens = self.tokenizer(
            item["chosen"],
            truncation=True,
            max_length=self.max_length//2,
            padding="max_length",
            return_tensors="pt"
        )
        
        rejected_tokens = self.tokenizer(
            item["rejected"],
            truncation=True,
            max_length=self.max_length//2,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "prompt_input_ids": prompt_tokens["input_ids"].squeeze(),
            "prompt_attention_mask": prompt_tokens["attention_mask"].squeeze(),
            "chosen_input_ids": chosen_tokens["input_ids"].squeeze(),
            "chosen_attention_mask": chosen_tokens["attention_mask"].squeeze(),
            "rejected_input_ids": rejected_tokens["input_ids"].squeeze(),
            "rejected_attention_mask": rejected_tokens["attention_mask"].squeeze(),
        }

# 3. DPO损失函数
class DPOLoss(nn.Module):
    def __init__(self, beta: float = 0.1):
        super().__init__()
        self.beta = beta
    
    def forward(self, policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps):
        """
        计算DPO损失
        """
        # 计算log概率差异
        chosen_logps = policy_chosen_logps - reference_chosen_logps
        rejected_logps = policy_rejected_logps - reference_rejected_logps
        
        # 计算DPO损失
        losses = -torch.log(torch.sigmoid(self.beta * (chosen_logps - rejected_logps)))
        return losses.mean()

# 4. DPO训练器
class DPOTrainer:
    def __init__(self, model_name: str, beta: float = 0.1, learning_rate: float = 5e-5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.beta = beta
        self.learning_rate = learning_rate
        
        # 加载模型和tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # 添加PEFT配置（LoRA）
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"]
        )
        self.model = get_peft_model(self.model, lora_config)
        
        # 创建参考模型（用于计算KL散度）
        self.reference_model = AutoModelForCausalLM.from_pretrained(model_name)
        self.reference_model.eval()
        
        # 移动到设备
        self.model.to(self.device)
        self.reference_model.to(self.device)
        
        # 损失函数和优化器
        self.criterion = DPOLoss(beta=beta)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
    
    def compute_logps(self, input_ids, attention_mask, model):
        """计算log概率"""
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # 计算每个token的log概率
            log_probs = torch.log_softmax(logits, dim=-1)
            
            # 获取实际token的log概率
            token_logps = torch.gather(log_probs, -1, input_ids.unsqueeze(-1)).squeeze(-1)
            
            # 根据attention_mask计算平均log概率
            masked_logps = token_logps * attention_mask
            total_logps = masked_logps.sum(dim=-1)
            token_counts = attention_mask.sum(dim=-1)
            
            return total_logps / token_counts
    
    def train_step(self, batch):
        """单步训练"""
        self.model.train()
        
        # 移动数据到设备
        prompt_input_ids = batch["prompt_input_ids"].to(self.device)
        prompt_attention_mask = batch["prompt_attention_mask"].to(self.device)
        chosen_input_ids = batch["chosen_input_ids"].to(self.device)
        chosen_attention_mask = batch["chosen_attention_mask"].to(self.device)
        rejected_input_ids = batch["rejected_input_ids"].to(self.device)
        rejected_attention_mask = batch["rejected_attention_mask"].to(self.device)
        
        # 计算策略模型的log概率
        policy_chosen_logps = self.compute_logps(
            chosen_input_ids, chosen_attention_mask, self.model
        )
        policy_rejected_logps = self.compute_logps(
            rejected_input_ids, rejected_attention_mask, self.model
        )
        
        # 计算参考模型的log概率
        reference_chosen_logps = self.compute_logps(
            chosen_input_ids, chosen_attention_mask, self.reference_model
        )
        reference_rejected_logps = self.compute_logps(
            rejected_input_ids, rejected_attention_mask, self.reference_model
        )
        
        # 计算DPO损失
        loss = self.criterion(
            policy_chosen_logps, policy_rejected_logps,
            reference_chosen_logps, reference_rejected_logps
        )
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, train_dataloader, num_epochs: int = 3):
        """训练循环"""
        print(f"开始DPO训练，共{num_epochs}个epoch...")
        
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0
            
            for batch in train_dataloader:
                loss = self.train_step(batch)
                total_loss += loss
                num_batches += 1
                
                if num_batches % 10 == 0:
                    print(f"Epoch {epoch+1}, Batch {num_batches}, Loss: {loss:.4f}")
            
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1} 完成，平均损失: {avg_loss:.4f}")
    
    def save_model(self, save_path: str):
        """保存模型"""
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"模型已保存到: {save_path}")

# 5. 主训练流程
def main():
    # 原始数据
    raw_data = [
        {
            "query": "阿拉斯加能吃红薯吗",
            "base_response": "阿拉斯加能吃红薯，但要控制好红薯的量。红薯是富含丰富营养的食物，也富含膳食纤维，对狗狗的胃肠道有一定的促进作用，也能补充维生素。但是阿拉斯加的胃肠道承受能力差，所以红薯也不能食用太多，否则易导致阿拉斯加腹泻和消化不良等问题。",
            "expanded_answer": "<think>\n嗯，用户问的是阿拉斯加能不能吃红薯。首先，我得确认红薯对狗狗来说是否安全。红薯含有丰富的营养，比如膳食纤维和维生素，这对狗狗的消化有好处。但阿拉斯加可能肠胃比较敏感，所以不能吃太多。\n\n接下来，我应该考虑用户可能的深层需求。他们可能想知道如何正确喂食红薯，或者担心过量带来的副作用。所以，回答时不仅要说明可以吃，还要给出具体建议，比如适量的分量和注意事项。\n\n然后，我需要整理这些信息，确保涵盖所有关键点：红薯的营养价值、适合的原因、适量原则、注意事项以及替代建议。这样用户就能全面了解，避免误区。\n\n最后，语言要简洁明了，结构清晰，让用户容易理解和应用这些建议。这样不仅回答了问题，还提供了实用的指导，帮助用户更好地照顾他们的宠物。\n</think>\n\n阿拉斯加犬是可以适量食用红薯的，但需要注意控制好红薯的摄入量。\n\n红薯是一种营养丰富的食物，含有以下对阿拉斯加犬有益的营养成分：\n1. 膳食纤维：有助于促进肠道蠕动，改善消化功能\n2. 维生素：包括维生素A、维生素C等，能为阿拉斯加提供必要的营养补充\n\n不过，由于阿拉斯加犬的胃肠道较为敏感，因此在喂食红薯时需要注意以下几点：\n\n1. 适量原则：红薯的分量不宜过多，建议控制在每日饮食的5%以内\n2. 注意事项：\n   - 建议将红薯蒸熟后喂食，避免高温油炸或添加调料\n   - 观察狗狗的消化情况，如出现腹泻、呕吐等症状应立即停止喂食\n   - 不要将红薯作为主食长期喂食\n\n如果担心红薯的消化问题，可以选择其他更容易消化的蔬果，如胡萝卜、南瓜等作为替代品。"
        },
        {
            "query": "猫咪经常出现抽搐式喘息，可能是什么原因？",
            "base_response": "建议主人带猫咪检查一下呼吸系统，同时排查出病因。例如有些猫咪会因为空气中有刺激性气体刺激到呼吸道，因此建议主人做好房间的通风换气工作以及日常的清洁卫生工作。如果是呼吸道感染或者疱疹病毒感染，则需要消除病因、对症治疗，并且做好猫咪的隔离工作，以免传染给其它猫咪。解决方案: 1.癫痫，癫安舒/左乙拉西坦。建议做高阶影像CT/MRI\n2.中毒，及时就医\n3.肝性脑病，血清中血氨异常\n4.脑膜炎",
            "expanded_answer": "<think>\n嗯，我现在需要帮助用户完善关于猫咪抽搐式喘息的原因和处理建议的回答。首先，我得仔细阅读用户提供的基础回答，确保所有信息点都被包含进去，同时保持逻辑性和条理性。\n\n基础回答中提到了呼吸系统的问题，比如刺激性气体、呼吸道感染和疱疹病毒。这些都是重要的原因，我应该把它们归类到呼吸系统疾病里，并详细解释每种情况。然后，解决方案部分有四个点：癫痫、中毒、肝性脑病和脑膜炎，这些属于神经系统疾病，需要分别说明。\n\n接下来，我需要考虑如何组织这些信息，使结构更清晰。可能分为原因分析和处理建议两部分。原因分析下再细分呼吸系统和神经系统问题，每个类别下详细列出可能的原因和症状。\n\n在处理建议部分，我应该给出具体的步骤，比如及时就医、环境改善、药物治疗和预防措施。这样用户可以一步步按照建议操作，更容易理解和实施。\n\n另外，我需要注意语言的口语化，避免使用过于专业的术语，让普通用户也能明白。同时，确保每个建议都有足够的解释，帮助用户理解背后的原因。\n\n最后，检查是否有遗漏的信息点，比如隔离患病猫咪的重要性，或者预防措施如疫苗接种和定期体检，这些都需要包含进去，确保回答全面。\n\n总结一下，我需要将信息重新组织，分成原因分析和处理建议，每个部分再细分，确保逻辑清晰，内容全面，同时语言通俗易懂，方便用户理解和操作。\n</think>\n\n猫咪出现抽搐式喘息，可能是由多种原因引起的，建议主人及时带猫咪就医并采取相应的处理措施。以下是可能的原因及建议：\n\n### 一、原因分析\n1. **呼吸系统疾病**\n   - **呼吸道刺激**：空气中的刺激性气体（如二手烟、清洁剂等）可能刺激猫咪的呼吸道，导致喘息。建议保持室内通风，做好日常清洁。\n   - **呼吸道感染**：细菌或病毒感染（如猫疱疹病毒）可能导致呼吸道炎症，出现喘息症状。需要对症治疗并隔离患病猫咪，防止传染。\n\n2. **神经系统疾病**\n   - **癫痫**：癫痫发作时猫咪可能出现抽搐和喘息。建议进行CT或MRI检查，必要时使用抗癫痫药物（如癫安舒或左乙拉西坦）。\n   - **中毒**：误食有毒物质可能导致神经系统异常，出现抽搐和喘息。需立即就医。\n   - **肝性脑病**：肝脏功能异常导致血氨水平升高，可能引起神经症状，包括抽搐和喘息。\n   - **脑膜炎**：脑部感染可能导致中枢神经系统异常，出现抽搐等症状。\n\n### 二、处理建议\n1. **及时就医**：尽快带猫咪到宠物医院进行全面检查，明确病因。\n2. **改善环境**：保持室内空气流通，减少刺激性气体，做好日常清洁。\n3. **药物治疗**：根据诊断结果，在兽医指导下使用相应药物（如抗癫痫药、抗生素等）。\n4. **隔离患病猫咪**：特别是呼吸道感染或疱疹病毒感染时，避免传染给其他猫咪。\n5. **预防措施**：定期为猫咪接种疫苗，保持良好的卫生习惯，定期体检。\n\n希望以上信息能帮助您更好地了解猫咪的健康问题，并采取适当的措施。"
        }
    ]
    
    # 转换为DPO格式
    print("正在转换数据格式...")
    dpo_data = convert_to_dpo_format(raw_data)
    print(f"成功转换 {len(dpo_data)} 条DPO训练数据")
    
    # 保存DPO格式数据
    with open("dpo_training_data.json", "w", encoding="utf-8") as f:
        json.dump(dpo_data, f, ensure_ascii=False, indent=2)
    print("DPO训练数据已保存到 dpo_training_data.json")
    
    # 初始化DPO训练器
    model_name = "THUDM/chatglm2-6b"  # 可以根据需要更换模型
    print(f"正在加载模型: {model_name}")
    
    try:
        dpo_trainer = DPOTrainer(
            model_name=model_name,
            beta=0.1,  # DPO温度参数
            learning_rate=5e-5
        )
        
        # 创建数据集和数据加载器
        dataset = DPODataset(dpo_data, dpo_trainer.tokenizer, max_length=512)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        
        # 开始训练
        dpo_trainer.train(dataloader, num_epochs=3)
        
        # 保存训练好的模型
        dpo_trainer.save_model("./dpo_trained_model")
        
        print("DPO训练完成！")
        
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        print("请检查模型名称是否正确，或者尝试使用其他可用的模型")

if __name__ == "__main__":
    main()