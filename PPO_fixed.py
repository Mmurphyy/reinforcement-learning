import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CorrectedPPOTrainer:
    """修正版的PPO训练器"""
    
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.policy_model = AutoModelForCausalLM.from_pretrained(model_path)
        self.value_model = AutoModelForCausalLM.from_pretrained(model_path)
        
        self.policy_model.to(self.device)
        self.value_model.to(self.device)
        
    def compute_action_probabilities(self, prompts: List[str], responses: List[str]):
        """正确计算动作概率的方法"""
        
        # 1. 正确的方式：对完整的prompt+response计算概率
        full_texts = [prompt + response for prompt, response in zip(prompts, responses)]
        
        # 编码完整的文本
        full_inputs = self.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # 获取每个prompt的长度
        prompt_lengths = [len(self.tokenizer.encode(p)) for p in prompts]
        
        # 2. 计算完整的logits
        with torch.no_grad():
            full_logits = self.policy_model(**full_inputs).logits  # [batch_size, seq_len, vocab_size]
        
        # 3. 提取response部分的logits（这才是真正的动作概率）
        response_logits = []
        response_log_probs = []
        
        for i, (prompt_len, response) in enumerate(zip(prompt_lengths, responses)):
            # 获取response部分的logits
            response_tokens = self.tokenizer.encode(response)
            response_logits_i = full_logits[i, prompt_len:prompt_len+len(response_tokens)]
            
            # 计算每个token的概率
            response_probs_i = F.softmax(response_logits_i, dim=-1)
            response_log_probs_i = F.log_softmax(response_logits_i, dim=-1)
            
            # 获取实际选择的token的概率
            actual_tokens = full_inputs["input_ids"][i, prompt_len:prompt_len+len(response_tokens)]
            actual_log_probs = response_log_probs_i.gather(-1, actual_tokens.unsqueeze(-1)).squeeze(-1)
            
            response_logits.append(response_logits_i)
            response_log_probs.append(actual_log_probs)
        
        return response_logits, response_log_probs
    
    def compute_value(self, prompts: List[str], responses: List[str]):
        """
        正确计算价值函数

        返回的values是什么意思？
        返回的values是每个(prompt, response)对的“价值估计”，即value function的输出。
        在PPO等强化学习算法中，value用于衡量在当前状态（即给定prompt和response的上下文）下，模型后续能够获得的期望奖励。
        这里的实现是：对每个样本，将prompt和response拼接后输入value_model，取最后一个token位置的logits（对vocab的分数），
        然后对所有vocab维度取均值，作为该序列的value估计。返回的values是一个一维张量，每个元素对应一个样本的value。

        注意：实际应用中，value head通常是一个单独的线性层，这里只是用logits均值做近似。
        """
        full_texts = [prompt + response for prompt, response in zip(prompts, responses)]
        
        full_inputs = self.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # 计算完整序列的价值
        with torch.no_grad():
            values = self.value_model(**full_inputs).logits[:, -1, :].mean(dim=-1)
        
        return values  # 返回每个样本的value，代表该(prompt, response)对的期望奖励
    
    def train_step_corrected(self, batch: Dict):
        """修正的训练步骤"""
        prompts = batch["prompts"]
        responses = batch["responses"]
        rewards = batch["rewards"]
        
        # 1. 正确计算当前策略的动作概率
        current_logits, current_log_probs = self.compute_action_probabilities(prompts, responses)
        
        # 2. 正确计算价值
        values = self.compute_value(prompts, responses)
        
        # 3. 计算每个response的总对数概率
        response_total_log_probs = [log_probs.sum() for log_probs in current_log_probs]
        
        # 4. 计算优势（这里简化处理，实际应该用GAE）
        advantages = torch.tensor(rewards, device=self.device) - values.detach()
        
        # 5. PPO更新
        for _ in range(4):
            # 重新计算策略概率
            new_logits, new_log_probs = self.compute_action_probabilities(prompts, responses)
            new_response_total_log_probs = [log_probs.sum() for log_probs in new_log_probs]
            
            # 计算概率比率
            ratios = torch.exp(torch.stack(new_response_total_log_probs) - torch.stack(response_total_log_probs))
            
            # PPO clipped objective
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 0.8, 1.2) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 价值损失
            value_loss = F.mse_loss(values, torch.tensor(rewards, device=self.device))
            
            # 总损失
            total_loss = policy_loss + 0.5 * value_loss
            
            # 反向传播
            total_loss.backward()
            
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "total_loss": total_loss.item()
        }

# 使用示例
def demonstrate_correct_approach():
    """演示正确的PPO实现方法"""
    
    # 示例数据
    prompts = [
        "请解释什么是机器学习？",
        "写一个Python函数来计算斐波那契数列"
    ]
    
    responses = [
        "机器学习是人工智能的一个分支，它使计算机能够在没有明确编程的情况下学习和改进。",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"
    ]
    
    rewards = [0.8, 0.9]
    
    # 创建训练器
    trainer = CorrectedPPOTrainer("/home/zp/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B")
    
    # 演示正确的概率计算
    batch = {"prompts": prompts, "responses": responses, "rewards": rewards}
    
    print("=== 正确的PPO实现演示 ===")
    print("1. 对完整的prompt+response计算概率")
    print("2. 提取response部分的动作概率")
    print("3. 计算每个response的总概率")
    print("4. 基于完整上下文计算价值函数")
    
    # 执行训练步骤
    losses = trainer.train_step_corrected(batch)
    print(f"训练损失: {losses}")

if __name__ == "__main__":
    demonstrate_correct_approach() 