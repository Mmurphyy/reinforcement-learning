import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import numpy as np
from typing import List, Dict, Tuple
import logging
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RewardCalculator:
    """奖励计算器 - 集成多种评估指标"""
    
    def __init__(self):
        self.sentence_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
    def compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """计算语义相似度"""
        embeddings = self.sentence_encoder.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)
    
    def compute_length_penalty(self, text: str, target_length: int = 100) -> float:
        """计算长度惩罚"""
        current_length = len(text)
        penalty = 1.0 - abs(current_length - target_length) / target_length
        return max(0.0, penalty)
    
    def check_safety(self, text: str) -> float:
        """简单的安全性检查"""
        harmful_patterns = [
            r'暴力|暴力行为',
            r'歧视|种族歧视',
            r'仇恨|仇恨言论',
            r'自杀|自残',
            r'非法|违法'
        ]
        
        for pattern in harmful_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return 0.0
        return 1.0
    
    def compute_fluency_score(self, text: str) -> float:
        """简单的流畅性评分（基于标点符号和句子结构）"""
        # 简单的启发式方法
        sentences = text.split('。')
        if len(sentences) < 2:
            return 0.5
        
        # 检查句子长度分布
        sentence_lengths = [len(s) for s in sentences if s.strip()]
        if not sentence_lengths:
            return 0.5
            
        avg_length = np.mean(sentence_lengths)
        # 理想句子长度在10-50之间
        if 10 <= avg_length <= 50:
            return 1.0
        elif 5 <= avg_length <= 80:
            return 0.8
        else:
            return 0.4

class RewardModel(nn.Module):
    """奖励模型 - 基于预训练模型微调"""
    
    def __init__(self, base_model_path: str):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            base_model_path, 
            num_labels=1
        )
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def forward(self, prompt: str, response: str) -> torch.Tensor:
        full_text = prompt + " [SEP] " + response
        inputs = self.tokenizer(
            full_text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512,
            padding=True
        )
        
        reward = self.model(**inputs).logits.squeeze()
        return torch.sigmoid(reward)  # 归一化到[0,1]

class RealisticPPOTrainer:
    """基于真实奖励的PPO训练器"""
    
    def __init__(self, model_path: str, reward_model_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.policy_model = AutoModelForCausalLM.from_pretrained(model_path)
        self.value_model = AutoModelForCausalLM.from_pretrained(model_path)
        
        self.policy_model.to(self.device)
        self.value_model.to(self.device)
        
        # 初始化奖励计算器
        self.reward_calculator = RewardCalculator()
        
        # 可选的奖励模型
        if reward_model_path:
            self.reward_model = RewardModel(reward_model_path)
            self.reward_model.to(self.device)
        else:
            self.reward_model = None
    
    def compute_realistic_reward(self, prompt: str, response: str, 
                               reference_response: str = None) -> float:
        """计算真实的奖励值"""
        
        # 1. 基础指标计算
        relevance_score = 0.5  # 默认值，实际应该基于prompt计算
        fluency_score = self.reward_calculator.compute_fluency_score(response)
        safety_score = self.reward_calculator.check_safety(response)
        length_penalty = self.reward_calculator.compute_length_penalty(response)
        
        # 2. 如果有参考答案，计算事实性
        if reference_response:
            factual_score = self.reward_calculator.compute_semantic_similarity(
                response, reference_response
            )
        else:
            factual_score = 0.5
        
        # 3. 使用奖励模型（如果有）
        if self.reward_model:
            with torch.no_grad():
                model_reward = self.reward_model(prompt, response).item()
        else:
            model_reward = 0.5
        
        # 4. 综合计算最终奖励
        final_reward = (
            0.25 * relevance_score +
            0.20 * fluency_score +
            0.20 * safety_score +
            0.15 * factual_score +
            0.10 * length_penalty +
            0.10 * model_reward
        )
        
        return final_reward
    
    def compute_action_probabilities(self, prompts: List[str], responses: List[str]):
        """
        【详细中文解释】

        你问的几个问题如下：
        1. 这个函数是怎么计算概率的？
        2. 计算response的logits，是计算整个response的logits吗？
        3. 最终返回的是什么？是整个response中每个单词的logits吗？
        4. logits和log_probs是什么关系？

        【逐条解答】

        1. 这个函数的作用是：对于每个(prompt, response)对，计算response部分每个token的logits（未归一化分数）和实际生成token的对数概率（log_prob）。
           - logits：是模型在每个位置上对所有词的打分（还没经过softmax归一化）。
           - log_probs：是logits经过log_softmax归一化后，得到的对数概率分布。

        2. 计算response的logits，是指：对于每个样本，把prompt和response拼接后送入模型，取出response部分（即从prompt后面开始，长度为response token数）的logits。
           - 例如：prompt长度为10，response长度为5，则取第10~14位置的logits。

        3. 最终返回的内容：
           - response_logits: List[Tensor]，每个元素形状为[response长度, vocab_size]，即response每个token位置上，对所有词的logits分数。
           - response_log_probs: List[Tensor]，每个元素形状为[response长度]，即response每个token实际生成的token的对数概率（log_prob）。
             例如response为"你好"，则返回"你"和"好"两个token在各自位置被选中的log_prob。

        4. logits和log_probs的关系：
           - logits是原始分数，log_probs = F.log_softmax(logits, dim=-1)，即对logits做softmax归一化后再取对数，得到每个词的对数概率分布。

        【代码流程】
        1. 拼接prompt+response，编码成模型输入。
        2. 前向推理，得到完整序列的logits。
        3. 对于每个样本，定位response部分的logits（即只取response对应的token位置）。
        4. 对response部分的logits做log_softmax，得到每个位置所有词的log_probs。
        5. 用gather操作，提取出每个位置实际生成token的log_prob（即模型在该位置实际选择的token的概率）。
        6. 返回response部分的logits（每个token的logits）和实际生成token的log_probs。

        这些log_probs在PPO中用于计算策略损失和重要性采样比率。

        【返回值举例】
        - response_logits: [Tensor([token1_logits, token2_logits, ...]), ...]
        - response_log_probs: [Tensor([token1_log_prob, token2_log_prob, ...]), ...]
        """
        # 1. 拼接prompt+response，编码输入
        full_texts = [prompt + response for prompt, response in zip(prompts, responses)]
        full_inputs = self.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        # 2. 计算每个prompt的长度，用于定位response部分
        prompt_lengths = [len(self.tokenizer.encode(p)) for p in prompts]

        # 3. 前向推理，得到完整序列的logits
        with torch.no_grad():
            full_logits = self.policy_model(**full_inputs).logits  # [batch, seq_len, vocab_size]

        response_logits = []
        response_log_probs = []

        # 4. 对每个样本，提取response部分的logits和log_probs
        for i, (prompt_len, response) in enumerate(zip(prompt_lengths, responses)):
            response_tokens = self.tokenizer.encode(response)
            # 取出response部分的logits，形状[response长度, vocab_size]
            response_logits_i = full_logits[i, prompt_len:prompt_len+len(response_tokens)]
            # 对logits做log_softmax，得到对数概率分布
            response_log_probs_i = F.log_softmax(response_logits_i, dim=-1)
            # 取出实际生成的token id
            actual_tokens = full_inputs["input_ids"][i, prompt_len:prompt_len+len(response_tokens)]
            # 用gather提取每个位置实际生成token的log_prob
            actual_log_probs = response_log_probs_i.gather(-1, actual_tokens.unsqueeze(-1)).squeeze(-1)
            # 保存
            response_logits.append(response_logits_i)
            response_log_probs.append(actual_log_probs)

        return response_logits, response_log_probs
    
    def compute_value(self, prompts: List[str], responses: List[str]):
        """
        这是什么意思

        该函数用于计算每个 (prompt, response) 对的“价值函数”估计值（value）。
        在PPO等强化学习算法中，价值函数用于衡量在给定状态下，模型后续能够获得的期望奖励。

        具体做法：
        - 对每个prompt和response拼接成完整输入，编码后送入value_model。
        - value_model输出logits（通常是语言模型的输出），这里取最后一个token位置的logits，并对其所有vocab维度取均值，作为该序列的value估计。
        - 返回每个样本的value。

        注意：这里的实现是用最后一个token的logits均值作为value，实际应用中可能会有更合理的value head设计。
        """
        full_texts = [prompt + response for prompt, response in zip(prompts, responses)]
        
        full_inputs = self.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            values = self.value_model(**full_inputs).logits[:, -1, :].mean(dim=-1)
        
        return values
    
    def train_step_with_real_rewards(self, batch: Dict):
        """使用真实奖励的训练步骤"""
        prompts = batch["prompts"]
        responses = batch["responses"]
        reference_responses = batch.get("reference_responses", [None] * len(prompts))
        
        # 计算真实奖励
        real_rewards = []
        for prompt, response, ref_response in zip(prompts, responses, reference_responses):
            reward = self.compute_realistic_reward(prompt, response, ref_response)
            real_rewards.append(reward)
        
        logger.info(f"计算的真实奖励: {real_rewards}")
        
        # 后续训练步骤与之前相同
        current_logits, current_log_probs = self.compute_action_probabilities(prompts, responses)
        values = self.compute_value(prompts, responses)
        
        response_total_log_probs = [log_probs.sum() for log_probs in current_log_probs]
        advantages = torch.tensor(real_rewards, device=self.device) - values.detach()
        
        # PPO更新
        for _ in range(4):
            new_logits, new_log_probs = self.compute_action_probabilities(prompts, responses)
            new_response_total_log_probs = [log_probs.sum() for log_probs in new_log_probs]
            
            ratios = torch.exp(torch.stack(new_response_total_log_probs) - torch.stack(response_total_log_probs))
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 0.8, 1.2) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            value_loss = F.mse_loss(values, torch.tensor(real_rewards, device=self.device))
            total_loss = policy_loss + 0.5 * value_loss
            
            total_loss.backward()
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "total_loss": total_loss.item(),
            "rewards": real_rewards
        }

def demonstrate_realistic_rewards():
    """演示真实奖励计算"""
    
    prompts = [
        "请解释什么是机器学习？",
        "写一个Python函数来计算斐波那契数列",
        "如何保护环境？"
    ]
    
    responses = [
        "机器学习是人工智能的一个分支，它使计算机能够在没有明确编程的情况下学习和改进。通过分析大量数据，机器学习算法可以发现模式并做出预测。",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "保护环境需要从多个方面入手：减少碳排放、节约用水、垃圾分类、使用可再生能源等。"
    ]
    
    reference_responses = [
        "机器学习是让计算机从数据中学习的技术。",
        "def fib(n): return n if n < 2 else fib(n-1) + fib(n-2)",
        "保护环境的方法包括减少污染、节约资源、植树造林等。"
    ]
    
    # 创建训练器
    trainer = RealisticPPOTrainer("/home/zp/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B")
    
    batch = {
        "prompts": prompts, 
        "responses": responses, 
        "reference_responses": reference_responses
    }
    
    print("=== 真实奖励计算演示 ===")
    print("1. 计算语义相似度")
    print("2. 评估流畅性")
    print("3. 检查安全性")
    print("4. 计算长度惩罚")
    print("5. 综合评分")
    
    # 执行训练步骤
    losses = trainer.train_step_with_real_rewards(batch)
    print(f"训练损失: {losses}")

if __name__ == "__main__":
    demonstrate_realistic_rewards() 