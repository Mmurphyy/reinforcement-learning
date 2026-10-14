import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers import Trainer, TrainingArguments
import numpy as np
import json
import random
from typing import List, Dict, Tuple, Optional
import os
from dataclasses import dataclass
from tqdm import tqdm
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PPOTrainingConfig:
    """PPO训练配置"""
    model_path: str = "/home/zp/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B"
    max_length: int = 512
    batch_size: int = 4
    learning_rate: float = 1e-5
    num_epochs: int = 3
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    gamma: float = 0.99
    gae_lambda: float = 0.95
    max_grad_norm: float = 1.0
    save_steps: int = 500
    eval_steps: int = 100
    warmup_steps: int = 100
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    output_dir: str = "./ppo_output"
    seed: int = 42

class RewardModel(nn.Module):
    """奖励模型"""
    def __init__(self, base_model_path: str):
        super().__init__()
        self.config = AutoConfig.from_pretrained(base_model_path)
        self.config.num_labels = 1  # 回归任务
        self.model = AutoModelForCausalLM.from_pretrained(base_model_path, config=self.config)
        
        # 添加奖励头
        self.reward_head = nn.Linear(self.config.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # 使用最后一个token的隐藏状态
        last_hidden_state = outputs.hidden_states[-1]
        last_token_hidden = last_hidden_state[:, -1, :]  # [batch_size, hidden_size]
        reward = self.reward_head(last_token_hidden)  # [batch_size, 1]
        return reward

class PPOTrainer:
    """PPO训练器"""
    def __init__(self, config: PPOTrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 设置随机种子
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        random.seed(config.seed)
        
        # 加载tokenizer和模型
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.policy_model = AutoModelForCausalLM.from_pretrained(config.model_path)
        self.value_model = AutoModelForCausalLM.from_pretrained(config.model_path)
        self.reward_model = RewardModel(config.model_path)
        
        # 移动到设备
        self.policy_model.to(self.device)
        self.value_model.to(self.device)
        self.reward_model.to(self.device)
        
        # 优化器
        self.policy_optimizer = torch.optim.AdamW(
            self.policy_model.parameters(), 
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.value_optimizer = torch.optim.AdamW(
            self.value_model.parameters(), 
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 学习率调度器
        self.policy_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.policy_optimizer, T_max=config.num_epochs
        )
        self.value_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.value_optimizer, T_max=config.num_epochs
        )
        
    def generate_training_data(self, num_samples: int = 1000) -> List[Dict]:
        """生成训练数据"""
        logger.info(f"生成 {num_samples} 个训练样本...")
        
        # 定义一些示例任务和提示
        tasks = [
            {
                "task": "问答",
                "prompts": [
                    "请解释什么是机器学习？",
                    "什么是深度学习？",
                    "强化学习的基本原理是什么？",
                    "请介绍一下神经网络。",
                    "什么是过拟合？如何避免？"
                ]
            },
            {
                "task": "代码生成",
                "prompts": [
                    "写一个Python函数来计算斐波那契数列",
                    "实现一个快速排序算法",
                    "写一个函数来检查字符串是否是回文",
                    "实现一个简单的神经网络类",
                    "写一个函数来查找数组中的最大值"
                ]
            },
            {
                "task": "文本摘要",
                "prompts": [
                    "请总结以下文章的主要内容：人工智能正在改变我们的生活方式...",
                    "请概括这段文字的核心观点：机器学习算法通过分析大量数据...",
                    "请简要说明以下内容：深度学习是机器学习的一个分支...",
                    "请总结这段技术文档：强化学习是一种通过与环境交互来学习最优策略的方法...",
                    "请概括以下研究论文的要点：Transformer架构在自然语言处理领域取得了突破性进展..."
                ]
            },
            {
                "task": "情感分析",
                "prompts": [
                    "分析这句话的情感倾向：今天天气真好，心情很愉快。",
                    "判断这句话的情感：这个产品质量太差了，非常失望。",
                    "分析情感：这部电影太精彩了，强烈推荐！",
                    "判断情感倾向：今天工作很累，但是很有成就感。",
                    "分析这句话的情感：这个餐厅的服务态度让人很不满意。"
                ]
            }
        ]
        
        training_data = []
        
        for _ in range(num_samples):
            # 随机选择一个任务
            task = random.choice(tasks)
            prompt = random.choice(task["prompts"])
            
            # 生成一些变体
            prompt_variants = [
                prompt,
                f"请回答：{prompt}",
                f"问题：{prompt}",
                f"任务：{prompt}",
                f"请详细解释：{prompt}"
            ]
            
            selected_prompt = random.choice(prompt_variants)
            
            # 生成一些示例回答（这里简化处理，实际应用中应该有更复杂的生成逻辑）
            responses = [
                f"这是一个关于{task['task']}的回答示例。",
                f"根据我的理解，{prompt}的答案是...",
                f"让我来回答这个问题：{prompt}",
                f"关于{task['task']}，我的回答是...",
                f"这是一个很好的问题，让我详细解释一下..."
            ]
            
            # 为每个回答分配奖励分数
            rewards = [random.uniform(0.3, 0.9) for _ in responses]
            
            training_data.append({
                "prompt": selected_prompt,
                "responses": responses,
                "rewards": rewards,
                "task": task["task"]
            })
        
        logger.info(f"成功生成 {len(training_data)} 个训练样本")
        return training_data
    
    def compute_advantages(self, rewards: torch.Tensor, values: torch.Tensor, 
                          dones: torch.Tensor) -> torch.Tensor:
        """
        详细中文解释（通俗易懂版）：

        这个函数的作用是：计算每一步的“优势”（Advantage），也就是当前动作到底比平均水平好多少。
        在PPO等强化学习算法中，优势越大，说明当前动作越值得被采纳。

        这里用的是GAE（广义优势估计，Generalized Advantage Estimation）方法，目的是让优势的估计更平滑、更稳定。

        参数说明：
        - rewards: 每一步的即时奖励（比如模型生成一句话后得到的分数），形状是[T]
        - values: 每一步的价值函数估计（模型对当前状态的“好坏”预测），形状是[T]
        - dones: 每一步是否为episode结束（True/False），形状是[T]

        计算思路（用大白话解释）：
        1. 我们要从最后一步往前算，因为优势的计算要用到“后面”的信息。
        2. 对于每一步t，先算一个delta，表示“当前奖励+下一个状态的价值-当前状态的价值”。
           - 如果这一步是终止（dones[t]=True），那后面就没了，所以delta = rewards[t] - values[t]
           - 如果不是终止，delta = rewards[t] + gamma * 下一个状态的价值 - values[t]
        3. 优势的递推公式是：
           - last_advantage = delta + gamma * lambda * 上一步的last_advantage
           - 这样可以把后面多步的信息都考虑进来（lambda越小越偏向一步，越大越平滑）
        4. 每一步算完，把last_advantage存到advantages[t]里。

        返回值：
        - advantages: 每一步的优势估计，形状是[T]

        例子（假设gamma=0.99, lambda=0.95）：
        假如有3步，rewards=[1, 0, 2]，values=[0.5, 0.2, 0.1]，dones=[False, False, True]
        就会从t=2,1,0依次往前算，把后面信息都累计进来。

        总结：这个函数就是“从后往前，把每一步的优势都算出来”，让PPO训练时能更好地判断哪些动作值得被强化。
        """
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        last_value = 0

        # 从最后一步往前递推
        for t in reversed(range(len(rewards))):
            if dones[t]:
                # 如果这一步是episode终止，后面没有价值了
                delta = rewards[t] - values[t]
                last_advantage = delta
            else:
                # 如果还没终止，要加上下一个状态的价值
                delta = rewards[t] + self.config.gamma * last_value - values[t]
                last_advantage = delta + self.config.gamma * self.config.gae_lambda * last_advantage

            advantages[t] = last_advantage
            last_value = values[t]

        return advantages
    
    def compute_returns(self, rewards: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """计算回报"""
        returns = torch.zeros_like(rewards)
        last_return = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                returns[t] = rewards[t]
                last_return = 0
            else:
                returns[t] = rewards[t] + self.config.gamma * last_return
                last_return = returns[t]
        
        return returns
    
    def train_step(self, batch: Dict) -> Dict:
        """单步训练"""
        prompts = batch["prompts"]
        responses = batch["responses"]
        rewards = batch["rewards"]
        
        # 编码输入
        inputs = self.tokenizer(
            prompts + responses,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length
        ).to(self.device)
        
        # 分离prompt和response的输入
        prompt_lengths = [len(self.tokenizer.encode(p)) for p in prompts]
        response_inputs = {
            "input_ids": inputs["input_ids"][len(prompts):],
            "attention_mask": inputs["attention_mask"][len(prompts):]
        }
        
        # 获取当前策略的动作概率
        with torch.no_grad():
            # 这行代码的意思是：通过当前策略模型（policy_model）对response的输入进行前向推理，得到每个token的logits（即未归一化的预测分数）。
            # 返回的是一个形状为 [batch_size, seq_len, vocab_size] 的张量，表示每个位置上每个词的logits分数。
            current_logits = self.policy_model(**response_inputs).logits
            
            # 这行代码的意思是：对logits进行softmax归一化，得到每个位置上每个词的概率分布。
            # 返回的是一个形状为 [batch_size, seq_len, vocab_size] 的张量，表示每个位置上每个词的概率。
            current_probs = F.softmax(current_logits, dim=-1)
            
            # 这行代码的意思是：对logits进行log_softmax归一化，得到每个位置上每个词的对数概率。
            # 返回的是一个形状为 [batch_size, seq_len, vocab_size] 的张量，表示每个位置上每个词的对数概率。
            current_log_probs = F.log_softmax(current_logits, dim=-1)
        
        # 计算价值
        values = self.value_model(**response_inputs).logits[:, -1, :].mean(dim=-1)
        
        # 计算优势
        advantages = self.compute_advantages(
            torch.tensor(rewards, device=self.device),
            values.detach(),
            torch.ones(len(rewards), device=self.device)
        )
        
        # 计算回报
        returns = self.compute_returns(
            torch.tensor(rewards, device=self.device),
            torch.ones(len(rewards), device=self.device)
        )
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO更新
        for _ in range(4):  # 多次更新
            # 重新计算策略概率
            logits = self.policy_model(**response_inputs).logits
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            
            # 计算比率
            ratio = torch.exp(log_probs - current_log_probs)
            
            # 计算PPO损失
            surr1 = ratio * advantages
            # 这行代码的意思是：对比率ratio进行裁剪，使其在[1 - clip_epsilon, 1 + clip_epsilon]区间内，然后再乘以优势advantages
            surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 价值损失
            value_loss = F.mse_loss(values, returns)
            
            # 熵损失
            entropy_loss = -(probs * log_probs).sum(dim=-1).mean()
            
            # 总损失
            total_loss = policy_loss + self.config.value_coef * value_loss + self.config.entropy_coef * entropy_loss
            
            # 反向传播
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.config.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), self.config.max_grad_norm)
            
            self.policy_optimizer.step()
            self.value_optimizer.step()
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "total_loss": total_loss.item(),
            "mean_reward": np.mean(rewards)
        }
    
    def train(self, training_data: List[Dict]):
        """训练循环"""
        logger.info("开始PPO训练...")
        
        # 创建输出目录
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        best_reward = float('-inf')
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            epoch_losses = []
            epoch_rewards = []
            
            # 打乱数据
            random.shuffle(training_data)
            
            for i, data in enumerate(tqdm(training_data, desc=f"Epoch {epoch + 1}")):
                # 准备批次数据
                batch = {
                    "prompts": [data["prompt"]] * len(data["responses"]),
                    "responses": data["responses"],
                    "rewards": data["rewards"]
                }
                
                # 训练步骤
                step_losses = self.train_step(batch)
                epoch_losses.append(step_losses)
                epoch_rewards.append(step_losses["mean_reward"])
                
                # 定期保存和评估
                if (i + 1) % self.config.save_steps == 0:
                    self.save_model(f"checkpoint_epoch_{epoch}_step_{i+1}")
                
                if (i + 1) % self.config.eval_steps == 0:
                    avg_reward = np.mean(epoch_rewards[-self.config.eval_steps:])
                    logger.info(f"Step {i+1}, Average Reward: {avg_reward:.4f}")
                    
                    if avg_reward > best_reward:
                        best_reward = avg_reward
                        self.save_model("best_model")
            
            # 更新学习率
            self.policy_scheduler.step()
            self.value_scheduler.step()
            
            # 计算epoch统计
            avg_policy_loss = np.mean([l["policy_loss"] for l in epoch_losses])
            avg_value_loss = np.mean([l["value_loss"] for l in epoch_losses])
            avg_total_loss = np.mean([l["total_loss"] for l in epoch_losses])
            avg_reward = np.mean(epoch_rewards)
            
            logger.info(f"Epoch {epoch + 1} Summary:")
            logger.info(f"  Policy Loss: {avg_policy_loss:.4f}")
            logger.info(f"  Value Loss: {avg_value_loss:.4f}")
            logger.info(f"  Total Loss: {avg_total_loss:.4f}")
            logger.info(f"  Average Reward: {avg_reward:.4f}")
            
            # 保存epoch模型
            self.save_model(f"epoch_{epoch + 1}")
        
        logger.info("训练完成！")
    
    def save_model(self, name: str):
        """保存模型"""
        save_path = os.path.join(self.config.output_dir, name)
        os.makedirs(save_path, exist_ok=True)
        
        # 保存策略模型
        self.policy_model.save_pretrained(os.path.join(save_path, "policy_model"))
        self.tokenizer.save_pretrained(os.path.join(save_path, "policy_model"))
        
        # 保存价值模型
        self.value_model.save_pretrained(os.path.join(save_path, "value_model"))
        
        # 保存奖励模型
        torch.save(self.reward_model.state_dict(), os.path.join(save_path, "reward_model.pt"))
        
        logger.info(f"模型已保存到: {save_path}")
    
    def load_model(self, path: str):
        """加载模型"""
        # 加载策略模型
        policy_path = os.path.join(path, "policy_model")
        self.policy_model = AutoModelForCausalLM.from_pretrained(policy_path)
        self.policy_model.to(self.device)
        
        # 加载价值模型
        value_path = os.path.join(path, "value_model")
        self.value_model = AutoModelForCausalLM.from_pretrained(value_path)
        self.value_model.to(self.device)
        
        # 加载奖励模型
        reward_path = os.path.join(path, "reward_model.pt")
        self.reward_model.load_state_dict(torch.load(reward_path, map_location=self.device))
        
        logger.info(f"模型已从 {path} 加载")

def main():
    """主函数"""
    # 配置
    config = PPOTrainingConfig()
    
    # 创建训练器
    trainer = PPOTrainer(config)
    
    # 生成训练数据
    training_data = trainer.generate_training_data(num_samples=1000)
    
    # 保存训练数据
    with open(os.path.join(config.output_dir, "training_data.json"), "w", encoding="utf-8") as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)
    
    # 开始训练
    trainer.train(training_data)
    
    logger.info("PPO微调完成！")

if __name__ == "__main__":
    main()
