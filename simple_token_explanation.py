import torch
import torch.nn.functional as F

def explain_actual_token_selection():
    """
    解释"实际选择的token的概率"
    
    在语言模型中：
    1. 每个位置输出一个概率分布（对所有可能的token）
    2. 但我们只关心实际选择的那个token的概率
    3. 这个概率用于PPO的损失计算
    """
    
    print("=" * 60)
    print("解释：什么是'实际选择的token的概率'")
    print("=" * 60)
    
    # 1. 设置简单场景
    print("\n=== 场景设置 ===")
    print("假设词汇表: ['我', '是', '一个', '模型', '。']")
    print("Token IDs:   [ 0,   1,    2,     3,   4 ]")
    
    vocab_size = 5  # 词汇表大小
    seq_length = 3  # 序列长度
    
    # 2. 模拟模型输出的logits
    print(f"\n=== 模型输出 ===")
    print(f"序列长度: {seq_length}")
    print(f"词汇表大小: {vocab_size}")
    
    # 创建假的logits矩阵 [seq_length, vocab_size]
    logits = torch.tensor([
        [1.0, 2.0, 0.5, 3.0, 1.5],  # 位置0: 对5个词的评分
        [0.8, 3.2, 1.0, 2.0, 0.5],  # 位置1: 对5个词的评分
        [2.0, 1.0, 3.5, 0.5, 0.8],  # 位置2: 对5个词的评分
    ])
    
    print("Logits矩阵 (每行是一个位置的评分):")
    print(f"Shape: {logits.shape}")
    for i, row in enumerate(logits):
        print(f"  位置{i}: {row.tolist()}")
    
    # 3. 转换为概率分布
    probs = F.softmax(logits, dim=-1)  # [seq_length, vocab_size]
    log_probs = F.log_softmax(logits, dim=-1)
    
    print(f"\n=== 概率分布 ===")
    print("经过softmax后的概率分布:")
    vocab = ['我', '是', '一个', '模型', '。']
    for i, prob_row in enumerate(probs):
        print(f"位置{i}:")
        for j, prob in enumerate(prob_row):
            print(f"  '{vocab[j]}': {prob.item():.4f}")
        print(f"  总和: {prob_row.sum().item():.4f}")
        print()
    
    # 4. 关键概念："实际选择的token"
    print("=== 关键概念：实际选择的token ===")
    
    # 假设实际生成的序列是：[3, 1, 2] = ["模型", "是", "一个"] 
    actual_tokens = torch.tensor([3, 1, 2])
    actual_sequence = [vocab[i] for i in actual_tokens]
    
    print(f"实际生成的token IDs: {actual_tokens.tolist()}")
    print(f"实际生成的文本: {actual_sequence}")
    print()
    
    # 5. 使用gather提取实际选择的token的概率
    print("=== 提取实际选择的token的概率 ===")
    
    print("步骤1: actual_tokens.unsqueeze(-1)")
    unsqueezed = actual_tokens.unsqueeze(-1)
    print(f"  原始: {actual_tokens}")
    print(f"  unsqueeze后: {unsqueezed}")
    print(f"  作用: 将形状从 [3] 变为 [3, 1]，为gather操作准备")
    print()
    
    print("步骤2: log_probs.gather(-1, actual_tokens.unsqueeze(-1))")
    gathered = log_probs.gather(-1, unsqueezed)
    print(f"  Gathered结果: {gathered}")
    print(f"  形状: {gathered.shape}")
    print("  gather的作用:")
    for i, token_id in enumerate(actual_tokens):
        selected_log_prob = log_probs[i, token_id]
        print(f"    位置{i}: 从概率分布中选择token {token_id} ('{vocab[token_id]}') 的对数概率: {selected_log_prob.item():.4f}")
    print()
    
    print("步骤3: .squeeze(-1)")
    actual_log_probs = gathered.squeeze(-1)
    print(f"  Squeeze后: {actual_log_probs}")
    print(f"  形状: {actual_log_probs.shape}")
    print(f"  作用: 移除最后一个维度，从 [3, 1] 变为 [3]")
    print()
    
    # 6. 实际应用
    print("=== 在PPO中的应用 ===")
    print("这些概率用于:")
    print("1. 计算策略梯度")
    print("2. 计算重要性采样比率 ratio = new_prob / old_prob")
    print("3. PPO的clipped objective")
    print()
    
    # 计算整个序列的概率
    total_log_prob = actual_log_probs.sum()
    total_prob = torch.exp(total_log_prob)
    
    print(f"整个序列的对数概率: {total_log_prob.item():.4f}")
    print(f"整个序列的概率: {total_prob.item():.6f}")
    print(f"序列: {' '.join(actual_sequence)}")

def demonstrate_why_this_matters():
    """演示为什么需要actual token的概率"""
    
    print("\n" + "=" * 60)
    print("为什么需要'实际选择的token'的概率？")
    print("=" * 60)
    
    # 模拟两个不同的策略
    print("\n=== 策略更新前后的比较 ===")
    
    # 定义词汇表
    vocab = ['我', '是', '一个', '模型', '。']
    
    # 旧策略的logits (训练开始时的策略)
    old_logits = torch.tensor([
        [1.0, 2.0, 0.5, 3.0, 1.5],
        [0.8, 3.2, 1.0, 2.0, 0.5],
    ])
    
    # 新策略的logits (经过梯度更新后的策略)
    new_logits = torch.tensor([
        [1.1, 2.1, 0.6, 2.9, 1.4],
        [0.9, 3.1, 1.1, 2.1, 0.4],
    ])
    
    # 实际生成的tokens
    actual_tokens = torch.tensor([3, 1])  # ["模型", "是"]
    
    print("实际生成的tokens:", [vocab[i] for i in actual_tokens])
    print()
    
    # 计算old和new策略下实际tokens的概率
    old_log_probs = F.log_softmax(old_logits, dim=-1)
    new_log_probs = F.log_softmax(new_logits, dim=-1)
    
    old_actual_probs = old_log_probs.gather(-1, actual_tokens.unsqueeze(-1)).squeeze(-1)
    new_actual_probs = new_log_probs.gather(-1, actual_tokens.unsqueeze(-1)).squeeze(-1)
    
    print("旧策略下实际tokens的对数概率:", old_actual_probs)
    print("新策略下实际tokens的对数概率:", new_actual_probs)
    print()
    
    # 转换为线性概率
    old_linear_probs = torch.exp(old_actual_probs)
    new_linear_probs = torch.exp(new_actual_probs)
    
    print("旧策略下实际tokens的线性概率:", old_linear_probs)
    print("新策略下实际tokens的线性概率:", new_linear_probs)
    print()
    
    # 计算重要性采样比率 (正确方式)
    ratio = new_linear_probs / old_linear_probs
    print(f"重要性采样比率 (new/old): {ratio}")
    
    print("\n=== 为什么这个比率很重要？ ===")
    print("1. 如果 ratio ≈ 1.0: 新旧策略相似，更新安全")
    print("2. 如果 ratio >> 1.0: 新策略更倾向于这个动作，需要小心更新")
    print("3. 如果 ratio << 1.0: 新策略不太倾向于这个动作，更新可能过大")
    print()
    
    print("PPO使用这个比率来:")
    print("1. 防止策略更新过大 (通过clipping到 [0.8, 1.2] 范围)")
    print("2. 保持训练稳定性")
    print("3. 计算策略梯度")
    print("4. 避免灾难性遗忘")

if __name__ == "__main__":
    explain_actual_token_selection()
    demonstrate_why_this_matters() 