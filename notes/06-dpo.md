# 06 - DPO 偏好学习

> 本章对应 [`trainer/train_dpo.py`](https://github.com/jingyaogong/minimind/blob/master/trainer/train_dpo.py) 与 [`dataset/lm_dataset.py:DPODataset`](https://github.com/jingyaogong/minimind/blob/master/dataset/lm_dataset.py#L122)。DPO 是 MiniMind 中最简单的 RLHF 形式，也是从“指令对齐”走向“人类偏好对齐”的第一步。

---

## 1. 什么是 RLHF / DPO？

RLHF（Reinforcement Learning from Human Feedback）是让模型贴合**人类偏好**的标准三阶段流程：

1. **SFT**：让模型先学会听话；
2. **训练 Reward Model**：用人类对“哪个回答更好”的标注训练一个打分模型；
3. **PPO 等强化学习算法**：用 RM 作为奖励信号，进一步微调 SFT 模型。

DPO（Direct Preference Optimization，Rafailov et al., 2023）的核心洞察是：

> RLHF 第三阶段的优化目标，可以**绕过 RM**，直接用 `(chosen, rejected)` 偏好对数据推导出一个**闭式损失**。

理论推导的关键是：把带 KL 约束的 PPO 奖励最大化目标的最优 policy 解出来，代入后发现 RM 项可以消掉，最终只剩下 policy 与 ref（参考模型）的对数概率比。这让 DPO 变成**纯监督学习**：跑两个模型（policy + ref），不需要训练 RM，**显存省一半**，训练稳定。

DPO 损失：

$$
\mathcal{L}_{\text{DPO}} = -\mathbb{E}\!\left[\log\sigma\!\left(\beta\!\left[\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right]\right)\right]
$$

其中 $y_w$ 是 chosen（好的回答），$y_l$ 是 rejected（差的回答），$\beta$ 是温度系数。

---

## 2. MiniMind 的 DPO 实现（[train_dpo.py:25-50](https://github.com/jingyaogong/minimind/blob/master/trainer/train_dpo.py#L25)）

整个 DPO 的损失计算只有几十行：

### 2.1 `logits_to_log_probs`

```python
def logits_to_log_probs(logits, labels):
    # logits: (B, T, V), labels: (B, T)
    log_probs = F.log_softmax(logits, dim=2)
    log_probs_per_token = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return log_probs_per_token  # (B, T)
```

把模型输出的 logits 转化为**每个 token 的对数概率**——DPO 关心的是 chosen / rejected 整段的 log_prob 之和，所以接下来用 mask 求和。

### 2.2 `dpo_loss`

```python
def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    # 先用 mask 求和，得到每个样本整段的对数概率
    ref_log_probs = (ref_log_probs * mask).sum(dim=1)
    policy_log_probs = (policy_log_probs * mask).sum(dim=1)

    # batch 前一半是 chosen，后一半是 rejected
    batch_size = ref_log_probs.shape[0]
    chosen_ref_log_probs = ref_log_probs[:batch_size // 2]
    reject_ref_log_probs = ref_log_probs[batch_size // 2:]
    chosen_policy_log_probs = policy_log_probs[:batch_size // 2]
    reject_policy_log_probs = policy_log_probs[batch_size // 2:]

    # DPO 核心：policy / ref 的对数比之差
    pi_logratios = chosen_policy_log_probs - reject_policy_log_probs
    ref_logratios = chosen_ref_log_probs - reject_ref_log_probs
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()
```

数学对照：

- `pi_logratios` = $\log\frac{\pi_\theta(y_w)}{\pi_\theta(y_l)}$
- `ref_logratios` = $\log\frac{\pi_{\text{ref}}(y_w)}{\pi_{\text{ref}}(y_l)}$
- `logits` = `pi_logratios - ref_logratios` = $\beta$ 内括号里的内容
- `-F.logsigmoid(beta * logits)` = $-\log\sigma(\beta \cdot \Delta)$ = DPO loss

当 policy 让 chosen 比 rejected 提升得**比 ref 模型更明显**时，`pi_logratios - ref_logratios > 0`，`logits > 0`，loss 减小——这正是 DPO 想驱动的方向。

---

## 3. 训练循环的关键差异

### 3.1 双模型：policy + ref

```python
model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)
ref_model.eval()
ref_model.requires_grad_(False)
```

- `model` 是要训练的策略模型；
- `ref_model` 加载相同初始权重，但**冻结**——它提供一个 KL 锚点，防止 policy 偏离太远。

显存占用上，`ref_model` 不需要保存梯度和优化器状态，因此总显存约等于“policy 训练 + 一个 inference 推理”。相比 PPO（需要 actor + critic + ref + RM）省一半以上。

### 3.2 拼接 chosen + rejected

```python
x = torch.cat([x_chosen, x_rejected], dim=0)
y = torch.cat([y_chosen, y_rejected], dim=0)
mask = torch.cat([mask_chosen, mask_rejected], dim=0)

ref_outputs = ref_model(x)            # ref 前向（无梯度）
ref_log_probs = logits_to_log_probs(ref_outputs.logits, y)
outputs = model(x)                    # policy 前向
policy_log_probs = logits_to_log_probs(outputs.logits, y)

dpo_loss_val = dpo_loss(ref_log_probs, policy_log_probs, mask, beta=beta)
loss = dpo_loss_val + outputs.aux_loss
```

注意几个细节：

- chosen 与 rejected 拼成一个大 batch 一次前向，更省显存；
- `outputs.aux_loss` 是 MoE 的负载均衡 loss（仅当 `use_moe=1` 时非零）；
- `mask` 把 assistant 段以外的 token 都置 0，确保只对回答部分计算对数概率。

### 3.3 学习率要小

```python
parser.add_argument("--learning_rate", type=float, default=4e-8, help="初始学习率（建议<=5e-8避免遗忘）")
```

DPO 的学习率比 SFT（5e-5）又低了一个数量级。原因：

- DPO 是在 SFT 基础上做**微调**，目标只是把偏好贴合一点；
- 学习率太大会快速冲掉 SFT 阶段学到的对话能力（即“灾难性遗忘”）；
- `beta`（默认 0.15）越大，KL 锚点越强，policy 越不会偏离 ref；学习率也可以相应放大一点。

---

## 4. 数据格式

```json
{
  "chosen": [
    {"role": "user", "content": "Q"},
    {"role": "assistant", "content": "good answer"}
  ],
  "rejected": [
    {"role": "user", "content": "Q"},
    {"role": "assistant", "content": "bad answer"}
  ]
}
```

`DPODataset.__getitem__`（[lm_dataset.py:135-174](https://github.com/jingyaogong/minimind/blob/master/dataset/lm_dataset.py#L135)）的处理流程：

1. chosen 与 rejected 各跑一次 chat_template；
2. 分别 tokenize + padding + 截断；
3. 用 `generate_loss_mask` 生成只对 assistant 段置 1 的 mask；
4. 返回 `(x_chosen, y_chosen, mask_chosen, x_rejected, y_rejected, mask_rejected)`。

主线 DPO 数据为 `dpo.jsonl`（53MB），采样自 [DPO-En-Zh-20k](https://huggingface.co/datasets/llamafactory/DPO-En-Zh-20k) 并补充了中文数据。

---

## 5. 启动训练

```bash
cd trainer && python train_dpo.py
```

或者：

```bash
torchrun --nproc_per_node N train_dpo.py
```

参数要点：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--from_weight` | `full_sft` | 从哪个 SFT 权重继续 |
| `--data_path` | `../dataset/dpo.jsonl` | 偏好数据路径 |
| `--beta` | 0.15 | DPO 温度系数 |
| `--learning_rate` | 4e-8 | 一定要小 |
| `--max_seq_len` | 1024 | 偏好数据偏长，建议开大一点 |

输出权重保存为 `../out/dpo_*.pth`。

---

## 6. 推理测试

```bash
python eval_llm.py --weight dpo
```

主观上会感觉模型回答**更稳定、更贴合人类偏好**——比如更倾向直接回答、避免无意义的客套、拒绝不确定的问题时会主动说“不知道”而不是强行编答案。

但要注意，DPO 数据如果规模不大或质量不高，很容易出现**反向过拟合**——模型可能学会某些特定的句式套路，而不是真正的“偏好对齐”。

---

## 7. DPO 的局限

1. **Off-Policy，缺少探索**：只用静态偏好数据，无法“在线探索”新的生成方向；
2. **对“是否能做对题”提升有限**：DPO 主要是“偏好/安全”层面的对齐，对智力能力的提升效果取决于偏好数据是否覆盖了对应题型；
3. **数据质量敏感**：偏好对的“chosen vs rejected 差距”如果不够明显，DPO 训练信号会很弱。

针对 (3)，MiniMind 的解决方案是**用 RLAIF（AI feedback）替代或补充人类反馈**——见 [07 - PPO / GRPO / CISPO（RLAIF）](./07-ppo-grpo.md)。

---

## 8. DPO 与 PPO 的关系

可以把 DPO 视作 PPO 的“离线特化版本”：

| 维度 | DPO | PPO |
|------|-----|-----|
| 奖励来源 | 静态偏好对 | 实时 RM 评分 |
| 训练模型数 | 1 (前向参与 2) | 2（actor + critic） |
| KL 约束 | 通过 ref 模型隐式实现 | 显式 KL 惩罚项 |
| 探索能力 | 弱 | 强 |
| 实现难度 | 低 | 高 |
| 稳定性 | 高 | 易训崩 |

如果把 PPO 的目标函数在最优 policy 处“展开”，会发现 DPO 的损失其实是它在 off-policy 设定下的解析解。所以工程上常见的组合是：

- 先用 DPO 做一次粗粒度对齐；
- 再用 PPO / GRPO 在更难的任务上做精细对齐。

---

## 9. 自检问题

1. DPO 与 RLHF 三阶段流程相比，省掉了哪个模型？为什么能省？
2. DPO 损失中的 `beta` 起什么作用？调大会发生什么？调小呢？
3. 为什么 DPO 的学习率必须设得非常小（默认 4e-8）？
4. DPO 数据中 `chosen` 与 `rejected` 的区分是怎样影响训练信号的？
5. DPO 与 PPO 在“探索能力”上的根本差异是什么？

---

## 10. 推荐阅读

- [DPO 原始论文](https://arxiv.org/abs/2305.18290)——Rafailov et al., *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*.
- [ORPO](https://arxiv.org/abs/2403.07691)——把 SFT 与 DPO 合并到一个损失里。
- [IPO](https://arxiv.org/abs/2310.12036)——DPO 的改进版本，缓解过拟合问题。
- [KTO](https://arxiv.org/abs/2402.01306)——用 Kahneman-Tversky 模型的偏好优化，不需要成对数据。

---

下一章：[07 - PPO / GRPO / CISPO（RLAIF）](./07-ppo-grpo.md)