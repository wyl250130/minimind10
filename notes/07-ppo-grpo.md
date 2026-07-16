# 07 - PPO / GRPO / CISPO（RLAIF）

> 本章对应 [`trainer/train_ppo.py`](https://github.com/jingyaogong/minimind/blob/master/trainer/train_ppo.py) 与 [`trainer/train_grpo.py`](https://github.com/jingyaogong/minimind/blob/master/trainer/train_grpo.py)。当 DPO 的“离线偏好”无法继续推动能力提升时，就需要进入 **On-Policy + 可扩展奖励信号**的世界——也就是 RLAIF。本章用一个统一视角把 PPO / GRPO / CISPO / DPO 全部串起来。

---

## 1. 统一视角：PO 算法的三个核心组件

作者在 README 中给出了一个非常漂亮的统一框架：

$$
\mathcal{J}_{\text{PO}} = \mathbb{E}_{q \sim P(Q), o \sim \pi(O|q)}\Big[\underbrace{f(r_t)}_{\text{策略项}} \cdot \underbrace{g(A_t)}_{\text{优势项}} - \underbrace{h(\text{KL}_t)}_{\text{正则项}}\Big]
$$

训练时最小化 $-\mathcal{J}_{\text{PO}}$。

| 符号 | 含义 |
|------|------|
| $r_t = \frac{\pi_\theta(o_t \mid q, o_{<t})}{\pi_{\text{ref}}(o_t \mid q, o_{<t})}$ | 概率比（policy vs ref）|
| $A_t$ | 优势函数，衡量“某动作相对基线有多好”|
| $\text{KL}_t$ | 防止 policy 偏离 ref 太远 |

不同的 **xxPO 算法**本质上只是对这三个组件的**不同实例化**：

| 算法 | 策略项 $f(r_t)$ | 优势项 $g(A_t)$ | 正则项 $h(\text{KL}_t)$ | 训练模型数 |
|------|----------------|----------------|--------------------------|-----------|
| **DPO** | $\log r_w - \log r_l$ | 无显式优势项 | 隐含在 $\beta$ 中 | 1（前向参与 2）|
| **PPO** | $\min(r, \text{clip}(r))$ | $R - V(s)$ | $\beta \cdot \mathbb{E}[\text{KL}]$ | 2 |
| **GRPO** | $\min(r, \text{clip}(r))$ | $\frac{R - \mu}{\sigma}$ | $\beta \cdot \text{KL}_t$ | 1 |
| **CISPO** | $\text{clip}(r, 0, \epsilon_{\max}) \cdot A_t \cdot \log\pi_\theta$ | $\frac{R - \mu}{\sigma}$ | $\beta \cdot \text{KL}_t$ | 1 |

记住这张表，本章其余内容只是“按表逐项展开”。

---

## 2. PPO（Proximal Policy Optimization）

PPO 是 OpenAI 2017 年提出的经典算法，是 LLM RL 领域最常见的基线方法之一。其损失：

$$
\mathcal{L}_{\text{PPO}} = -\mathbb{E}\!\Big[\min(r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t)\Big] + \beta \cdot \mathbb{E}[\text{KL}]
$$

三个组件：

- **策略项**：$\min(r_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon))$ —— 裁剪概率比，防止更新过激；
- **优势项**：$R - V(s)$ —— 由 Critic 网络估计的价值函数；
- **正则项**：$\beta \cdot \mathbb{E}[\text{KL}]$ —— 全局 KL 散度约束。

MiniMind 的 PPO 实现包含：

- **Actor**：策略模型（生成回答）；
- **Critic**：价值模型（评估回答价值）；
- **GAE**：Generalized Advantage Estimation 用于优势估计；
- **Reward Model**：外置评分模型（如 InternLM2-1.8B-Reward）。

训练时显存占用约为单网络方法的 1.5–2 倍（因为要同时维护 actor + critic + ref + RM）。

### 2.1 PPO 的训练曲线特点

作者观察到一个非常一致的现象：**reward 提升缓慢**。原因是：

1. Critic 需要逐步收敛才能准确估计价值函数；
2. Actor 的策略更新依赖 Critic 提供的优势估计，两者相互依赖形成复杂耦合；
3. 训练初期 Critic 估计不准会误导 Actor 的梯度方向，导致整体收敛缓慢。

因此在工程上 PPO 通常被认为“训得动但慢”，GRPO 在这点上有显著改善。

---

## 3. GRPO（Group Relative Policy Optimization）

GRPO 来自 DeepSeekMath（2024 年初）——也就是后来推动 DeepSeek-R1 的核心算法。

$$
\mathcal{L}_{\text{GRPO}} = -\mathbb{E}\!\Big[\min(r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t) - \beta \cdot \text{KL}_t\Big]
$$

- **策略项**：和 PPO 一样使用裁剪概率比；
- **优势项**：$\frac{R - \mu_{\text{group}}}{\sigma_{\text{group}}}$ —— **同一 prompt 生成 N 个回答，用组内均值 / 标准差归一化作为优势**；
- **正则项**：$\beta \cdot \text{KL}_t$ —— token 级别的 KL 散度约束。

最大变化：**不需要 Critic**。优势通过组内统计量估算，节省一半显存，训练更稳定。

### 3.1 MiniMind 中的 GRPO 实现（[train_grpo.py](https://github.com/jingyaogong/minimind/blob/master/trainer/train_grpo.py)）

整个训练循环可以拆成 6 步：

#### Step 1：构造 prompt 与 rollout

```python
prompts = batch['prompt']                          # list[str]
prompt_inputs = tokenizer(prompts, ...).to(args.device)
rollout_result = rollout_engine.rollout(
    prompt_ids=prompt_inputs["input_ids"],
    attention_mask=prompt_inputs["attention_mask"],
    num_generations=args.num_generations,          # 默认 6
    max_new_tokens=args.max_gen_len,
    temperature=0.8,
)
```

`rollout_engine` 是 MiniMind 自己设计的**训推分离抽象层**，支持本地 PyTorch rollout 和 SGLang 远端 rollout。

#### Step 2：计算奖励

```python
def calculate_rewards(prompts, responses, reward_model):
    rewards = torch.zeros(len(responses), device=args.device)
    with torch.no_grad():
        for i in range(batch_size):
            for j in range(num_generations):
                ...
                rewards[idx] += 0.5 if 20 <= len(response) <= 800 else -0.5
                if '</think>' in response:
                    rewards[idx] += 1.0 if 20 <= len(thinking) <= 300 else -0.5
                    rewards[idx] += 0.25 if response.count('</think>') == 1 else -0.25
                rewards[idx] -= rep_penalty(answer)
                score = reward_model.get_score(messages, answer)
                rewards[idx] += score
    return rewards
```

奖励来源是**混合**的：

1. **长度奖励**：合理的回答长度获得正分；
2. **思考格式奖励**：包含 `<think>...</think>` 标签且格式正确获得正分；
3. **重复惩罚**：n-gram 重复率过高扣分；
4. **RM 分数**：外置 reward model 的连续分数。

这种**稠密且连续**的奖励信号，对小模型特别重要——避免了规则二元奖励在小模型上“全零”的奖励稀疏问题。

#### Step 3：组内归一化优势

```python
grouped_rewards = rewards.view(-1, args.num_generations)  # [B, num_gen]
mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)
std_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)
advantages = (rewards - mean_r) / (std_r + 1e-4)
```

把每个 prompt 的 `num_generations` 个回答视为一组，用组内均值 / 标准差做归一化——这就是“组相对”名字的由来。

#### Step 4：计算重要性采样比与 KL

```python
ratio = torch.exp(per_token_logps - old_per_token_logps)
kl_div = ref_per_token_logps - per_token_logps
per_token_kl = torch.exp(kl_div) - kl_div - 1   # k3 estimator
```

注意几个细节：

- `per_token_logps` 是当前 policy 在 completion token 上的 log_prob；
- `old_per_token_logps` 是 rollout 时记录的 log_prob（用于 PPO 风格的 off-policy correction）；
- KL 用 k3 estimator（无偏估计）而不是直接 `kl_div`。

#### Step 5：GRPO / CISPO 损失

```python
if args.loss_type == "cispo":
    clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()
    per_token_loss = -(clamped_ratio * advantages.unsqueeze(1) * per_token_logps - args.beta * per_token_kl)
else:  # grpo
    clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
    per_token_loss1 = ratio * advantages.unsqueeze(1)
    per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
    per_token_loss = -(torch.min(per_token_loss1, per_token_loss2) - args.beta * per_token_kl)
```

两者仅在**策略项**的实现上有差异：

- GRPO：标准 PPO-style 的 $\min(r, \text{clip}(r))$；
- CISPO：把 ratio 裁剪后作为权重乘 $\log\pi$。

CISPO 的关键洞察：PPO/GRPO 里当 ratio 被 clip 掉时，**梯度直接被截断**——而 CISPO 把 ratio 改成“裁剪权重 × log 概率”，即使 ratio 被 clip，梯度流依然保留。这就是 CISPO 比 GRPO 更稳定的根源。

#### Step 6：聚合 + 反向

```python
policy_loss = ((per_token_loss * completion_mask).sum(dim=1) /
               completion_mask.sum(dim=1).clamp(min=1)).mean()
loss = (policy_loss + aux_loss) / args.accumulation_steps
loss.backward()
```

`completion_mask` 是只对生成 token 段置 1 的 mask，与 SFT 的 loss mask 思路一致。

---

## 4. Rollout Engine：训推分离的抽象层

MiniMind 的 [`trainer/rollout_engine.py`](https://github.com/jingyaogong/minimind/blob/master/trainer/rollout_engine.py) 是项目里一个很有意思的设计：

```text
            ┌─────────────────────────────┐
            │        训练侧 (Trainer)      │
            │  policy update / optimizer  │
            └────────────┬────────────────┘
                         │  prompt → rollout → samples
                         ▼
            ┌─────────────────────────────┐
            │       Rollout Engine        │
            │  (torch.generate / SGLang)  │
            └─────────────────────────────┘
```

好处是：

1. **训练脚本不用关心底层推理实现**——统一接口 `rollout(prompt_ids, ..., num_generations, ...)`；
2. **可以无缝切换到 SGLang / vLLM** 等高性能推理引擎，只需注册新的 engine；
3. **支持训推异步化**（虽然 MiniMind 当前仍是同步模式，但已经具备了演进到异步 / 分布式 rollout 的基础结构）。

启动方式：

```bash
# 默认：本地 PyTorch rollout
python train_grpo.py --rollout_engine torch

# 高性能：SGLang rollout
python -m sglang.launch_server --model-path ./minimind-3 --attention-backend triton --host 0.0.0.0 --port 8998
python train_grpo.py --rollout_engine sglang --sglang_base_url http://localhost:8998 ...
```

---

## 5. 启动训练

```bash
# PPO
torchrun --nproc_per_node N train_ppo.py

# GRPO
torchrun --nproc_per_node N train_grpo.py

# CISPO（GRPO 脚本加一个参数即可）
python train_grpo.py --loss_type cispo
```

关键参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--from_weight` | `full_sft` | 从哪个 SFT 权重继续 |
| `--data_path` | `../dataset/rlaif.jsonl` | RLAIF 数据路径 |
| `--num_generations` | 6 | 每个 prompt 生成的样本数 |
| `--max_gen_len` | 1024 | 单次 rollout 最大长度 |
| `--beta` | 0.1 | KL 系数 |
| `--loss_type` | `cispo` | `grpo` 或 `cispo` |
| `--reward_model_path` | `../../internlm2-1_8b-reward` | 外置 RM 路径 |
| `--rollout_engine` | `torch` | `torch` 或 `sglang` |

输出权重：

- PPO：`ppo_actor_*.pth` + `ppo_critic_*.pth`
- GRPO / CISPO：`grpo_*.pth`

---

## 6. 训练曲线对比

MiniMind 在相同 SFT 起点、相同随机种子下的主观对比（详见 README）：

- **SFT 模型**：风格稳定、回答完整但容易“凑答案”；
- **GRPO 模型**：能更好地贴合 RM 分数，回答更“讨喜”，但偶尔会出现“为了高分强行加修饰词”的现象；
- **CISPO**：与 GRPO 类似但更稳定，loss 抖动更小。

这是非常典型的**后训练对齐税（alignment tax）**：

> 在某个奖励维度上变强，通常会牺牲掉一部分通用能力、事实性或自然分布下的稳定性。

---

## 7. RLAIF 与 RLHF 的关系

RLAIF 的核心思想是用**机器生成的反馈**代替人类反馈，常见来源：

- **Model-based**：外置 reward model（如 InternLM2-Reward）；
- **Rule-based**：规则函数（数学题答案正确性、SQL 执行成功率、代码 pass@k、JSON/XML 格式合规等）；
- **Environment-based**：环境反馈（游戏得分、研究完整度、API 状态等）；
- **混合**：$r_{\text{total}} = \alpha \cdot r_{\text{model}} + \beta \cdot r_{\text{rule}}$

RLAIF 的优势是**可扩展性强**——不需要昂贵的人工标注，可以生成海量训练样本；劣势是**可能偏离人类真实偏好**。

对 MiniMind 这种 0.1B 参数、能力较弱的模型，**奖励稀疏（reward sparsity）** 是头号难题：

- 模型生成的候选回答几乎全部错误 → 所有奖励分数接近 0；
- 优势函数 $A(x, y) = r(x, y) - b(x) \approx 0$；
- 策略梯度信号消失，参数无法有效更新。

MiniMind 的解决方案：

1. **优先使用 Model-based 连续奖励**（避免规则二元 0/1）；
2. **混用多种奖励源**（长度 + 格式 + 重复惩罚 + RM）；
3. **避免直接用 rule-based 奖励 + 超纲难度数据**（如 MATH500）。

---

## 8. 实践建议

1. **先用 DPO 打底**——便宜、稳定、能看出偏好对齐的趋势；
2. **DPO 收敛后切到 GRPO / CISPO**——继续推动能力提升；
3. **如果是 Tool-Use 任务，直接走 Agentic RL**——见 [08 - Agentic RL](./08-agentic-rl.md)；
4. **RM 选择很关键**——用 InternLM2-1.8B-Reward 这种连续分数的 RM，不要用 GPT-4 风格的 judge prompt（信号稀疏）；
5. **监控 advantage.std**——如果优势方差持续接近 0，说明奖励信号失效，要换 RM 或换数据；
6. **学习率仍然要小**——默认 3e-7（GRPO），DPO 的 1/10 量级。

---

## 9. 自检问题

1. PPO / GRPO / CISPO 在“策略项”上有什么区别？这些差异如何影响训练稳定性？
2. GRPO 为什么能省掉 Critic？组内归一化的优势是什么？
3. CISPO 相对 GRPO 的关键 trick 是什么？为什么它能保留被裁剪 token 的梯度流？
4. 什么是“奖励稀疏”？MiniMind 是如何缓解这个问题的？
5. 训推分离的 rollout engine 设计有什么工程意义？

---

## 10. 推荐阅读

- [PPO 原始论文](https://arxiv.org/abs/1707.06347)——Schulman et al., *Proximal Policy Optimization Algorithms*.
- [GRPO 论文](https://arxiv.org/abs/2402.03300)——Shao et al., *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*.
- [CISPO 论文](https://arxiv.org/abs/2506.13585)——*Clipped Importance Sampling Policy Optimization*.
- [DPO 原始论文](https://arxiv.org/abs/2305.18290)——Rafailov et al., *Direct Preference Optimization*.
- [Reward Model：InternLM2-Reward](https://huggingface.co/internlm/internlm2-1_8b-reward)——MiniMind 主线使用的 RM。

---

下一章：[08 - Agentic RL（Tool-Use）](./08-agentic-rl.md)