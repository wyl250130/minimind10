# 08 - Agentic RL（Tool-Use）

> 本章对应 [`trainer/train_agent.py`](https://github.com/jingyaogong/minimind/blob/master/trainer/train_agent.py) 与 [`dataset/lm_dataset.py:AgentRLDataset`](https://github.com/jingyaogong/minimind/blob/master/dataset/lm_dataset.py#L226)。Agentic RL 是 MiniMind 训练链路里**最复杂、也是最接近真实应用场景**的阶段——模型不仅要会生成，还要会“调工具、看结果、再规划”。

---

## 1. 什么是 Agentic RL？

传统 RLHF / RLAIF 都是**单轮对话**的优化：给定一个 prompt，生成一个回答，按 RM 分数更新。

但现实中很多任务是**多轮交互**的：

- “帮我算一下 256 乘以 37”——模型需要主动调 `calculate_math` 工具；
- “现在北京几点了？明天杭州天气怎么样？”——需要两次工具调用；
- “翻译这句话，然后用结果写一首诗”——需要先调翻译工具，再生成。

这就是 **Agentic RL** 要解决的问题：在**多轮、含工具调用、环境反馈延迟**的场景下训练 policy 模型。

MiniMind 的 Agentic RL 是一套**狭义版本**的实现，专注于：

- 让模型学会基础的**调用工具**；
- 学会**观察工具返回**并把结果拼回上下文；
- 学会**继续规划**直到任务完成。

它**不**覆盖完整 Agent 系统里更大的状态管理、长期记忆、复杂工作流编排——这些可以基于这套最小实现逐步扩展。

---

## 2. 训练链路一览

```text
           ┌────────────────────────────────────────┐
           │           Agentic RL 训练循环          │
           └────────────────────────────────────────┘

   Prompt (用户问题)
       │
       ▼
   ┌──────────────┐
   │ Rollout      │  ←─ policy 模型 + rollout engine
   │ 多轮对话展开  │
   └──────┬───────┘
          │  每轮生成 assistant 回复
          ▼
   ┌──────────────┐
   │ 解析 tool_call│  ←─ 正则 + JSON.loads
   └──────┬───────┘
          │  若含 tool_call
          ▼
   ┌──────────────┐
   │ 执行工具     │  ←─ 模拟数据 / 真实 API
   └──────┬───────┘
          │  返回 tool_response
          ▼
   ┌──────────────┐
   │ 拼回上下文   │
   └──────┬───────┘
          │  继续让模型生成下一轮
          ▼
   ┌──────────────┐
   │ 计算整条轨迹  │
   │ 的总 reward   │
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │ 策略更新     │  ←─ GRPO / CISPO
   └──────────────┘
```

主线流程可以压缩成：

$$
\texttt{rollout batch} \rightarrow \texttt{calculate rewards} \rightarrow \texttt{policy update}
$$

---

## 3. 数据格式

Agentic RL 在多轮 tool-use 场景下训练，数据格式与 SFT 类似，但多了一个 `gt` 字段，用来事后校验最终答案是否正确：

```json
{
  "conversations": [
    {"role": "system", "content": "# Tools ...", "tools": "[...]"},
    {"role": "user", "content": "帮我算一下 256 乘以 37 等于多少"},
    {"role": "assistant", "content": "", "tool_calls": "[{...}]"},
    {"role": "tool", "content": "{\"result\":\"9472\"}"},
    {"role": "assistant", "content": "256 乘以 37 等于 9472。"}
  ],
  "gt": "9472"
}
```

主线数据：

- `agent_rl.jsonl`（86MB）：通用 Agentic RL 数据；
- `agent_rl_math.jsonl`（18MB）：数学题为主的 Agentic RL 数据（适合 RLVR）。

`AgentRLDataset.__getitem__`（[lm_dataset.py:249-252](https://github.com/jingyaogong/minimind/blob/master/dataset/lm_dataset.py#L249)）直接返回 `(messages, tools, gt)`，不进行 tokenize——因为多轮 rollout 过程中需要动态修改上下文，由 rollout engine 负责每轮重新构造 prompt。

---

## 4. 模拟工具集

MiniMind 在 `train_agent.py` 中内置了一组**模拟工具**用于训练（[train_agent.py:40-47](https://github.com/jingyaogong/minimind/blob/master/trainer/train_agent.py#L40)）：

```python
TOOLS = [
    {"type": "function", "function": {"name": "calculate_math", ...}},
    {"type": "function", "function": {"name": "unit_converter", ...}},
    {"type": "function", "function": {"name": "get_current_weather", ...}},
    {"type": "function", "function": {"name": "get_current_time", ...}},
    {"type": "function", "function": {"name": "get_exchange_rate", ...}},
    {"type": "function", "function": {"name": "translate_text", ...}},
]
```

每个工具都有：

- 模拟数据（`WEATHER_DATA`、`TIME_DATA`、`EXCHANGE_DATA`、`TRANSLATE_DATA`、`UNIT_DATA`）；
- 参数校验函数（`CHECK_ARGS`）；
- 模拟执行函数（`MOCK_RESULTS`）。

> ⚠️ 这里用 `eval(...)` 来执行 `calculate_math`，但**限制了 `__builtins__` 为空**，并且只暴露 `math` 模块；同时用 `signal.alarm(1)` 做 1 秒超时保护——所以是相对安全的。但生产环境中**强烈建议**换成 sympy / 真计算服务。

这层 mock 让训练**不需要任何外部 API**——你可以本地直接跑，不需要联网、付费 token 或 API key。

---

## 5. 多轮 Rollout

`rollout_single`（[train_agent.py:98-159](https://github.com/jingyaogong/minimind/blob/master/trainer/train_agent.py#L98)）是整个 Agentic RL 的核心，把单条样本展开成一条完整轨迹 $\tau$：

$$
\tau = (a_1, o_1, a_2, o_2, \dots, a_T), \quad a_t \sim \pi_\theta(\cdot \mid s_t, \mathcal{T})
$$

其中：

- $s_t$ 是第 $t$ 步的状态（对话上下文）；
- $\mathcal{T}$ 是工具列表；
- $a_t$ 是模型生成的动作（assistant 回复）；
- $o_t$ 是工具返回的 observation。

### 5.1 单轮流程

```python
def rollout_single(rollout_engine, tokenizer, messages, tools, max_turns=3, ...):
    for turn in range(max_turns):
        # 1. 构造当前上下文（含 chat_template）
        context = tokenizer.apply_chat_template(messages, ..., open_thinking=open_thinking)
        inputs = tokenizer(context, return_tensors="pt", ...).to(device)

        # 2. 用 rollout engine 生成
        rollout_result = rollout_engine.rollout(
            prompt_ids=inputs["input_ids"],
            num_generations=1,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
        )
        new_ids = rollout_result.completion_ids[0].tolist()

        # 3. 解码、解析
        response_text = tokenizer.decode(new_ids, skip_special_tokens=True)

        # 4. 检测是否含 tool_call
        tool_calls = parse_tool_calls(response_text)

        if tool_calls:
            # 5a. 执行工具，得到 observation
            observation = execute_tool(tool_calls[0]["name"], tool_calls[0]["arguments"])
            tool_response = json.dumps(observation, ensure_ascii=False)

            # 5b. 把 tool_call + tool_response 拼回 messages
            messages.append({"role": "assistant", "content": "", "tool_calls": json.dumps(tool_calls, ensure_ascii=False)})
            messages.append({"role": "tool", "content": tool_response})
        else:
            # 5c. 没有 tool_call，rollout 结束
            messages.append({"role": "assistant", "content": response_text})
            break
```

关键设计：

- **每轮重新构造 prompt**：把 messages 渲染成完整上下文，再用 rollout engine 生成；
- **动态修改 messages**：检测到 tool_call 就执行工具，把 assistant 消息 + tool_response 拼回；
- **最多 3 轮**（`max_turns=3`）：避免无限循环；
- **每轮单独记录 ids / mask / old_logps**：用于后续 GRPO 风格的优势计算。

### 5.2 终止条件

```text
- 触发 EOS
- 检测到不含 tool_call 的 assistant 回复（模型主动结束）
- 达到 max_turns 上限
- 单条样本 rollout 耗时超过阈值
```

任意一个条件触发，rollout 就停止，进入 reward 计算阶段。

---

## 6. Reward 设计

Agentic RL 的 reward 是对**整条轨迹**联合打分：

$$
R(\tau) = R_{\text{answer}} + R_{\text{tool}} + R_{\text{format}} + R_{\text{rm}} - R_{\text{unfinished}}
$$

各项含义：

| Reward 项 | 含义 | 来源 |
|-----------|------|------|
| $R_{\text{answer}}$ | 最终答案是否正确 | 与 `gt` 字段比对（可模糊匹配、可数字比对） |
| $R_{\text{tool}}$ | 工具调用是否合法（参数是否齐全、是否能解析） | 规则函数 |
| $R_{\text{format}}$ | `<think>` / `tool_call` 标签是否成对闭合 | 正则校验 |
| $R_{\text{rm}}$ | RM 给的连续分数 | InternLM2-Reward |
| $R_{\text{unfinished}}$ | 未完成任务的惩罚 | rollout 状态判断 |

实现细节：

```python
rewards[idx] += 1.0 if gt == final_answer else 0.0    # R_answer
rewards[idx] += 0.5 if tool_call_valid else -0.5        # R_tool
rewards[idx] += 0.25 if '<think>' in resp and '</think>' in resp else 0.0  # R_format
rewards[idx] -= 1.0 if unfinished else 0.0              # R_unfinished
rewards[idx] += reward_model.get_score(...)              # R_rm
```

混合多种奖励源是 MiniMind 处理**奖励稀疏**的核心思路——对小模型来说，纯规则二元奖励往往全 0，必须配合连续信号。

---

## 7. 损失与策略更新

Agentic RL 的策略更新和 GRPO / CISPO **几乎一致**，差别仅在：

- **Rollout 是多轮的**，但 loss 计算时把整条轨迹的所有生成 token 当作一个长 completion 处理；
- **KL 锚点**仍然用 ref 模型（policy 初始化时的 SFT 权重）；
- **Policy ratio 和 advantage** 完全沿用 GRPO / CISPO 的实现。

参见 [07 - PPO / GRPO / CISPO](./07-ppo-grpo.md)。

---

## 8. Rollout Engine 的复用

Agentic RL 与 GRPO **共用同一个 rollout engine**（[`trainer/rollout_engine.py`](https://github.com/jingyaogong/minimind/blob/master/trainer/rollout_engine.py)）。它的接口非常简洁：

```python
rollout_result = rollout_engine.rollout(
    prompt_ids=...,
    attention_mask=...,
    num_generations=1,
    max_new_tokens=...,
    temperature=0.8,
)
# rollout_result.output_ids, .completion_ids, .completion_mask, .per_token_logps, .prompt_lens
```

训练侧不关心底层到底是本地 PyTorch generate 还是 SGLang 远端推理——切换只需要：

```bash
python train_agent.py --rollout_engine torch    # 默认
python train_agent.py --rollout_engine sglang --sglang_base_url http://localhost:8998
```

---

## 9. 启动训练

```bash
# ① 默认 torch rollout
torchrun --nproc_per_node N train_agent.py

# ② 使用 SGLang rollout（推荐：吞吐更高）
python -m sglang.launch_server --model-path ./minimind-3 --attention-backend triton --host 0.0.0.0 --port 8998
python train_agent.py --rollout_engine sglang --sglang_base_url http://localhost:8998 \
                      --sglang_shared_path ./ckpt_mm \
                      --data_path ../dataset/agent_rl_math.jsonl \
                      --use_wandb
```

输出权重保存为 `../out/agent_*.pth`。

推理测试：

```bash
python eval_toolcall.py --weight agent
```

预期效果（来自 README 实际样例）：

```text
💬: 帮我生成一个1到1000的随机数，然后计算它的平方
🧠: <tool_call>{"name": "random_number", "arguments": {"min": 1, "max": 1000}}</tool_call>
📞 [Tool Calling]: random_number
✅ [Tool Called]: {"result": 71}
🧠: <tool_call>{"name": "calculate_math", "arguments": {"expression": "71**2"}}</tool_call>
📞 [Tool Calling]: calculate_math
✅ [Tool Called]: {"result": "5041"}
🧠: 生成的1到1000的随机数是71，根据计算结果，71的平方等于5041。
```

注意这里有几个关键能力：

1. **连续多次工具调用**——先生成随机数，再调用计算；
2. **把工具结果拼回上下文**——把 71 作为参数传给 calculate_math；
3. **最终答案与工具结果一致**——回答里也提到了 71 和 5041。

---

## 10. Agentic RL 的设计哲学

MiniMind 的 README 里有一段非常坦诚的总结：

> “我自己很喜欢的一个训练脚本：它把 RLVR / RLAIF 风格的数据组织方式与 online RL 的 rollout 过程揉在了一起，中间来回调过很多版，也踩过收敛失败、奖励 hack、多轮上下文错位之类的 bug，最后仍然保持了 MiniMind 一贯的简洁性和可读性。”

具体来说：

- **最小串联**：模板组织 + 工具执行 + 多轮 rollout + 延迟奖励 + 训推分离，这些关键元素都被**最小化**实现；
- **可演进**：当前是**同步**模式（采样完一批再更新），但已经具备异步 rollout 的接口雏形；
- **保持原生**：不依赖 verl / openrlhf / slime 等大规模 RL 框架，但接口风格类似——训练侧负责 policy 更新，rollout 侧负责采样，中间通过轨迹和权重同步衔接。

作者自己评价：“虽然还远不是工业级 Agent 训练框架，但已经把关键元素真正实现了最小串联（也许目前没有比它更简洁的了）”。

---

## 11. 与普通 GRPO 的对比

| 维度 | 普通 GRPO | Agentic RL |
|------|-----------|-----------|
| 任务类型 | 单轮问答 | 多轮 Tool-Use |
| Rollout 长度 | 固定 `max_new_tokens` | 取决于模型何时停止调工具 |
| Reward 时机 | 回答结束立即打分 | 整条轨迹结束后统一打分 |
| Reward 来源 | RM + 长度 + 格式 | gt 校验 + 工具合法性 + RM + 格式 + 未完成惩罚 |
| 难度 | 容易上手 | 需要仔细设计 reward 与多轮 rollout 逻辑 |
| 业务场景 | 通用对话 / 推理 | 工具调用、Agent 工作流 |

---

## 12. 自检问题

1. Agentic RL 与单轮 GRPO 在数据格式、rollout 流程、reward 设计上的核心差异是什么？
2. `parse_tool_calls` 用什么方式解析模型生成的工具调用？为什么选择正则 + JSON 而不是某些专用解析器？
3. 整条轨迹的 reward 是怎么聚合的？为什么需要把 `gt` 校验、tool 合法性、格式、RM、未完成惩罚这几类信号一起用？
4. 多轮 rollout 中为什么要每轮重新构造 prompt，而不是把上一轮生成的 token 直接续到下一轮？
5. `signal.alarm(1)` 的作用是什么？它在什么场景下会触发？

---

## 13. 推荐阅读

- [Toolformer](https://arxiv.org/abs/2302.04761)——把工具调用融入 LLM 的早期工作。
- [ReAct](https://arxiv.org/abs/2210.03629)——Reasoning + Acting 的 Agent 范式。
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)——把 RL 推向大规模推理能力的代表。
- [verl / openrlhf](https://github.com/volcengine/verl)——工业级 Agentic RL 框架，MiniMind 的训推分离设计思路与之类似。
- [Tool Calling 数据：qwen3-4b 蒸馏](https://www.modelscope.cn/organization/Magpie-Align)——MiniMind 的 tool call 数据来源。

---

下一章：[09 - 推理与部署](./09-inference.md)