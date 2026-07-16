# 02 - Tokenizer 与数据集

> 本章对应 [`model/`](https://github.com/jingyaogong/minimind/tree/master/model) 与 [`dataset/lm_dataset.py`](https://github.com/jingyaogong/minimind/blob/master/dataset/lm_dataset.py)。Tokenizer 决定了模型看到的世界长什么样，Dataset 决定了模型学什么。本章先讲 MiniMind 的自研分词器，再讲 5 类训练数据集的格式与构造方式。

---

## 1. Tokenizer 概述

MiniMind 的分词器（[`model/tokenizer.json`](https://github.com/jingyaogong/minimind/blob/master/model/tokenizer.json) + [`tokenizer_config.json`](https://github.com/jingyaogong/minimind/blob/master/model/tokenizer_config.json)）有以下特点：

| 维度 | MiniMind Tokenizer | 主流对比 |
|------|--------------------|----------|
| 词表大小 | **6,400** | Llama3=128,000；Qwen2=151,643 |
| 算法 | BPE + ByteLevel | 同 GPT-2 / RoBERTa 系列 |
| 中文压缩比 | ≈1.5~1.7 字符/token | Qwen2 ≈1.0~1.2 字符/token |
| 英文压缩比 | ≈4~5 字符/token | Llama3 ≈4~5 字符/token |
| 特殊 token | BOS / EOS / PAD / `<tool_call>` / `<tool_response>` / `<think>` | — |

### 1.1 为什么词表这么小？

对小模型来说，词表大小直接影响 embedding 层（`vocab × hidden`）和 lm_head（`hidden × vocab`）的参数占比：

$$
\text{embedding 参数占比} = \frac{2 \cdot \text{vocab\_size} \cdot \text{hidden\_size}}{\text{总参数量}}
$$

主线 `minimind-3` 总参 64M、hidden 768、vocab 6400，embedding 占约 7.7%。如果用 Llama3 的 128k 词表，embedding 会变成约 154M，比 dense 主体还大，完全不划算。

### 1.2 训练自定义 Tokenizer

仓库提供 [`trainer/train_tokenizer.py`](https://github.com/jingyaogong/minimind/blob/master/trainer/train_tokenizer.py) 作为训练示例。**但作者强烈不建议重新训练**：

- 词表和切分规则一旦变化，模型权重、数据格式、推理接口与社区生态的兼容性都会下降；
- 会削弱模型的“传播性”；
- 跨 tokenizer 比较时，BPB（Bits Per Byte）比 PPL 更可靠。

主线训练统一沿用现成的 `minimind_tokenizer`，仅在历史早期版本（v1）尝试过 Mistral tokenizer，已下线。

---

## 2. `lm_dataset.py` 概览

[`dataset/lm_dataset.py`](https://github.com/jingyaogong/minimind/blob/master/dataset/lm_dataset.py) 定义了 5 个核心 Dataset 类：

| 类名 | 数据格式 | 训练阶段 |
|------|---------|----------|
| `PretrainDataset` | `{"text": "..."}` | Pretrain |
| `SFTDataset` | `{"conversations": [...]}` | Full SFT、LoRA、Reason |
| `DPODataset` | `{"chosen": [...], "rejected": [...]}` | DPO（RLHF） |
| `RLAIFDataset` | `{"conversations": [...]}` | PPO / GRPO / CISPO |
| `AgentRLDataset` | `{"conversations": [...], "gt": "..."}` | Agentic RL |

外加两个 chat 预处理函数：

- `pre_processing_chat`：按 20% 概率补一条 system prompt（tool-call 样本直接保留，不补）；
- `post_processing_chat`：按 20% 概率移除空 `<think>\n\n</think>\n\n` 标签，强制模型学习“该思考时思考、该直答时直答”的混合模式。

---

## 3. `PretrainDataset`（[lm_dataset.py:37-55](https://github.com/jingyaogong/minimind/blob/master/dataset/lm_dataset.py#L37)）

最朴素的 next-token prediction 数据集：

```python
def __getitem__(self, index):
    sample = self.samples[index]
    tokens = self.tokenizer(str(sample['text']), add_special_tokens=False,
                            max_length=self.max_length - 2, truncation=True).input_ids
    tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
    input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    labels = input_ids.clone()
    labels[input_ids == self.tokenizer.pad_token_id] = -100
    return input_ids, labels
```

要点：

- 句子截断长度留 2 给 BOS / EOS；
- 右侧 padding；
- labels 把 padding 位置设为 `-100`，在 `F.cross_entropy(ignore_index=-100)` 中被忽略。

预训练数据格式：

```jsonl
{"text": "如何才能摆脱拖延症？治愈拖延症并不容易，但以下建议可能有所帮助。"}
{"text": "清晨的阳光透过窗帘洒进房间，桌上的书页被风轻轻翻动。"}
```

主线预训练数据为 `pretrain_t2t_mini.jsonl`（约 1.2GB）和 `pretrain_t2t.jsonl`（约 10GB）。

---

## 4. `SFTDataset`（[lm_dataset.py:58-119](https://github.com/jingyaogong/minimind/blob/master/dataset/lm_dataset.py#L58)）

SFT 数据集是 MiniMind 中最复杂的一个，因为它需要处理**多角色对话 + tool_call + thinking** 三种情况。

### 4.1 数据格式

```jsonl
{
  "conversations": [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！"},
    {"role": "user", "content": "再见"},
    {"role": "assistant", "content": "再见！"}
  ]
}
{
  "conversations": [
    {"role": "system", "content": "# Tools ...", "tools": "[...]"},
    {"role": "user", "content": "把'你好世界'翻译成english"},
    {"role": "assistant", "content": "", "tool_calls": "[{...}]"},
    {"role": "tool", "content": "{\"translated_text\":\"Hello World\"}"},
    {"role": "assistant", "content": "Hello World"}
  ]
}
```

注意 `tool_calls` 挂在 assistant 消息上、`tools` 挂在 system 消息上。这种结构与 OpenAI Chat Completions API 风格一致，便于后续接入第三方 UI。

### 4.2 `create_chat_prompt`

```python
def create_chat_prompt(self, conversations):
    messages = []
    tools = None
    for message in conversations:
        message = dict(message)
        if message.get("role") == "system" and message.get("tools"):
            tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
        if message.get("tool_calls") and isinstance(message["tool_calls"], str):
            message["tool_calls"] = json.loads(message["tool_calls"])
        messages.append(message)
    return self.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, tools=tools
    )
```

核心逻辑：

1. 把 `tools` 从 system message 里抽出来；
2. 把 `tool_calls` 字符串反序列化成 list；
3. 调用 tokenizer 内置的 chat_template，把多轮消息**渲染成一个完整字符串**。

`chat_template`（定义在 `tokenizer_config.json` 里）负责把每条消息加上 `<|im_start|>{role}\n{content}<|im_end|>\n` 这样的包装，并在 tool_call / tool_response 处插入 `<tool_call>...</tool_call>` / `<tool_response>...</tool_response>` 标签。

### 4.3 `generate_labels`——Loss Mask 的关键

SFT 的一个核心问题是：**只对 assistant 的回复计算 loss**，不能让模型学“用户说的话”。MiniMind 用一个非常巧妙的方式实现 loss mask：

```python
def generate_labels(self, input_ids):
    labels = [-100] * len(input_ids)
    i = 0
    while i < len(input_ids):
        if input_ids[i:i + len(self.bos_id)] == self.bos_id:
            # bos_id 对应 "<BOS>assistant\n"
            start = i + len(self.bos_id)
            end = start
            while end < len(input_ids):
                if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                    break
                end += 1
            for j in range(start, min(end + len(self.eos_id), self.max_length)):
                labels[j] = input_ids[j]
            i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
        else:
            i += 1
    return labels
```

逻辑：

- 找到每一段 `<BOS>assistant\n` → `<EOS>` 之间的 token 区间；
- 把这段区间的 labels 设为原始 token id，其余保持 `-100`；
- 这样训练时只有 assistant 的回复会贡献 loss。

这套机制完全靠 tokenizer 内置的 `<|im_start|>` / `<|im_end|>` 边界实现，所以**前提是 chat_template 输出里 assistant 段的前后边界要和 `bos_id` / `eos_id` 精确对齐**——这正是 MiniMind 的 `tokenizer.json` 设计如此精细的原因。

---

## 5. `DPODataset`（[lm_dataset.py:122-192](https://github.com/jingyaogong/minimind/blob/master/dataset/lm_dataset.py#L122)）

DPO 数据集接收的是成对的 `(chosen, rejected)` 多轮对话：

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

`__getitem__` 把 chosen 和 rejected 各跑一遍 chat_template，生成 `(x_chosen, y_chosen, mask_chosen)` 与 `(x_rejected, y_rejected, mask_rejected)`。这里的 `mask` 同样是只对 assistant 段置 1 的 loss mask。

数据采样自 [DPO-En-Zh-20k](https://huggingface.co/datasets/llamafactory/DPO-En-Zh-20k)，中文数据做了额外补充。

---

## 6. `RLAIFDataset`（[lm_dataset.py:195-224](https://github.com/jingyaogong/minimind/blob/master/dataset/lm_dataset.py#L195)）

RLAIF（PPO / GRPO / CISPO）阶段的数据和 SFT 几乎一样，只是最后一个 assistant 字段是空的——因为这一阶段的回复由**当前策略模型实时采样生成**。

```json
{
  "conversations": [
    {"role": "user", "content": "请解释一下什么是光合作用？"},
    {"role": "assistant", "content": "无"}
  ]
}
```

`__getitem__` 直接返回 `prompt` 字符串 + 空 `answer` 字符串，由 rollout engine 决定如何采样。

`create_chat_prompt` 里有一点小细节：

```python
use_thinking = random.random() < self.thinking_ratio
return self.tokenizer.apply_chat_template(
    conversations[:-1], tokenize=False,
    open_thinking=use_thinking,
    add_generation_prompt=True,
)
```

- `conversations[:-1]` 把最后那条 assistant 消息去掉；
- `add_generation_prompt=True` 让模板自动补上 `<BOS>assistant\n`；
- `open_thinking=use_thinking` 控制是否预先注入 `<think>` 标签。

`thinking_ratio` 默认 0.5，让模型在训练时一半时间思考、一半时间直答。

---

## 7. `AgentRLDataset`（[lm_dataset.py:226-252](https://github.com/jingyaogong/minimind/blob/master/dataset/lm_dataset.py#L226)）

Agentic RL 在多轮 tool-use 场景下训练，数据格式多了一个 `gt`（ground truth）字段，用来事后校验最终答案是否正确：

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

`__getitem__` 返回 `(messages, tools, gt)`，rollout engine 会：

1. 用当前 policy 模型生成第一条 assistant 回复（可能含 tool_call）；
2. 检测 tool_call 并执行工具，把结果拼回上下文；
3. 继续让模型生成下一条 assistant 回复；
4. 直到触发 EOS 或达到最大步数；
5. 用 `gt` 校验最终答案，结合规则 / RM 给出整条轨迹的总 reward。

---

## 8. Chat 模板的几个细节

### 8.1 System 注入概率

```python
def pre_processing_chat(conversations, add_system_ratio=0.2):
    # tool use 数据完整保留不做处理
    if any(conv.get('tools') for conv in conversations): return conversations
    ...
    if conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations
```

20% 的概率注入一条随机 system prompt，目的是**让模型见过“有时有 system、有时没有 system”的训练分布**——这与推理时常见的灵活调用方式（用户可选是否带 system）相匹配。

### 8.2 空思考标签的处理

```python
def post_processing_chat(prompt_content, empty_think_ratio=0.2):
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content
```

以 80% 的概率移除“空思考”标签，避免模型学会“机械地输出空 think 再回答”的退化模式。这是 MiniMind v3 引入 **Adaptive Thinking** 的关键一环。

### 8.3 Tool Use 数据完整性

Tool-Use 样本**不会**被注入 system prompt，也不会被处理空 think 标签——因为这些样本的训练目标是精确学会 OpenAI 风格的 tool call 协议，多余的扰动会影响对齐。

---

## 9. 数据量参考

主线训练所需的核心数据集：

```text
./dataset/
├── agent_rl.jsonl         (86MB)   ← Agentic RL
├── agent_rl_math.jsonl    (18MB)   ← Agentic RL（数学）
├── dpo.jsonl              (53MB)   ← DPO 偏好
├── pretrain_t2t_mini.jsonl(1.2GB) ✨ ← 轻量预训练
├── pretrain_t2t.jsonl     (10GB)   ← 主线预训练
├── rlaif.jsonl            (24MB) ✨ ← RLAIF
├── sft_t2t_mini.jsonl     (1.6GB) ✨ ← 轻量 SFT（已含 Tool Call）
└── sft_t2t.jsonl          (14GB)   ← 主线 SFT
```

✨为推荐项。最快复现 `MiniMind Zero` 对话模型，仅需 `pretrain_t2t_mini` + `sft_t2t_mini`。

---

## 10. 自检问题

1. MiniMind 为什么选择自研 tokenizer 而不是直接用 Qwen2 / Llama3 的 tokenizer？词表大小对小模型的参数占比影响有多大？
2. SFT 数据集的 `loss mask` 是怎么实现的？为什么需要 loss mask？
3. `pre_processing_chat` 在什么情况下会跳过？为什么 tool-call 样本必须跳过？
4. `RLAIFDataset` 为什么要把 `conversations[:-1]` 截断掉最后一条 assistant？`thinking_ratio` 控制的是什么？
5. `AgentRLDataset` 与 `RLAIFDataset` 的本质区别是什么？为什么 Agentic RL 需要 `gt` 字段？

---

## 11. 推荐阅读

- [BPE 算法原始论文](https://arxiv.org/abs/1508.07909)——Sennrich et al., *Neural Machine Translation of Rare Words with Subword Units*.
- [SentencePiece / ByteLevel BPE 综述](https://huggingface.co/docs/transformers/tokenizer_summary)
- [DPO 数据集：DPO-En-Zh-20k](https://huggingface.co/datasets/llamafactory/DPO-En-Zh-20k)
- [Magpie-Align](https://www.modelscope.cn/organization/Magpie-Align)——SFT 主线数据来源之一。
- [R1-Distill-SFT](https://www.modelscope.cn/datasets/AI-ModelScope/R1-Distill-SFT)——reasoning 数据来源。

---

下一章：[03 - 预训练 Pretrain](./03-pretrain.md)