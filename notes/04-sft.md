# 04 - 全参微调 Full SFT

> 本章对应 [`trainer/train_full_sft.py`](https://github.com/jingyaogong/minimind/blob/master/trainer/train_full_sft.py) 与 [`dataset/lm_dataset.py:SFTDataset`](https://github.com/jingyaogong/minimind/blob/master/dataset/lm_dataset.py#L58)。SFT 是主线训练的第二步、也是让模型“从能接龙到能对话”的关键。

---

## 1. SFT 在做什么？

预训练后的模型虽然语言能力不错，但还**不会对话**——它只会接龙，你问它问题，它可能继续把问题“接”下去。

SFT（Supervised Fine-Tuning）的目标，就是把模型从“词语接龙器”调成“指令跟随的助手”：

- 学会多轮对话的格式（`<|im_start|>` / `<|im_end|>` 包裹的多角色结构）；
- 学会在恰当位置停止（`<EOS>`）；
- 学会基本的工具调用（`<tool_call>` / `<tool_response>`）；
- 学会显式思考（`<think>...</think>`）。

SFT 并不只是“把模型调成更会聊天”。当数据规模足够大时（主线 SFT 数据高达 14GB），它还兼具**mid-training** 的功能——继续把特定知识分布、任务模式、助手风格压进参数。

---

## 2. 数据格式回顾

详见 [02 - Tokenizer 与数据集](./02-tokenizer-dataset.md)。SFT 数据的主格式：

```jsonl
{
  "conversations": [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！"},
    {"role": "user", "content": "请介绍一下自己"},
    {"role": "assistant", "content": "我是一个 AI 助手..."}
  ]
}
```

主线数据已经混入 tool_call 样本（约 10w 条，基于 qwen3-4b 蒸馏），所以默认的 full_sft 权重就具备基础 Tool Call 能力。

---

## 3. 启动训练

```bash
cd trainer && python train_full_sft.py
```

常用参数（与 train_pretrain 几乎一致，只是默认值略不同）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--from_weight` | `pretrain` | 从哪个权重继续训练 |
| `--data_path` | `../dataset/sft_t2t_mini.jsonl` | SFT 数据路径 |
| `--max_seq_len` | 2048 | 最大截断长度 |
| `--epochs` | 2 | 训练轮数 |
| `--batch_size` | 16 | 单卡 batch size |
| `--learning_rate` | 5e-5（远小于 pretrain 的 5e-4）| 学习率 |
| `--save_weight` | `full_sft` | 权重前缀名 |

> ⚠️ **学习率**：SFT 阶段通常需要把学习率降到 pretrain 的 1/10 量级。Pretrain 是从零学知识，SFT 是把已有知识按特定格式“对齐”，太大步长会冲掉 pretrain 阶段的能力。

输出权重保存为 `../out/full_sft_*.pth`（`*` 为 hidden_size），用于后续推理或下游对齐。

---

## 4. 训练循环的关键差异

相比 `train_pretrain.py`，`train_full_sft.py` 在主循环上几乎是“复制粘贴”。但有几个**关键差异**：

### 4.1 数据集不同

```python
from dataset.lm_dataset import SFTDataset
train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
```

`SFTDataset` 内部会自动调用 `tokenizer.apply_chat_template` 渲染多轮对话，再通过 `generate_labels` 生成只对 assistant 段置 1 的 loss mask。

### 4.2 系统提示与空思考处理

```python
def __getitem__(self, index):
    sample = self.samples[index]
    conversations = pre_processing_chat(sample['conversations'])  # 20% 概率补 system
    prompt = self.create_chat_prompt(conversations)
    prompt = post_processing_chat(prompt)                          # 80% 概率移除空 think
    input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
    input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
    labels = self.generate_labels(input_ids)
    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
```

这套处理让模型在训练时见过：

- **有时**带 system prompt、有时不带；
- **有时**含完整思考、有时是空思考、有时没有思考。

从而学到“该思考时思考、该直答时直答”的混合模式（详见 [08 - Agentic RL](./08-agentic-rl.md) 提到的 Adaptive Thinking）。

### 4.3 续训起点不同

```python
model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
```

`from_weight` 默认是 `pretrain`，也可以指定 `full_sft` 继续微调（这通常用于在已有对话模型基础上做领域适配）。

---

## 5. 为什么 SFT 后效果提升明显？

主要原因有三：

1. **数据规模大**：主线 SFT 数据 14GB，相当于在 pretrain 1.2GB mini 数据上继续塞入一个数量级的额外知识；
2. **格式对齐**：模型学会了 chat_template 的完整结构，包括 system / user / assistant / tool 四种角色的边界；
3. **任务多样**：SFT 数据覆盖问答、写作、代码、翻译、推理、tool call 等多种任务，使模型成为一个“通才助手”。

简单测试 SFT 后的模型：

```bash
python eval_llm.py --weight full_sft
```

会看到模型能够稳定以助手身份回答问题，并具备基本的工具调用格式。

---

## 6. 进阶玩法：垂域 SFT

SFT 的一个常见用法是**在已有 full_sft 基础上继续微调特定领域**。例如想做一个医疗助手：

```jsonl
{"conversations": [
  {"role": "user", "content": "请问颈椎病的人枕头多高才最好？"},
  {"role": "assistant", "content": "颈椎病患者选择枕头的高度应该根据..."}
]}
{"conversations": [
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}
]}
```

可以单独建一个 `medical.jsonl`，然后：

```bash
python train_full_sft.py \
  --data_path ../dataset/medical.jsonl \
  --from_weight full_sft \
  --epochs 3 --learning_rate 1e-5
```

但要注意：

- 一定要混入通用数据，否则容易“灾难性遗忘”，把已有对话能力破坏掉；
- 全参 SFT 比 LoRA 更容易过拟合垂域，谨慎调节学习率和 epoch。

如果希望更低成本做垂域适配，参见 [05 - LoRA 微调](./05-lora.md)。

---

## 7. 训练曲线与典型现象

观察 SFT 阶段的 loss 曲线可以发现几个现象：

1. **前期 loss 下降快**：模型刚学会 chat 格式，loss 从一个相对高的位置快速下降；
2. **中期平稳收敛**：模型开始拟合具体任务模式；
3. **后期微抖**：典型的“训练-验证”gap 开始显现，必要时可以提前停止。

如果 loss 下降很慢或抖动剧烈，通常需要检查：

- 数据格式是否正确（`apply_chat_template` 输出是否符合预期）；
- loss mask 是否生效（debug 时可以把 labels 中非 `-100` 的位置打印出来核对）；
- 学习率是否过大（pretrain 阶段的学习率套到 SFT 是非常常见的错误）。

---

## 8. SFT 的局限

SFT 把模型的“对齐格式”学得很牢，但也带来一个微妙的副作用——**模型会倾向于模仿 SFT 数据中的回答风格**，包括：

- 表达冗长（很多 SFT 答案偏长）；
- 礼貌口癖（“当然可以”、“以下是...”）；
- 在不确定时“凑答案”而不是说“不知道”。

这些副作用正是后续 RLHF / RLAIF 阶段需要修正的问题。

---

## 9. 自检问题

1. SFT 与 Pretrain 在数据格式、训练目标、学习率设置上的关键区别是什么？
2. `loss mask` 在 SFT 中为什么比 Pretrain 更重要？
3. `pre_processing_chat` 的两个开关分别模拟了推理时的哪些真实场景？
4. SFT 数据集为什么已经混入 tool_call 样本？这与 v1 时代的设计有什么不同？
5. 在 full_sft 基础上继续做垂域微调时，最容易出现的灾难性遗忘是什么？如何缓解？

---

## 10. 推荐阅读

- [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206)——探讨“高质量少量数据”的 SFT 哲学。
- [Self-Instruct](https://arxiv.org/abs/2212.10560)——SFT 数据合成方法之一，MiniMind 的部分合成数据也用了类似思路。
- [Toolformer](https://arxiv.org/abs/2302.04761)——把工具调用能力融入 SFT 的早期工作。

---

下一章：[05 - LoRA 微调](./05-lora.md)