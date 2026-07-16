# 10 - 评测与对比

> 本章对应 MiniMind 评测相关的全部内容：模型间主观对比、benchmark 客观评测、RoPE 外推效果、Agent 任务测试等。**评测不是为了证明谁更强，而是为了理解不同训练阶段对模型行为的具体影响**。

---

## 1. 评测的多重维度

MiniMind 主线包含三种评测方式：

1. **RL 模型间主观对比**：同一 SFT 起点，分别用 GRPO / CISPO 训练，对比问答效果；
2. **不同模型间主观对比**：MiniMind / MiniMind-MoE / baby-llama2-chinese / chatlm-mini-chinese 在同一组 prompt 上的输出对比；
3. **客观 benchmark 评测**：用 lm-evaluation-harness 在 C-Eval / CMMLU / ARC-Easy / PIQA / OpenBookQA / HellaSwag / Social-IQa 上的得分；
4. **Tool-Use 评测**：在 20 道数学 Tool-Use 题上对比 `full_sft` 与 `agent` 权重的准确率；
5. **RoPE 外推评测**：在不同文本长度下，启用 YaRN 前后的 PPL 对比。

每一种评测方式都揭示了不同的训练现象，下面逐一展开。

---

## 2. RL 模型对比（SFT vs GRPO vs Agent-CISPO）

主线 README 给出了一个非常细致的对比实验——同一 SFT 起点，分别走三条路线：

- `[A] minimind-3 (64M, SFT)`
- `[B] minimind-3 (64M, GRPO)`
- `[C] minimind-3 (64M, Agent-CISPO)`

### 2.1 主观问答对比

节选自 README：

```text
[Q]: 鲁迅的《狂人日记》是如何批判封建礼教的？

[A] (SFT): 鲁迅的《狂人日记》是其作品中对封建礼教的批判，主要通过以下几个方面
        进行批判：1. 文学结构的变革：……

[B] (GRPO): 鲁迅的《狂人日记》是中国古典四大名著之一，全称为《后传》。
         这部作品通过细腻的笔触，展现了中国社会的复杂与深邃。……

[C] (Agent-CISPO): 鲁迅是中国现代文学史上第一位作家，他于1912年出版，
         自诞生以来便以诗歌为题，通过多次诠释封建礼教的复杂性与多面性。
         …… 共舞、共鸣、共舞、共进、共鸣、共舞……
```

观察：

- **A（SFT）**：基本事实准确，结构完整，逻辑清晰——是“标准答案型”回答；
- **B（GRPO）**：事实错误严重（《狂人日记》不是四大名著之一），但表达流畅、有结构；
- **C（Agent-CISPO）**：严重的重复循环（“共舞”反复出现），这是典型的 reward hacking——模型为了拿 RM 分数倾向于使用某些“讨喜”的修饰词。

这就是典型的**对齐税（alignment tax）**——RL 后训练虽然在某些奖励维度上更强，但通用能力、事实性会下降。

### 2.2 轻 Agent 任务对比

```text
[A] minimind-3 (full_sft)
1/20 ✅ (94)-35 gt=59 pred=59
2/20 ❌ 3**2 gt=9 pred=8
3/20 ✅ (29)+64 gt=93 pred=93
...
20/20 ✅ (348/(12))-(28)*(8) gt=-195 pred=-195
full_sft: 12/20 = 60.00%

[C] minimind-3 (agent)
1/20 ✅ (94)-35 gt=59 pred=59
2/20 ✅ 3**2 gt=9 pred=9
3/20 ✅ (29)+64 gt=93 pred=93
...
20/20 ❌ (348/(12))-(28)*(8) gt=-195 pred=3.625
agent: 17/20 = 85.00%
```

观察：

- `agent` 权重在 Tool-Use 任务上从 60% 提升到 85%——证明 Agentic RL 确实让模型更擅长“决定要不要调工具、再把可验证结果做对”；
- 但代价是**通用问答的事实性下降**——这也是 [07 - PPO / GRPO / CISPO](./07-ppo-grpo.md) 提到的“alignment tax”。

### 2.3 综合点评（来自 GPT-5.4 Thinking 的评判）

| 排名 | 模型 | 准确性 | 完整性 | 逻辑性 | 代码质量 | 总分 |
|----|------|-------|-------|-------|---------|------|
| 1  | B (GRPO) | 11 | 23 | 16 | 18 | 68 |
| 2  | D (chatlm-mini-chinese) | 25 | 19 | 15 | 3 | 62 |
| 3  | A (SFT) | 10 | 21 | 13 | 17 | 61 |
| 4  | C (baby-llama2-chinese) | 8 | 6 | 5 | 2 | 21 |

> B 虽然在知识准确性上幻觉严重（大熊猫吃海产品），但胜在表达流畅、结构清晰、代码题实现正确；
> D 的知识储备明显领先（牛顿1687年、长江发源地等全部正确），但代码题条件写反；
> A 和 B 风格接近，代码同样可用，但稳定性和知识准确性都不如 B；
> C 在事实性、展开能力和整体可读性上都有明显差距。

值得注意的是 D 和 A 的总分非常接近（62 vs 61），但优劣势分布几乎是互补的：

- D 赢在知识准确性（25 vs 10）；
- A 赢在代码能力（17 vs 3）。

这其实也反映了小参数模型的一个典型现象——**在有限的参数预算下，“说得好”和“说得对”往往很难兼得**。

---

## 3. 不同模型对比

与 baby-llama2-chinese、chatlm-mini-chinese 在同一组 prompt 上的对比：

```text
[Q]: 海水为什么是咸的？

[A] (minimind-3, 64M): 海水的咸度是由海水的重量以及海水的盐度共同作用的结果。
                         这种变化主要是由于海水在阳光照射下会吸收水中的盐分……
                         （答非所问 + 编造"光散射"）

[B] (minimind-3-moe, 198M): 海水的咸度是由其所含物质的盐分占据的，这一现象
                            可以从几个方面来解释：1. 盐分占据的作用……
                            （表达流畅但知识混乱）

[C] (baby-llama2-chinese, 200M): 因为水是许多生物的生存所必需的，它能够吸收和
                                释放二氧化碳，释放氧气……
                                （完全答非所问）

[D] (chatlm-mini-chinese, 200M): 海水是咸的主要原因是因为其中含有大量的溶解盐类，
                                其中最主要的是氯化钠（NaCl）。氯化钠是一种非常常见的
                                盐类……（答对了核心，后段出现重复退化）
```

观察：

- B 的表达最流畅、结构最好，但知识错误严重（“共舞”修饰词泛滥）；
- D 的知识准确性最高，但生成后段容易重复退化；
- A 风格接近 B，但稳定性和准确性都更弱；
- C 在事实性、展开能力、可读性上都有明显差距。

整体排名（按主观观感）：`B > D > A > C`。

---

## 4. 客观 Benchmark 评测

### 4.1 评测框架与数据集

使用 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)：

```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness && pip install -e .
```

```bash
HF_ENDPOINT=https://hf-mirror.com lm_eval \
  --model hf \
  --model_args pretrained="/path/to/model",dtype=auto \
  --tasks "ceval-valid,cmmlu,arc_easy,piqa,openbookqa,hellaswag,social_iqa" \
  --batch_size 16 \
  --device cpu \
  --trust_remote_code \
  --apply_chat_template
```

> ⚠️ 这类选择题测评通常不是让模型自由生成完整答案，而是给定题目上下文 `y` 和若干候选项 `x`，直接比较各候选项的条件概率 `p(x | y)`，并取概率最大的选项作为预测结果。**随机作答的准确率往往就是很强的下界**，这个量级的模型也确实长期徘徊在这个附近。

### 4.2 评测结果

| 模型 | 来源 | 参数量 | zh (ceval / cmmlu) | en (arc / piqa / obqa / hellaswag / siqa) |
|------|------|--------|---------------------|----------------------------------------|
| minimind-3 | current | 64M | 24.89 / 25.38 | 28.49 / 50.65 / 23.60 / 28.28 / 34.19 |
| minimind-3-moe | current | 198M | 25.48 / 24.32 | 27.74 / 50.71 / 26.20 / 27.43 / 34.03 |
| minimind-3-exam | current | 64M | 30.98 / 26.12 | 35.61 / 56.26 / 24.20 / 28.40 / 34.19 |
| Steel-LLM | ZhanShiJin | 1121M | 24.89 / 25.32 | 39.69 / 65.13 / 26.00 / 35.73 / 39.15 |
| gpt2-medium | OpenAI | 360M | 23.18 / 25.00 | 43.60 / 66.38 / 30.20 / 39.38 / 39.10 |
| TinyLlama-1.1B | TinyLlama | 1100M | 25.71 / 25.03 | 54.80 / 74.43 / 35.60 / 60.38 / 43.09 |
| SmolLM2-135M | HuggingFace | 135M | 24.44 / 24.71 | 58.50 / 68.17 / 32.80 / 43.15 / 39.46 |
| Aquila-135M | BAAI | 135M | 25.19 / 25.10 | 54.59 / 67.52 / 34.40 / 41.67 / 39.66 |

### 4.3 结果解读

#### 中文任务（CEval / CMMLU）

- 所有小模型都在 25% 左右——基本就是“随机选 4 选 1”附近的水平；
- MiniMind 数据偏中文，因此与同尺寸模型相比相对有优势；
- `minimind-3-exam` 通过在 CEval/MMLU test 子集上做轻量 LoRA 对齐，把中文选择题从 ~25% 拉到 ~30%——说明**小模型的瓶颈未必完全在知识本身，也可能在于输入格式没有对齐**。

#### 英文任务

- MiniMind 英文数据少，因此英文表现弱于专门训练英文的同尺寸模型；
- SmolLM2-135M 和 Aquila-135M 因为专门训练英文，在 ARC-Easy 等任务上明显领先；
- TinyLlama-1.1B 凭借大参数量在所有任务上都更强。

#### minimind-3-exam 的来源

> minimind-3-exam 不是更大的基座模型，也几乎没有额外注入新知识。它仅基于 minimind-3 在 [lora_exam.jsonl](https://huggingface.co/datasets/jingyaogong/minimind_dataset/blob/main/lora_exam.jsonl) 上做了一次轻量 LoRA 对齐。这部分数据由 ceval 与 mmlu（英文）的 test 子集抽样构成，并做了前缀、后缀等格式增强，主要作用是对齐选择题评测中常见的上下文与候选项组织形式，而不是学习题目答案。

#### 数据污染警告

> 本节评估所用的 7 个数据集与上述对齐数据无样本交集，因此可以认为不存在数据污染。
> 相反，若直接在有交集的数据上微调，这类小模型的分数往往会明显失真；例如 minimind-3 在有污染的 ceval / cmmlu 子集上实验跑到过约 97% 准确率，但这种结果没有参考意义。

---

## 5. RoPE 长度外推评测

主线 README 给出了在不同文本长度下启用 YaRN 前后的 PPL 对比：

- 文本长度 < 训练长度（2048）：YaRN 启用前后 PPL 几乎无差异；
- 文本长度 > 训练长度：不启用 YaRN，PPL 急剧上升；启用 YaRN，PPL 显著下降，但仍高于训练长度内的 PPL。

这说明 YaRN 是一种**实用但有损**的外推方案——在多数场景下够用，但与原生训练到对应长度相比仍有差距。

---

## 6. 评测方法论的几点提醒

1. **选择题 ≠ 真实能力**：选择题的随机下界就很高，对小模型意义有限；
2. **主观对比 ≠ 客观结论**：GPT-5.4 Thinking 的评分有一定主观性，**仅供娱乐**；
3. **数据规模决定知识量**：MiniMind 训练数据远小于表中其他模型（特别是英文），英文表现弱是必然的；
4. **格式对齐 vs 真实能力**：`minimind-3-exam` 案例提醒我们，对小模型来说，**让模型见过评测格式**可能比“多塞知识”更立竿见影；
5. **数据污染的隐性影响**：所有评测前都要确认数据无样本交集，否则分数不可信；
6. **RLHF 后训练的 alignment tax**：用 RM 优化出来的模型可能在某些维度更弱（事实性、通用对话），需要权衡。

---

## 7. 自检问题

1. 同一 SFT 起点分别走 GRPO 和 Agent-CISPO，为什么前者在通用问答上更强、后者在 Tool-Use 上更强？
2. `minimind-3-exam` 在选择题评测上大幅提升（24.89 → 30.98），这个提升的本质是什么？是不是数据污染？
3. 客观选择题 benchmark 对小模型评测有什么固有限制？为什么 MiniMind 在这些任务上都徘徊在 25% 左右？
4. YaRN 在什么长度下效果最好？什么长度下效果会退化？

---

## 8. 推荐阅读

- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)——目前事实标准的大模型评测框架。
- [C-Eval](https://cevalbenchmark.com/)——中文综合评测 benchmark。
- [CMMLU](https://github.com/haonan-li/CMMLU)——中文多任务语言理解 benchmark。
- [MobileLLM](https://arxiv.org/abs/2402.14905)——小模型 scaling law 研究；解释了 MiniMind 64M 与同尺寸模型的能力天花板。
- [Alignment Tax](https://arxiv.org/abs/2009.01325)——RLHF 后训练带来的通用能力损失现象。

---

## 收束语

到这里，MiniMind 学习笔记的 11 个章节已经全部走完。

回到 [00 - 项目概览](./00-overview.md) 里那张训练链路图，你会发现：每个章节其实都在回答这张图里的某个具体问题。

- **模型架构**（[01](./01-model-architecture.md)）定义了图里的“模型节点”；
- **Tokenizer 与数据集**（[02](./02-tokenizer-dataset.md)）定义了图的“输入边”；
- **Pretrain / SFT / LoRA**（[03](./03-pretrain.md) / [04](./04-sft.md) / [05](./05-lora.md)）是三条核心训练路径；
- **DPO / PPO/GRPO/CISPO**（[06](./06-dpo.md) / [07](./07-ppo-grpo.md)）是对齐的两条支线；
- **Agentic RL**（[08](./08-agentic-rl.md)）是把图拓展到多轮交互场景的尝试；
- **推理与部署**（[09](./09-inference.md)）是把训练产物释放出去的通道；
- **评测与对比**（[10](./10-evaluation.md)）是对整张图的反向度量。

希望这份笔记能帮助你从一个“小白”变成一个“看得懂 MiniMind 每一行代码”的读者。

如果有任何疏漏或错误，欢迎通过 Issues / PR 反馈，让我们一起把这套 LLM 入门教程打磨得更好。

— wyl250130