# 00 - 项目概览与代码地图

> 本章对应 MiniMind 仓库根目录与 [`README.md`](https://github.com/jingyaogong/minimind)，目标是对项目形成一个**全局视图**——它是什么、为什么存在、由哪些模块组成、读完代码后能在脑子里画出一张什么样的地图。

---

## 1. MiniMind 是什么？

MiniMind 是一个**从零开始训练的超小语言模型**项目。它最显眼的两个标签是：

- **小**：主线版本 `minimind-3` 只有 64M 参数，约为 GPT-3 (175B) 的 1/2700；
- **便宜**：在单卡 NVIDIA 3090 上，预训练 + SFT 全流程只需约 **2 小时、3 元人民币**。

更重要的是，它的所有核心算法都用 **PyTorch 原生实现**，不依赖 `transformers` / `trl` / `peft` 提供的高层抽象——这一点决定了它的真正定位：一份**LLM 入门教程式的可读源码**，而不是又一个大模型 API 封装。

主线版本沿用 Qwen3 / Qwen3-MoE 的设计语言：

- Decoder-Only Transformer；
- Pre-Norm + **RMSNorm**；
- **RoPE** 旋转位置编码（支持 YaRN 外推）；
- **SwiGLU** 激活函数；
- GQA（`q_heads=8, kv_heads=4`）；
- 可选 **MoE** 前馈层（去除 shared expert，对齐 Qwen3-MoE）。

---

## 2. 为什么要做这件事？

读 MiniMind 的 README 时，最值得反复咀嚼的是这一段：

> “用乐高自己拼出一架飞机，远比坐在头等舱里飞行更让人兴奋。”

当下 LLM 学习的现实是：

- 商业大模型规模太大，个人 GPU 跑不动也训不动；
- `transformers` / `trl` / `peft` 提供了非常高层、非常便利的接口，10～20 行代码就能完成“加载模型 + 数据集 + 训练 + 推理”，但这层便利把底层实现完全藏起来了；
- 网络上充斥着付费课程与营销内容，但很多讲解本身就漏洞百出。

MiniMind 想做的，就是**把门槛降到最低**：用最小的模型、最便宜的成本、**最贴近底层的代码**，让每个人都能从理解每一行代码开始，亲手训练出一个语言模型。

---

## 3. 训练链路全景

MiniMind 当前的训练链路，可以浓缩成下面这张图：

```text
┌────────────────────────────────────────────────────────────────┐
│                       MiniMind 训练链路                         │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   Pretrain          SFT           LoRA / DPO       RLHF/RLAIF  │
│  ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐   │
│  │文本语料  │ ──► │对话模板  │ ──► │垂域数据  │ ──► │偏好/奖励 │   │
│  │next-token│     │多轮对话  │     │参数高效  │     │PPO/GRPO │   │
│  │prediction│    │tool-call │     │fine-tune │     │CISPO    │   │
│  └─────────┘     └─────────┘     └─────────┘     └─────────┘   │
│       │              │                │                │        │
│       ▼              ▼                ▼                ▼        │
│   pretrain_*.pth  full_sft_*.pth   lora_xxx_*.pth   grpo_*.pth  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

- **必做**：Pretrain → Full SFT；
- **可选**：LoRA（垂域适配）、DPO（偏好对齐）、PPO/GRPO/CISPO（RLAIF）、Agentic RL（Tool-Use 多轮）、Knowledge Distillation（教师模型蒸馏）。

主线训练数据已经开源到 [ModelScope](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files) 与 [HuggingFace](https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/main)。

---

## 4. 代码地图

MiniMind 仓库的目录结构非常克制，可以一眼看完：

```text
minimind/
├── README.md                ← 主文档
├── README_en.md             ← 英文版
├── LICENSE                  ← Apache-2.0
├── CODE_OF_CONDUCT.md
├── requirements.txt
├── eval_llm.py              ← CLI 推理入口（必读）
│
├── model/                   ← 模型架构（核心：model_minimind.py）
│   ├── model_minimind.py    ← Transformer + RoPE + RMSNorm + Attention + MoE
│   ├── model_lora.py        ← LoRA 手写实现
│   ├── tokenizer.json       ← 自研 tokenizer
│   └── tokenizer_config.json
│
├── dataset/                 ← 数据集（核心：lm_dataset.py）
│   ├── lm_dataset.py        ← Pretrain / SFT / DPO / RLAIF / AgentRL Dataset
│   ├── __init__.py
│   └── dataset.md
│
├── trainer/                 ← 训练脚本
│   ├── train_pretrain.py    ← 预训练
│   ├── train_full_sft.py    ← 全参微调
│   ├── train_lora.py        ← LoRA 微调
│   ├── train_dpo.py         ← DPO 偏好学习
│   ├── train_ppo.py         ← PPO
│   ├── train_grpo.py        ← GRPO / CISPO
│   ├── train_agent.py       ← Agentic RL（Tool-Use）
│   ├── train_distillation.py← 知识蒸馏
│   ├── train_tokenizer.py   ← tokenizer 训练脚本
│   ├── rollout_engine.py    ← 训推分离的 rollout 抽象层
│   └── trainer_utils.py     ← 训练工具函数（init_model、checkpoint 等）
│
└── scripts/                 ← 推理 / 部署 / 工具脚本
    ├── web_demo.py          ← Streamlit WebUI
    ├── serve_openai_api.py  ← OpenAI-API 兼容服务端
    ├── chat_api.py          ← 客户端测试脚本
    ├── eval_toolcall.py     ← Tool Calling 评测
    └── convert_model.py     ← torch ↔ transformers 格式互转
```

如果你是第一次接触这个项目，建议按下面顺序阅读源码：

1. [`model/model_minimind.py`](https://github.com/jingyaogong/minimind/blob/master/model/model_minimind.py)——理解模型结构；
2. [`dataset/lm_dataset.py`](https://github.com/jingyaogong/minimind/blob/master/dataset/lm_dataset.py)——理解数据怎么变成模型输入；
3. [`trainer/train_pretrain.py`](https://github.com/jingyaogong/minimind/blob/master/trainer/train_pretrain.py) 与 [`trainer/train_full_sft.py`](https://github.com/jingyaogong/minimind/blob/master/trainer/train_full_sft.py)——理解训练主循环；
4. [`eval_llm.py`](https://github.com/jingyaogong/minimind/blob/master/eval_llm.py)——理解推理流程；
5. 再回头看 DPO / GRPO / Agent 等高级训练脚本。

---

## 5. 已发布模型一览

| 模型 | 参数量 | 架构 | 发布日期 | 备注 |
|------|--------|------|---------|------|
| minimind-3 | 64M | Dense | 2026.04.01 | 当前主线 |
| minimind-3-moe | 198M-A64M | MoE (4 experts / top-1) | 2026.04.01 | 当前主线 |
| minimind2-small | 26M | Dense | 2025.04.26 | 历史版本 |
| minimind2-moe | 145M | MoE | 2025.04.26 | 历史版本 |
| minimind2 | 104M | Dense (16 层) | 2025.04.26 | 历史版本 |
| minimind-v1-small | 26M | Dense | 2024.08.28 | 历史版本 |
| minimind-v1-moe | 4×26M | MoE | 2024.09.17 | 历史版本 |
| minimind-v1 | 108M | Dense | 2024.09.01 | 历史版本 |

主线笔记围绕 `minimind-3` / `minimind-3-moe` 展开。

---

## 6. 模型规模与硬件参考

训练开销的官方参考值（单卡 NVIDIA 3090）：

| 模型 | Pretrain (mini) | SFT (mini) | ToolCall | RLAIF |
|------|----------------|-----------|----------|-------|
| minimind-3 (64M) | ≈1.21h / ≈1.57¥ | ≈1.10h / ≈1.43¥ | ≈0.9h / ≈1.17¥ | ≈1.1h / ≈1.43¥ |
| minimind-3-moe (198M-A64M) | ≈1.69h / ≈2.20¥ | ≈1.54h / ≈2.00¥ | ≈1.26h / ≈1.64¥ | ≈1.54h / ≈2.00¥ |

也就是说，**P40 等级的消费级 GPU 也可以动手**，而 3090 / 4090 则能非常舒服地跑完全流程。CPU / MPS 理论上也能跑，但速度会显著变慢。

---

## 7. 一些值得提前知道的细节

- **Tokenizer**：自研 `minimind_tokenizer`，词表 6,400，使用 BPE + ByteLevel；中文压缩比约 `1.5~1.7 字符/token`，英文约 `4~5 字符/token`。主线训练统一沿用此词表，**不建议重新训练**。
- **位置编码**：默认 RoPE + `rope_theta=1e6`，最大位置 32,768。推理时支持 YaRN 外推到 16 倍甚至更高。
- **归一化**：RMSNorm（而不是 LayerNorm），与 Llama / Qwen 一致。
- **激活函数**：SwiGLU（`act_fn(gate(x)) * up(x)`），`intermediate_size ≈ ceil(hidden * π / 64) * 64`。
- **注意力**：Qwen3 风格的 GQA（8 query heads / 4 kv heads），Flash Attention 在 PyTorch ≥ 2.x 时自动启用。
- **MoE**：原生 PyTorch 实现 4 experts / top-1 routing；**未**采用 Triton / Megatron-style fused kernel，因此在 MoE 训练速度上有一定折中（比同尺寸 dense 慢约 50%），但保持了源码可读性。
- **chat_template**：内置 `<|im_start|>` / `<|im_end|>` 风格的多角色模板，支持 `<tool_call>` / `<tool_response>` / `<think>` 等特殊 token。
- **可视化**：[SwanLab](https://swanlab.cn/)（兼容 wandb API，国内访问友好）。

---

## 8. 自检问题

读完整章后，建议确认自己能否回答下列问题：

1. MiniMind 项目的核心定位是什么？和直接调用 `transformers` 库训练模型相比，它强调的最大差异是什么？
2. 主线训练链路必做的两步是什么？可选的高级训练阶段又有哪些？
3. `minimind-3` 的 Dense 版本和 MoE 版本在结构上有什么主要区别？两者默认的 `hidden_size` 与 `num_hidden_layers` 是多少？
4. 项目代码结构按 `model/`、`dataset/`、`trainer/`、`scripts/` 四个目录组织，每个目录大致承担什么职责？
5. 主线 `minimind-3` 的词表大小是多少？与主流大模型（Llama3 / Qwen2）相比词表差异有什么设计含义？

---

## 9. 推荐阅读

- [MiniMind README](https://github.com/jingyaogong/minimind/blob/master/README.md)——务必通读一遍，再回到本笔记精读。
- [MiniMind Discussions](https://github.com/jingyaogong/minimind/discussions)——作者维护的延伸话题，包括 dLM、Linear Attention、generate 方法详解等。
- [MLNLP-World/DeepLearning-MuLi-Notes](https://github.com/MLNLP-World/DeepLearning-MuLi-Notes)——本笔记的目录结构参考来源，建议同时阅读李沐《动手学习深度学习》对应章节作为深度学习基础补充。

---

下一章：[01 - 模型架构（Dense + MoE）](./01-model-architecture.md)