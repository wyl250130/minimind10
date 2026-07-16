# 03 - 预训练 Pretrain

> 本章对应 [`trainer/train_pretrain.py`](https://github.com/jingyaogong/minimind/blob/master/trainer/train_pretrain.py) 与 [`dataset/lm_dataset.py:PretrainDataset`](https://github.com/jingyaogong/minimind/blob/master/dataset/lm_dataset.py#L37)。预训练是 LLM “学会说话”的起点，也是 MiniMind 主线必做的第一步。

---

## 1. 预训练在做什么？

预训练的本质是**让模型学会高质量的“词语接龙”**。给定一段文本，模型的任务是预测下一个 token；通过海量文本训练，模型逐渐内化：

- 词与词之间的搭配规律；
- 句法与语义结构；
- 世界知识与常识。

这一阶段是**无监督 / 自监督**的——不需要人工标注，只需要构造好的纯文本语料。

具体到 MiniMind，`PretrainDataset` 把每条 JSONL 文本样本 tokenize 后拼成：

```text
[BOS] token_1 token_2 ... token_n [EOS] [PAD] [PAD] ...
```

模型通过 `F.cross_entropy(logits[..., :-1, :], labels[..., 1:])` 学会“在看到 `token_1...token_k` 时预测 `token_{k+1}`”。

---

## 2. 启动训练

主线命令：

```bash
cd trainer && python train_pretrain.py
```

或者多卡 DDP：

```bash
torchrun --nproc_per_node N train_pretrain.py
```

输出权重文件保存在 `../out/pretrain_*.pth`，`*` 为 `hidden_size`（默认 768）。

常用参数（[train_pretrain.py:84-107](https://github.com/jingyaogong/minimind/blob/master/trainer/train_pretrain.py#L84)）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 2 | 训练轮数 |
| `--batch_size` | 32 | 单卡 batch size |
| `--learning_rate` | 5e-4 | 初始学习率 |
| `--accumulation_steps` | 8 | 梯度累积步数 |
| `--max_seq_len` | 340 | 单条样本最大 token 长度 |
| `--data_path` | `../dataset/pretrain_t2t_mini.jsonl` | 数据路径 |
| `--from_weight` | `none` | 从哪个权重继续训练，`none` 表示从零 |
| `--from_resume` | 0 | 是否自动检测 checkpoint 续训 |
| `--use_moe` | 0 | 是否使用 MoE |
| `--use_wandb` / `--use_compile` | 0 | 可视化 / torch.compile |

---

## 3. 训练主循环（[train_pretrain.py:24-80](https://github.com/jingyaogong/minimind/blob/master/trainer/train_pretrain.py#L24)）

整个训练脚本可以分成 9 段（脚本注释里也是这么划分的），最核心的是第 8 段：

```python
def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)

        # 1) 动态调整学习率（cosine schedule）
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 2) 前向 + 损失
        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss = res.loss + res.aux_loss       # aux_loss 仅在 MoE 下非零
            loss = loss / args.accumulation_steps

        # 3) 反向 + 累积
        scaler.scale(loss).backward()

        # 4) 到累积步数时统一更新参数
        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # 5) 日志
        if step % args.log_interval == 0 or step == iters:
            ... Logger(...); wandb.log(...)

        # 6) 周期保存
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            ... torch.save(...); lm_checkpoint(...)
```

要点逐条解释：

### 3.1 学习率调度

`get_lr`（[trainer_utils.py:40](https://github.com/jingyaogong/minimind/blob/master/trainer/trainer_utils.py#L40)）：

```python
def get_lr(current_step, total_steps, lr):
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))
```

是一个从 `lr*0.55` 起、先平稳再 cosine 衰减到 `lr*0.1` 的调度器。**没有 warmup**，这是 MiniMind 的一个工程取舍——它的训练周期短（小时级），warmup 的边际收益不大。

### 3.2 混合精度

```python
autocast_ctx = torch.cuda.amp.autocast(dtype=dtype)  # bfloat16 或 float16
scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
```

- 默认 `bfloat16`：数值范围与 float32 一致，无需 GradScaler；
- 选 `float16` 时启用 GradScaler 防止下溢。

### 3.3 梯度累积

```python
loss = loss / args.accumulation_steps
...
if step % args.accumulation_steps == 0:
    ... step(optimizer)
```

通过把一个 mini-batch 拆成 8 步累积再更新，等效 batch size = `32 × 8 = 256`。这是小显存训练大 batch 的常用 trick。

### 3.4 梯度裁剪

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # 默认 1.0
```

防止 loss spike / 数据噪声造成的梯度爆炸。

### 3.5 训练状态保存

每次保存权重会同时落两份文件：

- `<weight>_<hidden_size>.pth`（或 `_<hidden_size>_moe.pth`）：纯 state_dict，用于推理；
- `<weight>_<hidden_size>_resume.pth`（在 `../checkpoints/` 下）：包含 model / optimizer / scaler / epoch / step / wandb_id 等，用于续训。

`--from_resume 1` 会自动检测 `../checkpoints/` 下对应的 resume 文件并恢复——**注意它甚至会自动调整 step**（[trainer_utils.py:111-114](https://github.com/jingyaogong/minimind/blob/master/trainer/trainer_utils.py#L111)）：

```python
saved_ws = ckp_data.get('world_size', 1)
current_ws = dist.get_world_size() if dist.is_initialized() else 1
if saved_ws != current_ws:
    ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
```

这是为了支持**跨 GPU 数量**的续训——把 8 卡训练中断后换 4 卡继续，不会丢步数。

---

## 4. 几个值得品味的工程细节

### 4.1 续训与随机种子

```python
setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
...
setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
```

每个 epoch 都用 `42 + epoch` 作为新种子，并重新生成随机索引——这样即使中途续训，**同一 epoch 内每个进程看到的样本仍然不同**，但不同 epoch 之间又是确定性的，便于复现实验。

### 4.2 SkipBatchSampler

`SkipBatchSampler` 是 MiniMind 自己实现的、用于跳过若干个已训练 step 的 batch sampler，结构上和 PyTorch 内置 `BatchSampler` 类似，但接收一个 `skip` 参数：

```python
class SkipBatchSampler(Sampler):
    ...
    def __iter__(self):
        for i in range(self.skip, len(self.sampler)):
            if (i - self.skip) % self.batch_size == 0:
                yield list(self.sampler[i:i + self.batch_size])
```

这是实现 `--from_resume 1` 时不丢 batch、不重训的关键。

### 4.3 Aux Loss

```python
loss = res.loss + res.aux_loss
```

`res.aux_loss` 是所有 MoE 层 `router_aux_loss_coef × load × score` 的总和，仅在 `use_moe=True` 时非零。Dense 模型训练时它恒为 0，可以安全相加。

---

## 5. 训练成本与效果参考

`minimind-3` (64M) 在 3090 单卡上的经验值：

| 数据 | 时长 | 成本 |
|------|------|------|
| `pretrain_t2t_mini` (1.2GB) | ≈ 1.21 h | ≈ 1.57 ¥ |
| `pretrain_t2t` (10GB) | ≈ 多日级 | — |

零模型（仅 pretrain）就已经具备**基本的词语接龙**能力——能写出“天空之所以看起来是蓝色的，主要是因为……”这样的回答，但还不能稳定地回答问题、保持角色。简单测试：

```bash
python eval_llm.py --weight pretrain
```

会看到回答通顺但缺少 SFT 阶段学会的“对话格式”和“指令跟随”能力。

---

## 6. pretrain 与 SFT 的关系

作者有一句非常精辟的总结：

> “模型在这一阶段的核心目标就是**学会高质量地词语接龙**。例如输入‘秦始皇’，它要能够继续生成‘是中国历史上的第一位皇帝’这类符合语义与常识的后续内容。”

Pretrain 学的是**语言模型本身**，SFT 才学的是**对话格式 + 指令跟随 + 工具调用**。二者不能互相替代——这正是主线必做两步的根本原因。

---

## 7. 自检问题

1. 为什么 MiniMind 的学习率调度器是 cosine 但没有 warmup？这在小模型上是否合理？
2. 梯度累积的本质是什么？`accumulation_steps=8, batch_size=32` 等效 batch size 是多少？
3. `--from_resume 1` 的核心机制是什么？为什么需要按 GPU 数量做 step 调整？
4. 在 MoE 模型上，`loss = res.loss + res.aux_loss` 的两 part 分别是什么？为什么要相加？
5. Pretrain 与 SFT 在数据格式、训练目标上的核心区别是什么？

---

## 8. 推荐阅读

- [Improving Language Models by Retrieving from Trillions of Tokens](https://arxiv.org/abs/2112.04426)——讨论了“数据 > 参数”的小模型现象。
- [MobileLLM](https://arxiv.org/abs/2402.14905)——深入研究小模型的 scaling law；MiniMind 的 `dim=768, n_layers=8` 配置参考了此文。
- [Karpathy nanoGPT](https://github.com/karpathy/nanoGPT)——另一个值得对照阅读的“小而美”训练循环实现。

---

下一章：[04 - 全参微调 Full SFT](./04-sft.md)