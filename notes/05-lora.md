# 05 - LoRA 微调

> 本章对应 [`model/model_lora.py`](https://github.com/jingyaogong/minimind/blob/master/model/model_lora.py)（约 60 行）与 [`trainer/train_lora.py`](https://github.com/jingyaogong/minimind/blob/master/trainer/train_lora.py)。LoRA 是 MiniMind 主线里**最便宜的微调方式**，整个 LoRA 文件只有几十行，非常适合作为 PEFT 入门读物。

---

## 1. LoRA 的核心思想

LoRA（Low-Rank Adaptation）出自 Microsoft 2021 年的论文 [*LoRA: Low-Rank Adaptation of Large Language Models*](https://arxiv.org/abs/2106.09685)。它的核心假设是：

> 模型在适配下游任务时，权重的更新量 $\Delta W$ 是**低秩**的——可以用两个低秩矩阵的乘积来表示。

因此，对一个 `in_features × out_features` 的权重矩阵 $W_0$，LoRA 把更新建模为：

$$
W = W_0 + \Delta W = W_0 + B A
$$

其中 $A \in \mathbb{R}^{r \times \text{in}}, B \in \mathbb{R}^{\text{out} \times r}$，秩 $r \ll \min(\text{in}, \text{out})$。

前向计算：

$$
y = W_0 x + B A x
$$

训练时**冻结** $W_0$，**只更新** $A, B$：

- 参数量从 $d \times k$ 降到 $r \times (d + k)$，当 $r \ll d, k$ 时大幅减少；
- 多个任务可以共享 $W_0$，只切换 $(A, B)$，节省存储。

---

## 2. MiniMind 的 LoRA 实现（[model_lora.py:6-18](https://github.com/jingyaogong/minimind/blob/master/model/model_lora.py#L6)）

整个 LoRA 模块只有十几行：

```python
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        # 矩阵 A 高斯初始化
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # 矩阵 B 全 0 初始化
        self.B.weight.data.zero_()

    def forward(self, x):
        return self.B(self.A(x))
```

两点细节非常关键：

1. **A 用高斯初始化，B 用 0 初始化**：保证初始时 $BA = 0$，即 $\Delta W = 0$，模型输出与原模型完全一致——LoRA 训练起点和原模型“无缝衔接”。
2. **没有 bias**：和原始 Linear 保持一致，不引入额外参数。

---

## 3. `apply_lora`——把 LoRA 挂到模型上（[model_lora.py:21-32](https://github.com/jingyaogong/minimind/blob/master/model/model_lora.py#L21)）

```python
def apply_lora(model, rank=16):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.in_features == module.out_features:
            lora = LoRA(module.in_features, module.out_features, rank=rank).to(model.device)
            setattr(module, "lora", lora)
            original_forward = module.forward

            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)

            module.forward = forward_with_lora
```

它做了三件事：

1. **遍历所有 Linear 层**，筛选出**方阵**（`in_features == out_features`，典型如 Q/K/V/O 投影、FFN 的 gate/up/down 的部分投影）；
2. **挂一个 `lora` 子模块**到原 Linear 层；
3. **替换 forward**，使其输出 = 原始输出 + LoRA 输出。

> ⚠️ MiniMind 的实现只对方阵 Linear 加 LoRA。这与 PEFT 库默认对所有 Linear 都加的做法不同——优点是参数更省，缺点是灵活性低。如果你想对非方阵（如 FFN 的 down_proj）也加 LoRA，需要修改 `apply_lora` 的判断条件。

`forward_with_lora` 中用 `layer1=..., layer2=lora` 的默认参数把外层变量固化进闭包，避免了 `for` 循环中常见的“闭包晚绑定”陷阱——这是一个值得学习的 Python 小技巧。

---

## 4. 加载、保存、合并 LoRA 权重

### 4.1 `save_lora`（[model_lora.py:45-53](https://github.com/jingyaogong/minimind/blob/master/model/model_lora.py#L45)）

```python
def save_lora(model, path):
    raw_model = getattr(model, '_orig_mod', model)
    state_dict = {}
    for name, module in raw_model.named_modules():
        if hasattr(module, 'lora'):
            clean_name = name[7:] if name.startswith("module.") else name
            lora_state = {f'{clean_name}.lora.{k}': v.cpu().half() for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
    torch.save(state_dict, path)
```

只保存 LoRA 的 `A` / `B` 权重，文件名形如 `lora_medical_768.pth`，体积极小（一般几 MB）。

### 4.2 `load_lora`（[model_lora.py:35-42](https://github.com/jingyaogong/minimind/blob/master/model/model_lora.py#L35)）

```python
def load_lora(model, path):
    state_dict = torch.load(path, map_location=model.device)
    state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}

    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            lora_state = {k.replace(f'{name}.lora.', ''): v for k, v in state_dict.items() if f'{name}.lora.' in k}
            module.lora.load_state_dict(lora_state)
```

按层名匹配，把 `A` / `B` 权重回填。

### 4.3 `merge_lora`（[model_lora.py:56-65](https://github.com/jingyaogong/minimind/blob/master/model/model_lora.py#L56)）——导出完整模型

```python
def merge_lora(model, lora_path, save_path):
    load_lora(model, lora_path)
    raw_model = getattr(model, '_orig_mod', model)
    state_dict = {k: v.cpu().half() for k, v in raw_model.state_dict().items() if '.lora.' not in k}
    for name, module in raw_model.named_modules():
        if isinstance(module, nn.Linear) and '.lora.' not in name:
            state_dict[f'{name}.weight'] = module.weight.data.clone().cpu().half()
            if hasattr(module, 'lora'):
                state_dict[f'{name}.weight'] += (module.lora.B.weight.data @ module.lora.A.weight.data).cpu().half()
    torch.save(state_dict, save_path)
```

把 $W = W_0 + BA$ 直接合并成一个新的完整 Linear 权重，再保存——这样就可以把 LoRA “固化”回基础模型，得到一个不依赖额外模块的完整模型，方便部署到 vllm / ollama 等不支持动态 LoRA 的推理引擎。

仓库还提供了 [`scripts/convert_model.py`](https://github.com/jingyaogong/minimind/blob/master/scripts/convert_model.py) 包装这个流程：

```bash
cd scripts && python convert_model.py
```

可以选择 `merge_base_lora` 把基础模型与 LoRA 权重合并成新的完整权重。

---

## 5. 训练循环（[trainer/train_lora.py](https://github.com/jingyaogong/minimind/blob/master/trainer/train_lora.py)）

LoRA 的训练循环与 full_sft 几乎一样，只有两个差异：

1. **加载基础模型后立刻 `apply_lora`**；
2. **优化器只更新 LoRA 参数**：

```python
model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
apply_lora(model, rank=args.lora_rank)

lora_params = []
for name, param in model.named_parameters():
    if 'lora' in name:
        lora_params.append(param)
optimizer = optim.AdamW(lora_params, lr=args.learning_rate)
```

训练脚本默认基础模型是 `full_sft`，可以换成 `pretrain` 或其他权重。

典型用法：

```bash
cd trainer && python train_lora.py \
  --data_path ../dataset/lora_medical.jsonl \
  --from_weight full_sft \
  --lora_rank 8 \
  --epochs 5
```

训练过程中只有 LoRA 参数被更新，主干模型保持冻结——这意味着**显存占用大幅下降**，CPU 上也能跑出不错的速度。

---

## 6. 推理：基础模型 + LoRA

```bash
python eval_llm.py --weight full_sft --lora_weight lora_medical
```

`eval_llm.py` 内部会：

1. 加载 `full_sft` 基础模型；
2. 调用 `apply_lora` 把 LoRA 子模块挂上；
3. 调用 `load_lora` 加载对应权重。

详见 [09 - 推理与部署](./09-inference.md)。

---

## 7. 几个工程要点

### 7.1 rank 怎么选？

主线默认 `rank=16`，适合大部分任务：

- rank 太小（4/8）：表达能力不足，可能学不到领域差异；
- rank 太大（64/128）：参数过多，容易过拟合小数据；
- 通常 8～32 是合理区间。

### 7.2 应该对哪些层加 LoRA？

MiniMind 默认只对方阵 Linear 加 LoRA，这对应 MiniMind 模型中的：

- `q_proj`、`k_proj`、`v_proj`、`o_proj`（Attention 中的 Q/K/V/O 投影）；
- FFN 中 `in == out` 的部分（具体由模型定义决定）。

这种“只加 QVO”的策略在实践中通常已经够用，且参数量最小。如果需要进一步提升效果，可以改为对所有 Linear 加 LoRA：

```python
if isinstance(module, nn.Linear):
    ...
```

### 7.3 训练数据格式

LoRA 训练数据与 full_sft 完全相同——`{"conversations": [...]}` 多轮对话格式：

```jsonl
{"conversations": [
  {"role": "user", "content": "你叫什么名字？"},
  {"role": "assistant", "content": "您好，我叫 MiniMind..."}
]}
{"conversations": [
  {"role": "user", "content": "请问颈椎病的人枕头多高才最好？"},
  {"role": "assistant", "content": "颈椎病患者选择枕头的高度应该根据..."}
]}
```

---

## 8. LoRA vs Full SFT 的取舍

| 维度 | LoRA | Full SFT |
|------|------|----------|
| 显存 | 很低 | 高（与模型规模成正比） |
| 训练速度 | 快 | 慢 |
| 适用任务 | 垂域适配、轻量个性化 | 大规模风格/能力变化 |
| 数据需求 | 几千~几万条即可 | 通常需要更多 |
| 可合并 | 可以 merge 回基础模型 | 本身就是完整模型 |
| 风险 | 低（不会破坏基础能力） | 较高（容易遗忘） |

实际工程中一个非常常见的**最佳实践**是：

1. 先用 LoRA 快速验证数据和方案的可行性；
2. 确认可行后再考虑是否值得做 full_sft。

这也是 MiniMind 同时提供 `train_lora.py` 和 `train_full_sft.py` 的原因。

---

## 9. 自检问题

1. LoRA 的核心数学假设是什么？为什么“低秩”假设在小模型微调中往往也成立？
2. MiniMind 的 `LoRA.__init__` 中 A 用高斯初始化、B 用 0 初始化——这样设计的好处是什么？
3. `apply_lora` 中只对**方阵 Linear**加 LoRA，背后的设计取舍是什么？
4. `merge_lora` 的核心数学操作是什么？合并后的模型能否继续训练？为什么？
5. LoRA 与 Full SFT 在显存占用、训练数据需求、效果上各有什么优劣？

---

## 10. 推荐阅读

- [LoRA 原始论文](https://arxiv.org/abs/2106.09685)——Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*.
- [QLoRA](https://arxiv.org/abs/2305.14314)——把 LoRA 与 4-bit 量化结合，进一步降低显存。
- [AdaLoRA](https://arxiv.org/abs/2303.10512)——自适应调整每个 LoRA 模块的秩。
- [PEFT 库文档](https://huggingface.co/docs/peft)——工业界标准的 PEFT 实现。

---

下一章：[06 - DPO 偏好学习](./06-dpo.md)