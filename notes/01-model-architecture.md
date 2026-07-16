# 01 - 模型架构（Dense + MoE）

> 本章对应 [`model/model_minimind.py`](https://github.com/jingyaogong/minimind/blob/master/model/model_minimind.py)（约 290 行），这是整个 MiniMind 项目的“心脏”。我们会沿着 `MiniMindConfig → RMSNorm → RoPE → Attention → FeedForward/MoE → MiniMindBlock → MiniMindModel → MiniMindForCausalLM` 的顺序，把每一块拆开讲清楚。

---

## 1. 整体架构一览

MiniMind 是一个 **Decoder-Only Transformer**（即 GPT 风格），由 N 个相同的 Block 堆叠而成，每个 Block 包含两部分：

```text
输入 x
   │
   ├──► RMSNorm ──► Multi-Head Attention (with RoPE & GQA & Flash-Attn) ──► + ──► x1
   │                                                                    │
   x ◄──────────────────────────────────────────────────────────────────┘
   │
   ├──► RMSNorm ──► FeedForward (SwiGLU) 或 MOEFeedForward ──► + ──► x2
   │
   x1 ◄──────────────────────────────────────────────────────────────┘
   │
   ▼
   (下一层 Block)
```

最后再过一层 RMSNorm + 一个共享权重的 `lm_head`，输出词表上的 logits。

主线 `minimind-3` 默认配置：

| 参数 | 值 |
|------|---|
| `hidden_size` | 768 |
| `num_hidden_layers` | 8 |
| `num_attention_heads` | 8 |
| `num_key_value_heads` | 4（GQA） |
| `head_dim` | 96 |
| `intermediate_size` | ≈ 3072（`ceil(768 * π / 64) * 64`） |
| `max_position_embeddings` | 32,768 |
| `rope_theta` | 1e6 |
| `vocab_size` | 6,400 |
| `rms_norm_eps` | 1e-6 |
| `tie_word_embeddings` | True |

---

## 2. `MiniMindConfig`（[`model_minimind.py:10-45`](https://github.com/jingyaogong/minimind/blob/master/model/model_minimind.py#L10)）

MiniMind 的 Config 直接继承自 `transformers.PretrainedConfig`，因此可以无缝挂到 HuggingFace 生态上。配置项总体分为两类：

1. **基础架构参数**：`hidden_size`、`num_hidden_layers`、`num_attention_heads`、`num_key_value_heads`、`head_dim`、`intermediate_size`、`max_position_embeddings`、`vocab_size` 等；
2. **行为开关**：`flash_attn`、`use_moe`、`inference_rope_scaling`、`tie_word_embeddings` 等。

其中比较有意思的几个设计：

- `intermediate_size = ceil(hidden_size * π / 64) * 64`：用 π 做了一个近似取整，目的是让 FFN 的隐藏维度是 64 的倍数，从而在 cuBLAS / cuDNN 上获得更高效的 kernel 选择。Llama2 也用过类似 trick。
- `head_dim = hidden_size / num_attention_heads`：默认每头 96 维；如果不指定，会由 Config 自动算出。
- `inference_rope_scaling`：是否启用 YaRN 外推。打开后 `rope_scaling` 字段会自动填充一份默认的 YaRN 配置（`factor=16, beta_fast=32, beta_slow=1`）。
- `num_experts` / `num_experts_per_tok` / `moe_intermediate_size` / `router_aux_loss_coef`：仅在 `use_moe=True` 时生效。

---

## 3. `RMSNorm`（[`model_minimind.py:50-60`](https://github.com/jingyaogong/minimind/blob/master/model/model_minimind.py#L50)）

RMSNorm（Root Mean Square Layer Normalization）是 LayerNorm 的简化版，只用**均方根**做归一化，不减均值：

$$
\mathrm{RMSNorm}(x) = \frac{x}{\sqrt{\mathrm{mean}(x^2) + \epsilon}} \cdot \gamma
$$

实现细节：

- `weight` 初始化为 1，shape = `(dim,)`，即可学习的 scale；
- 用 `torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)` 计算反平方根，避免显式 `sqrt`；
- 在内部把 `x` 转成 float32 计算，最后再 `.type_as(x)` 转回去，避免低精度数值问题。

相比 LayerNorm，RMSNorm 少了一次均值计算与一次减法，训练更稳定、工程实现更便宜。Llama / Qwen / Gemma 等现代 LLM 都采用了 RMSNorm。

---

## 4. RoPE 旋转位置编码（[`model_minimind.py:62-84`](https://github.com/jingyaogong/minimind/blob/master/model/model_minimind.py#L62)）

### 4.1 基本公式

RoPE（Rotary Position Embedding）通过**对 Query / Key 做二维平面内的旋转变换**来编码位置信息。设第 $m$ 个位置的 $q$ 向量中第 $i$ 个二维子空间 $(q_{2i}, q_{2i+1})$，RoPE 的做法是把它旋转 $m\theta_i$：

$$
\begin{pmatrix} q'_{2i} \\ q'_{2i+1} \end{pmatrix} = \begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{pmatrix} \begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}
$$

其中 $\theta_i = \theta^{-2i/d}$，$\theta$ 即 `rope_theta`（MiniMind 默认 1e6）。

实现上更高效的等价形式：

```text
q' = q * cos + rotate_half(q) * sin
```

其中 `rotate_half(x)` 把后半段取负号拼到前面。这是 MiniMind 中 `apply_rotary_pos_emb` 的核心逻辑（[model_minimind.py:80-84](https://github.com/jingyaogong/minimind/blob/master/model/model_minimind.py#L80)）。

### 4.2 预计算 cos/sin

MiniMind 在 `MiniMindModel.__init__` 里一次性把 `[0, max_position_embeddings)` 内的 cos / sin 表都算好，作为 buffer 注册（`persistent=False`，不写入 state_dict）：

```python
freqs_cos, freqs_sin = precompute_freqs_cis(
    dim=config.head_dim,
    end=config.max_position_embeddings,
    rope_base=config.rope_theta,
    rope_scaling=config.rope_scaling,
)
self.register_buffer("freqs_cos", freqs_cos, persistent=False)
self.register_buffer("freqs_sin", freqs_sin, persistent=False)
```

推理时只需要 `freqs_cos[start_pos:start_pos + seq_length]` 切片即可，避免每步重新计算。

### 4.3 YaRN 外推（可选）

当 `inference_rope_scaling=True` 时，MiniMind 会启用 **YaRN** 算法（Yet another RoPE extensioN）：

- 用 ramp 函数把高频维度（与小位置相关）的频率保持不变，把低频维度（与大位置相关）的频率乘以 `1/factor`；
- 同时乘上 `attention_factor`（默认 1.0）作为 attention 的幅度补偿。

效果上，它能让一个仅在 2048 长度上训练过的模型，**无需重新训练**就能在 32k 长度上稳定推理。MiniMind 的 README 中给出了 YaRN 启用前后 PPL 的对比曲线，可以看到长文本下 PPL 显著下降。

---

## 5. `Attention`（[`model_minimind.py:91-134`](https://github.com/jingyaogong/minimind/blob/master/model/model_minimind.py#L91)）

MiniMind 的 Attention 沿用了 Qwen3 的几个关键设计：

### 5.1 GQA 分组查询注意力

```python
self.n_local_heads = config.num_attention_heads        # 8
self.n_local_kv_heads = config.num_key_value_heads     # 4
self.n_rep = self.n_local_heads // self.n_local_kv_heads  # 2
```

GQA（Grouped Query Attention）让多个 Query Head 共享同一组 K/V Head：

```text
Query Heads:   [Q0, Q1] [Q2, Q3] [Q4, Q5] [Q6, Q7]
                └─┬─┘    └─┬─┘    └─┬─┘    └─┬─┘
KV Heads:      [K0/V0]  [K1/V1]  [K2/V2]  [K3/V3]
```

参数上比 MQA（所有 Q 共享 1 组 K/V）多，但表达力更强；显存上比 MHA 少一半 K/V。`repeat_kv` ([model_minimind.py:86-89](https://github.com/jingyaogong/minimind/blob/master/model/model_minimind.py#L86)) 负责把 KV 头重复扩展到与 Q 头对齐。

### 5.2 QK Norm

每个 head 的 Q / K 都先过一次 RMSNorm：

```python
self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
...
xq, xk = self.q_norm(xq), self.k_norm(xk)
```

这是一种**注意力层面的归一化** trick，能显著稳定训练，避免 QK 内积爆炸。Llama3 等最新模型也采用了类似做法。

### 5.3 Flash Attention

当 PyTorch 版本支持 `scaled_dot_product_attention` 并且配置允许时，MiniMind 会优先调用 Flash Attention：

```python
self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn
...
if self.flash and ...:
    output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=..., is_causal=self.is_causal)
```

否则回退到手写实现：上三角 mask + softmax + 加权求和。手写实现里有一个非常容易踩坑的小细节——causal mask 用的是 `[..., -seq_len:]`，这样在增量解码（KV cache）阶段也能正确处理。

### 5.4 KV Cache

`forward` 接受 `past_key_value` 和 `use_cache` 两个参数：

```python
if past_key_value is not None:
    xk = torch.cat([past_key_value[0], xk], dim=1)
    xv = torch.cat([past_key_value[1], xv], dim=1)
past_kv = (xk, xv) if use_cache else None
```

这种写法比 `transformers` 库里的 `DynamicCache` 更“手写”——更适合教学场景，但功能一致。

---

## 6. `FeedForward` 与 `MOEFeedForward`

### 6.1 `FeedForward`（SwiGLU）（[`model_minimind.py:136-146`](https://github.com/jingyaogong/minimind/blob/master/model/model_minimind.py#L136)）

```python
def forward(self, x):
    return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

SwiGLU 激活（来自 PaLM / Llama）：

$$
\mathrm{FFN}(x) = \mathrm{down}\!\left(\mathrm{SiLU}(\mathrm{gate}(x)) \odot \mathrm{up}(x)\right)
$$

- `gate_proj`：从 `hidden → intermediate`；
- `up_proj`：从 `hidden → intermediate`；
- `act_fn`：SiLU（也叫 Swish）；
- `down_proj`：从 `intermediate → hidden`。

中间用 element-wise 乘法把两条通路“门控”起来，效果比纯 ReLU/GELU 更平滑。

### 6.2 `MOEFeedForward`（[`model_minimind.py:148-176`](https://github.com/jingyaogong/minimind/blob/master/model/model_minimind.py#L148)）

```python
class MOEFeedForward(nn.Module):
    def __init__(self, config):
        ...
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([FeedForward(...) for _ in range(config.num_experts)])
```

每个 token 经过一个**路由（Gate）**网络，输出 `num_experts` 个概率分布，然后选 top-k 个专家：

```python
scores = F.softmax(self.gate(x_flat), dim=-1)
topk_weight, topk_idx = torch.topk(scores, k=config.num_experts_per_tok, dim=-1, sorted=False)
if config.norm_topk_prob:
    topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
```

接着对每个专家，把分给它的 token 单独 forward 一次，再按权重加权写回：

```python
y = torch.zeros_like(x_flat)
for i, expert in enumerate(self.experts):
    mask = (topk_idx == i)
    if mask.any():
        token_idx = mask.any(dim=-1).nonzero().flatten()
        weight = topk_weight[mask].view(-1, 1)
        y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
```

**负载均衡 Loss**：

```python
load = F.one_hot(topk_idx, config.num_experts).float().mean(0)
self.aux_loss = (load * scores.mean(0)).sum() * config.num_experts * config.router_aux_loss_coef
```

这一项 loss 会在最终 `loss + aux_loss` 中以 `router_aux_loss_coef`（默认 5e-4）的系数加入，避免所有 token 都路由到同一个专家。

> ⚠️ MiniMind 的 MoE 实现是**原生 PyTorch + 循环**的写法，不依赖 fused kernel。优势是代码可读；代价是比同尺寸 dense 慢约 50%，且 expert 数量继续增加时调度开销会急剧变大。

---

## 7. `MiniMindBlock`、`MiniMindModel`、`MiniMindForCausalLM`

### 7.1 `MiniMindBlock`（[`model_minimind.py:178-194`](https://github.com/jingyaogong/minimind/blob/master/model/model_minimind.py#L178)）

标准 Pre-Norm Decoder Block：

```python
def forward(self, hidden_states, position_embeddings, ...):
    residual = hidden_states
    hidden_states, present_key_value = self.self_attn(
        self.input_layernorm(hidden_states), position_embeddings, ...
    )
    hidden_states += residual
    hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
    return hidden_states, present_key_value
```

注意两点：

- Pre-Norm：norm 在残差前；与 Post-Norm 相比训练更稳定；
- MLP 路由到 `FeedForward` 或 `MOEFeedForward`，由 `config.use_moe` 决定。

### 7.2 `MiniMindModel`（[`model_minimind.py:196-232`](https://github.com/jingyaogong/minimind/blob/master/model/model_minimind.py#L196)）

把 N 个 Block 串起来，加上 embedding、dropout、最后一层 norm：

- `embed_tokens`：词嵌入；
- `dropout`：embedding 之后的 dropout；
- `layers`：N 个 Block；
- `norm`：最后一层 RMSNorm；
- `freqs_cos / freqs_sin`：RoPE buffer。

`forward` 流程：

1. 拿到 `input_ids` → embedding；
2. 切出当前 step 对应的 RoPE 位置编码；
3. 依次过 N 个 Block，累积 `presents`（KV cache）；
4. 最后一层 norm；
5. 汇总所有 MoE 层的 `aux_loss`，作为返回值之一。

### 7.3 `MiniMindForCausalLM`（[`model_minimind.py:234-288`](https://github.com/jingyaogong/minimind/blob/master/model/model_minimind.py#L234)）

包一层 `lm_head`，把 hidden states 映射到词表 logits：

```python
self.model = MiniMindModel(self.config)
self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
if self.config.tie_word_embeddings:
    self.model.embed_tokens.weight = self.lm_head.weight
```

`tie_word_embeddings=True` 表示 embedding 与 lm_head 共享权重，能省下约 `vocab * hidden` ≈ 6400×768 ≈ 4.9M 参数，对小模型来说比例可观。

`forward` 返回 `MoeCausalLMOutputWithPast`（兼容 HuggingFace 的命名），内部把 aux_loss 一并返回供训练循环使用。

损失计算直接用 `F.cross_entropy`：

```python
loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
```

`ignore_index=-100` 是关键：训练时数据集会用 `-100` 标记需要 mask 的位置（如 system / user 段、padding），这些位置就不参与 loss 计算。

### 7.4 `generate`（[model_minimind.py:256-288](https://github.com/jingyaogong/minimind/blob/master/model/model_minimind.py#L256)）

MiniMind 自己手写了一个 `generate`，覆盖了 KV cache、temperature、top_k、top_p、repetition_penalty 等常见采样策略，**不依赖** `transformers.GenerationMixin` 的高层接口：

```python
for _ in range(max_new_tokens):
    past_len = past_key_values[0][0].shape[1] if past_key_values else 0
    outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache, ...)
    logits = outputs.logits[:, -1, :] / temperature
    if repetition_penalty != 1.0: ...
    if top_k > 0: ...
    if top_p < 1.0: ...
    next_token = torch.multinomial(...) if do_sample else torch.argmax(...)
    input_ids = torch.cat([input_ids, next_token], dim=-1)
    past_key_values = outputs.past_key_values if use_cache else None
    if finished.all(): break
```

这种“自给自足”的实现有几个好处：

- 教学场景下，KV cache 与采样策略是一目了然的；
- 不依赖 `transformers` 的版本兼容；
- 与 MiniMind 自己的 MoE / YaRN 等细节完全对齐。

---

## 8. 几个值得品味的工程细节

1. **RoPE buffer 在 meta device init 后丢失**：[model_minimind.py:215-218](https://github.com/jingyaogong/minimind/blob/master/model/model_minimind.py#L215) 显式做了检测——如果 `freqs_cos[0, 0] == 0`，说明是在 meta device 上初始化的（transformers ≥ 5.x 的行为），需要重新计算。这是个非常隐蔽但常见的 bug。

2. **Flash 路径的判断条件**：[model_minimind.py:125](https://github.com/jingyaogong/minimind/blob/master/model/model_minimind.py#L125) 同时检查了 `seq_len > 1`、`not self.is_causal or past_key_value is None`、`attention_mask is None or all 1`，避免在奇异输入下走 Flash 出现 NaN。

3. **QK Norm**：这个细节很多 LLM 教程都不会提，但它对训练稳定性帮助非常大。读源码时值得专门留意。

4. **tie_word_embeddings**：默认开启，能省下 `vocab_size * hidden_size` ≈ 4.9M 参数。如果打开 MoE 之后再考虑 share-expert，模型的参数效率还能再上一个台阶。

---

## 9. 自检问题

1. MiniMind 为什么不直接用 `transformers` 的 `LlamaConfig` / `LlamaForCausalLM`？它自己实现的 Config / Model 有哪些**多出来**的设计？
2. RMSNorm 与 LayerNorm 的核心差异是什么？为什么现代 LLM 普遍选择 RMSNorm？
3. RoPE 的核心思想是什么？MiniMind 在哪一步做了 RoPE？为什么对 Q / K 都施加，而不是仅对 K？
4. GQA 与 MHA / MQA 的区别是什么？MiniMind 默认的 `q_heads / kv_heads` 是几比几？
5. MoE 的负载均衡 loss 是怎么计算的？它的目标是什么？
6. `tie_word_embeddings=True` 能省下多少参数？什么情况下你会想关掉它？

---

## 10. 推荐阅读

- [RoPE 原始论文](https://arxiv.org/abs/2104.09864)——Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*.
- [YaRN 论文](https://arxiv.org/abs/2309.00071)——Peng et al., *YaRN: Efficient Context Window Extension of Large Language Models*.
- [GQA 论文](https://arxiv.org/abs/2305.13245)——Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*.
- [SwiGLU 论文](https://arxiv.org/abs/2002.05202)——Shazeer, *GLU Variants Improve Transformer*.
- [Llama2 源码](https://github.com/meta-llama/llama)——MiniMind 整体结构与 Llama 高度对齐，可以对照阅读。

---

下一章：[02 - Tokenizer 与数据集](./02-tokenizer-dataset.md)