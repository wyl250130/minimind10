# 09 - 推理与部署

> 本章对应 [`eval_llm.py`](https://github.com/jingyaogong/minimind/blob/master/eval_llm.py) 与 [`scripts/`](https://github.com/jingyaogong/minimind/tree/master/scripts)。模型训练完后，要把它跑起来给别人用——CLI 推理、WebUI、OpenAI-API 兼容服务、第三方推理引擎都是常见路径。本章把 MiniMind 的整套部署生态梳理一遍。

---

## 1. 推理流程总览

```text
                     ┌────────────────────────┐
                     │      训练产物           │
                     │  pretrain_*.pth         │
                     │  full_sft_*.pth         │
                     │  dpo_*.pth / grpo_*.pth │
                     │  lora_xxx_*.pth         │
                     └───────────┬────────────┘
                                 │
                                 ▼
                     ┌────────────────────────┐
                     │   convert_model.py     │
                     │  torch ↔ transformers  │
                     └───────────┬────────────┘
                                 │
            ┌────────────────────┼────────────────────┐
            │                    │                    │
            ▼                    ▼                    ▼
    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
    │  eval_llm.py │    │ serve_openai │    │  第三方引擎  │
    │   CLI 推理   │    │   _api.py    │    │ vllm/ollama/ │
    │              │    │ OpenAI-API   │    │  llama.cpp   │
    └──────┬───────┘    └──────┬───────┘    └──────┬───────┘
           │                   │                   │
           ▼                   ▼                   ▼
     用户本地对话        FastGPT / OpenWebUI     生产级部署
```

主线流程：

1. 用 [`scripts/convert_model.py`](https://github.com/jingyaogong/minimind/blob/master/scripts/convert_model.py) 把 torch 权重转成 HuggingFace `transformers` 格式（除非直接用 torch 权重）；
2. CLI 推理用 `eval_llm.py`，快速做模型自测；
3. WebUI 用 `scripts/web_demo.py`，本地交互体验；
4. OpenAI-API 兼容服务用 `scripts/serve_openai_api.py`，接入第三方 Chat UI；
5. 生产级部署可以选 vllm / ollama / llama.cpp / SGLang / MNN。

---

## 2. CLI 推理：[`eval_llm.py`](https://github.com/jingyaogong/minimind/blob/master/eval_llm.py)

### 2.1 启动

下载主线模型（[HuggingFace](https://huggingface.co/collections/jingyaogong/minimind) / [ModelScope](https://www.modelscope.cn/collections/MiniMind-b72f4cfeb74b47)）：

```bash
modelscope download --model gongjy/minimind-3 --local_dir ./minimind-3
```

然后：

```bash
# 方式 1：transformers 格式
python eval_llm.py --load_from ./minimind-3

# 方式 2：原生 torch 权重（需 ./out/ 下有相应文件）
python eval_llm.py --load_from ./model --weight full_sft
```

启动后会进入交互界面：

```text
[0] 自动测试
[1] 手动输入
```

选 `0` 会用 8 个内置 prompt 测试模型；选 `1` 则进入交互对话模式。

### 2.2 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--load_from` | `model` | 加载路径；`model` = 原生 torch 权重，其他 = transformers 格式 |
| `--weight` | `full_sft` | 权重前缀名（pretrain / full_sft / dpo / ppo_actor / grpo / spo）|
| `--lora_weight` | `None` | LoRA 权重名（None / lora_identity / lora_medical）|
| `--hidden_size` | 768 | 必须与训练时一致 |
| `--use_moe` | 0 | 是否 MoE |
| `--inference_rope_scaling` | False | 启用 YaRN 外推 |
| `--max_new_tokens` | 8192 | 最大生成长度 |
| `--temperature` | 0.85 | 采样温度 |
| `--top_p` | 0.95 | nucleus 采样阈值 |
| `--open_thinking` | 0 | 是否开启自适应思考 |
| `--historys` | 0 | 多轮对话携带的历史轮数 |

### 2.3 推理流程

```python
def main():
    ...
    conversation = []
    model, tokenizer = init_model(args)
    for prompt in prompt_iter:
        conversation = conversation[-args.historys:] if args.historys else []
        conversation.append({"role": "user", "content": prompt})

        if 'pretrain' in args.weight:
            inputs = tokenizer.bos_token + prompt
        else:
            inputs = tokenizer.apply_chat_template(
                conversation, tokenize=False,
                add_generation_prompt=True,
                open_thinking=bool(args.open_thinking),
            )

        inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)

        generated_ids = model.generate(
            inputs=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens,
            do_sample=True, streamer=streamer,
            top_p=args.top_p, temperature=args.temperature,
            repetition_penalty=1,
        )
        response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        conversation.append({"role": "assistant", "content": response})
```

几个细节：

- **`pretrain` 权重走裸续写模式**：直接 `tokenizer.bos_token + prompt`，不套 chat_template；
- **其他权重走 chat_template 模式**：构造多轮上下文，让模型扮演助手；
- **open_thinking** 控制是否预先注入 `<think>` 标签；
- **`repetition_penalty=1`** 关闭重复惩罚（MiniMind 自带的 generate 内部实现了重复惩罚逻辑）。

### 2.4 LoRA 推理

```bash
python eval_llm.py --weight full_sft --lora_weight lora_medical
```

`init_model` 内部会：

```python
model = MiniMindForCausalLM(MiniMindConfig(...))
model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
if args.lora_weight != 'None':
    apply_lora(model)
    load_lora(model, f'./{args.save_dir}/{args.lora_weight}_{args.hidden_size}.pth')
```

详见 [05 - LoRA 微调](./05-lora.md)。

---

## 3. 思考开关：Open Thinking

MiniMind 的一个非常独特的设计是**Adaptive Thinking**——同一个模型在推理时通过 `open_thinking` 开关动态切换“是否显式思考”：

```bash
# 直答模式
python eval_llm.py --load_from ./minimind-3

# 思考模式
python eval_llm.py --load_from ./minimind-3 --open_thinking 1
```

底层机制是 chat_template 的一个开关：

- `open_thinking=0`：模板预注入空的 `<think>\n\n</think>`，模型倾向直接回答；
- `open_thinking=1`：模板预注入 `<think>`，模型继续输出显式思考与最终答案。

训练时通过混合空 think、显式 `reasoning_content` 与 `thinking_ratio` 采样，让模型见过“该想时想、该直答时直答”的混合模式，从而在推理时能根据开关切换风格。

> ⚠️ 当前同时开启 Tool Call 与显式思考时，模型通常并不太会稳定地输出思考过程——原因是训练数据里还缺少“reasoning 与 tool call 同时存在”的联合蒸馏样本。

---

## 4. WebUI：[`scripts/web_demo.py`](https://github.com/jingyaogong/minimind/blob/master/scripts/web_demo.py)

基于 Streamlit 的极简聊天 WebUI：

```bash
cd scripts && streamlit run web_demo.py
```

特点：

- 支持思考展示（自动检测 `<think>...</think>` 并单独渲染）；
- 支持工具调用可视化（自动检测 `<tool_call>` / `<tool_response>`）；
- 多轮对话保留历史；
- 与 `eval_llm.py` 共用同一套 `apply_chat_template`。

部署前需要把 transformers 格式的模型目录复制到 `scripts/` 下：

```bash
cp -r minimind-3 ./scripts/minimind-3
```

---

## 5. OpenAI-API 兼容服务：[`scripts/serve_openai_api.py`](https://github.com/jingyaogong/minimind/blob/master/scripts/serve_openai_api.py)

把 MiniMind 包装成兼容 OpenAI Chat Completions API 的服务端，方便接入 FastGPT / OpenWebUI / Dify 等第三方 UI：

```bash
cd scripts && python serve_openai_api.py
```

默认端口 8998，请求示例：

```bash
curl http://localhost:8998/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minimind",
    "messages": [{ "role": "user", "content": "世界上最高的山是什么？" }],
    "temperature": 0.7, "max_tokens": 1024,
    "stream": true,
    "open_thinking": true
  }'
```

扩展字段：

- `reasoning_content`：返回模型的思考过程；
- `tool_calls`：返回工具调用请求（与 OpenAI 风格一致）；
- `open_thinking`：在请求中控制是否开启思考。

客户端测试：

```bash
cd scripts && python chat_api.py
```

---

## 6. Tool Call 评测：[`scripts/eval_toolcall.py`](https://github.com/jingyaogong/minimind/blob/master/scripts/eval_toolcall.py)

专门的 tool-call 测试脚本，会自动检测模型生成的 `<tool_call>` 标签、解析参数、执行模拟工具、回填 `<tool_response>`、再让模型继续生成。

```bash
python eval_toolcall.py --weight full_sft
# 或者
python eval_toolcall.py --weight agent
```

典型输出：

```text
💬: 现在几点了？
🧠: <tool_call>{"name": "get_current_time", "arguments": {"timezone": "Asia/Shanghai"}}</tool_call>
📞 [Tool Calling]: get_current_time
✅ [Tool Called]: {"datetime": "2026-03-15 17:18:22", "timezone": "Asia/Shanghai"}
🧠: 现在是2026年3月15日17时18分22秒。
```

---

## 7. 模型格式互转：[`scripts/convert_model.py`](https://github.com/jingyaogong/minimind/blob/master/scripts/convert_model.py)

MiniMind 同时支持 **原生 torch 权重**和 **HuggingFace transformers 权重**两种格式，二者通过 `convert_model.py` 互转：

| 转换 | 命令 | 用途 |
|------|------|------|
| torch → transformers | `python convert_model.py` | 把训练完的 `out/pretrain_*.pth` 等转成 HF 格式，方便部署到 vllm / ollama |
| transformers → torch | `python convert_model.py` 反向 | 把 HF 格式权重转回 torch，方便继续用 `eval_llm.py --load_from model` |
| LoRA 合并 | `python convert_model.py --mode merge_base_lora` | 把基础模型与 LoRA 权重合并成新的完整模型 |

合并 LoRA 的代码实质上就是 [model_lora.py:56-65](https://github.com/jingyaogong/minimind/blob/master/model/model_lora.py#L56) 的 `merge_lora`。

---

## 8. 第三方推理引擎

### 8.1 vLLM

```bash
vllm serve /path/to/model --model-impl transformers --served-model-name "minimind" --port 8998
```

特点：PagedAttention + 连续批处理，吞吐量高，社区活跃，主流生产部署选择。

### 8.2 SGLang

```bash
python -m sglang.launch_server --model-path /path/to/model --attention-backend triton --host 0.0.0.0 --port 8998
```

特点：RadixAttention + 结构化生成，适合 Agent 类场景；MiniMind 在 RL 训练中也支持作为 rollout engine。

### 8.3 ollama

```bash
# 创建 Modelfile（MiniMind 在 README 中给出了完整模板）
ollama create -f minimind.modelfile minimind-local
ollama run minimind-local
```

也可以直接拉取官方发布的版本：

```bash
ollama run jingyaogong/minimind-3
```

特点：本地使用最方便，适合个人开发者体验。

### 8.4 llama.cpp

llama.cpp 是 C++ 推理框架，需要先把 HF 模型转成 GGUF：

```bash
# 1. 在 llama.cpp/convert_hf_to_gguf.py 末尾补一行兼容（qwen2 兜底）
# 2. 转 GGUF
python convert_hf_to_gguf.py /path/to/minimind-model
# 3. 量化（可选）
./build/bin/llama-quantize /path/to/model.gguf /path/to/model.q8.gguf Q8_0
# 4. 推理
./build/bin/llama-cli -m /path/to/model.gguf
```

特点：CPU 也能跑，适合端侧部署。

### 8.5 MNN（端侧）

```bash
# 4-bit HQQ 量化导出
python llmexport.py --path /path/to/model/ --export mnn --hqq --dst_path model-mnn

# 在 Mac / 手机上跑
./llm_demo /path/to/model-mnn/config.json prompt.txt
```

特点：阿里巴巴开源的端侧推理引擎，iOS / Android 都能部署。

---

## 9. RoPE 长度外推（YaRN）

MiniMind 默认训练长度 2048（mini）/ 32k（full），推理时可以通过 YaRN 外推到更长：

```bash
# torch 权重
python eval_llm.py --weight full_sft --inference_rope_scaling

# transformers 权重：在 config.json 中添加
"rope_scaling": {
    "type": "yarn",
    "factor": 16.0,
    "original_max_position_embeddings": 2048,
    "beta_fast": 32.0,
    "beta_slow": 1.0,
    "attention_factor": 1.0
}
```

YaRN 的核心思想（见 [01 - 模型架构](./01-model-architecture.md)）：

- 高频维度保持原频率，对应小位置；
- 低频维度乘以 `1/factor`，对应大位置；
- 通过 ramp 函数平滑过渡；
- 注意力分数乘以 `attention_factor` 做幅度补偿。

效果：在长文本下 PPL 显著下降（README 给出了详细的对比图）。

---

## 10. 模型权重目录约定

训练产物：

```text
out/
├── pretrain_768.pth            # Pretrain 权重
├── full_sft_768.pth            # SFT 权重
├── dpo_768.pth                 # DPO 权重
├── ppo_actor_768.pth           # PPO actor 权重
├── ppo_critic_768.pth          # PPO critic 权重
├── grpo_768.pth                # GRPO / CISPO 权重
├── agent_768.pth               # Agentic RL 权重
├── lora_medical_768.pth        # LoRA 权重（垂域适配）
└── lora_identity_768.pth       # LoRA 权重（自我认知）

checkpoints/
├── full_sft_768_resume.pth     # 训练续训 checkpoint（含 optimizer 等）
└── ...
```

发布到 HuggingFace 时通常是 transformers 格式：

```text
minimind-3/
├── config.json
├── generation_config.json
├── model_minimind.py          # 可选，导出方式决定
├── pytorch_model.bin or model.safetensors
├── special_tokens_map.json
├── tokenizer_config.json
└── tokenizer.json
```

---

## 11. 自检问题

1. `eval_llm.py` 在加载 `pretrain` 权重时为什么不套 chat_template？加载 `full_sft` 权重时为什么要套？
2. `--open_thinking 1` 在底层是如何生效的？训练时模型是怎么学会“该思考时思考、该直答时直答”的？
3. OpenAI-API 兼容服务端支持哪些 OpenAI 标准之外的扩展字段？这些扩展分别适用于什么场景？
4. 为什么 vLLM / ollama / llama.cpp 等第三方引擎只能消费 transformers 格式权重？什么场景下应该优先选哪一个？
5. YaRN 外推的三个超参数（`factor`、`beta_fast`、`beta_slow`）分别控制什么？调大会发生什么？

---

## 12. 推荐阅读

- [vLLM 论文](https://arxiv.org/abs/2309.06180)——PagedAttention 的提出。
- [SGLang 论文](https://arxiv.org/abs/2312.07104)——RadixAttention 的提出。
- [YaRN 论文](https://arxiv.org/abs/2309.00071)——RoPE 外推。
- [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat)——理解第三方 Chat UI 与 MiniMind 服务端的契约。

---

下一章：[10 - 评测与对比](./10-evaluation.md)