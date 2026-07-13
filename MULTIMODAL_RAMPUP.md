# 多模态大模型 Ramp-up：从原理到 SGLang 实现

> 面向第一次接触多模态（VLM/MLLM）的工程师。前半部分讲原理、引用领域经典工作，后半部分对照 SGLang 这个仓库的真实代码，把"论文里的图"落到"工程里的数据流"。

---

## 0. 一句话直觉

一个多模态大语言模型（Multimodal LLM，下称 **MLLM** 或 **VLM**）做的事情，可以浓缩成一句话：

> **把图片/视频/音频"翻译"成一串和文字 token 平起平坐的向量，塞进语言模型的输入序列里，然后照常做自回归生成。**

语言模型本身（LLM backbone）几乎不需要改。难点全在"怎么翻译"和"怎么塞"——以及在推理引擎里"怎么塞得又快又省"。整篇文档就是在展开这句话。

---

## 1. 为什么需要多模态：从「文字接龙」到「看图说话」

标准 LLM 的输入是 token id 序列，第一步永远是查一张 embedding 表（`nn.Embedding`），把每个整数 id 变成一个 `hidden_size` 维的向量。模型后面所有的 Transformer 层都只跟"向量序列"打交道——**它根本不知道这些向量原本是文字**。

这个观察是多模态的全部基础：既然模型只认向量，那只要我能把一张图也变成"同一个向量空间里、同样 `hidden_size` 维"的若干向量，我就能把图当成"特殊的词"喂进去。模型在自注意力里会让文字 token 去 attend 这些图像向量，于是就具备了"看图回答"的能力。

所以一个 VLM 通常由三部分拼成：

```
        ┌─────────────┐   ┌──────────────┐   ┌────────────────────┐
 图片 → │ Vision      │ → │ Projector /  │ → │  LLM backbone      │ → 文本
        │ Encoder     │   │ Connector    │   │ (Llama/Qwen/...)   │
        └─────────────┘   └──────────────┘   └────────────────────┘
   文字 ────────────────────────────────────────┘ (走原来的 embedding 表)
```

1. **Vision Encoder（视觉编码器）**：把像素变成一组视觉特征向量。
2. **Projector / Connector（连接器/对齐模块）**：把视觉特征"对齐"到 LLM 的词向量空间和维度。
3. **LLM backbone（语言模型主干）**：原封不动的解码器，负责理解+生成。

---

## 2. 三大组件，逐个拆解（含经典工作）

### 2.1 Vision Encoder：怎么"看"

主流做法源自 **ViT（Vision Transformer, Dosovitskiy et al., ICLR 2021）**：把图片切成不重叠的小块（patch，比如 14×14 像素一块），每块拉平后线性投影成一个向量，加上位置编码，就得到了一串"视觉 token"，再过标准 Transformer。一张 224×224 的图、patch=14，就是 16×16 = 256 个视觉 token。

关键的不是 ViT 本身，而是**用什么目标训练它**。经典分水岭是 **CLIP（Radford et al., ICML 2021）**：用 4 亿「图-文」对做对比学习，让"匹配的图和文"在向量空间里靠近、"不匹配的"远离。CLIP 训出来的视觉编码器，其特征天然就和"语言语义"对齐——这正是 VLM 想要的。绝大多数开源 VLM（LLaVA、Qwen-VL 早期等）的视觉塔都是 CLIP/SigLIP 系。

> **SigLIP（Zhai et al., 2023）** 把 CLIP 的 softmax 对比损失换成 sigmoid，训练更稳、小 batch 也能用，是近两年新模型（如 Gemma3、PaliGemma）的常见选择。

近年的趋势是**原生分辨率 / 动态分辨率**：早期 ViT 强制把图缩放成固定 224×224，文字密集的图（文档、表格）会糊掉。**Qwen2-VL（Wang et al., 2024）** 引入 *Naive Dynamic Resolution*，让图按原始长宽比切成可变数量的 patch——大图给更多 token，小图给更少。这就是为什么后面你会在代码里反复看到 `image_grid_thw`（grid 的 Time/Height/Width 三个维度）这个字段。

### 2.2 Connector：怎么"对齐"

视觉编码器吐出的向量，维度和语义空间未必和 LLM 一致，需要一个连接器搬运。主要有两大流派：

**(a) 线性 / MLP 投影（LLaVA 流派）**
**LLaVA（Liu et al., NeurIPS 2023）** 用最简单的方案：一个线性层（后来升级成两层 MLP）把视觉 token 投到 LLM 维度，然后 **N 个视觉 token 就原样占 N 个序列位置**。简单、信息无损，但图越大 token 越多，长上下文压力大。LLaVA 的另一个历史贡献是提出了"两阶段训练 + GPT 生成指令数据"的范式，让学术界能低成本复现 VLM。

**(b) Query-based 重采样（BLIP-2 / Flamingo 流派）**
**Flamingo（Alayrac et al., NeurIPS 2022）** 和 **BLIP-2（Li et al., ICML 2023）** 用一个叫 **Q-Former / Perceiver Resampler** 的模块：用固定数量（比如 32 个）可学习的 query 向量，通过 cross-attention 从几百个视觉 token 里"抽取"出固定 32 个向量。好处是无论图多大，喂给 LLM 的永远是固定且很少的 token，省算力；代价是有信息瓶颈。Flamingo 还开创了用 **gated cross-attention** 把视觉信息插进冻结 LLM 的中间层（而不是只在输入端）的做法。

> **两条路线的工程含义**：LLaVA 路线下"图占多少 token"是动态的、跟分辨率走；Q-Former 路线下是固定的。SGLang 这种推理引擎必须两者都支持，所以你会看到"每个多模态 item 自己算 token 数 / offset"的通用设计。

### 2.3 LLM backbone 与「占位符」机制

LLM 主干一般是现成的 Llama / Qwen / Mistral。融合视觉的最常见方式（也是 SGLang 用的方式）是 **占位符替换（placeholder substitution）**：

1. 在文本里插入特殊 token，例如 `<image>`，或者 Qwen 系列的 `<|image_pad|>`。
2. tokenizer 把它展开成 N 个占位 token（N = 这张图实际会产生的视觉 token 数）。
3. 模型 forward 时，先查 embedding 表得到所有 token 的文本 embedding，然后**把占位 token 对应位置的 embedding，原地替换成视觉编码器算出来的视觉向量**。

这样语言模型拿到的就是一条"文字向量 + 视觉向量"无缝拼接的序列，自回归生成逻辑完全不变。后面 SGLang 的 `embed_mm_inputs` 干的就是这件事。

### 2.4 位置编码的坑：M-RoPE

LLM 用 RoPE（旋转位置编码）给每个 token 一个一维位置。但图像是二维的、视频是三维的（加时间轴）。如果把图像 token 当成一长串一维序列，会丢掉空间结构。**Qwen2-VL 的 M-RoPE（Multimodal RoPE）** 把位置拆成 (时间 t, 高 h, 宽 w) 三个分量分别做 RoPE，让模型知道"这个 patch 在图里的二维位置 / 视频里的第几帧"。代价是位置不再是一个标量而是一个三维向量——这就是代码里 `mrope_positions` 形状是 `(3, seq_len)` 的来由。

---

## 3. 推理引擎视角：SGLang 怎么落地

理解了原理，现在看 SGLang 的真实代码。推理引擎额外要操心三件原理课不讲的事：**(1) 异步预处理**（CPU 上解码图片、跑 processor）、**(2) 缓存复用**（同一张图别重复编码）、**(3) 和 RadixAttention 前缀缓存共存**。

整体数据流（一次带图请求的生命周期）：

```
HTTP 请求 (text + image_url)
      │
      ▼
[TokenizerManager 进程]  multimodal_processor.py / base_processor.py
   - 下载/解码图片，跑 HF processor → pixel_values, image_grid_thw
   - 文本 tokenize，把 <image> 展开成 N 个占位 token
   - 产出 MultimodalDataItem（feature + hash + pad_value + offsets）
      │  (通过 IPC / 共享内存传给调度器)
      ▼
[Scheduler 进程]  schedule_batch.py
   - 把占位 token 用 pad_value 填充进 input_ids（给 RadixAttention 当缓存 key）
   - 组 batch
      │
      ▼
[ModelRunner / GPU]  mm_utils.py: general_mm_embed_routine
   - 文本 embedding 查表
   - 命中视觉 token 的位置 → 调 model.get_image_feature() 跑 Vision Encoder
   - scatter：把视觉向量原地写进文本 embedding 的对应位置
   - 喂给 LLM backbone 自回归生成
```

### 3.1 核心数据结构：`MultimodalDataItem`

[schedule_batch.py:238](python/sglang/srt/managers/schedule_batch.py#L238) 定义了贯穿全程的载体。一张图 / 一段视频 / 一段音频 = 一个 item：

```python
@dataclasses.dataclass
class MultimodalDataItem:
    modality: Modality          # IMAGE / VIDEO / AUDIO
    hash: int = None            # 这份数据内容的哈希
    pad_value: int = None       # 用哈希派生的占位值，填进 input_ids
    offsets: Optional[list] = None   # 这个 item 占 input_ids 里哪些位置
    feature: ... = None              # processor 产出的原始特征 (pixel_values)
    precomputed_embeddings: ... = None  # 或者：已经算好的视觉向量
    model_specific_data: dict = ...     # 如 image_grid_thw 等模型特有字段
```

注意它的设计哲学：**通用字段提到顶层，模型特有字段（如 Qwen 的 `image_grid_thw`）塞进 `model_specific_data`**，通过 `__getattr__` 让 `item.image_grid_thw` 这样的访问透明工作（[schedule_batch.py:264](python/sglang/srt/managers/schedule_batch.py#L264)）。这正是第 2 节里"两大流派字段不同"在工程上的妥协。

`feature` 和 `precomputed_embeddings` 二选一：前者是"原始像素特征、需要 GPU 上跑 encoder"，后者是"上游已经编码好、直接用"——对应分离式部署（把视觉编码拆到单独的 encode server，见 `epd_disaggregation`）。

### 3.2 `pad_value`：一份哈希，串起两套缓存

这是 SGLang 多模态最妙的设计，新人最容易看懵的地方。看 `set_pad_value`（[schedule_batch.py:290](python/sglang/srt/managers/schedule_batch.py#L290)）：

```python
self.hash = hash_feature(hashed_feature)   # 对图像特征算哈希
self.pad_value = _compute_pad_value(self.hash)
```

然后调度器把占位 token 在 `input_ids` 里**全部填成这个 `pad_value`**（而不是真实的词表 id）。为什么？因为 SGLang 的 **RadixAttention 前缀缓存**是靠 `input_ids` 序列做匹配的。如果两个请求带的是同一张图，它们的占位 token 就会是同一串 `pad_value`，前缀树就能命中、复用 KV cache。换句话说：

> **用"内容哈希"当占位 token，让基于文本序列的 RadixAttention 缓存，"免费"获得了对图像的内容感知能力。** 不同图哈希不同，自然不会误命中。

当然这些 `pad_value` 是超出词表范围的假 id，真正 forward 前会被 `clamp` 回合法范围（反正它们的 embedding 马上要被视觉向量覆盖掉），见 [mm_utils.py:882](python/sglang/srt/managers/mm_utils.py#L882) 那段注释。

### 3.3 GPU 上的融合：`general_mm_embed_routine`

[mm_utils.py:1022](python/sglang/srt/managers/mm_utils.py#L1022) 是所有 VLM 模型 `forward` 都会调用的统一入口。逻辑就是第 2.3 节的"占位符替换"，但加了工程优化：

- **只在 prefill 阶段做**：`forward_mode.is_decode()` 时跳过——图早在 prefill 编码过、向量进了 KV cache，decode 阶段一个新文本 token 不需要再碰视觉编码器（[mm_utils.py:1051](python/sglang/srt/managers/mm_utils.py#L1051)）。这是巨大的省算力点。
- **编码完把 feature 卸载到 CPU**：[mm_utils.py:1107](python/sglang/srt/managers/mm_utils.py#L1107)，因为 chunked-prefill 下一个请求跨多个 batch，原始特征要留着兜底，但不该占着 GPU 显存。

真正的拼接在 `embed_mm_inputs`（[mm_utils.py:781](python/sglang/srt/managers/mm_utils.py#L781)）：

```python
# 1. 按 modality 分组所有 item
# 2. 调用对应的 embedder 跑视觉编码器，例如 model.get_image_feature(items)
embedding, mask, input_ids = get_embedding_and_mask(...)
# 3. 查文本 embedding 表
input_embeds = input_embedding(input_ids)
# 4. 用 mask 找到占位位置，原地 scatter 写入视觉向量
indices = torch.where(mask.squeeze(dim=-1))[0]
input_embeds[indices] = embedding.to(...)
```

第 4 步那行 `input_embeds[indices] = embedding` 就是整个多模态原理的物理落点——**"把图塞进文字序列"在这里只是一次张量按下标赋值**。

而 `get_image_feature` 由各模型自己实现。看 Qwen2-VL（[qwen2_vl.py:489](python/sglang/srt/models/qwen2_vl.py#L489)）：

```python
def get_image_feature(self, items):
    pixel_values = torch.cat([item.feature for item in items], dim=0)
    image_grid_thw = torch.concat([item.image_grid_thw for item in items], dim=0)
    image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)  # 跑 ViT
    return image_embeds
```

`self.visual` 就是第 2.1 节的 Vision Encoder（这里是支持动态分辨率的 Qwen2 ViT），`grid_thw` 就是 2.1 节说的原生分辨率网格。一条龙串起来了。

### 3.4 模型怎么接进来：`general_mm_embed_routine` 的调用方

每个 VLM 的 `forward` 都长这样（[qwen2_vl.py:558](python/sglang/srt/models/qwen2_vl.py#L558)）：

```python
hidden_states = general_mm_embed_routine(
    input_ids=input_ids,
    forward_batch=forward_batch,
    language_model=self.model,    # LLM backbone
    multimodal_model=self,        # 提供 get_image_feature 等方法
    positions=positions,
)
```

这就是"组件三件套"在代码里的对应：`language_model` 是 backbone，`multimodal_model=self` 暴露视觉编码能力，连接器逻辑藏在各自的 `get_xxx_feature` 里。新增一个模型，核心就是实现 vision encoder + `get_image_feature` + 注册一个 processor。

### 3.5 预处理侧：Processor

[base_processor.py:180](python/sglang/srt/multimodal/processors/base_processor.py#L180) 的 `BaseMultimodalProcessor` 是所有模型 processor 的基类，跑在 TokenizerManager 进程（CPU），关键方法：

- `load_mm_data` / `fast_load_mm_data`（[base_processor.py:774](python/sglang/srt/multimodal/processors/base_processor.py#L774)）：异步下载、解码图片/抽帧视频。异步是因为这是 IO 密集，不能阻塞调度。
- `process_mm_data`（[base_processor.py:401](python/sglang/srt/multimodal/processors/base_processor.py#L401)）：调 HuggingFace processor 得到 `pixel_values`。
- `collect_mm_items_from_processor_output`（[base_processor.py:1087](python/sglang/srt/multimodal/processors/base_processor.py#L1087)）：把结果打包成上面的 `MultimodalDataItem`。

`python/sglang/srt/multimodal/processors/` 下每个文件对应一个模型家族（`qwen_vl.py`、`llava.py`、`internvl.py`、`pixtral.py`……），这是仓库里支持几十种 VLM 的扩展点。

### 3.6 缓存与性能优化（推理引擎的"看家本领"）

原理课不会讲，但这是 SGLang 的价值所在：

1. **多模态 embedding 缓存** `MultiModalStaticCache`（[multimodal_cache.py:76](python/sglang/srt/mem_cache/multimodal_cache.py#L76)）：服务级 LRU 缓存，key 是 item 哈希（或多个 item 的 `combine_hashes`），存的是已编码好的视觉向量。同一张图二次请求直接取，跳过整个 ViT。
2. **RadixAttention 前缀缓存复用**：靠 3.2 的 `pad_value` 机制，图像也能参与前缀匹配，复用 KV cache。
3. **Vision Encoder 的 CUDA Graph**：`multimodal/vit_cuda_graph_runner.py`，把视觉塔的 forward 也 capture 成 CUDA graph 降低 kernel launch 开销（详见 `docs/advanced_features/cuda_graph_for_multi_modal_encoder.md`）。
4. **EPD 分离部署**：把 Encode（视觉编码）、Prefill、Decode 拆到不同 GPU/进程，视觉编码不再和文本生成抢资源——这就是 `precomputed_embeddings` 字段存在的原因（`docs/advanced_features/epd_disaggregation.md`）。
5. **特征哈希优化**：算哈希时临时把 feature 搬上 GPU 加速（`SGLANG_MM_BUFFER_SIZE_MB`，[schedule_batch.py:512](python/sglang/srt/managers/schedule_batch.py#L459) 附近），以及跨 TP rank 复用哈希避免重复计算。

---

## 4. 一张图总结映射关系

| 原理概念 | 经典工作 | SGLang 代码落点 |
|---|---|---|
| 图片切 patch → 视觉 token | ViT (2021) | 各模型 `self.visual`，如 `Qwen2VisionTransformer` |
| 视觉-语言对齐预训练 | CLIP/SigLIP | 视觉塔权重（推理引擎不训练，直接加载） |
| 线性投影连接器 / 占位符替换 | LLaVA (2023) | `embed_mm_inputs` 的 scatter（[mm_utils.py:908](python/sglang/srt/managers/mm_utils.py#L908)） |
| Query 重采样连接器 | Flamingo / BLIP-2 | `get_image_feature` 内部（模型相关） |
| 动态/原生分辨率 | Qwen2-VL (2024) | `image_grid_thw` / `model_specific_data` |
| 多模态位置编码 | M-RoPE (Qwen2-VL) | `mrope_positions` `(3, seq_len)` |
| 统一融合入口 | —（工程） | `general_mm_embed_routine` |
| 数据载体 | —（工程） | `MultimodalDataItem` |
| 图像内容感知缓存 | —（工程） | `pad_value` + RadixAttention + `MultiModalStaticCache` |

---

## 5. 建议的上手路径

1. **跑通一次**：用 `python -m sglang.launch_server --model-path Qwen/Qwen2-VL-7B-Instruct`，发一个带图请求，确认链路。
2. **读一条最短链路**：`qwen2_vl.py` 的 `forward` → `general_mm_embed_routine` → `embed_mm_inputs` → `get_image_feature`。把这四个函数读透，多模态就懂了一大半。
3. **读一个 processor**：`multimodal/processors/qwen_vl.py`，看 `<image>` 怎么变成 N 个占位 token、`pixel_values` 怎么来的。
4. **理解 pad_value 的双重身份**：这是 SGLang 区别于"照搬 HF"的精髓，3.2 节那段想清楚。
5. **再扩展**：想加新模型，照着一个已支持的模型抄 `models/xxx.py` + `processors/xxx.py` 两个文件。

### 经典论文清单（按阅读优先级）
- **ViT** — Dosovitskiy et al., *An Image is Worth 16x16 Words*, ICLR 2021
- **CLIP** — Radford et al., *Learning Transferable Visual Models From Natural Language Supervision*, ICML 2021
- **LLaVA** — Liu et al., *Visual Instruction Tuning*, NeurIPS 2023（连接器 + 训练范式，必读）
- **BLIP-2** — Li et al., 2023（Q-Former）
- **Flamingo** — Alayrac et al., NeurIPS 2022（gated cross-attention）
- **Qwen2-VL** — Wang et al., 2024（动态分辨率 + M-RoPE，最贴近当下工程）
- **SigLIP** — Zhai et al., 2023（新一代视觉塔）

---

*本文档结合 SGLang 当前实现撰写；代码引用为撰写时的行号，迭代后可能漂移，以实际文件为准。*
