# 配图方案：blueprint（推荐）

**文章**：`study-notes/推理优化全景-六个动词-codex.md`
**语言**：中文
**风格**：工程蓝图。米白工程纸底、浅灰网格、深灰文字、工程蓝主路径、琥珀色瓶颈或代价。技术标签清楚，装饰克制。
**配图数**：3 张新图；保留已有 Roofline 图。

## 图 1：后厨直觉与系统资源对照

**插入位置**：“一、先把 GPU 想成一家后厨”的对照表之后。
**目的**：让灶火、送菜通道、案板/冷藏区、出菜顺序成为一套完整比喻。
**画面**：左侧后厨俯视图，四个区域用连线映射到右侧 GPU 系统图；灶台→计算能力，传菜通道→显存带宽，案板/冷藏区→显存容量，订单依赖箭头→逐 token 串行。
**文件名**：`illustration-kitchen-resource-map.png`

## 图 2：四层架构 × 六个动作全景图

**插入位置**：“三、全景图：六个动作发生在哪四层”的文字架构图之后。
**目的**：同时表达技术发生的位置和主要节省的资源，替代平铺的技术矩阵。
**画面**：纵向四层为模型结构层、算子执行层、缓存与运行时层、服务与集群层；横向六列为不算、少算、少搬、少占、打满、少走几轮。技术卡片放进对应交叉区域，跨格技术用短连线连接，不强行复制。右侧单独用琥珀色列出各层主要代价。
**文件名**：`illustration-four-layers-six-actions.png`

## 图 3：SGLang 请求链路架构图

**插入位置**：“四、这些技术在 SGLang 里怎样串起来”标题后，替换当前注释。
**目的**：用读者可直接理解的系统图替代多个源码路径。
**画面**：横向主链路为“客户端/API → Tokenizer Manager → Scheduler → Model Worker / Model Runner → Detokenizer Manager → 流式返回”。Scheduler 下方连接“Radix Cache、Paged KV Pool、HiCache”；Model Runner 内标“Attention 后端、MoE/量化、CUDA Graph、投机解码”。用底色把组件归入服务与集群、缓存与运行时、算子执行、模型结构四层。图底部另画虚线扩展：“Router；Prefill 实例 → KV 传输 → Decode 实例；TP/PP/EP 位于执行层内部”。
**文件名**：`illustration-sglang-request-architecture.png`
