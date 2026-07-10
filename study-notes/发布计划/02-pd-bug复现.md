# 02 · 《从一个 GitHub Issue 到可复现实验：PD 分离的正确性 bug》

- **源料**：`pr30233-repro/`（README、PR_DRAFT.md、PD_ABORTED_REQUEST_DEEP_DIVE.md、
  PD_ABORT_LIFECYCLE_REVIEW.md、remote-artifacts/ 的前后 metrics 与 repro_result.json）
- **目标读者**：想学"如何把一个 issue 变成可复现实验、再推到 PR"的工程师；对 PD 分离好奇的人
- **一句话卖点**：一个"本该返回 400 的超长请求"如何在 PD 分离下变成 decode 侧乱码/hang，我如何复现并修好它
- **字数**：5–7k　**优先级**：★★★（故事性最强，建议**首发打样**）

## 钩子（前 200 字）

「一个超长输入请求，本该被干脆利落地拒绝（HTTP 400）。但在 PD 分离架构下，它没有——
它让 decode 节点在**一块从没被正确填充的 KV cache** 上开始解码，吐出乱码，甚至把整个
节点拖挂。这篇文章讲我如何从一行 issue 复现出这个 bug，再把它修好。」

## 结构（分节 → 源料 → 必引数字/事实）

| 节 | 内容 | 源料 | 必引硬料 |
|---|---|---|---|
| 1 | 先讲清 PD 分离：为什么要拆 prefill / decode，KV transfer 是怎么跨节点搬的 | DEEP_DIVE §2-3 | 一次请求经过哪些组件 |
| 2 | 这个 bug 长什么样：`context_len=1048576` 但 `max_req_input_len=284922`，请求 284937 tokens | PR_DRAFT Motivation | 三个长度数字的对照 |
| 3 | 根因：`allow_auto_truncate=False` → `set_finish_with_abort()` 只在本地对象上记了 abort，**没传播成 KV 协议的终止态** | PR_DRAFT、DEEP_DIVE §1 | "abort 状态没被转成 decode 能观察到的终止态" |
| 4 | 我如何复现：脚本化拉起 P/D 两卡 + mini-LB，打 short → over-long → short 三连击 | README、`launch_pd.sh`、`repro.py` | 非对称配置（prefill 2048 / decode 8192）作为稳定入口 |
| 5 | buggy 现象 vs fixed 现象（附前后 metrics） | remote-artifacts/、repro_result.*.json | buggy：乱码/hang/500；fixed：400 in 0.02s、follow-up 200 in 0.07s |
| 6 | 修复要点：入队前拦截已 abort 请求 + 通过 `KVPoll.Failed` 把 reason/status 传播到 decode + Mooncake 孤儿 room 清理 | PR_DRAFT Modifications | 4 层改动清单 |
| 7 | 如何验证不是碰运气：17 个单测 + e2e 三连击稳定复现 | PR_DRAFT Accuracy Tests | `17 passed` |
| 8 | 复盘：从 issue 到 PR 的完整方法（去重、blame、写最小复现） | PD_ABORT_LIFECYCLE_REVIEW | 引流到主线 03 |

## 配图清单

- PD 分离架构图（prefill/decode/LB/KV transfer 的数据流，DEEP_DIVE 里有 ASCII，重画成图）
- 时序图：正常请求 vs aborted 请求的两条生命周期线（哪一步状态错位）
- 前后对比截图：buggy 乱码输出 vs fixed 的 400（从 repro_result.json 取）

## 诚实时刻

- 一开始以为 bug 是"prefill/decode 配了不同 max length"——错，非对称只是**稳定复现入口**，真正的 bug 是 abort 没传播
- 讲清"我怎么排除了误判方向"

## 复现命令（贴给读者，注意脱敏远程路径）

```bash
bash launch_pd.sh                 # 起 P(GPU0)+D(GPU1)+mini-LB
python repro.py --lb http://127.0.0.1:8000 --ctx 8192 --overlong-tokens 3000
bash capture_metrics.sh after-overlong
```

## ⚠️ 发布前脱敏

- `pr30233-repro/env.sh`、remote-artifacts 里的**远程主机 IP / autodl 路径 / HF mirror token**
- 确认 PR 是否已提交/合并——若已进 upstream，文章可直接链 PR 增加可信度

## 状态：未开工 → **建议首发**
