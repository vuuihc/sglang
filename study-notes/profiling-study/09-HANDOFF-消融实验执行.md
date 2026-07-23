# 09 · HANDOFF:消融实验执行交接(给接手的 agent）

> 这份文档让另一个 agent（codex）冷启动接手"跑消融实验 → 出 V2 数据篇 → 回填修订 V1"这件事。
> 写于 2026-07-15，前一个 agent 已完成环境准备和模型下载，卡在"等用户切有卡模式"。

---

## 0. 一句话现状

租的 H20 单卡机(AutoDL，就是 06/07 那台)已删旧模型腾空间、下好 P0 全部 4 个模型、上传好跑分脚本；**只等用户把机器从"无卡模式"切到"有卡模式"，就能开跑**。软件栈现成（sglang 0.5.13.post1），不用装环境。

---

## 1. 目标（三步，按顺序）

1. **跑实验**：在 H20 上执行 `/root/autodl-tmp/ablation_battery.sh`，采集"关掉每个优化性能掉多少"的数据。
2. **出 V2 文章**：《把推理优化一个个关掉，性能会掉多少？——六个动词的实测账单》，姊妹篇，不是改 V1。
3. **回填 V1**：给 `study-notes/推理优化全景-六个动词.md` 的 roofline/少搬/破串行等处补实测价签 + 文末挂 V2 链接。

计划全文见 [08-消融实验计划-V2数据篇.md](08-消融实验计划-V2数据篇.md)。实验设计理由、账单格、预期看点都在那里，**先读它**。

---

## 2. 连接方式（关键，含一个坑）

连接信息在仓库根 `.env`：
```
CMD="ssh -p 47720 root@region-42.seetacloud.com"
PASSWORD=<在 .env 里>
```
⚠️ **用户切有卡模式重启后，端口/密码可能变**。开工前先让用户确认 `.env` 是最新的，或直接问一次当前 SSH 信息。

本地没有 sshpass，只有 `expect`。用这个封装（已在 `/tmp/rssh.exp`，重启 agent 后需重建）：
```bash
cat > /tmp/rssh.exp << 'EXPECT'
#!/usr/bin/expect -f
set timeout [expr {[info exists env(RSSH_TIMEOUT)] ? $env(RSSH_TIMEOUT) : 180}]
set port [lindex $argv 0]; set userhost [lindex $argv 1]
set pw [lindex $argv 2]; set cmd [lindex $argv 3]
spawn ssh -p $port -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 $userhost $cmd
expect { -re {[Pp]assword:} { send "$pw\r"; exp_continue }
         -re {yes/no} { send "yes\r"; exp_continue } eof }
catch wait result; exit [lindex $result 3]
EXPECT
chmod +x /tmp/rssh.exp
```
调用（**注意**：远端命令必须作为**单个字符串**传第 4 个参数）：
```bash
set -a; source /Users/vuuihc/Workspace/study/sglang/.env; set +a
R() { RSSH_TIMEOUT=60 /tmp/rssh.exp 47720 root@region-42.seetacloud.com "$PASSWORD" "$1" 2>/dev/null | grep -vE "spawn ssh|password:"; }
R 'echo hi; nvidia-smi -L'
```
**坑**：expect 会把 `spawn ssh...` 和 `password:` 打到 stdout，污染输出。所以**解析远端输出前务必 `grep -vE "spawn ssh|password:"`**。前一个 agent 的后台轮询脚本就是因为没过滤，把密码提示行当成结果，误报 timeout（实际早跑完了）。别按行号 `sed -n '1p'` 解析，用唯一标记 grep。

---

## 3. 远端环境（已确认，无需重建）

- 机器：AutoDL 单卡 **H20 96GB**，`root@region-42.seetacloud.com`，容器 `autodl-container-0aae4b9e20-...`
- **sglang 0.5.13.post1** 装在 base conda（`/root/miniconda3`），`hf`/`huggingface-cli` 都在。**base python 直接能 import sglang**（但非交互 SSH 不 source .bashrc，若 conda 没初始化就用 `bash -lc "..."`）。
- **环境修复脚本 `/root/sglang_env.sh`**（跑 sglang 前必 source，battery 脚本已内置 source）：
  ```
  export LD_LIBRARY_PATH=".../nvidia/cu13/lib:$LD_LIBRARY_PATH"   # 06 的 cu13 坑
  export HF_ENDPOINT=https://hf-mirror.com                        # 直连 hf.co 不通，走镜像
  export HF_HOME=/root/autodl-tmp/hf                              # 模型都在数据盘这里
  ```
- 磁盘：`/root/autodl-tmp` 50G 总量，**当前 ~20G 可用**（已删 06/07 的 35B 老模型）。系统盘 `/` 只有 30G，别往那儿写模型。

### 已下好的模型（HF_HOME=/root/autodl-tmp/hf）
| 模型 | 体积 | 用于 |
|---|---|---|
| Qwen/Qwen3-8B | 16G | 基线，几乎每个实验 |
| Qwen/Qwen3-8B-FP8 | 8.9G | e2 少算 |
| Qwen/Qwen3-8B-AWQ | 5.7G | e3 少搬 |
| Tengyunw/qwen3_8b_eagle3 | 764M | e4 破串行 draft |

### 还没下（第二批，见 §5 磁盘编排）
- `deepseek-ai/DeepSeek-V2-Lite-Chat`（~31G，MLA，e11 用）——**现在磁盘装不下，必须先跑完 Qwen 实验、删掉部分 Qwen 模型再下**。

---

## 4. 怎么跑

跑分脚本已上传并语法校验通过：`/root/autodl-tmp/ablation_battery.sh`（本地源在 `study-notes/scripts/ablation_battery.sh`，246 行）。

**第一批（P0 + kernel 层，全用已下的 Qwen 模型，不碰 MLA）**：
```bash
# 用户切有卡模式后，先确认 GPU 可见
R 'nvidia-smi -L'
# 后台跑，别在前台等（2-3 小时）。注意排除 e11（MLA 还没下模型）
R 'cd /root/autodl-tmp && nohup bash ablation_battery.sh e5 e1 e9 e2 e7 e8 e10 e12 e3 e6 e4 > /root/autodl-tmp/battery_run1.log 2>&1 & echo started $!'
```
监控（每隔 3-5 分钟查一次即可，别高频）：
```bash
R 'tail -20 /root/autodl-tmp/battery_run1.log; echo ---; grep -E "EXP_BEGIN|EXP_END" study-notes/contribution-scan/exp3-ablation/phases.log 2>/dev/null | tail'
```
注意：battery 脚本里 `OUT=study-notes/contribution-scan/exp3-ablation`是**相对 cwd** 的，所以必须 `cd /root/autodl-tmp` 再跑（数据会落在 `/root/autodl-tmp/study-notes/contribution-scan/exp3-ablation/`）。或跑前 `export OUT=/root/autodl-tmp/exp3-ablation`。**建议显式设 OUT 绝对路径**，省得找不到数据。

### 卫生铁律（07 §七的教训，脚本已内置但要盯）
每个 server 起来后先跑一轮丢弃的 warmup 再采数——**冷启动的第一波压测数据不可信**（lazy JIT / triton autotune 未热）。脚本里 `warmup()` 已做，但看数据时如果某个实验的第一个点异常慢，先怀疑是不是 warmup 没生效。

### 已知风险点（跑的时候留意）
- **e4 EAGLE3**：draft 模型是 `Tengyunw/qwen3_8b_eagle3`（用户已确认 HF 上有）。spec 参数 `--speculative-num-steps 6 --speculative-eagle-topk 10 --speculative-num-draft-tokens 32` 是常见值，若 server 起不来看日志调。
- **e10 torch_native 后端**：大概率在 8K prefill OOM 或极慢——**这本身就是数据点**（朴素注意力有多差），脚本里 `|| echo 记录之` 已容错，别当失败。
- **e2/e3 的模型名**：Qwen3-8B-FP8/AWQ 都已下好，直接能用。
- **mem-fraction**：默认 0.85，H20 96G 够。若某模型 OOM 调 `MEMFRAC=0.8`。

---

## 5. 第二批：MLA（e11）的磁盘编排

Qwen 实验（第一批）全部跑完后：
```bash
# 1. 删掉不再需要的 Qwen 变体，给 DeepSeek 腾地方（保留 Qwen3-8B 基线以防复测）
R 'rm -rf /root/autodl-tmp/hf/hub/models--Qwen--Qwen3-8B-FP8 /root/autodl-tmp/hf/hub/models--Qwen--Qwen3-8B-AWQ; df -h /root/autodl-tmp'
# 2. 下 DeepSeek-V2-Lite（~31G）
R 'source /root/sglang_env.sh; nohup hf download deepseek-ai/DeepSeek-V2-Lite-Chat > /root/autodl-tmp/dl_mla.log 2>&1 & echo $!'
# 3. 下完跑 e11
R 'cd /root/autodl-tmp && nohup bash ablation_battery.sh e11 > /root/autodl-tmp/battery_e11.log 2>&1 & echo $!'
```
e11 三连测：(a) KV 池容量对账（和 Qwen3-8B 的 e5 日志比每 token 字节）(b) MLA 后端 A/B（flashinfer/triton/flashmla，"结构红利要 kernel 接住"的证据）(c) ITL vs 上下文曲线。DeepSeek-V2-Lite 要 `--trust-remote-code`（脚本已加）。

---

## 6. 数据回收 & 出稿

跑完把整个结果目录拉回本地：
```bash
# 用 scp（同样要 expect 喂密码，或临时装 sshpass）。目标：
# study-notes/contribution-scan/exp3-ablation/  下的所有 *.log + phases.log + dmon.log
```
出稿要点：
- **V2 结构**：每个动词一节 = 开关 + 数字表 + 一句为什么 + 回链 V1 对应章节；结尾一张"关掉 X 掉多少"总条形图。已有数据（03 的 CUDA Graph 2.3×、07 的两面墙计数器）直接入账，见 08 文档 §0。
- **诚实局限照写**（07 的风格：64%≠100%、小样本±10%），这是知乎加分项。
- **配图**：用 gpt-image-2 出。中转站配置和坑见 memory `gpt-image-relay`（`.env` 里 `GPT_IMAGE_2_KEY_NEW` + `GPT_ENDPOINT`，model `gpt-image-2`，竖版 1024×1536 用 `quality=medium` ~80s，high 会撞 Cloudflare 524）。已有 5 张图在 `infographic/inference-opt-5verbs/`。
- **文风**：见 memory `writing-tone-preference`——用「我们」不用「你」，忌爹味/教师爷腔（别写"记住/注意/留个作业"），结论收着说，专业黑话（MECE 这种）要么不用要么解释。V1 已经按这个调过一轮，V2 保持一致。

---

## 7. V1 现状（别搞混）

已发布的 V1 是 `study-notes/推理优化全景-六个动词.md`（知乎已发，几十赞收）。框架：一个问题 → 后厨四约束（灶火=算力/送菜=带宽/案板=容量/出菜顺序=串行）→ 一条时间公式 → 六动词（不算/少算/少搬/少占/打满/破串行）→ 一次请求的一生 → 四层架构 → 技术×动词矩阵 → 四问。V2 是补"每格多少钱"，**不要重写 V1 的框架**。`推理优化全景-六个动词-codex.md` 是当时的平行参考版，已吸收进 V1，别动。

---

## 8. 待办 checklist

- [ ] 确认 `.env` SSH 信息最新（重启后可能变）
- [ ] 确认 GPU 可见（`nvidia-smi -L` 出 H20）
- [ ] 显式设 `OUT` 绝对路径，后台跑第一批 `e5 e1 e9 e2 e7 e8 e10 e12 e3 e6 e4`
- [ ] 监控 + 抽查前几个实验数据合理（尤其 e10 torch_native 的"失败即数据"）
- [ ] 第一批完 → 删 FP8/AWQ → 下 DeepSeek-V2-Lite → 跑 e11
- [ ] 数据拉回 `study-notes/contribution-scan/exp3-ablation/`
- [ ] 写 V2 姊妹篇 + 配图 + 回填 V1
