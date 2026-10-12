# 声刻 GPU 运行手册

状态（2026-10-12）：AutoDL RTX 4090D 24 GB、Torch 2.3.1+cu118、FP32 下，四模型各 36 次正式请求，共 144/144 成功。三个环境 CUDA 预检通过，证据已取回，实际消费 2.34 元并确认关机。Mac 四模型与工作台已恢复。结果及听感验收边界见 [验收记录](shengke-acceptance.md)。以下保留复现步骤与历史接入记录。

## 执行前需要的信息

需要可连接的 SSH 主机别名、端口、GPU 型号、小时价及停止计费的操作方式。通过已有 SSH 密钥或 SSH agent 登录，不把私钥、密码写入仓库或聊天。总云预算上限为 300 元，实例费、存储费及流量费都应计入；是否按分钟或整小时结算以服务商实际账单为准。

先在本机备齐公开模型与代码，再开启计费实例。目标 GPU 约 24 GB，四个模型顺序运行。每个模型验证一条合成后再跑完整矩阵；发生环境或权重错误时先停止基准，避免不断提交失败任务。

## 本次实际实例（2026-10-12）

用户充值 30 元后创建重庆 A 区 123 机实例 `5af44aa208-e5b03b67`：RTX 4090D 24 GB ×1、16 核 Xeon Platinum 8481C、80 GB 内存，系统盘 30 GB、免费数据盘 50 GB，未扩容。最终创建页配置费 1.88 元/小时、日常费用 0.00 元/日。基础镜像为官方 Miniconda / conda3 / Python 3.10（Ubuntu 22.04）/ CUDA 11.8。

平台服务器时间约 08:17 创建并运行；已在平台设置 2026-10-12 12:17:00 定时关机，设置时显示余下预计费用 7.49 元。该设置是停止运行，不是释放实例；提前完成或阻塞时仍应提前关机。页面提示连续关机 15 天后自动释放并清除数据，因此证据必须及时取回。

JupyterLab 终端 `nvidia-smi` 显示 NVIDIA GeForce RTX 4090 D、24564 MiB、驱动 580.76.05（驱动支持 CUDA 13.0）。数据盘独立环境 `models/environments/cosyvoice` 的 Torch/Torchaudio 2.3.1 cu118 已安装成功，日志位于远端 `artifacts/install-torch.log`。执行仓库同版 `shengke_gpu_preflight.py` 后，2026-10-12T00:31:21Z 的预检返回 `ready=true`，Python 3.10.8、Torch 2.3.1+cu118、CUDA 11.8，真实完成 64×64 CUDA 矩阵运算。远端 JSON 为 `artifacts/gpu-preflight-cosyvoice.json`，本机截图为 `artifacts/shengke/gpu-autodl-20261012/cuda-preflight.png`。NumPy 及模型其余依赖尚未安装，不能声称完整模型环境通过。

网页复制按钮提示成功，但自动化剪贴板未返回 SSH 信息；Jupyter 上传又被浏览器扩展的本地文件权限阻止，模型包尚未上传。已请用户将该实例登录指令和密码分别存入 `/tmp/shengke-gpu-login.txt` 前两行；临时空文件权限为 0600，凭证不进入仓库或报告。认证完成后删除临时文件。CUDA 预检结束后主动关机，待连接信息准备好再开机，避免空闲等待计费。

## 准备代码与权重

远端保持与本仓库相同的相对目录：`services/tts-runtime/`、`scripts/shengke_*.py`、`configs/shengke/`、`models/repos/`、`models/shengke/`、`models/environments/`。平台数据库和对象存储保留在本机，不需要上传实习资料、`.env.local`、数据库卷或私有录音。

- CosyVoice：源码固定 `074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc`，同时按该提交初始化 `third_party/Matcha-TTS` 子模块。迁移现有权重到 `models/shengke/cosyvoice2`，执行 `python scripts/shengke_verify_cosyvoice.py models/shengke/cosyvoice2`，验证十二个实际必需文件。公开源版本尚未核实，文件哈希锁可验证迁移一致性。
- GPT-SoVITS：源码固定官方 `20240821v2` 标签对应提交 `a7e38f93f1ad8d263093ca26ce0dea3fb7080ea2`。运行 `python scripts/shengke_fetch_models.py gpt-sovits-v2`，或迁移本机已校验权重后再次运行该命令复验。
- ESPnet：固定 `espnet==202402`。分别运行 `python scripts/shengke_fetch_models.py vits-csmsc` 与 `python scripts/shengke_fetch_models.py transformer-csmsc`。
- Transformer 声码器：在装有 `gdown==5.2.0` 的环境运行 `python scripts/shengke_prepare_vocoder.py`，下载和校验与 CPU 实测相同的三个文件。

所有下载命令可在本机完成。下载过程中不把 `.partial` 当成可加载权重；完整文件也必须通过哈希校验。

## 模型环境与 CUDA 预检

### 已准备的离线迁移包（2026-10-11）

本机 `artifacts/shengke/gpu-transfer-20261011/shengke-gpu.tar` 为 5,706,731,520 字节，SHA-256 为 `c58527cdbcac32000f17f27c5d33cbe9a0123042a85a905aca0e054c9d3a4199`。同目录 `manifest.json` 列出 50 个文件及各自大小和 SHA-256，`verification.json` 记录逐成员读回校验及三个 Git bundle 的克隆还原验证。仅用于该版本的迁移；运行时代码或权重变化后须重建。

包内包括显式选取的模型服务代码、四模型必要权重和三个固定提交的公开源码 bundle。权重已按现有锁文件复验；不包含本机 Python 环境、登录凭证、数据库、实习资料或参考录音。GPT-SoVITS 原目录是部分克隆，直接打包全部历史会触发额外下载，因此从本机建立深度 1 的副本后导出 bundle，保留原提交号。CosyVoice 两份未跟踪的本地笔记本不会进入 bundle。

获得主机后传输压缩包和清单，先比对整个 tar 的 SHA-256，再在专用空目录解包。源码还原：

```bash
git clone sources/CosyVoice.bundle models/repos/CosyVoice
git clone sources/Matcha-TTS.bundle models/repos/CosyVoice/third_party/Matcha-TTS
git clone sources/GPT-SoVITS.bundle models/repos/GPT-SoVITS
```

`git rev-parse HEAD` 必须与 `manifest.json` 的 `source_revisions` 一致。Git bundle 保留提交元数据，运行时可继续记录真实上游版本。本轮实际采用 25.39 MB 引导包加云端下载权重：完整迁移包上传过慢，已中止。CosyVoice 权重来自 ModelScope，其他权重使用固定版本的 Hugging Face 镜像；云端 50 个部署文件均与原始清单 SHA-256 匹配。三个独立 Linux 环境已安装并实测，冻结依赖见 `configs/shengke/*-linux-cu118-py310.lock.txt`。旧版 Whisper 构建使用 setuptools 69.5.1 与 `--no-build-isolation`；未启用可选 TensorRT 或 DeepSpeed 加速。

使用三个独立的 Python 3.10 环境：CosyVoice、GPT-SoVITS、ESPnet（两个基线共用）。Mac 环境锁文件位于 `configs/shengke/*macos-py310.lock.txt`，可用于核对依赖，不能直接宣称适用于 Linux CUDA。先安装与目标驱动兼容的 CUDA 版 Torch，再结合固定源码的依赖安装；依赖调整必须记录并重新验证。

已验证 CPU 环境的关键版本为 Torch/Torchaudio 2.3.1。ESPnet 使用 NumPy 1.23.5、SciPy 1.10.1、Parallel WaveGAN 0.6.1；GPT-SoVITS 的依赖入口见 `services/tts-runtime/requirements-gpt-v2-macos.txt`，并需运行 `scripts/shengke_patch_gpt_dependencies.py` 修复已固定的 LangSegment 0.2.0 导出问题。CosyVoice 的 Linux 上游依赖包含 CUDA ONNX Runtime；安装后的提供程序与实际运行设备都需要检查。

在每个模型环境的 Python 下执行：

```bash
python scripts/shengke_gpu_preflight.py --output artifacts/gpu-preflight-<环境名>.json
python -m pip check
```

预检要求 Torch 真正完成 CUDA 矩阵运算，并记录 GPU、显存容量、驱动、Torch、CUDA 与 Python 版本。失败返回非零状态。CPU 开发机已验证它会正确拒绝 CUDA 不可用的环境；2026-10-12 三个 Linux 环境均已通过真实 CUDA 预检，证据见 `artifacts/shengke/gpu-autodl-20261012/remote/artifacts/gpu-preflight-*.json`。

## 顺序启动与基准

首先使用四模型统一 FP32 的口径，避免把精度差异混入模型差异。FP16 优化属于后续单独对照，ESPnet 当前仅开放 FP32。

```bash
# 远端：每次只启动一个模型；依次替换模型 ID。
python scripts/shengke_model.py cosyvoice2 --device cuda
```

模型 ID / 端口：CosyVoice2 9881、GPT-SoVITS v2 9882、VITS 9883、Transformer 9884。启动器默认使用 `models/environments/` 中相应环境，可用 `--python` 指定别的位置。服务只监听远端 loopback。

```bash
# 本机：本地用不同端口，避免覆盖已经运行的 CPU 服务。
ssh -N -L 19881:127.0.0.1:9881 \
  -L 19882:127.0.0.1:9882 -L 19883:127.0.0.1:9883 \
  -L 19884:127.0.0.1:9884 <SSH主机别名>

# 待现有队列清空后，重启本机工作台启动器并使用隧道端点。
export SHENGKE_MODEL_ENDPOINTS='{"cosyvoice2":"http://127.0.0.1:19881","gpt-sovits-v2":"http://127.0.0.1:19882","vits-csmsc":"http://127.0.0.1:19883","transformer-csmsc":"http://127.0.0.1:19884"}'
python scripts/shengke_local.py --ui-python <Gradio环境Python>
```

在工作台先验证当前模型一条真实合成，再运行：

```bash
PYTHONPATH=packages/audio-core/src:apps/tts-eval/src python -m tts_eval.cli \
  data/tts-eval/mandarin-v1.jsonl \
  --voice-id c96637f6-3e88-4bef-9e7d-62b18e876c30 \
  --models cosyvoice2 --repeats 3 \
  --output artifacts/shengke/benchmark-cosyvoice2-gpu-public-D12
```

上述音色编号属于当前本机数据库；新数据库应先导入公开参考，并使用导入结果中的 D12_892 编号。依次替换模型与输出目录，始终使用同一 GPU、FP32、参考音频和十二条文本。每模型一次预热不计入分布，36 次正式请求；运行期间不要做界面生成或另启模型争用 GPU。

## 保存与验收

保存每个模型的 `report.json`、`runs.jsonl`、`warmup.jsonl`、WAV、环境预检文件和服务日志。检查每条正式记录中 `device=cuda`、实际 GPU 名称、精度、权重哈希、CUDA 峰值 allocated/reserved 字段；失败项保留在总请求数中。汇总 RTF 与推理耗时 P50/P95、失败率、显存，CPU RSS 单列。

每个模型完成后停止其服务再加载下一个。所有证据复制回本机并验证可读取后，按服务商实际流程关闭计费实例；关闭 SSH 窗口或停止模型进程不等于停止计费。保存最终账单以核对 300 元预算。无人值守情况下不得仅依据本手册声称已自动关机或已完成 GPU 验收。

GPU 性能验收与听感分开。两轮八样本均已完成 N=1 真人预评测；第二轮情绪与语速调整未达到一致改善，默认仍使用 `baseline`，不能把首轮分数用于新候选。六份分档参考已由用户核对，D12 十秒逐字稿已按反馈更正；在 GPU 导入时使用当前素材清单，保留更正及发音疑点记录。

## 阿里云接入准备（2026-10-11）

用户已确认已有阿里云账号。优先核对 `ecs.gn7i-c8g1.2xlarge`：官方规格为 8 vCPU、30 GiB 内存、单张 NVIDIA A10 24 GB，满足计划中单卡约 24 GB 的方向，不要求必须 RTX 4090。来源：https://help.aliyun.com/zh/egs/gpu-accelerated-compute-optimized-instance-families/ 。库存、按量单价和可用区必须以登录后账号实际购买页为准，不使用开发者社区文章的价格代替报价。

用户已登录。2026-10-11 22:00 左右，杭州购买页可选上述规格，参考实例单价 9.5326 元/小时；100 GiB ESSD PL0 系统盘 0.105 元/小时，合计 9.6376 元/小时。公网按实际出网流量 0.8 元/GB，草稿峰值 100 Mbps；流量未计入小时价。价格是当前账号页面报价，库存仍以提交结果为准。

已准备未提交草稿：实例名 `shengke-gpu-eval`，Ubuntu 22.04 官方预装 NVIDIA 580.126.09 / CUDA 12.8 镜像，默认免费安全加固，元数据仅加固模式。SSH、RDP、HTTP、HTTPS 公网入站端口均未勾选；登录凭证选择创建后设置，实际连接配置尚未完成。未启用付费备份、快照或 CDT。购买数量 1。

草稿自动释放时间为北京时间 2026-10-12 04:15，系统盘随实例释放；这会删除临时实例及云盘数据，需要提交前明确确认。若按六小时估算，实例和磁盘约 57.83 元，另加实际出网流量；总预算仍为 300 元。提交前重新核对当前时间和剩余测试窗口，不能沿用过期释放日期。所有测试证据须在释放前持续复制回本机。

截至这份阿里云草稿记录时，没有接受服务协议、没有下单，也没有产生 GPU 实测结果。后续改用 AutoDL，阿里云草稿未提交。浏览器操作规则要求在接受服务协议及不可恢复释放安排前取得当次确认；配置已保留供用户查看。确认后还需完成实例创建、受限连接配置与 CUDA 实机预检，才能运行四模型矩阵。

## AutoDL 低成本方案核对（2026-10-11 晚）

用户改为登录 AutoDL 后，已核对实际市场页面：西北 B 区 RTX 4090 暂无空闲，北京 B 区 RTX 4090 24 GB 有库存，列价 2.29 元/小时、会员展示价 2.18 元/小时。重庆 A 区 RTX 4090D 24 GB 有库存，列价 1.98 元/小时、会员展示价 1.88 元/小时；不能把会员展示价当作账号最终结算价。候选为重庆 A 区 086 机（页面标识 `5d5147b55f`），单卡分配 15 核 CPU、80 GB 内存、30 GB 系统盘及 50 GB 数据盘，数据盘当时可扩容 665 GB。库存为即时观察，并未锁定。

以下是首次选机时的历史记录，后续绑定、充值、开机与实测均已完成：点击候选的“1卡可租”后，平台提示“请先绑定微信后开始租用GPU”。已导航至账号安全页，微信状态为未绑定；需要用户自行绑定。尚未到镜像与最终价格确认页，没有创建实例或发生 GPU 租用费用。4090D 与标准 4090 应在后续实测报告中准确区分；不得标为标准 4090 的性能。此前阿里云配置仍未提交，不应继续误购该高价草稿。


## AutoDL 实测完成（2026-10-12）

实际实例 `5af44aa208-e5b03b67`，重庆 A 区，RTX 4090D 24 GB，16 核 CPU / 80 GB RAM、30 GB 系统盘及 50 GB 免费数据盘，无付费扩容。SSH 已连接成功，认证后删除本机临时密码文件。四模型正式测试 144/144 通过，另完成 GPT 六种参考回归及 CosyVoice 开心原版/候选对照；这些结果不能替代真人听感评分。

全部原始音频、日志、环境锁及校验结果已保存本机。09:34 确认实例已关机，三笔实际费用 0.46 + 0.80 + 1.08 = 2.34 元；账单和最终关机截图见 `artifacts/shengke/gpu-autodl-20261012/`。Mac 四模型已恢复，工作台切回本地 CPU 推理。平台提示连续关机 15 天后释放，请使用本机归档复现，不依赖云实例长期留存。
