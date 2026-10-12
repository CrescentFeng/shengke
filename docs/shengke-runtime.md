# 声刻模型服务部署

四种模型必须使用各自的上游环境。`services/tts-runtime/server.py` 只依赖标准 HTTP 协议，不导入平台包；附加依赖见同目录 `requirements.txt`。四种模型的本机 CPU 环境均已完成真实推理，RTX 4090D FP32 的四模型 144/144 次正式请求也已通过，见 [验收记录](shengke-acceptance.md)。

GPT-SoVITS 和两个 ESPnet 基线的固定版本权重可用 `python scripts/shengke_fetch_models.py <模型ID>` 下载，默认目录 `models/shengke/<模型ID>`。脚本只选择清单中的文件，按 Git blob / LFS SHA-256 校验，保存 `download-lock.json`；`.partial` 文件不被当作完整权重。`--metadata-only` 可先查看大小。下载可重新运行以继续未完成的文件；下载成功仍不代表推理验证通过。CosyVoice2 当前通过文末的本地文件哈希锁迁移与复验。

VITS 和 Transformer 还支持 `--source zenodo`，从作者发布的原始归档提取同一权重，同时验证归档 MD5 和固定 Hugging Face 版本的文件哈希。Transformer 的缓存复验命令已通过。网络中断时重新运行同一命令即可继续，不应删除哈希检查或把 `.partial` 改名成权重。

若 Python TLS 连接反复中断，可指定 `SHENGKE_DOWNLOAD_TRANSPORT=curl python scripts/shengke_fetch_models.py <模型ID>`，使用系统 curl 续传同一来源；文件大小与哈希的验收规则不变。参考素材准备脚本也支持这一环境变量。

## 统一启动与退出

日常使用 `scripts/shengke_service.py start --open` 或双击 `scripts/启动声刻.command`；`status` 查询，`stop` 停止。关闭终端不会停止后台服务；没有登录自动启动。配置文件为 `configs/shengke/local-macos.json`，保留独立模型环境和 GPT 停顿辅助环境。启动器先运行基础服务与平台预检，再加载模型；所有服务就绪后才打开网页。

后台管理使用操作锁、实例锁和仅本机可访问的控制通道，不按端口杀进程。重复启动复用健康实例；未知占用拒绝启动。停止只清理本次管理的模型与工作台，不停止 Docker 基础服务或其他项目。状态与监督日志在 `artifacts/shengke/service/`，各模型日志在 `artifacts/shengke/logs/stack/`。首次安装仍需准备模型环境并运行数据库迁移。

前台开发仍使用 `scripts/shengke_stack.py --start-infra`，保留终端并通过 Ctrl+C 停止；不可与后台入口同时抢占固定端口。完整日常说明及验收见 [v1.0 发布说明](shengke-v1-release.md)。

## 通用约定

- `SHENGKE_MODEL`：`cosyvoice2` / `gpt-sovits-v2` / `vits-csmsc` / `transformer-csmsc`。
- `SHENGKE_REPO_ROOT`：对应固定提交的代码仓库绝对路径。
- `SHENGKE_MODEL_DIR`：本次模型所有权重的公共父目录，用于计算 SHA-256；应包含 BERT/HuBERT 或声码器等实际使用的依赖。
- `SHENGKE_DEVICE`：`cpu` 或 `cuda`。请求 CUDA 而不可用时直接失败；GPU 主机做 CPU 对照需显式设置 `CUDA_VISIBLE_DEVICES=''`。
- `SHENGKE_FP16`：`0` 或 `1`。FP16 只允许 CUDA；ESPnet 基线目前要求 FP32。
- `SHENGKE_CPU_THREADS`：默认 4。
- `SHENGKE_MODEL_TOKEN`：可选服务令牌，worker 与模型服务需一致；不要写入仓库。

启动方式：在所选模型的独立环境执行 `python -m uvicorn server:app --app-dir /绝对路径/项目/services/tts-runtime --host 127.0.0.1 --port 9881`。服务同步加载模型、一次接受一个推理请求，忙时返回 429；每个模型一个进程，不使用 Uvicorn 多 workers 重复加载权重。

## CosyVoice2

固定代码提交见 `configs/shengke/models.json`；本机权重为 CosyVoice2-0.5B。原生调用 `inference_zero_shot` / `inference_instruct2`。关闭自动文本规范化前端，避免运行时隐式下载未固定的资源。中文标点与数字读法由输入文本明确给出。

## GPT-SoVITS v2（本机 CPU 已验证）

额外指定 `SHENGKE_GPT_CONFIG`，必须是专用的配置文件，不直接修改用户原有配置。上游配置的 `custom` 部分应固定：

```yaml
custom:
  device: cpu
  is_half: false
  version: v2
  t2s_weights_path: /models/gpt-sovits/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt
  vits_weights_path: /models/gpt-sovits/gsv-v2final-pretrained/s2G2333k.pth
  bert_base_path: /models/gpt-sovits/chinese-roberta-wwm-ext-large
  cnhuhbert_base_path: /models/gpt-sovits/chinese-hubert-base
```

路径必须是绝对路径且全部存在，避免依赖工作目录。服务会用 `SHENGKE_DEVICE` 与 `SHENGKE_FP16` 覆盖运行设备。模型只开放普通话、短参考克隆和中性风格；不把 V2 的音色克隆包装成已验证的情绪控制。配置字段来源：[上游配置](https://github.com/RVC-Boss/GPT-SoVITS/blob/main/GPT_SoVITS/configs/tts_infer.yaml)，实际安装时仍需记录固定提交。

## ESPnet 中文基线（两种架构已完成真实合成）

指定 `SHENGKE_ESPNET_CONFIG` 与 `SHENGKE_ESPNET_WEIGHTS`，分别指向对应下载包的训练配置与权重。VITS 使用 [CSMSC VITS](https://huggingface.co/espnet/kan-bayashi_csmsc_vits) 的端到端波形；Transformer 使用 [CSMSC Transformer](https://huggingface.co/espnet/kan-bayashi_csmsc_transformer) 的声学模型，额外指定 `SHENGKE_VOCODER_CONFIG` / `SHENGKE_VOCODER_WEIGHTS`，使用匹配 CSMSC 特征配置的 Parallel WaveGAN。

不能用普通 vocoder-only 输出替代 Transformer TTS，也不能用其他架构来冒充 Transformer。部署时检查采样率、hop length、mel 数量和归一化统计一致；验证输出波形非空后才加入正式基准。固定音色基线不评估克隆相似度。

声码器可以用 `models/environments/espnet/bin/python scripts/shengke_prepare_vocoder.py` 准备。没有缓存时需要该环境中的 `gdown==5.2.0`；脚本验证归档 SHA-256，只提取 checkpoint、配置和归一化统计三个普通文件，再逐个校验哈希。已在空目录中成功复现解包，结果与实测服务使用的文件一致。

## 云 GPU 验证

建议按模型顺序启动、预热、跑基准、保存证据、停止进程，避免四模型同时争抢显存。使用同一台约 24 GB NVIDIA GPU；记录 GPU 型号、驱动、CUDA/Torch 与实际精度，不能用估算显存或 CPU 结果代替 GPU 实测。

通过 SSH 把远程模型服务转发到本地，例如 `ssh -N -L 9881:127.0.0.1:9881 <用户>@<主机>`。远端仍绑定 loopback，MinIO 和数据库留在本机。端口按模型区分：CosyVoice2 9881、GPT-SoVITS 9882、VITS 9883、Transformer 9884。平台配置示例：

```bash
export SHENGKE_MODEL_ENDPOINTS='{"cosyvoice2":"http://127.0.0.1:9881","gpt-sovits-v2":"http://127.0.0.1:9882","vits-csmsc":"http://127.0.0.1:9883","transformer-csmsc":"http://127.0.0.1:9884"}'
```

控制总预算 300 元：开机前确认计费单位和小时价；先验证单条，再跑完整矩阵，完成后关闭计费实例并保留证据。2026-10-12 已完成 AutoDL RTX 4090D 实测，实际消费 2.34 元，证据取回后已确认关机。

详细执行与验收步骤见 [GPU 运行手册](shengke-gpu-runbook.md)。预检脚本会实际执行 CUDA 矩阵运算，单凭驱动列表或文件存在不算通过。

## 已验证的 Mac Transformer 启动方式

专用环境：`models/environments/espnet`，Python 3.10、Torch 2.3.1、ESPnet 202402、SciPy 1.10.1、Parallel WaveGAN 0.6.1。完整已安装依赖见 `configs/shengke/espnet-macos-py310.lock.txt`；该锁文件描述 Mac CPU 环境；Linux CUDA 已安装环境分别记录在 `configs/shengke/espnet-linux-cu118-py310.lock.txt`、`gpt-v2-linux-cu118-py310.lock.txt` 和 `cosyvoice2-linux-cu118-py310.lock.txt`。

```bash
python scripts/shengke_model.py transformer-csmsc
```

该命令检查权重、训练配置、归一化统计及声码器文件，再在 9884 端口启动服务。服务还会拒绝错误架构或不匹配的声学特征。实际 checkpoint 和声码器哈希分别记录在任务报告和 `configs/shengke/csmsc-vocoder.json`。

平台启动时加入已运行端点：

```bash
export SHENGKE_MODEL_ENDPOINTS='{"cosyvoice2":"http://127.0.0.1:9881","transformer-csmsc":"http://127.0.0.1:9884"}'
python scripts/shengke_local.py --ui-python /opt/anaconda3/envs/game_tts/bin/python
```

GPT-SoVITS 源码固定到官方 `20240821v2` 标签（`a7e38f93f1ad8d263093ca26ce0dea3fb7080ea2`），使用专用环境 `models/environments/gpt-sovits`。推理依赖入口为 `services/tts-runtime/requirements-gpt-v2-macos.txt`，已通过推理模块导入、`pip check`、权重加载与真实克隆。`python scripts/shengke_model.py gpt-sovits-v2` 会生成专用配置并在 9882 端口启动。

PyPI 的 LangSegment 0.2.0 导出两个不存在的旧别名，导致上游模块无法导入。安装依赖后，用该环境运行 `models/environments/gpt-sovits/bin/python scripts/shengke_patch_gpt_dependencies.py`。脚本检查版本与原文件 SHA-256，只把旧别名绑定到包内现有的 `setfilters` / `getfilters`，不改动模型或分词实现。该环境的复现需要同时安装锁定依赖并执行此修复。

VITS 使用同一 ESPnet 隔离环境：`python scripts/shengke_model.py vits-csmsc`，服务端口 9883。首条 CPU 推理已生成 3.04 秒、22.05 kHz 音频，RTF 0.42；该条是功能验证，不能代替重复基准。

GPT-SoVITS 对 MVP 最多 200 字的输入默认使用上游 `cut0` 整句模式。早期 `cut5` 按标点切出的极短片段出现提前 EOS，前后音频分别保留于 `artifacts/shengke/runtime-gpt-v2-cpu/` 和 `runtime-gpt-v2-cpu-cut0/`，不得把接口成功当成文本完整性或听感验收。两轮实际试听仍反馈逗号停顿不足；整体降速未带来改善。

显式实验配置 `gpt_pause_v3` 仅限 GPT-SoVITS：对纯汉字、受支持标点和空格构成的文本，在至少两个分句且每句至少八个汉字时启用原生 `cut5`；其他文本继续 `cut0`。保持速度 1.0、种子 42 和原生片段间隔 0.3 秒。八字门槛只是规避已知极短片段问题的保守实验条件，并不保证完整性或自然度。实际切分模式、间隔及选择原因写入每份报告的 `delivery`，顶层 `text_split_method` 同步记录。

用户实际使用十秒参考时发现该门槛仍会触发提前 EOS。运行时现在在声码器前拒绝零、负或越界语义长度，防止上游 `[-0:]` 将参考语义当成目标解码。`cut5` 失败时清除该次全部输出并整句重试一次；报告记录 `requested_text_split_method=cut5`、实际 `text_split_method=cut0` 和 `fallback_reason=early_semantic_termination`。整句也失败则返回明确错误，不发布上游异常处理中临时 yield 的静音。该保护适用于 GPT 基线和实验配置，不修改上游源码或替换用户选择的参考音色。

修复后已用两位说话人各 3/5/10 秒参考复测同一句目标文本，六个条件均生成成功：五个原生分句，一个触发保护后整句重试。本地 Whisper base 无提示转写支持旧错误的大段参考内容混入已在该案例中消除；识别仍有少量字词差异，不作为逐字准确或自然度验收。完整证据见 `artifacts/shengke/gpt-early-eos-regression/`。

“清晨的阳光……”真实候选中间检测到约 0.302 秒的零值区间；原始音频仅末尾存在 0.3 秒零值区间。第三轮 N=1 配对试听已完成：候选自然度 4、原始 3，相似度均 5；用户反馈停顿更自然但略长。工作台已提供默认关闭的“按标点停顿（试用）”选项，保留已试听参数；单一文本结果不代表一般化效果。证据位于 `artifacts/shengke/pause-trial-v3/`。

CosyVoice2 的现有本地权重已逐文件锁定于 `configs/shengke/cosyvoice2-weights-lock.json`，校验命令为 `python scripts/shengke_verify_cosyvoice.py <权重目录>`。这份清单表示本机已实测文件的身份，不表示已核实上游仓库版本；官方元数据网络连接失败，未填写猜测的提交号。迁移现有文件后应在目标机器再次运行校验。Mac 推理环境快照另存于 `configs/shengke/cosyvoice2-macos-py310.lock.txt`。
# 可重复的本地内容诊断

`scripts/shengke_check_content.py` 使用已下载的 OpenAI Whisper 权重，在 CPU 上独立转写输出音频。只在转写结束后比较目标文本，不向 ASR 提供目标句提示，不自动下载权重，不上传音频，不覆盖既有报告。

```bash
PATH=/opt/anaconda3/bin:$PATH /opt/anaconda3/envs/game_tts/bin/python \
  scripts/shengke_check_content.py \
  --audio artifacts/shengke/gpt-early-eos-regression/original-error.wav \
          artifacts/shengke/gpt-early-eos-regression/same-input-fixed.wav \
  --expected-text '清晨的阳光穿过窗帘，落在书桌旁的一杯热茶上。' \
  --weights models/cache/whisper-openai/base.pt \
  --weights-sha256 ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e \
  --output artifacts/shengke/content-diagnostic-new.json
```

输出记录权重与音频哈希、转写参数、分段文字及原始字符编辑距离。默认 0.6 阈值仅用于提示严重内容偏离，没有校准为产品验收门槛。繁简体、同音字、数字形式尚未折叠，可能造成误报；ASR 本身也会漏字或错字。`no_gross_mismatch_detected` 不等于逐字正确，`human_acceptance` 始终为 `not_assessed`。音色、情绪和自然度仍需真人试听。

进程退出码 0 只表示诊断执行完毕，报告可能仍有 `review_required`；2 表示至少一个文件诊断失败。参数或权重校验失败也会非零退出。每次使用新的输出路径保留历史证据。

## 可选舒缓节奏

`cosy_rhythm_v5` 仅适用于 CosyVoice2 开心情绪，保留原生 speed=1.0 后进行 FFmpeg atempo=0.85 处理。模型进程 PATH 必须包含 FFmpeg（本机验证版本 6.1.1）；缺失或处理失败会报错。报告保存处理链与输入/输出哈希，inference_s 包含后处理。默认 baseline 不受影响；此配置仅完成 Mac CPU 及单句听感验证，未纳入此前 GPU 矩阵。

## GPT 舒缓语速与停顿（试用）

显式配置 `gpt_rhythm_v6` 保留 GPT 分句早停保护，再调用独立 CPU Whisper 环境定位生成音频中的标点边界，最后进行 atempo=0.9 处理。适用范围为中文及标点；数字、外语、低置信度对齐不作为可靠标点位置。识别文本差异过大、边界附近缺字或时间不可靠时跳过该处，报告和界面均保留跳过数量；不保证每一个标点都能自动补入停顿。已有至少约 120 毫秒的局部静音会保留，避免重复加长。

本机启动 GPT 时显式提供辅助环境（已有本地权重，不自动下载）：

```bash
CUDA_VISIBLE_DEVICES='' PATH=/opt/anaconda3/bin:$PATH \
  /opt/anaconda3/envs/audio-ai/bin/python scripts/shengke_model.py gpt-sovits-v2 \
  --device cpu --rhythm-python /opt/anaconda3/envs/game_tts/bin/python \
  --rhythm-weights models/cache/whisper-openai/base.pt
```

部署到其他机器时替换 Python 路径，安装已有锁文件中的 Whisper、Torch、soundfile 及 FFmpeg。Whisper base 权重校验 SHA-256 `ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e`。没有配置辅助环境会明确报错；默认模型推理不依赖该辅助环境。辅助进程最多执行 180 秒，使用 CPU。推理耗时包含后处理；现有 peak_rss 只采样主模型进程，不包含 Whisper/FFmpeg 子进程，因此不能当成整条处理链的内存峰值。

五个 CPU 端到端用例成功，原接受样本 PCM 完全一致。新模式未纳入此前 GPU 基准，新增 ASR 延时也不能混入旧 RTF 数值。
