# 共享音频底座

服务于三个上层产品的共同基础设施：

- `清听 · 音审通`：音频修复、质量检测、内容审核；
- `听析`：ASR、分类、RAG 与 Agent 编排；
- `声刻`：参考音频质检、语音克隆与情感 TTS。

声刻 **1.0.0 工程验证版**：[日常使用与发布说明](docs/shengke-v1-release.md)。

GitHub 归档范围、未提交的大型本地资产和恢复方式见
[仓库归档说明](docs/shengke-repository.md)。

声刻开发入口：[MVP 使用与验证状态](docs/shengke-mvp.md)、[独立模型服务部署](docs/shengke-runtime.md)。CosyVoice2、GPT-SoVITS v2、中文 VITS 和 Transformer TTS 均已完成本机真实合成与 CPU 基准；四轮 N=1 自听共 21 份评分已保存，RTX 4090D 正式测试 144/144 通过并已关机。GPT 当前样本节奏获用户接受，CosyVoice 机械感减轻；开心强度与跨文本听感仍有改进空间。本机可双击 [启动声刻](scripts/启动声刻.command)，统一启动四模型与工作台。

完整交付状态见 [验收清单](docs/shengke-acceptance.md)；演示可按 [五分钟演示稿](docs/shengke-demo.md) 进行，GPU 操作见 [运行手册](docs/shengke-gpu-runbook.md)。

网站与推理采用分离部署：[个人网站展示与本地后端方案](docs/shengke-web-deployment.md)。展示页源文件在 `apps/shengke-showcase/`，已接入个人作品集的本地版本；实时生成继续使用 Mac 工作台，GPU 按需通过 SSH 隧道接入。

当前版本 `v0.9.0` 已建立可持久化 ingest、可插拔 VAD 与校准、说话人分离及 DER/JER 评测、带 speaker 和术语热词的 ASR 时间轴、音质分析及可重复模型评测闭环：

`API 创建作业 -> PostgreSQL 排队 -> worker 原子抢占 -> S3/MinIO 拉取 -> SHA-256 / ffprobe -> VAD/切片 -> Pyannote 说话人时间轴 -> Faster-Whisper/FunASR -> speaker 归属与全局时间戳合并 -> 音质分析/归一化 -> 制品与报告落库`

## 目录

```text
apps/platform-api/       FastAPI 作业入口
apps/ingest-worker/      PostgreSQL 队列消费者与 ingest 流水线
apps/asr-eval/           清单驱动的离线 ASR 基准评测
apps/vad-eval/           VAD 标注集评测与参数校准
apps/diarization-eval/   说话人分离 DER/JER 基准评测
packages/audio-core/     无业务耦合的音频核心库
contracts/               对外 JSON Schema
tests/                   单元与 FFmpeg 集成测试
infra/                   Docker 与本地编排
docs/                    架构决策和开发说明
```

## 本地运行

要求 Python 3.11+，以及可执行的 `ffmpeg` / `ffprobe`。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
make test
audio-platform-api
```

完整本地栈：

```bash
docker compose -f infra/compose.yaml up --build
python scripts/smoke_stack.py
```

完整 Docker 栈默认安装并启用 Faster-Whisper ASR，默认模型为 `small`：

```bash
docker compose -f infra/compose.yaml up --build
python scripts/smoke_stack.py --transcribe
```

使用 `data/input` 下的真实音频验证中文与术语热词：

```bash
python scripts/smoke_stack.py --transcribe \
  --input-file data/input/sample.wav \
  --language zh \
  --hotword 音频智能 \
  --hotword 术语热词
```

如需临时关闭 ASR，可设置 `AUDIO_PLATFORM_ASR_BACKEND=disabled`；如需换模型，可通过 `AUDIO_PLATFORM_ASR_MODEL` 覆盖。

启用 FunASR 时将 extra 改为 `asr-funasr`、backend 改为 `funasr`，并把 `AUDIO_PLATFORM_ASR_MODEL` 设为模型 ID 或完整本地目录。模型缓存保存在 Docker `model-cache` volume。

启用 WebRTC VAD 时安装 `vad-webrtc` extra；Docker 可用逗号组合多个 extra，例如 `AUDIO_PLATFORM_EXTRAS=asr-whisper,vad-webrtc`。

### 说话人分离

Pyannote 后端使用 `speaker-diarization-community-1`，需要先接受模型使用条件。Compose
通过只读文件挂载读取本机 Hugging Face token，不应把 token 明文写入命令、Compose 文件或日志：

```bash
cp .env.example .env.local
# 在 .env.local 中启用两个 extra/后端，并把 AUDIO_PLATFORM_HF_TOKEN_HOST_PATH
# 指向本机 Hugging Face token 文件；不要复制或粘贴 token 内容。
docker compose --env-file .env.local -f infra/compose.yaml up --build
```

作业增加 `diarize` 操作，并可用 `num_speakers` 或 `min_speakers` / `max_speakers` 约束 speaker 数量：

```json
{"operations":["probe","vad","diarize","transcribe"],"parameters":{"diarization":{"min_speakers":2,"max_speakers":4,"use_exclusive":true}}}
```

分离结果作为 JSON 制品进入 S3，并持久化到 PostgreSQL。制品同时保留
`regular_turns`、`exclusive_turns`，`turns` 表示实际用于对齐的时间轴。VAD segment 与
ASR transcript span 都会携带 speaker。配置、模型权限和时间轴归属规则见
[说话人分离说明](docs/speaker-diarization.md)。

完整六操作功能验证：

```bash
python scripts/smoke_stack.py \
  --input-file data/input/two-speaker-functional.wav \
  --diarize --num-speakers 2 --transcribe --language zh --timeout 900
```

人工标注真实多人语音后，可以从 RTTM 生成带 SHA-256 的可复现 JSONL 清单，并运行
regular/exclusive × oracle人数/人数范围/无约束的六配置准确率矩阵：

```bash
python scripts/prepare_diarization_manifest.py \
  --audio-root data/diarization-eval/audio \
  --rttm-root data/diarization-eval/rttm \
  --output data/diarization-eval/manifest.jsonl \
  --dataset internal-meetings --dataset-version 2026-09-27 \
  --split test --license internal-approved

docker compose --env-file .env.local -f infra/compose.yaml run --rm ingest-worker \
  audio-diarization-eval /data/diarization-eval/manifest.jsonl \
  --model pyannote/speaker-diarization-community-1 \
  --matrix \
  --frame-ms 10 \
  --collar-ms 0 \
  --boundary-tolerance-ms 250 \
  --speaker-bound-padding 1 \
  --output /artifacts/diarization-eval/community-1
```

工具会拒绝合成或未声明为人工标注的样本，并报告 DER、JER、speaker-count accuracy、
边界 MAE/P95/召回率、RTF、失败样本和生产门槛建议。详见
[说话人评测说明](docs/diarization-evaluation.md)。

VoxConverse 0.3 dev 的固定版本清单为
`data/diarization-eval/voxconverse-dev-v0.3-full.jsonl`；确定性 20 条验收子集为
`data/diarization-eval/voxconverse-dev-v0.3-acceptance.jsonl`。本地运行后的正式结果位于
`artifacts/diarization-eval/voxconverse-community-1-v0.3/`，其中
`acceptance-summary.md` 明确区分子集验收与 216 条全量 dev 评测。

### ASR 术语热词

作业可以在 `parameters.asr.hotwords` 中携带最多 100 个术语：

```json
{"asr":{"language":"zh","hotwords":["音频智能","客服质检","DeepFilterNet"]}}
```

术语会去除首尾空白并按大小写折叠去重。Faster-Whisper 接收空格拼接的 `hotwords`，FunASR 接收模型相关的 `hotword` 参数；使用前仍应通过 ASR 基准清单比较有无热词时的 WER/CER。详见 [ASR 热词说明](docs/asr-hotwords.md)。

### ASR 基准评测

准备逐行 JSON 的数据清单后，可以用生产流水线相同的适配器测量 WER、CER、RTF、语种识别、时间戳边界误差及资源峰值：

```bash
audio-asr-eval data/eval/manifest.jsonl \
  --backend faster-whisper \
  --model small \
  --output artifacts/asr-eval/faster-whisper-small
```

命令会同时生成 `report.json` 与 `report.md`。清单字段、指标口径和 FunASR 示例见 [ASR 评测说明](docs/asr-evaluation.md)。

### VAD 后端与校准

生产作业默认继续使用 `adaptive_energy_v1`。要切换 WebRTC，在作业参数中传入：

```json
{"vad":{"backend":"webrtc_v1","webrtc_aggressiveness":2,"minimum_speech_ms":150,"maximum_silence_ms":300,"padding_ms":100}}
```

也可以对人工标注数据批量扫描能量阈值或 WebRTC 模式：

```bash
audio-vad-eval data/vad-eval/manifest.jsonl \
  --backend adaptive_energy_v1 \
  --energy-threshold -48 \
  --energy-threshold -42 \
  --energy-threshold -36 \
  --output artifacts/vad-eval/baseline
```

报告根据时长级 F1 自动选择完整运行的最佳候选，并同时输出 Precision、Recall、IoU、边界 MAE 和 RTF。格式与指标说明见 [VAD 校准说明](docs/vad-evaluation.md)。

服务：API `http://localhost:8000`、MinIO S3 `http://localhost:9000`、MinIO Console `http://localhost:9001`、PostgreSQL `localhost:55433`。可通过 `AUDIO_PLATFORM_POSTGRES_PORT` 覆盖 PostgreSQL 的主机端口。

存储层使用标准 S3 API，通过环境变量可连接 MinIO、AWS S3 或其他 S3 兼容存储。由于 MinIO 社区版现为 source-only 发布，本地 Compose 会从固定发布标签构建开发镜像；生产环境应配置经审计的 MinIO/AWS S3 端点与密钥，并单独评估许可证和安全维护策略。

### 提交一个持久化作业

```bash
curl -X POST http://localhost:8000/v1/audio/jobs \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: demo-001' \
  -d '{"input_uri":"file:///data/input/sample.wav","operations":["probe","vad","diarize","transcribe","loudness","normalize"],"idempotency_key":"demo-001","parameters":{"sample_rate_hz":16000,"channels":1,"vad":{"backend":"adaptive_energy_v1","minimum_speech_ms":150,"maximum_silence_ms":300,"padding_ms":100},"diarization":{"min_speakers":1,"max_speakers":4,"use_exclusive":true},"asr":{"language":"zh","hotwords":["音频智能","客服质检"]},"quality_policy":{"minimum_integrated_lufs":-24,"maximum_integrated_lufs":-14,"maximum_true_peak_dbtp":-1,"maximum_clipping_ratio":0.0001,"maximum_silence_ratio":0.5}}}'
```

worker 会从 PostgreSQL 原子领取作业，将原始文件、派生 WAV 和 JSON 报告写入 S3，并把作业事件、制品、工具执行记录与报告回写 PostgreSQL。

API 启动后：

- `GET /healthz`
- `GET /readyz`
- `POST /v1/audio/jobs`
- `GET /v1/audio/jobs/{job_id}`
- `POST /v1/audio/jobs/{job_id}/cancel`
- `GET /v1/audio/jobs/{job_id}/assets`
- `GET /v1/audio/jobs/{job_id}/segments`
- `GET /v1/audio/jobs/{job_id}/transcript`
- `GET /v1/audio/jobs/{job_id}/diarization`
- `GET /v1/audio/jobs/{job_id}/report`
- `GET /docs`

## 直接使用 Audio Core

```bash
audio-core probe sample.wav
audio-core vad sample.wav
audio-core quality sample.wav
audio-core normalize sample.wav artifacts/sample.wav --sample-rate 16000 --channels 1
```

## 安全边界

- API 不直接接受任意本地路径；输入先登记为 URI，由后续受控 ingest worker 获取。
- `probe_file` 和 `normalize_audio` 只接受已存在的普通文件，且子进程不经过 shell。
- 原始音频不可覆盖；派生文件写入临时文件后原子替换。
- 数据、模型和音频制品默认不提交到 Git。

## 当前未包含

VAD 已提供 `adaptive_energy_v1` 与可选 `webrtc_v1` 后端及统一校准框架；ASR 已提供 Faster-Whisper、FunASR、术语热词及统一评测框架；说话人分离已提供 Pyannote 4.x 适配器、持久化时间轴和帧级 DER/JER 评测。模型权重、访问授权和业务指标仍需经审校数据集验证。租户权限、跨文件 speaker linking 与完整迁移框架仍在后续里程碑。未配置 `AUDIO_PLATFORM_DATABASE_URL` 时 API 仍可使用内存 Store 进行轻量开发，但生产环境必须使用 PostgreSQL。
