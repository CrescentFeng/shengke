# 说话人分离与 speaker 时间轴

v0.8 在 ingest worker 中增加可插拔 `DiarizationBackend`，首个后端为 Pyannote 4.x 的 `pyannote/speaker-diarization-community-1`。该模型提供常规说话人时间轴和 exclusive 时间轴；默认优先使用 exclusive 结果，以便与单一文本时间轴对齐。

## 模型准备

1. 在 Hugging Face 接受 `pyannote/speaker-diarization-community-1` 的使用条件。
2. 创建有读取权限的 access token，并使用 `hf auth login` 安全保存到本机；不要把 token
   发送到聊天、写入已跟踪文件或放入会被打印的命令参数。
3. 安装可选依赖并配置 worker。

```bash
pip install -e ".[diarization-pyannote]"
hf auth login
export AUDIO_PLATFORM_DIARIZATION_BACKEND=pyannote
export AUDIO_PLATFORM_DIARIZATION_MODEL=pyannote/speaker-diarization-community-1
export AUDIO_PLATFORM_DIARIZATION_DEVICE=cpu
```

`hf auth login` 把凭据保存在 Hugging Face 本机 token 文件中，后端会自动读取；不要把
token 放在 shell 历史或命令参数中。Docker Compose 推荐在未跟踪的
`.env.local` 中设置 `AUDIO_PLATFORM_HF_TOKEN_HOST_PATH`，把本机 token 文件只读挂载为
`HF_TOKEN_PATH`：

```dotenv
AUDIO_PLATFORM_EXTRAS=asr-whisper,diarization-pyannote
AUDIO_PLATFORM_DIARIZATION_BACKEND=pyannote
AUDIO_PLATFORM_HF_TOKEN_HOST_PATH=/absolute/path/to/huggingface/token
```

```bash
docker compose --env-file .env.local -f infra/compose.yaml up --build
```

`.env.local`、`.env.*.local` 和 `.secrets/` 都被 `.gitignore` 排除。模型也可以预先下载到
受控目录，并把 `AUDIO_PLATFORM_DIARIZATION_MODEL` 设置为本地路径。Docker Compose 将
Hugging Face 模型缓存保存在持久化 `model-cache` volume，容器重建或重启不会重复下载。

## 作业参数

```json
{
  "operations": ["probe", "vad", "diarize", "transcribe"],
  "parameters": {
    "diarization": {
      "min_speakers": 2,
      "max_speakers": 4,
      "use_exclusive": true
    }
  }
}
```

已知人数时可只设置 `num_speakers`。它不能与 `min_speakers`、`max_speakers` 同时使用，且最小人数不得大于最大人数。

## 数据流

1. Pyannote 在原音频的全局毫秒时间轴上生成 `SpeakerTurn`。
2. `DiarizationDocument` 同时保存常规 `regular_turns`、exclusive
   `exclusive_turns`，以及本次用于对齐的 `turns`；完整 JSON 写入 S3，并通过
   `audio_diarizations` 表保存 URI、speaker 数量和 JSON payload。
3. worker 计算每个 VAD segment 与各 speaker turn 的重叠时长，把重叠最多的 speaker 赋给 segment。
4. 平局时按 speaker 标识稳定排序，保证重放结果确定。
5. ASR 的每个 `TranscriptSpan` 继承 segment speaker。

可通过以下接口分别查询原始 turn 时间轴和归属后的片段：

```text
GET /v1/audio/jobs/{job_id}/diarization
GET /v1/audio/jobs/{job_id}/segments
GET /v1/audio/jobs/{job_id}/transcript
```

segment 的 `speaker_coverage` 标签表示获胜 speaker 的重叠时长占 segment 时长的比例。它是对齐覆盖率，不是模型置信度。

## 端到端功能测试

下面的样本应包含至少两位说话人。合成语音只能用于验证加载、编排、持久化与 API，必须
明确标记为“功能测试、非准确率评测”；DER/JER 准确率验收必须换成人工标注的真实语音。

```bash
python scripts/smoke_stack.py \
  --input-file data/input/two-speaker-functional.wav \
  --diarize --num-speakers 2 \
  --transcribe --language zh --timeout 900
```

输出应包含 `speaker_count`、三类 turn 数量、segment speaker、transcript speaker 以及
MinIO URI。数据库可检查 `audio_diarizations`、`audio_segments`、`audio_transcripts` 和
`tool_runs`；MinIO 中对应 job 前缀应包含原始音频、切片、归一化音频、diarization、
transcript 和 report JSON。

## 当前边界

- 模型需要单独授权和下载，默认 worker 不启用 diarization。
- 重叠说话保留在原始 turn 时间轴；单个 VAD segment 目前只有一个主 speaker。
- DER/JER 基准评测已由 `audio-diarization-eval` 提供；当前尚未实现跨文件 speaker identity linking。
- 生产前应在客服、会议和播客标注集上验证 speaker 数量、边界与领域泛化。
