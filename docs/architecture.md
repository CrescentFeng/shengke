# 共享音频底座架构 v0.9

## 设计原则

1. 原始输入不可变，所有派生结果都记录来源、工具版本和参数。
2. 音频处理与业务编排分离。`audio_core` 不依赖 FastAPI、数据库或具体模型。
3. 先定义数据契约，再接入 ASR、增强、分类和 TTS。
4. 所有耗时操作最终进入异步 worker；HTTP 层只创建、查询或取消作业。
5. 模型输出必须可回链到输入音频、时间段和运行版本。
6. worker 使用可恢复租约：启动时将超过阈值的 `running` 作业重新排队。

## 当前组件

```text
Client
  |
  v
Platform API ----> PostgreSQL metadata + DB-backed queue
                         |
                         v
                  Ingest Worker ----> S3 API (MinIO / AWS S3)
                         |
                         v
   ffprobe + VAD/slicing + diarization + ASR + quality + normalization

Audio Core
  +---- SHA-256
  +---- ffprobe metadata
  +---- LUFS / true peak / clipping / silence metrics
  +---- adaptive-energy / WebRTC VAD + millisecond timeline
  +---- speech slice export
  +---- pluggable Faster-Whisper / FunASR adapters
  +---- global transcript timeline merge
  +---- speaker turn timeline + segment/span attribution
  +---- ffmpeg normalization
  +---- path and output safety checks

ASR Eval
  +---- JSONL benchmark manifest
  +---- shared production ASR adapters
  +---- WER / CER / RTF / language / timestamp / memory metrics
  +---- versioned JSON contract + Markdown report
  +---- per-sample hotword regression

VAD Eval
  +---- human-annotated interval manifest
  +---- energy threshold / WebRTC mode sweep
  +---- duration precision / recall / F1 / IoU + boundary MAE
  +---- best complete candidate selection

Diarization Eval
  +---- human-annotated speaker turns
  +---- optimal speaker label mapping
  +---- frame-based FA / miss / confusion / DER / JER
  +---- JSON contract + Markdown report
```

## 恢复与幂等

- `Idempotency-Key` 防止重复提交，相同 key 但 payload 不同时拒绝请求。
- worker 使用 `FOR UPDATE SKIP LOCKED` 领取作业，支持水平扩展。
- 制品和 tool run 使用确定性 UUID，同一作业重放不会制造重复记录。
- worker 启动时将超过租约阈值的 `running` 作业重新入队。

## ASR 评测边界

`asr_eval` 与在线 worker 共用 `AsrBackend`，但保持离线运行：输入是经人工审校的 JSONL 清单，输出是 `AsrBenchmarkReport` JSON 与 Markdown。报告不进入在线作业表，避免基准实验污染生产流水线；后续排行榜服务可按报告契约独立入库。

## VAD 后端与校准边界

在线 worker 通过作业级 `VadSpec.backend` 在自适应能量与 WebRTC 后端之间切换，两者输出相同 `VadResult`。`vad_eval` 对人工标注区间执行候选参数扫描，只有覆盖全部样本且无失败的候选才有资格成为最佳候选。报告保持离线，不自动修改生产默认值；阈值升级必须经评审和版本发布。

## 说话人时间轴

`DiarizationBackend` 在原始音频全局时间轴上输出 `SpeakerTurn`。Pyannote Community-1 可优先使用 exclusive diarization，减少与 ASR 区间对齐时的重叠歧义。worker 按重叠时长为每个 VAD segment 选择主 speaker；ASR span 继承所在 segment 的 speaker。原始 turn 时间轴独立写入 S3 和 PostgreSQL，避免 segment 级归属丢失重叠说话信息。

`diarization_eval` 使用人工标注 turn 和生产后端，以固定帧长建立参考/预测 speaker 活跃集合。评测先用最优一对一映射消除任意 speaker 标签差异，再拆分 false alarm、miss 和 confusion。collar、重叠语音处理、exclusive 选择和帧长全部写入报告，避免指标口径漂移。

## ASR 热词边界

热词是逐作业、逐评测样本的解码提示，不进入全局 worker 状态。请求层限制数量、单词长度和合计长度，运行记录保留最终去重后的术语。不同 FunASR checkpoint 对热词的支持不同，因此“成功传参”不代表模型一定消费热词，必须用带/不带热词的同一 ASR 清单做回归比较。

## 下一个架构增量

- Silero VAD 后端与真实业务标注集建设；
- speaker-attributed WER、跨文件 speaker linking 与文本对齐时间戳评测；
- 完整的制品 lineage 查询与处理链版本管理；
- 长作业租约心跳与指数退避重试；
- OpenTelemetry trace 和 Prometheus 指标。
