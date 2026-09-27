# ADR-0005：可插拔逐段 ASR 与全局时间轴

状态：已接受

## 决定

ASR 不直接绑定到单一模型。worker 依赖 `AsrBackend` 协议，首批提供 Faster-Whisper 与 FunASR 适配器；模型依赖延迟导入，未启用 ASR backend 时不增加启动成本。

转写流程：

1. 使用已有 VAD 切片进行逐段推理；
2. 将模型返回的切片内相对时间戳平移到原始音频毫秒时间轴；
3. 回填 `AudioSegment.transcript` 和 `language`；
4. 生成 `TranscriptDocument`，写入 PostgreSQL 并上传 S3/MinIO JSON 制品；
5. 在 `AudioReport` 中记录全文、语种、时间戳数量和制品 URI。

## 原因

- Whisper 和 FunASR 的显存、语种和时间戳特性不同，底座应稳定输出而非锁定实现；
- 逐段转写可复用 VAD 制品，并为重试、并行和说话人分离保留边界；
- 确定性 transcript ID 与 upsert 保证 worker 重放幂等。

## 限制

当前自动测试使用可控假后端，不下载模型权重。每个候选模型仍需在目标 GPU/CPU 环境中测量 WER/CER、RTF、显存、语种识别和时间戳偏差。

## 参考

- [Faster-Whisper 官方文档](https://github.com/SYSTRAN/faster-whisper)
- [FunASR Python API](https://github.com/modelscope/FunASR/blob/main/docs/python_api.md)
