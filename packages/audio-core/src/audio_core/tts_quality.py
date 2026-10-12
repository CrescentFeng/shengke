from pathlib import Path

from .probe import probe_file
from .quality import QualityAnalysisSpec, analyze_quality


def inspect_reference(path: Path, *, ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe"):
    probe = probe_file(path, ffprobe=ffprobe)
    duration = probe.duration_seconds or 0
    if not 3.0 <= duration <= 10.0:
        raise ValueError(f"参考音频应为 3–10 秒，当前为 {duration:.3f} 秒；请截取完整短句")
    metrics = analyze_quality(
        path,
        duration_seconds=duration,
        spec=QualityAnalysisSpec(minimum_silence_seconds=0.2),
        ffmpeg=ffmpeg,
    )
    if metrics.rms_dbfs is None or metrics.rms_dbfs < -60 or metrics.silence_ratio >= 0.98:
        raise ValueError("参考音频全静音或电平过低，请更换清晰人声")
    warnings = []
    if metrics.clipping_ratio > 0.0001:
        warnings.append("检测到削波，建议更换无失真的参考音频")
    if metrics.silence_ratio > 0.4:
        warnings.append("静音比例较高，建议减少首尾静音")
    if probe.channels > 1:
        warnings.append("原始音频为多声道；推理时将创建单声道副本")
    return probe, metrics, warnings
