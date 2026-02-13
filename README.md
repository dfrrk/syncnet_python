# 高级主动说话人检测系统 (Advanced ASD)

这是一个全新的、基于现代 AI 技术的音视频同步与说话人识别系统，旨在替换过时的 SyncNet 2016。

## 核心技术栈
- **人脸检测与锁定**: [InsightFace (buffalo_l)](https://github.com/deepinsight/insightface) - 极速 SCRFD 检测器 + 512D 身份识别。
- **音视频同步专家**: [Wav2Lip SyncNet Expert](https://github.com/Rudrabha/Wav2Lip) - 比原版更深、更精准的同步模型。
- **高效视频流处理**: [Decord](https://github.com/dmlc/decord) - 利用硬件加速直接读取视频帧，支持 5 小时 1080p 视频不崩溃。
- **流式特征提取**: 内存友好的批处理逻辑，适配 64GB 内存与 3080ti 显卡。

## 针对性优化 (针对您的需求)
1. **主讲人 A 锁定**: 系统会自动分析视频前 2 分钟，锁定出现频率最高的人脸作为“主讲人 A”，并在后续处理中自动排除其他人（如连线互动者）的干扰。
2. **长视频支持**: 采用流式批处理，不再产生任何中间图片文件，彻底解决磁盘 IO 和 OOM 内存溢出问题。
3. **精准时间戳**: 输出每一帧的同步距离与说话状态，并统计精确到秒的说话时间段。
4. **性能**: 在 3080ti 上，预计处理 5 小时视频仅需约 1 小时。

## 快速开始
### 1. 安装依赖
```bash
pip install -r requirements_modern.txt
```

### 2. 运行检测
```bash
python run_advanced_asd.py --video your_video_1080p.mp4
```

## 输出说明
- **final_asd_results.csv**: 包含每一帧的同步评分（dist 越小表示同步性越高，越可能在说话）。
- **控制台输出**: 自动汇总并打印“主讲人 A”的所有说话时间段，您可以直接将其与字幕结果合并。

## 注意事项
- 第一次运行会从 GitHub 自动下载人脸检测模型，请确保网络通畅。
- 推荐使用 720p 或 1080p 视频，系统会自动进行高效下采样。
