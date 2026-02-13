import os
import cv2
import torch
import numpy as np
from decord import VideoReader, cpu
from insightface.app import FaceAnalysis
try:
    from asd_pro.models.syncnet import SyncNet_color
    from asd_pro import audio
except ImportError:
    from asd_pro.models.syncnet import SyncNet_color
    from asd_pro import audio
import time
import subprocess
from scipy.spatial.distance import cosine
from tqdm import tqdm
from scipy.ndimage import median_filter

class AdvancedModernASD:
    """
    针对 1-5 小时长视频优化的现代主动说话人检测 (ASD) 系统。
    集成 InsightFace (用于高精度检测与主讲人锁定) 与 Wav2Lip SyncNet (同步专家模型)。
    """
    def __init__(self, device='cuda'):
        self.device = 'cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        print(f"初始化系统 | 推理设备: {self.device}")

        # 1. 初始化人脸分析模型 (使用 buffalo_l 模型，包含检测和 512D 特征提取)
        # 模型会自动下载至 ~/.insightface/models/
        self.face_app = FaceAnalysis(name='buffalo_l', root='.', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.face_app.prepare(ctx_id=0 if self.device == 'cuda' else -1, det_size=(640, 640))

        # 2. 初始化同步专家模型 (Wav2Lip SyncNet)
        self.sync_model = SyncNet_color().to(self.device)
        self.load_sync_weights('weights/lipsync_expert.pth')
        self.sync_model.eval()

    def load_sync_weights(self, path):
        if not os.path.exists(path):
            print(f"错误: 未找到权重文件 {path}。请确保已下载 Wav2Lip 专家模型权重。")
            return
        checkpoint = torch.load(path, map_location=self.device)
        s = checkpoint["state_dict"]
        new_s = {}
        for k, v in s.items():
            new_s[k.replace('module.', '')] = v
        self.sync_model.load_state_dict(new_s)
        print(f"成功加载同步判别器权重")

    def get_mel(self, audio_path):
        wav = audio.load_wav(audio_path, 16000)
        mel = audio.melspectrogram(wav)
        return mel

    def crop_mouth(self, frame, kps):
        """
        根据人脸 5 个关键点 (EyeL, EyeR, Nose, MouthL, MouthR) 裁剪嘴部。
        MouthL 和 MouthR 的索引通常是 3 和 4。
        """
        ml, mr = kps[3], kps[4]
        center_mouth = (ml + mr) / 2
        dist = np.linalg.norm(ml - mr)
        # 动态扩大裁剪区域以包含完整嘴唇
        size = int(dist * 2.5)

        y1, y2 = int(center_mouth[1] - size//2), int(center_mouth[1] + size//2)
        x1, x2 = int(center_mouth[0] - size//2), int(center_mouth[0] + size//2)

        h, w = frame.shape[:2]
        y1, y2 = max(0, y1), min(h, y2)
        x1, x2 = max(0, x1), min(w, x2)

        mouth_crop = frame[y1:y2, x1:x2]
        if mouth_crop.size == 0: return None
        return cv2.resize(mouth_crop, (96, 96))

    def run(self, video_path, output_csv="final_asd_results.csv"):
        print(f"正在处理视频: {video_path}")

        # 1. 音频预处理
        audio_path = "temp_audio_pro.wav"
        print("正在提取并重采样音频...")
        subprocess.call(f"ffmpeg -y -i {video_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_path}",
                        shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        mel = self.get_mel(audio_path)

        # 2. 视频流读取 (Decord 高效读取)
        vr = VideoReader(video_path, ctx=cpu(0))
        fps = vr.get_avg_fps()
        total_frames = len(vr)

        # 3. 锁定主讲人 A (基于出现频率和特征相似度)
        print("正在分析视频以锁定主讲人 A...")
        embeddings = []
        # 采样前 2 分钟的视频帧进行身份建模
        sample_end = min(total_frames, int(fps * 120))
        for i in range(0, sample_end, int(fps * 2)):
            frame = vr[i].asnumpy()
            faces = self.face_app.get(frame)
            for f in faces:
                embeddings.append(f.normed_embedding)

        if not embeddings:
            print("未能检测到任何人脸，请检查视频内容。")
            return

        # 设定主讲人 A 的基准特征（取均值）
        target_emb = np.mean(embeddings, axis=0)

        # 4. 逐帧检测与同步推理
        results = []
        batch_size = 64 # 显存充足时可调大

        print(f"开始深度推理 (共 {total_frames} 帧)...")
        for i in tqdm(range(0, total_frames - 5, batch_size)):
            # 加载当前批次及其上下文（5帧窗口）
            batch_indices = range(i, min(i + batch_size + 5, total_frames))
            frames_batch = vr.get_batch(batch_indices).asnumpy()

            v_inputs, a_inputs, frame_meta = [], [], []

            for j in range(min(batch_size, total_frames - 5 - i)):
                global_idx = i + j
                center_frame = frames_batch[j + 2]

                # 在中心帧检测人脸并匹配主讲人 A
                faces = self.face_app.get(center_frame)
                best_face = None
                max_sim = -1
                for f in faces:
                    sim = 1 - cosine(f.normed_embedding, target_emb)
                    if sim > max_sim:
                        max_sim = sim
                        best_face = f

                # 如果匹配到主讲人 A 且置信度达标
                if best_face and max_sim > 0.6:
                    window_crops = []
                    for k in range(5):
                        crop = self.crop_mouth(frames_batch[j + k], best_face.kps)
                        if crop is not None: window_crops.append(crop)

                    if len(window_crops) == 5:
                        v_inp = np.concatenate(window_crops, axis=-1)
                        v_inp = np.transpose(v_inp, (2, 0, 1))

                        mel_start = int(global_idx * 4)
                        a_inp = mel[:, mel_start : mel_start + 20]

                        if a_inp.shape[1] == 20:
                            v_inputs.append(v_inp)
                            a_inputs.append(np.expand_dims(a_inp, 0))
                            frame_meta.append(global_idx)

            # 批量执行推理
            if v_inputs:
                v_tensor = torch.from_numpy(np.stack(v_inputs)).float().to(self.device) / 255.0
                a_tensor = torch.from_numpy(np.stack(a_inputs)).float().to(self.device)

                with torch.no_grad():
                    a_emb, v_emb = self.sync_model(a_tensor, v_tensor)
                    # 计算音视频特征向量的 L2 距离
                    dists = torch.norm(a_emb - v_emb, p=2, dim=1).cpu().numpy()
                    for idx, d in zip(frame_meta, dists):
                        results.append((idx, d))

        # 5. 结果聚合与平滑
        results.sort()
        indices = [r[0] for r in results]
        raw_dists = [r[1] for r in results]
        # 中值滤波去除瞬时噪声
        smooth_dists = median_filter(raw_dists, size=15)

        # 6. 保存与统计
        with open(output_csv, 'w') as f:
            f.write("frame,timestamp_ms,dist,speaking\n")
            for idx, d, sd in zip(indices, raw_dists, smooth_dists):
                ts = int(idx * 1000 / fps)
                is_speaking = 1 if sd < 1.1 else 0
                f.write(f"{idx},{ts},{d:.4f},{is_speaking}\n")

        print(f"\n任务完成。详细结果已保存至: {output_csv}")
        self.print_summary(indices, smooth_dists, fps)

    def print_summary(self, indices, dists, fps):
        print("\n" + "="*50)
        print("          主讲人 A 说话/唱歌时间标记统计")
        print("="*50)
        start = None
        count = 0
        for i in range(len(indices)):
            speaking = dists[i] < 1.1
            if speaking and start is None:
                start = indices[i]
            elif not speaking and start is not None:
                duration = (indices[i] - start) / fps
                if duration > 0.4:
                    print(f"段落 {count+1:03d} | {start/fps:8.2f}s ---> {indices[i]/fps:8.2f}s | 时长: {duration:5.2f}s")
                    count += 1
                start = None
        print("="*50)
        print(f"共识别到 {count} 段说话内容。")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="输入视频路径")
    args = parser.parse_args()

    # 启动高级 ASD 系统
    system = AdvancedModernASD()
    system.run(args.video)
