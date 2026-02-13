import os
import cv2
import torch
import numpy as np
from decord import VideoReader, cpu, gpu
from insightface.app import FaceAnalysis
from asd_pro.models.syncnet import SyncNet_color
from asd_pro import audio
import time
import subprocess
from scipy.spatial.distance import cosine

class AdvancedASD:
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'

        # 1. Initialize Face Analysis (Detection + Re-ID)
        # SCRFD detector is very fast
        self.face_app = FaceAnalysis(name='buffel', root='./weights/insightface', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.face_app.prepare(ctx_id=0 if self.device == 'cuda' else -1, det_size=(640, 640))

        # 2. Initialize SyncNet (Wav2Lip Expert)
        self.sync_model = SyncNet_color().to(self.device)
        self.load_sync_weights('./weights/lipsync_expert.pth')
        self.sync_model.eval()

    def load_sync_weights(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"SyncNet weights not found at {path}")
        checkpoint = torch.load(path, map_location=self.device)
        s = checkpoint["state_dict"]
        new_s = {}
        for k, v in s.items():
            new_s[k.replace('module.', '')] = v
        self.sync_model.load_state_dict(new_s)
        print(f"Loaded SyncNet weights from {path}")

    def get_mel(self, audio_path):
        wav = audio.load_wav(audio_path, 16000)
        mel = audio.melspectrogram(wav)
        return mel

    def crop_mouth(self, frame, face_kps):
        # face_kps contains landmarks. indices 3 and 4 are mouth corners in some models,
        # but InsightFace (buffel) returns 5 points: left eye, right eye, nose, left mouth, right mouth.
        # We use points 3 and 4 for mouth.
        lm = face_kps
        ml, mr = lm[3], lm[4]
        center_mouth = (ml + mr) / 2
        # Crop size based on distance between mouth corners
        dist = np.linalg.norm(ml - mr)
        size = int(dist * 2.5) # Heuristic

        y1, y2 = int(center_mouth[1] - size//2), int(center_mouth[1] + size//2)
        x1, x2 = int(center_mouth[0] - size//2), int(center_mouth[0] + size//2)

        # Boundary check
        h, w = frame.shape[:2]
        y1, y2 = max(0, y1), min(h, y2)
        x1, x2 = max(0, x1), min(w, x2)

        mouth_crop = frame[y1:y2, x1:x2]
        if mouth_crop.size == 0:
            return None
        return cv2.resize(mouth_crop, (96, 96))

    def process_video(self, video_path, speaker_a_template=None):
        # Extract audio first
        audio_path = "temp_audio.wav"
        subprocess.call(f"ffmpeg -y -i {video_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_path}", shell=True)

        mel = self.get_mel(audio_path)
        vr = VideoReader(video_path, ctx=cpu(0)) # Use GPU if possible, but CPU is safer for memory
        fps = vr.get_avg_fps()

        tracks = {} # face_id -> {frames: [], bboxes: [], embeddings: []}

        print(f"Processing {len(vr)} frames...")

        # Speaker A identification logic
        # We'll collect embeddings and find the most frequent person as Speaker A if no template provided
        all_embeddings = []

        step = 5 # Process every 5 frames for speed in detection
        for i in range(0, len(vr), step):
            frame = vr[i].asnumpy()
            faces = self.face_app.get(frame)

            for face in faces:
                # Store for Re-ID
                all_embeddings.append(face.normed_embedding)

        # Simple clustering or just similarity to template
        if speaker_a_template is not None:
            # target = speaker_a_template
            pass
        else:
            # Heuristic: Speaker A is the one most similar to the average of all detected faces
            # (In a fixed camera live stream, the main speaker occupies most frames)
            avg_emb = np.mean(all_embeddings, axis=0)
            target_emb = avg_emb # Placeholder

        # Real ASD Loop
        results = []
        batch_size = 32

        # Wav2Lip expert looks at 5 frames window
        for i in range(0, len(vr) - 5, batch_size):
            frames_batch = vr.get_batch(range(i, min(i + batch_size + 5, len(vr)))).asnumpy()

            # For each frame in the batch i to i+batch_size
            for j in range(min(batch_size, len(vr) - 5 - i)):
                global_idx = i + j

                # Get central frame for detection
                center_frame = frames_batch[j + 2]
                faces = self.face_app.get(center_frame)

                # Find Speaker A amongst faces
                best_face = None
                max_sim = -1
                for face in faces:
                    sim = 1 - cosine(face.normed_embedding, target_emb)
                    if sim > max_sim:
                        max_sim = sim
                        best_face = face

                if best_face and max_sim > 0.6: # Threshold for Speaker A
                    # Extract mouth crops for the 5-frame window
                    window_crops = []
                    for k in range(5):
                        crop = self.crop_mouth(frames_batch[j + k], best_face.kps)
                        if crop is None: break
                        window_crops.append(crop)

                    if len(window_crops) == 5:
                        # Prepare Visual Input (B, 15, 96, 96) - 5 frames stacked in channels
                        # RGB -> 3 channels * 5 frames = 15 channels
                        v_input = np.concatenate(window_crops, axis=-1) # (96, 96, 15)
                        v_input = np.transpose(v_input, (2, 0, 1)) # (15, 96, 96)
                        v_tensor = torch.from_numpy(v_input).float().unsqueeze(0).to(self.device) / 255.0

                        # Prepare Audio Input
                        # Each video frame @ 25fps is 40ms.
                        # Mel frames are usually 10ms hop. So 4 mel frames per video frame.
                        # 5 video frames = 20 mel frames.
                        mel_idx = int(global_idx * (16000 / fps) / 80) # Heuristic for mel alignment
                        mel_window = mel[:, mel_idx : mel_idx + 20]
                        if mel_window.shape[1] == 20:
                            a_tensor = torch.from_numpy(mel_window).float().unsqueeze(0).unsqueeze(0).to(self.device)

                            with torch.no_grad():
                                a_emb, v_emb = self.sync_model(a_tensor, v_tensor)
                                dist = torch.norm(a_emb - v_emb, p=2).item()
                                results.append({'frame': global_idx, 'dist': dist, 'active': dist < 1.0})

        return results

if __name__ == "__main__":
    # This is a template script. Actual execution requires specific weights paths.
    # User can run this on their 3080ti.
    # asd = AdvancedASD()
    # res = asd.process_video("input.mp4")
    # print(res)
    pass
