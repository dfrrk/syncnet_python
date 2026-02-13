#!/usr/bin/python
#-*- coding: utf-8 -*-
# 视频帧率 25 FPS, 音频采样率 16000HZ

import torch
import numpy
import time, pdb, argparse, subprocess, os, math, glob
import cv2
import python_speech_features

from scipy import signal
from scipy.io import wavfile
from SyncNetModel import *
from shutil import rmtree


# ==================== 计算特征间距离 (计算时间偏移) ====================

def calc_pdist(feat1, feat2, vshift=10):
    """
    计算音视频特征之间的成对距离，用于寻找最佳对齐偏移量。
    feat1: 视觉特征
    feat2: 音频特征
    vshift: 搜索的时间窗半径（前后漂移量）
    """
    win_size = vshift * 2 + 1

    # 对音频特征进行填充，以便在滑动窗口内计算距离
    feat2p = torch.nn.functional.pad(feat2, (0, 0, vshift, vshift))

    dists = []

    for i in range(0, len(feat1)):
        # 在窗口 win_size 内计算当前视觉帧与音频对应段的欧氏距离
        dists.append(torch.nn.functional.pairwise_distance(feat1[[i], :].repeat(win_size, 1), feat2p[i:i + win_size, :]))

    return dists

# ==================== SyncNet 核心实例类 ====================

class SyncNetInstance(torch.nn.Module):

    def __init__(self, dropout = 0, num_layers_in_fc_layers = 1024):
        super(SyncNetInstance, self).__init__();

        # 初始化核心网络结构 S
        self.__S__ = S(num_layers_in_fc_layers = num_layers_in_fc_layers)

        # 检查 GPU 是否可用并进行迁移
        if torch.cuda.is_available():
            self.__S__ = self.__S__.cuda()

    def evaluate(self, opt, videofile):
        """
        评估视频文件的音视频同步情况。
        优化点：采用流式分批提取特征，避免长视频下内存溢出（OOM）。
        """
        self.__S__.eval();

        # ========== 准备临时目录 ==========
        if os.path.exists(os.path.join(opt.tmp_dir, opt.reference)):
          rmtree(os.path.join(opt.tmp_dir, opt.reference))

        os.makedirs(os.path.join(opt.tmp_dir, opt.reference))

        # ========== 提取音频 ==========
        # 使用 ffmpeg 提取单声道、16000Hz 的 wav 音频文件
        audio_out = os.path.join(opt.tmp_dir, opt.reference, 'audio.wav')
        command = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 %s" % (videofile, audio_out))
        subprocess.call(command, shell=True, stdout=None)
        
        # ========== 加载音频并提取 MFCC 特征 ==========
        sample_rate, audio = wavfile.read(audio_out)
        mfcc = zip(*python_speech_features.mfcc(audio, sample_rate))
        mfcc = numpy.stack([numpy.array(i) for i in mfcc])
        cc = numpy.expand_dims(numpy.expand_dims(mfcc, axis=0), axis=0)
        cct = torch.from_numpy(cc.astype(float)).float()

        # ========== 加载视频 (流式批处理优化) ==========
        cap = cv2.VideoCapture(videofile)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 检查音视频输入长度匹配情况
        if (float(len(audio)) / 16000) != (float(num_frames) / 25) :
            print("警告: 音频 (%.4fs) 和视频 (%.4fs) 长度不匹配。" % (float(len(audio)) / 16000, float(num_frames) / 25))

        min_length = min(num_frames, math.floor(len(audio) / 640))
        lastframe = min_length - 5

        im_feat = []
        cc_feat = []

        # 视频读取缓存，仅保留当前批次所需的帧
        video_buffer = []
        frames_read_ptr = 0

        tS = time.time()
        for i in range(0, lastframe, opt.batch_size):

            # --- 1. 确保缓存中有足够的帧用于当前批次 ---
            # 每个样本需要 5 帧，批次结束位置为 min(lastframe, i + opt.batch_size)
            # 所以需要读取到第 (batch_end - 1) + 4 帧
            batch_end = min(lastframe, i + opt.batch_size)
            needed_frames_until = batch_end + 4
            
            while frames_read_ptr < needed_frames_until:
                ret, frame = cap.read()
                if not ret: break
                video_buffer.append(frame)
                frames_read_ptr += 1

            # --- 2. 构建批次张量 ---
            im_batch = []
            for vframe in range(i, batch_end):
                # 获取相对于当前缓存起始位置的索引
                # 注意：video_buffer 始终从全局帧索引 i 开始（在下文清理）
                idx = vframe - i
                sample = numpy.stack(video_buffer[idx : idx+5], axis=0) # (5, H, W, 3)
                sample = numpy.transpose(sample, (3, 0, 1, 2)) # (3, 5, H, W)
                im_batch.append(sample)

            if not im_batch: break

            # --- 3. 视觉特征提取 ---
            im_in = torch.from_numpy(numpy.stack(im_batch, axis=0)).float()
            if torch.cuda.is_available():
                im_out = self.__S__.forward_lip(im_in.cuda())
            else:
                im_out = self.__S__.forward_lip(im_in)
            im_feat.append(im_out.data.cpu())

            # --- 4. 对应音频特征提取 ---
            cc_batch = [ cct[:, :, :, vframe*4 : vframe*4+20] for vframe in range(i, batch_end) ]
            cc_in = torch.cat(cc_batch, 0)
            if torch.cuda.is_available():
                cc_out = self.__S__.forward_aud(cc_in.cuda())
            else:
                cc_out = self.__S__.forward_aud(cc_in)
            cc_feat.append(cc_out.data.cpu())

            # --- 5. 清理缓存，释放已处理帧的内存 ---
            # 删除当前批次起始位置之前的帧，因为滑动窗口不会再用到它们
            # 下一个批次 i 会增加 opt.batch_size，所以我们保留最后 4 帧（作为重叠部分）
            # 或者简单地，根据 i 的增量来清理
            video_buffer = video_buffer[opt.batch_size:]

        cap.release()
        im_feat = torch.cat(im_feat, 0)
        cc_feat = torch.cat(cc_feat, 0)

        # ========== 计算偏移量 (Offset) ==========
        print('特征提取耗时 %.3f 秒.' % (time.time() - tS))

        dists = calc_pdist(im_feat, cc_feat, vshift=opt.vshift)
        mdist = torch.mean(torch.stack(dists, 1), 1)

        minval, minidx = torch.min(mdist, 0)
        offset = opt.vshift - minidx
        conf   = torch.median(mdist) - minval

        fdist   = numpy.stack([dist[minidx].numpy() for dist in dists])
        fconf   = torch.median(mdist).numpy() - fdist
        fconfm  = signal.medfilt(fconf, kernel_size=9)
        
        numpy.set_printoptions(formatter={'float': '{: 0.3f}'.format})
        print('逐帧置信度: ')
        print(fconfm)
        print('音视频偏移 (AV offset): \t%d \n最小距离: \t%.3f\n置信度 (Confidence): \t%.3f' % (offset, minval, conf))

        dists_npy = numpy.array([ dist.numpy() for dist in dists ])
        return offset.numpy(), conf.numpy(), dists_npy

    def extract_feature(self, opt, videofile):
        """
        仅提取视频的唇部特征向量（流式处理优化版本）。
        """
        self.__S__.eval();
        
        cap = cv2.VideoCapture(videofile)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        lastframe = num_frames - 4
        
        im_feat = []
        video_buffer = []
        frames_read_ptr = 0

        tS = time.time()
        for i in range(0, lastframe, opt.batch_size):

            batch_end = min(lastframe, i + opt.batch_size)
            needed_frames_until = batch_end + 4
            
            while frames_read_ptr < needed_frames_until:
                ret, frame = cap.read()
                if not ret: break
                video_buffer.append(frame)
                frames_read_ptr += 1

            im_batch = []
            for vframe in range(i, batch_end):
                idx = vframe - i
                sample = numpy.stack(video_buffer[idx : idx+5], axis=0)
                sample = numpy.transpose(sample, (3, 0, 1, 2))
                im_batch.append(sample)

            if not im_batch: break

            im_in = torch.from_numpy(numpy.stack(im_batch, axis=0)).float()
            if torch.cuda.is_available():
                im_out = self.__S__.forward_lipfeat(im_in.cuda())
            else:
                im_out = self.__S__.forward_lipfeat(im_in)
            im_feat.append(im_out.data.cpu())

            video_buffer = video_buffer[opt.batch_size:]

        cap.release()
        im_feat = torch.cat(im_feat, 0)
        print('特征提取耗时 %.3f 秒.' % (time.time() - tS))

        return im_feat


    def loadParameters(self, path):
        """
        加载预训练模型参数。
        """
        loaded_state = torch.load(path, map_location=lambda storage, loc: storage);
        self_state = self.__S__.state_dict();
        for name, param in loaded_state.items():
            self_state[name].copy_(param);
