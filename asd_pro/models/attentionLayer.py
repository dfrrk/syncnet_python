import torch
import torch.nn as nn
import torch.nn.functional as F

class attentionLayer(nn.Module):
    """
    【注意力机制层】 (Transformer-style Attention)

    该模块是 TalkNet 能够超越传统 SyncNet 的关键原因。
    它支持：
    1. 跨模态交互 (Cross-Attention): 学习音频信号与视觉嘴部动作的细微对齐关系。
    2. 时序自注意力 (Self-Attention): 分析数秒内的长时说话证据，有效排除短时背景音乐、唱歌伴奏或突发噪声的干扰。
    """
    def __init__(self, d_model, nhead, dropout=0.1):
        super(attentionLayer, self).__init__()
        # 使用 PyTorch 原生的多头注意力机制，实现并行化高效计算
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # 前馈神经网络 (FFN): 进一步增强特征的非线性表达能力
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)

        # 层归一化 (Layer Normalization): 保证超长视频推理过程中的数值稳定性
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, tar):
        """
        核心前向传播逻辑。
        src: 源特征序列 (Batch, Seq, Dim)
        tar: 目标特征序列
        """
        # 注意：nn.MultiheadAttention 期望输入维度为 (Seq_Len, Batch, Dim)
        # 我们在此进行转置处理，确保模型内部逻辑正确
        src = src.transpose(0, 1)
        tar = tar.transpose(0, 1)

        # 1. 运行多头注意力计算，并使用残差连接 (Skip Connection)
        src2 = self.self_attn(src, tar, tar)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # 2. 运行前馈网络，进行第二次残差连接
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        # 恢复维度顺序为 (Batch, Seq_Len, Dim) 并返回结果
        return src.transpose(0, 1)
