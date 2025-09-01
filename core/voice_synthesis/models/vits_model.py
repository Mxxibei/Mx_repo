# vits_model.py：简化版VITS模型（适合零基础微调，不用改）
import torch
import torch.nn as nn
import torch.nn.functional as F

class VITS(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=1024, hop_length=256):
        super().__init__()
        # 简化的VITS结构（仅保留微调需要的部分）
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length

        # 1. 音频特征提取（Mel谱图）
        self.mel_layer = nn.Sequential(
            nn.Conv1d(1, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # 2. 声线编码（学习目标声线特征）
        self.speaker_encoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)  # 声线特征维度
        )

        # 3. 解码器（生成声线）
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1 * hop_length)  # 输出音频片段
        )

        # 损失函数（衡量模型输出和真实音频的差距）
        self.loss_fn = nn.MSELoss()

    def forward(self, audio):
        # 前向传播：输入音频→提取特征→编码声线→解码→计算损失
        # audio: (batch_size, 1, time_steps)
        batch_size = audio.size(0)

        # 1. 提取Mel谱图特征
        mel = self.mel_layer(audio)  # (batch_size, 512, time_steps//hop_length)
        mel = mel.mean(dim=2)  # 全局平均池化，得到特征向量 (batch_size, 512)

        # 2. 编码声线特征
        speaker_feat = self.speaker_encoder(mel)  # (batch_size, 128)

        # 3. 解码生成音频片段
        output = self.decoder(speaker_feat)  # (batch_size, hop_length)
        # 扩展维度，和输入音频格式匹配（用于计算损失）
        output = output.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, hop_length)

        # 4. 计算损失（用输入音频的前hop_length个样本作为目标）
        target = audio[:, :, :self.hop_length]  # (batch_size, 1, hop_length)
        target = target.unsqueeze(2)  # (batch_size, 1, 1, hop_length)
        loss = self.loss_fn(output, target)

        return loss

    def generate_voice(self, speaker_feat, length=16000):
        # 生成声线：输入声线特征→输出音频
        self.eval()  # 切换到评估模式
        with torch.no_grad():
            # 生成音频片段（按长度循环）
            num_frames = length // self.hop_length
            audio_frames = []
            for _ in range(num_frames):
                frame = self.decoder(speaker_feat)  # (1, hop_length)
                audio_frames.append(frame)
            audio = torch.cat(audio_frames, dim=1)  # (1, length)
        self.train()  # 切回训练模式
        return audio.squeeze(0).cpu().numpy()