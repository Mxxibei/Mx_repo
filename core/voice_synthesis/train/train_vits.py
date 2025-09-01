# train_vits.py：VITS模型微调声线的脚本
import os
import yaml
import torch
import numpy as np
from tqdm import tqdm  # 显示训练进度条
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
from utils.path_utils import get_project_root
from core.voice_synthesis.models.vits_model import VITS  # 后面会补VITS模型定义

# 1. 加载配置
def load_config():
    config_path = os.path.join(get_project_root(), "config", "model_config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config["voice_synthesis"]["model"]

# 2. 定义多声线-多语言数据集（替换原VoiceDataset）
class MultiVoiceDataset(Dataset):
    def __init__(self, metadata_path, sample_rate=16000):
        self.sample_rate = sample_rate
        self.metadata = self.load_metadata(metadata_path)  # 加载metadata
        # 声线和语言的映射（和模型一致）
        self.speaker_id_map = {"01":0, "02":1, "03":2}
        self.language_id_map = {"jp":0, "en":1, "cn":2}

    def load_metadata(self, metadata_path):
        """加载metadata.csv，返回列表：[(音频路径, 声线编号, 语言标签), ...]"""
        metadata = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                audio_path, speaker_label, lang_label = line.split(",")
                # 检查路径是否存在
                if not os.path.exists(audio_path):
                    print(f"警告：{audio_path}不存在，跳过")
                    continue
                metadata.append( (audio_path, speaker_label, lang_label) )
        if len(metadata) == 0:
            raise ValueError(f"metadata中没有有效音频数据：{metadata_path}")
        print(f"加载完成：共{len(metadata)}个样本（含{len(set([x[1] for x in metadata]))}个声线，{len(set([x[2] for x in metadata]))}种语言）")
        return metadata

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # 读取单个样本的“音频路径、声线编号、语言标签”
        audio_path, speaker_label, lang_label = self.metadata[idx]
        # 读取音频文件
        audio, sr = sf.read(audio_path)
        if sr != self.sample_rate:
            raise ValueError(f"音频{audio_path}采样率{sr}≠{self.sample_rate}")
        # 转成torch张量
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0)  # (1, time_steps)
        # 转声线/语言标签为数字ID
        speaker_id = torch.tensor(self.speaker_id_map[speaker_label], dtype=torch.long)
        language_id = torch.tensor(self.language_id_map[lang_label], dtype=torch.long)
        # 返回：音频数据 + 声线ID + 语言ID
        return audio_tensor, speaker_id, language_id

# 3. 初始化模型（加载预训练权重）
def init_model(config):
    # 确定训练设备（cpu或cuda）
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    if config["device"] == "cuda" and not torch.cuda.is_available():
        print("警告：没有找到GPU，自动切换到CPU训练（会比较慢）")
        device = torch.device("cpu")
    print(f"使用设备训练：{device}")

    # 加载VITS模型（后面补模型定义）
    model = VITS().to(device)
    # 加载预训练权重
    pretrained_path = os.path.join(get_project_root(), config["pretrained_model_path"])
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"预训练模型没找到：{pretrained_path}")
    model.load_state_dict(torch.load(pretrained_path, map_location=device), strict=False)
    print(f"已加载预训练模型：{pretrained_path}")

    # 设置模型为“训练模式”
    model.train()
    return model, device

# 4. 训练函数
def train(model, dataloader, device, config):
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    epochs = config["train_epochs"]
    save_dir = os.path.join(get_project_root(), "core", "voice_synthesis", "models")
    os.makedirs(save_dir, exist_ok=True)
    # 多声线模型保存名（区分单声线）
    save_path = os.path.join(save_dir, "multi_speaker_vits.pth")

    print(f"开始多声线-多语言训练，共{epochs}次（epoch），每次{config['batch_size']}个样本")
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        for batch in pbar:
            # ---------------------- 修改：批量数据包含“音频+声线ID+语言ID” ----------------------
            audio, speaker_id, language_id = batch
            # 把数据放到设备上（CPU/GPU）
            audio = audio.to(device)
            speaker_id = speaker_id.to(device)
            language_id = language_id.to(device)

            # 清空梯度→模型计算→反向传播→更新参数
            optimizer.zero_grad()
            # 调用模型forward时，传入audio、speaker_id、language_id
            loss = model(audio, speaker_id, language_id)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * audio.size(0)
            pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(dataloader.dataset)
        print(f"Epoch {epoch} 完成，平均损失：{avg_loss:.6f}")

        # 每10次保存中间模型
        if epoch % 10 == 0:
            mid_save_path = save_path.replace(".pth", f"_epoch{epoch}.pth")
            torch.save(model.state_dict(), mid_save_path)
            print(f"已保存中间模型：{mid_save_path}")

    # 保存最终多声线模型
    torch.save(model.state_dict(), save_path)
    print(f"===== 多声线训练完成！模型保存到：{save_path} =====")
    return save_path

# 主函数
def main():
    # 加载配置
    config = load_config()
    project_root = get_project_root()

    # ---------------------- 修改：加载metadata和多声线数据集 ----------------------
    metadata_path = os.path.join(project_root, "data", "input", "audio", "metadata.csv")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata文件不存在：{metadata_path}（请先运行generate_metadata.py）")
    # 加载预处理后的干净音频对应的metadata？不，直接用原始metadata，音频路径指向预处理后的文件！
    # （关键：预处理后的音频路径需要和metadata中的路径对应，后面预处理脚本会改）
    dataset = MultiVoiceDataset(metadata_path, sample_rate=16000)

    # 创建数据加载器（批量喂给模型）
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,  # 打乱样本顺序（让模型同时学不同声线/语言）
        num_workers=0  # 零基础用0（避免多线程报错）
    )

    # 初始化模型（新增set_device）
    model, device = init_model(config)
    model.set_device(device)  # 给模型设置设备（CPU/GPU）

    # 开始训练
    train(model, dataloader, device, config)
if __name__ == "__main__":
    main()