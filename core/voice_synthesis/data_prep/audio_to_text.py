import os
import whisper
import pandas as pd
from tqdm import tqdm  # 显示处理进度
from utils.path_utils import get_project_root  # 复用之前的路径工具


def init_asr_model(model_size="base", model_dir=None):
    """
    初始化多语言ASR模型（Whisper）
    :param model_size: 模型大小（tiny/base/small，零基础选base）
    :param model_dir: 模型本地保存路径（None则用默认路径）
    :return: 加载好的Whisper模型
    """
    print(f"正在加载Whisper {model_size}模型（多语言支持）...")
    # 加载模型（model_dir指定本地路径，避免重复下载）
    model = whisper.load_model(
        name=model_size,
        download_root=model_dir,  # 模型保存到项目内，如core/voice_synthesis/models/asr/
        device="cpu"  # 若有GPU，改为"cuda"（需安装PyTorch GPU版）
    )
    print(f"模型加载完成，保存路径：{model_dir if model_dir else '默认缓存路径'}")
    return model


def audio_to_text_single(model, audio_path, lang_label=None):
    """
    单音频文件转文本（支持自动识别语言或手动指定）
    :param model: Whisper模型
    :param audio_path: 音频文件路径（预处理后的干净WAV，如character01_jp_01_clean.wav）
    :param lang_label: 手动指定语言（jp/en/cn，None则自动识别）
    :return: 识别出的文本（str）
    """
    try:
        # 语言映射（Whisper的语言代码：日语=ja，英语=en，中文=zh）
        lang_map = {"jp": "ja", "en": "en", "cn": "zh"}
        whisper_lang = lang_map[lang_label] if lang_label else None

        # 运行ASR识别（transcribe=转录文本）
        result = model.transcribe(
            audio=audio_path,
            language=whisper_lang,  # 手动指定语言（提高准确率）
            fp16=False,  # CPU运行设为False，GPU设为True
            verbose=False  # 关闭详细日志（安静模式）
        )

        # 提取识别结果（result["text"]是完整文本）
        text = result["text"].strip()
        # 简单后处理（去除无意义的开头符号，如“。”“,”）
        text = text.lstrip("。,，、")
        print(f"音频{os.path.basename(audio_path)}识别完成，文本：{text}")
        return text
    except Exception as e:
        print(f"音频{os.path.basename(audio_path)}识别失败：{str(e)}")
        return ""  # 识别失败返回空字符串，后续人工处理


def batch_audio_to_text(model, metadata_path, output_metadata_path=None):
    """
    批量处理：读取metadata.csv中的音频，生成文本并更新metadata
    :param model: Whisper模型
    :param metadata_path: 原始metadata路径（含音频路径、声线、语言，无文本）
    :param output_metadata_path: 输出新metadata路径（含文本）
    :return: 新的metadata DataFrame
    """
    # 读取原始metadata（之前生成的，格式：音频路径,声线,语言）
    df = pd.read_csv(metadata_path, header=None, names=["audio_path", "speaker_id", "lang_label"])
    # 检查是否已有文本列（避免重复处理）
    if "text" in df.columns:
        print("metadata已包含文本列，跳过处理")
        return df

    # 批量识别每个音频的文本
    texts = []
    print(f"开始批量识别{len(df)}个音频的文本...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        audio_path = row["audio_path"]
        lang_label = row["lang_label"]  # 从metadata获取语言标签（jp/en/cn）
        # 识别文本
        text = audio_to_text_single(model, audio_path, lang_label=lang_label)
        texts.append(text)

    # 新增“text”列到metadata
    df["text"] = texts
    # 保存新的metadata（含文本，用于后续TTS训练）
    if output_metadata_path is None:
        # 默认覆盖原始metadata（或改为新文件名，如metadata_with_text.csv）
        output_metadata_path = metadata_path.replace(".csv", "_with_text.csv")
    df.to_csv(output_metadata_path, index=False, header=False)  # 不保留表头，符合之前格式
    print(f"批量处理完成！新metadata保存到：{output_metadata_path}")
    return df


def main():
    # 1. 项目路径配置（复用工具函数，无需改）
    project_root = get_project_root()
    # ASR模型保存路径（放到项目内，符合私有化管理）
    asr_model_dir = os.path.join(project_root, "core", "voice_synthesis", "models", "asr")
    os.makedirs(asr_model_dir, exist_ok=True)  # 自动创建文件夹

    # 2. 初始化ASR模型（零基础选base，GPU可选small）
    model = init_asr_model(
        model_size="base",  # 模型大小：tiny/base/small
        model_dir=asr_model_dir  # 模型保存到项目内
    )

    # 3. 批量处理音频→生成文本（用之前预处理后的metadata）
    # 原始metadata路径（预处理后生成的，含音频路径、声线、语言）
    metadata_path = os.path.join(project_root, "data", "input", "audio", "metadata.csv")
    # 运行批量识别
    batch_audio_to_text(
        model=model,
        metadata_path=metadata_path,
        output_metadata_path=None  # 自动生成带文本的metadata
    )

    # 4. 输出识别统计（提示用户人工校对）
    df = pd.read_csv(metadata_path.replace(".csv", "_with_text.csv"), header=None,
                     names=["audio_path", "speaker_id", "lang_label", "text"])
    # 统计识别成功/失败数量
    success_count = len(df[df["text"] != ""])
    fail_count = len(df[df["text"] == ""])
    print(f"\n识别统计：成功{success_count}个，失败{fail_count}个")
    if fail_count > 0:
        # 输出失败的音频路径，提示用户检查
        fail_audio = df[df["text"] == ""]["audio_path"].tolist()
        print("识别失败的音频：")
        for audio in fail_audio:
            print(f"  - {audio}")
        print("建议：检查音频是否损坏，或更换更大的ASR模型（如small）重试")
    # 关键提示：人工校对
    print(
        "\n⚠️  重要：请打开新生成的metadata_with_text.csv，人工校对所有文本（尤其是识别结果较短/语义不通的），避免错误文本影响TTS训练！")


if __name__ == "__main__":
    main()