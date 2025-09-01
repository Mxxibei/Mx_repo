# generate_voice.py：用训练好的声线模型生成声音
import os
import yaml
import torch
import soundfile as sf
from utils.path_utils import get_project_root
from core.voice_synthesis.models.vits_model import VITS

def main():
    project_root = get_project_root()
    config_path = os.path.join(project_root, "config", "model_config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)["voice_synthesis"]["model"]

    # 加载多声线模型
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    model = VITS(num_speakers=3, num_languages=3)  # 3个声线，3种语言
    model_path = os.path.join(project_root, "core", "voice_synthesis", "models", "multi_speaker_vits.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.set_device(device)
    print(f"已加载多声线模型：{model_path}")

    # ---------------------- 测试不同声线+语言组合 ----------------------
    test_cases = [
        ("01", "jp", "01声线日语"),  # 01声线+日语
        ("02", "en", "02声线英语"),  # 02声线+英语
        ("03", "cn", "03声线中文")   # 03声线+中文
    ]

    output_dir = os.path.join(project_root, "data", "output", "audio", "generated_multi")
    os.makedirs(output_dir, exist_ok=True)

    for speaker_label, lang_label, desc in test_cases:
        print(f"正在生成：{desc}")
        audio = model.generate_voice(speaker_label=speaker_label, language_label=lang_label, length=3*16000)
        # 保存生成的音频
        output_path = os.path.join(output_dir, f"{desc}_generated.wav")
        sf.write(output_path, audio, samplerate=16000)
        print(f"{desc}生成完成，保存到：{output_path}\n")

if __name__ == "__main__":
    main()

