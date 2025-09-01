import os
from utils.path_utils import get_project_root


def main():
    # 音频文件夹路径
    audio_dir = os.path.join(get_project_root(), "data", "input", "audio")
    # metadata保存路径
    metadata_path = os.path.join(audio_dir, "metadata.csv")

    with open(metadata_path, "w", encoding="utf-8") as f:
        # 遍历所有mp3文件
        for mp3_file in os.listdir(audio_dir):
            if not mp3_file.endswith(".mp3"):
                continue
            # 解析文件名：character01_jp_01.mp3 → 01, jp
            parts = mp3_file.replace(".mp3", "").split("_")  # ["character01", "jp", "01"]
            speaker_id = parts[0].replace("character", "")  # "01"
            lang = parts[1]  # "jp"
            # 写一行到metadata
            audio_path = os.path.join(audio_dir, mp3_file)  # 相对路径
            f.write(f"{audio_path},{speaker_id},{lang}\n")

    print(f"metadata生成完成，保存到：{metadata_path}")


if __name__ == "__main__":
    main()