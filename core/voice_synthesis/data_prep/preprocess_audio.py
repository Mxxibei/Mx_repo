# 导入需要的工具库（之前安装的依赖）
import os
import yaml
import wave
import pydub
from pydub import AudioSegment
import noisereduce as nr
import numpy as np
from utils.path_utils import get_project_root  # 项目路径工具（已在utils里）


# 1. 读取配置文件（获取音频路径、格式等参数）
def load_config():
    # 找到config/model_config.yaml文件的路径
    config_path = os.path.join(get_project_root(), "config", "model_config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config["voice_synthesis"]["data_prep"]  # 只取声音合成的预处理配置


# 2. 解析valid_segment01.txt，获取每个音频的有效段
def parse_valid_segments(segment_file_path):
    segments = {}  # 用字典存：音频编号→有效段列表，如{"01": ["17:24-17:26", ...]}
    current_audio = None
    with open(segment_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()  # 去掉换行符和空格
            if not line:
                continue  # 跳过空行
            # 识别“audio XX”行（如audio 01）
            if line.startswith("audio"):
                current_audio = line.split()[1]  # 取“01”“02”这样的编号
                segments[current_audio] = []
            else:
                # 识别有效段（如17：24-17：26），注意原文件可能有中文冒号，先替换成英文冒号
                line = line.replace("：", ":")
                if "-" in line and current_audio:
                    segments[current_audio].append(line)
    return segments


# 3. 把mp3转成wav（去噪库只支持wav格式）
def mp3_to_wav(mp3_path, wav_path):
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format="wav", parameters=["-ar", "16000"])  # 转成16000Hz（VITS模型要求）
    print(f"已将{mp3_path}转成wav，保存到{wav_path}")


# 4. 按有效段剪切wav音频
def cut_audio_by_segments(wav_path, segments, output_dir):
    audio = AudioSegment.from_wav(wav_path)
    audio_name = os.path.basename(wav_path).replace(".wav", "")  # 取音频名（如character01_01）
    cut_audios = []

    for i, segment in enumerate(segments):
        # 解析时间：如“17:24-17:26”→开始时间17分24秒，结束时间17分26秒
        start_str, end_str = segment.split("-")
        start_min, start_sec = map(int, start_str.split(":"))
        end_min, end_sec = map(int, end_str.split(":"))
        # 转成毫秒（AudioSegment用毫秒计算）
        start_ms = (start_min * 60 + start_sec) * 1000
        end_ms = (end_min * 60 + end_sec) * 1000
        # 剪切片段
        cut_audio = audio[start_ms:end_ms]
        # 保存剪切后的片段
        cut_path = os.path.join(output_dir, f"{audio_name}_cut_{i + 1}.wav")
        cut_audio.export(cut_path, format="wav")
        cut_audios.append(cut_path)
        print(f"已剪切有效段{segment}，保存到{cut_path}")
    return cut_audios


# 5. 去除音频背景噪音
def remove_noise(wav_path, output_path):
    # 读取wav文件
    with wave.open(wav_path, "rb") as wf:
        params = wf.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        audio_data = wf.readframes(nframes)
        # 转成numpy数组（去噪库需要）
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0  # 归一化

    # 提取噪音样本（取音频开头0.5秒，默认是噪音）
    noise_sample = audio_np[:int(framerate * 0.5)]  # 0.5秒的噪音
    # 去噪
    reduced_noise = nr.reduce_noise(y=audio_np, y_noise=noise_sample, sr=framerate)
    # 转回int16格式（wav文件要求）
    reduced_noise_int16 = (reduced_noise * 32768.0).astype(np.int16)

    # 保存去噪后的音频
    with wave.open(output_path, "wb") as wf:
        wf.setparams(params)
        wf.writeframes(reduced_noise_int16.tobytes())
    print(f"已去噪，保存到{output_path}")
    return output_path


# 主函数：整合所有步骤
def main():
    config = load_config()
    project_root = get_project_root()
    input_audio_dir = os.path.join(project_root, "data", "input", "audio")
    # ---------------------- 修改：标注文件用新的multi版本 ----------------------
    segment_file = os.path.join(input_audio_dir, "valid_segment_multi.txt")
    # ---------------------- 修改：预处理后的音频路径（和原始音频同名，加_clean） ----------------------
    output_dir = os.path.join(project_root, "data", "output", "audio", "processed")
    os.makedirs(output_dir, exist_ok=True)

    # 解析有效段（修改解析逻辑，支持audio 01_jp格式）
    print("开始解析多声线有效段标注文件...")
    segments_dict = parse_valid_segments(segment_file)  # 原parse函数不用改，会自动识别“01_jp”作为key
    print(f"解析完成，共找到{len(segments_dict)}个声线-语言组合的有效段")

    # 处理每个mp3音频
    for mp3_file in os.listdir(input_audio_dir):
        if not mp3_file.endswith(".mp3"):
            continue
        # 解析文件名：character01_jp_01.mp3 → 声线-语言组合（01_jp）
        mp3_name = mp3_file.replace(".mp3", "")  # character01_jp_01
        speaker_lang = "_".join(mp3_name.split("_")[1:3])  # 取“01_jp”（从第2个_到第3个_）
        # 检查该声线-语言组合是否有有效段
        if speaker_lang not in segments_dict:
            print(f"警告：{mp3_file}对应的{speaker_lang}没有有效段标注，跳过")
            continue

        # 后续步骤（mp3转wav、剪切、去噪）不变，但保存路径改到output_dir，文件名加_clean
        mp3_path = os.path.join(input_audio_dir, mp3_file)
        # 预处理后的wav文件名：character01_jp_01_clean.wav
        clean_wav_name = f"{mp3_name}_clean.wav"
        clean_wav_path = os.path.join(output_dir, clean_wav_name)

        # 步骤1：mp3转wav（临时文件）
        temp_wav_path = os.path.join(output_dir, f"{mp3_name}_temp.wav")
        mp3_to_wav(mp3_path, temp_wav_path)

        # 步骤2：按有效段剪切（取第一个有效段即可，或循环剪切所有段）
        valid_segments = segments_dict[speaker_lang]
        # 简化：取第一个有效段（如果要多段，可循环处理）
        first_segment = valid_segments[0]
        start_str, end_str = first_segment.split("-")
        start_str = start_str.replace("：", ":")
        end_str = end_str.replace("：", ":")
        start_min, start_sec = map(int, start_str.split(":"))
        end_min, end_sec = map(int, end_str.split(":"))
        start_ms = (start_min * 60 + start_sec) * 1000
        end_ms = (end_min * 60 + end_sec) * 1000
        audio = AudioSegment.from_wav(temp_wav_path)
        cut_audio = audio[start_ms:end_ms]

        # 步骤3：去噪并保存
        remove_noise(cut_audio.export(temp_wav_path, format="wav"), clean_wav_path)

        # 更新metadata中的音频路径（关键！把原始路径改成预处理后的路径）
        metadata_path = os.path.join(input_audio_dir, "metadata.csv")
        with open(metadata_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        with open(metadata_path, "w", encoding="utf-8") as f:
            for line in lines:
                if mp3_path in line:
                    # 替换原始路径为预处理后的路径
                    new_line = line.replace(mp3_path, clean_wav_path)
                    f.write(new_line)
                else:
                    f.write(line)

        os.remove(temp_wav_path)
        print(f"===== {mp3_file} 预处理完成，干净音频：{clean_wav_path} =====")

    print("所有多声线音频预处理完成！")

# 运行脚本（只有当直接运行这个文件时才执行main函数）
if __name__ == "__main__":
    main()