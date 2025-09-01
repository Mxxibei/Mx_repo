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
    # 加载配置
    config = load_config()
    # 1. 定义各种路径（不用改，配置文件里会设）
    project_root = get_project_root()
    input_audio_dir = os.path.join(project_root, "data", "input", "audio")  # 原始音频文件夹
    segment_file = os.path.join(input_audio_dir, "valid_segment01.txt")  # 有效段标注文件
    # 输出路径：预处理后的音频放在data/output/audio/processed/
    output_dir = os.path.join(project_root, "data", "output", "audio", "processed")
    os.makedirs(output_dir, exist_ok=True)  # 自动创建文件夹（如果没有）

    # 2. 解析有效段
    print("开始解析有效段标注文件...")
    segments_dict = parse_valid_segments(segment_file)
    print(f"解析完成，共找到{len(segments_dict)}个音频的有效段")

    # 3. 处理每个mp3音频
    # 遍历input_audio_dir里的所有mp3文件
    for mp3_file in os.listdir(input_audio_dir):
        if not mp3_file.endswith(".mp3"):
            continue  # 只处理mp3文件
        # 解析mp3文件名：如character01_01.mp3→声线编号01，样本序号01
        mp3_name = mp3_file.replace(".mp3", "")  # character01_01
        audio_num = mp3_name.split("_")[1]  # 取“01”“02”（对应audio 01、audio 02）

        # 检查该音频是否有有效段标注
        if audio_num not in segments_dict:
            print(f"警告：{mp3_file}没有对应的有效段标注，跳过处理")
            continue

        # 4. 步骤1：mp3转wav
        mp3_path = os.path.join(input_audio_dir, mp3_file)
        temp_wav_path = os.path.join(output_dir, f"{mp3_name}_temp.wav")
        mp3_to_wav(mp3_path, temp_wav_path)

        # 5. 步骤2：按有效段剪切
        valid_segments = segments_dict[audio_num]
        cut_wav_paths = cut_audio_by_segments(temp_wav_path, valid_segments, output_dir)

        # 6. 步骤3：对每个剪切片段去噪
        for cut_wav_path in cut_wav_paths:
            # 去噪后的文件名：如character01_01_cut_1_clean.wav
            clean_wav_name = os.path.basename(cut_wav_path).replace(".wav", "_clean.wav")
            clean_wav_path = os.path.join(output_dir, clean_wav_name)
            remove_noise(cut_wav_path, clean_wav_path)

        # 删除临时wav文件（转格式用的，没用了）
        os.remove(temp_wav_path)
        print(f"===== {mp3_file} 预处理完成 =====\n")

    print("所有音频预处理完成！干净的有效音频在：", output_dir)


# 运行脚本（只有当直接运行这个文件时才执行main函数）
if __name__ == "__main__":
    main()