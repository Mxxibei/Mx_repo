# path_utils.py：获取项目根目录的工具
import os

def get_project_root():
    """获取项目根目录（ai-personal-companion）"""
    # 从当前文件（path_utils.py）向上找，直到找到“config”文件夹（根目录有config）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while not os.path.exists(os.path.join(current_dir, "config")):
        current_dir = os.path.dirname(current_dir)
    return current_dir