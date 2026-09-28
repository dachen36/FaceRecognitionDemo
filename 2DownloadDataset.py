import os
import requests
from sklearn.datasets import fetch_lfw_people, get_data_home


def download_lfw_robust(url, destination):
    """使用 requests 实现带进度的稳健下载"""
    try:
        # 设置 stream=True 避免一次性加载到内存
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0

        print(f"开始下载数据集至: {destination}")
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024 * 128):  # 128KB 块大小
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    if total_size > 0:
                        done = int(50 * downloaded_size / total_size)
                        percent = (downloaded_size / total_size) * 100
                        print(
                            f"\r进度: [{'=' * done}{' ' * (50 - done)}] {percent:.1f}% ({downloaded_size / 1024 / 1024:.2f}MB)",
                            end='')
        print("\n下载成功！")
    except Exception as e:
        print(f"\n下载过程中出错: {e}")
        if os.path.exists(destination):
            os.remove(destination)  # 下载失败删除残余文件以免后续解压报错
        return False
    return True


# --- 1. 配置路径与URL ---
# 官方数据下载地址
LFW_URL = "https://ndownloader.figshare.com/files/5976015"
# 获取 sklearn 默认的数据集存储目录
lfw_home = os.path.join(get_data_home(), 'lfw_home')
if not os.path.exists(lfw_home):
    os.makedirs(lfw_home)

archive_path = os.path.join(lfw_home, 'lfw-funneled.tgz')

# --- 2. 执行下载逻辑 ---
if not os.path.exists(archive_path):
    print("未在本地找到数据集，准备从官网下载...")
    success = download_lfw_robust(LFW_URL, archive_path)
    if not success:
        print("下载失败，请检查网络连接或尝试手动下载后放入路径。")
        exit()
else:
    print(f"检测到本地已存在数据集: {archive_path}")

# --- 3. 加载数据集 ---
print("正在解压并加载数据集（此步可能需要一两分钟）...")
try:
    # fetch_lfw_people 会自动检测已存在的 tgz 文件并解压
    lfw_data = fetch_lfw_people(min_faces_per_person=20, resize=None)

    print("\n--- 数据集概况 ---")
    print(f"图片总量: {lfw_data.images.shape[0]} 张")
    print(f"人物类别: {len(lfw_data.target_names)} 类")
    print(f"图片尺寸: {lfw_data.images.shape[1]}x{lfw_data.images.shape[2]}")
except Exception as e:
    print(f"加载失败，可能是文件损坏，建议删除 {archive_path} 后重试。")
    print(f"错误详情: {e}")