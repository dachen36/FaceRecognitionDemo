# my_dataset/
# ├── images/          # 存放所有的 JPEG 照片
# │   ├── train/       # 训练用的
# │   └── val/         # 验证用的（至少分 20% 的照片出来）
# └── labels/          # 存放标注文件 (跟图片同名)
#     ├── train/       # 内容：0 0.5 0.5 0.2 0.2 (类别, x, y, w, h)
#     └── val/

from ultralytics import YOLO

def train_from_scratch():
    # 1. 核心操作：加载结构定义(yaml)，而不是加载权重(.pt)
    # 这一步相当于买了一张空的 CPU 电路图，里面还没存任何指令
    model = YOLO("yolov8n.yaml")  # 加载结构

    # 2. 启动炼丹炉
    # 如果你手头没数据，运行这行会报错。
    # 但你可以先看看参数，这些才是“调包侠”进阶成“算法工程师”的关键：
    model.train(
        data="my_data.yaml",    # 数据集配置
        epochs=100,             # 练 100 轮
        imgsz=640,              # 图像分辨率
        batch=16,               # 每次喂 16 张图，1050Ti 建议设 8 或 16
        device=0,               # 你的 1050Ti
        workers=4,              # CPU 搬运工线程数
        lr0=0.01,               # 初始学习率 (炼丹的火候)
        patience=20             # 如果 20 轮都不进步，就自动停（早停机制）
    )

    # 3. 保存你的心血
    # 训练完后，它会自动在 runs/detect/train/weights 下生成 best.pt
    print("训练完成！那是属于你自己的权重。")

if __name__ == "__main__":
    train_from_scratch()