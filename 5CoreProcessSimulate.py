import torch
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization

# ================= 配置区 =================
# 确保这两个文件就在你的 py 文件旁边
YOLO_MODEL_PATH = "yolov8n-face-lindevs.pt"
FACENET_WEIGHT_PATH = "20180402-114759-vggface2.pt"

# 设置 1050Ti 动力源
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 正在使用设备: {device}")

# ================= 核心模型加载 =================
# 1. 加载 YOLOv8-face
detector = YOLO(YOLO_MODEL_PATH)

# 2. 加载 FaceNet
# 修改点：加上 classify=True 匹配你下载的权重格式
# ================= 核心模型加载 =================
detector = YOLO(YOLO_MODEL_PATH)

# 1. 坚持用最简单的结构（不分类），不给它报错的机会
recognizer = InceptionResnetV1(pretrained=None).to(device)

try:
    # 2. 手动加载权重
    state_dict = torch.load(FACENET_WEIGHT_PATH, map_location=device)

    # 3. 关键动作：把权重字典里多出来的 "logits.weight" 和 "logits.bias" 给删掉
    # 这样它们就不会跟我们的模型“打架”了
    state_dict.pop('logits.weight', None)
    state_dict.pop('logits.bias', None)

    # 4. 现在加载就完全匹配了
    recognizer.load_state_dict(state_dict)
    recognizer.eval()  # 切换到预测模式

    print("✅ 完美！多余权重已剔除，特征提取器就绪。")
except Exception as e:
    print(f"❌ 加载还是卡住了: {e}")


# ================= 功能函数 =================
def get_face_embedding(image_path):
    """从一张大图中自动抠脸并返回特征向量"""
    # 1. YOLO 检测
    results = detector(image_path, verbose=False)[0]
    if len(results.boxes) == 0:
        print(f"⚠️ 警告: 在 {image_path} 中未检测到人脸")
        return None

    # 2. 获取置信度最高的脸的坐标
    img = cv2.imread(image_path)
    box = results.boxes.xyxy[0].cpu().numpy()
    x1, y1, x2, y2 = map(int, box)

    # 3. 实时抠图并处理格式
    face = img[y1:y2, x1:x2]
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    # 4. 预处理 (FaceNet 标准要求: 160x160)
    face_pil = Image.fromarray(face_rgb).resize((160, 160))
    face_tensor = fixed_image_standardization(
        torch.tensor(np.array(face_pil)).permute(2, 0, 1).float()
    ).to(device).unsqueeze(0)

    # 5. 提取 512 维特征向量
    with torch.no_grad():
        embedding = recognizer(face_tensor).cpu().numpy()[0]
    return embedding


# ================= 实战运行 =================
if __name__ == "__main__":
    # 替换为你自己的 JPEG 照片路径
    img1 = "CD.jpg"
    #img2 = "CDwGlasses.jpg"
    #img2="E:\PythonFaceRecog\CDStandard.jpeg"
    img2="E:\PythonFaceRecog\lfw\lfw_home\lfw_funneled\Aaron_Pena\Aaron_Pena_0001.jpg"

    print("\n--- 开始提取特征 ---")
    v1 = get_face_embedding(img1)
    v2 = get_face_embedding(img2)

    if v1 is not None and v2 is not None:
        # 计算欧氏距离
        dist = np.linalg.norm(v1 - v2)
        print(f"\n📊 最终比对距离: {dist:.4f}")

        # 判定标准 (通常 0.6 为分界线)
        if dist < 0.6:
            print("✨ 判定结果：【同一个人】")
        else:
            print("👥 判定结果：【不是同一个人】")

# 包装一下 不用搞2个.pt权重
# import numpy as np
#
#
# class FaceEngine:
#     def __init__(self, detector, recognizer):
#         self.detector = detector
#         self.recognizer = recognizer
#         self.threshold = 0.8  # 既然 0.6 太严，咱们手动放宽到 0.8
#
#     def compare(self, img1_path, img2_path):
#         v1 = self.get_vector(img1_path)
#         v2 = self.get_vector(img2_path)
#
#         if v1 is None or v2 is None:
#             return "检测不到人脸", 999
#
#         dist = np.linalg.norm(v1 - v2)
#
#         # 这里的逻辑就是你刚才发现的规律
#         is_same = dist < self.threshold
#         return is_same, dist
#
# # 使用方法：
# # engine = FaceEngine(yolo_model, facenet_model)
# # result, score = engine.compare("me1.jpg", "me2.jpg")