# 乔安网络摄像头实时监测 基本版 比较卡
import sys
import os
import cv2
import torch
import numpy as np
from PIL import Image
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel,
                             QPushButton, QVBoxLayout, QHBoxLayout, QGroupBox,
                             QLineEdit, QMessageBox, QFileDialog)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
from ultralytics import YOLO

# 强制使用 TCP 传输，防止 RTSP 丢包黑屏 (必须在所有 cv2 调用前执行)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

SamePersonThreshold = 1.14


class FaceRecognitionSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI 人脸识别系统 - 乔安网络流版")
        self.resize(1300, 750)

        # --- 样式配置 ---
        self.setStyleSheet("""
            QMainWindow { background-color: #F9F9F9; }
            QGroupBox { 
                color: #333333; font-weight: bold; 
                border: 1px solid #D1D1D1; margin-top: 15px; 
                padding: 10px; border-radius: 8px; 
            }
            QPushButton { 
                background-color: #FFFFFF; color: #333333; 
                border-radius: 5px; padding: 8px 15px; border: 1px solid #CCCCCC; 
            }
            QPushButton:hover { background-color: #F0F0F0; }
            QLineEdit { background-color: #FFFFFF; padding: 6px; border-radius: 4px; border: 1px solid #DCDCDC; }
        """)

        # --- 1. 初始化人脸识别引擎 ---
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"📡 正在初始化设备: {self.device}")

        # 加载模型
        try:
            self.detector = YOLO("yolov8n-face-lindevs.pt")
            self.recognizer = InceptionResnetV1(pretrained=None).to(self.device)
            self.recognizer.load_state_dict(torch.load("20180402-114759-vggface2.pt", map_location=self.device),
                                            strict=False)
            self.recognizer.eval()
            print("✅ 模型全部加载成功")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")

        self.threshold = SamePersonThreshold
        self.db_dir = "database"
        self.known_embeddings = []
        self.known_names = []
        self.load_database()

        # --- 2. 界面初始化 ---
        self.init_ui()

        # --- 3. 网络摄像头初始化 ---
        self.ip = "192.168.0.106"
        self.password = "admin123"
        self.username = "admin"
        # 刚才你测试通的路径
        self.rtsp_url = f"rtsp://{self.username}:{self.password}@{self.ip}:554/live/ch00_0"

        # 使用 CAP_FFMPEG 明确后端
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 极低缓冲区

        self.timer = QTimer()
        if not self.cap.isOpened():
            self.cam_label.setText("🚫 无法连接网络摄像头\n请检查IP和密码")
            self.cam_label.setStyleSheet("color: #FF5555; background-color: #000;")
        else:
            self.timer.timeout.connect(self.update_frame)
            self.timer.start(1)  # 高频触发，配合内部跳帧逻辑

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        self.setCentralWidget(main_widget)

        # 左侧管理
        left_panel = QVBoxLayout()
        reg_box = QGroupBox("人脸档案录入")
        reg_layout = QVBoxLayout()
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("输入姓名/ID")
        reg_layout.addWidget(self.name_input)
        btn_reg = QPushButton("📤 录入照片")
        btn_reg.clicked.connect(self.register_face)
        reg_layout.addWidget(btn_reg)
        reg_layout.addStretch()
        reg_box.setLayout(reg_layout)
        left_panel.addWidget(reg_box)
        main_layout.addLayout(left_panel, 1)

        # 中间预览
        mid_panel = QVBoxLayout()
        preview_box = QGroupBox("实时监控")
        preview_layout = QVBoxLayout()
        self.cam_label = QLabel("正在拉取视频流...")
        self.cam_label.setFixedSize(640, 480)
        self.cam_label.setAlignment(Qt.AlignCenter)
        self.cam_label.setStyleSheet("background-color: #000;")
        preview_layout.addWidget(self.cam_label)
        preview_box.setLayout(preview_layout)
        mid_panel.addWidget(preview_box)
        main_layout.addLayout(mid_panel, 2)

        # 右侧报告
        right_panel = QVBoxLayout()
        status_box = QGroupBox("识别报告")
        status_layout = QVBoxLayout()
        self.status_title = QLabel("待机中")
        self.status_title.setFont(QFont("Arial", 20, QFont.Bold))
        status_layout.addWidget(self.status_title)
        self.dist_label = QLabel("距离: --")
        status_layout.addWidget(self.dist_label)
        self.db_info = QLabel(f"库中人数: {len(self.known_names)}")
        status_layout.addWidget(self.db_info)
        status_layout.addStretch()
        status_box.setLayout(status_layout)
        right_panel.addWidget(status_box)
        main_layout.addLayout(right_panel, 1)

    def load_database(self):
        if not os.path.exists(self.db_dir): os.makedirs(self.db_dir)
        self.known_embeddings = []
        self.known_names = []
        files = [f for f in os.listdir(self.db_dir) if f.lower().endswith(('.jpg', '.png'))]
        for f in files:
            img = cv2.imread(os.path.join(self.db_dir, f))
            emb = self.get_embedding(img)
            if emb is not None:
                self.known_embeddings.append(emb)
                self.known_names.append(os.path.splitext(f)[0])
        if self.known_embeddings:
            self.known_embeddings = np.array(self.known_embeddings)

    def get_embedding(self, img_bgr):
        if img_bgr is None or img_bgr.size == 0: return None
        try:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(img_rgb).resize((160, 160))
            face_tensor = fixed_image_standardization(
                torch.tensor(np.array(face_pil)).permute(2, 0, 1).float()
            ).to(self.device).unsqueeze(0)
            with torch.no_grad():
                return self.recognizer(face_tensor).cpu().numpy()[0]
        except:
            return None

    def update_frame(self):
        # 核心技巧：丢弃旧帧，获取最新帧
        if not self.cap.isOpened(): return

        # 循环读取缓冲区直到最后一帧
        for _ in range(5):
            self.cap.grab()
        ret, frame = self.cap.retrieve()

        if ret:
            # 1. 运行 YOLO 检测
            results = self.detector(frame, verbose=False, conf=0.5)[0]
            face_crop = None

            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                # 画红色识别框
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # 抠出人脸
                face_crop = frame[max(0, y1):y2, max(0, x1):x2]
                break  # 演示版仅处理第一个检测到的人脸

            # 2. 运行识别比对
            if face_crop is not None and face_crop.size > 0:
                emb = self.get_embedding(face_crop)
                self.compare_and_report(emb)
            else:
                self.status_title.setText("未检出")
                self.status_title.setStyleSheet("color: #888;")

            # 3. 渲染画面
            self.display_image(frame)

    def compare_and_report(self, live_emb):
        if live_emb is None or len(self.known_embeddings) == 0: return
        dists = np.linalg.norm(self.known_embeddings - live_emb, axis=1)
        idx = np.argmin(dists)
        min_dist = dists[idx]
        self.dist_label.setText(f"距离: {min_dist:.4f}")
        if min_dist < self.threshold:
            self.status_title.setText(f"✅ {self.known_names[idx]}")
            self.status_title.setStyleSheet("color: #2E7D32;")
        else:
            self.status_title.setText("❌ 未知人员")
            self.status_title.setStyleSheet("color: #D32F2F;")

    def display_image(self, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        q_img = QImage(img_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.cam_label.setPixmap(QPixmap.fromImage(q_img).scaled(self.cam_label.size(), Qt.KeepAspectRatio))

    def register_face(self):
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "提示", "先输入姓名")
            return
        file_path, _ = QFileDialog.getOpenFileName(self, "选照片", "", "Images (*.jpg *.png)")
        if file_path:
            img = cv2.imread(file_path)
            cv2.imwrite(os.path.join(self.db_dir, f"{name}.jpg"), img)
            self.load_database()
            self.db_info.setText(f"库中人数: {len(self.known_names)}")
            QMessageBox.information(self, "成功", f"【{name}】已入库")

    def closeEvent(self, event):
        if self.cap.isOpened(): self.cap.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaceRecognitionSystem()
    window.show()
    sys.exit(app.exec_())