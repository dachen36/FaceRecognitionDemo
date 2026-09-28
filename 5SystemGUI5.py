# 摄像头视频流+ 静态图像识别 +  GUI 优化... 最好先打开手机APP(让快速连上然后截取视频流)

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

# 强制使用 TCP 传输，解决网络摄像头黑屏问题
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# RTSPURL 连接网络摄像头的配置  用户名(admin) 密码(admin123) IP地址(192.168.0.106)
rtsp_url = "rtsp://admin:admin123@192.168.0.106:554/live/ch00_0"

# 识别阈值 (欧氏距离 < 1.1 视为同一人)
SamePersonThreshold = 1.14


class FaceRecognitionSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI 人脸识别终极版 - 实时/静态/距离数据")
        self.resize(1600, 850)

        # --- 1. 模型初始化 ---
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            self.detector = YOLO("yolov8n-face-lindevs.pt")
            self.recognizer = InceptionResnetV1(pretrained=None).to(self.device)
            self.recognizer.load_state_dict(torch.load("20180402-114759-vggface2.pt", map_location=self.device),
                                            strict=False)
            self.recognizer.eval()
            print(f"✅ 系统就绪，运行设备: {self.device}")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")

        self.db_dir = "database"
        self.known_embeddings, self.known_names = [], []
        self.load_database()

        # --- 2. 界面初始化 ---
        self.init_ui()

        # --- 3. 摄像头初始化 ---
        self.rtsp_url = "rtsp://admin:admin123@192.168.0.106:554/live/ch00_0"
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(5)

    def init_ui(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #F0F2F5; }
            QGroupBox { font-weight: bold; border: 2px solid #D1D9E6; border-radius: 10px; margin-top: 10px; background: white; }
            QLabel { color: #2C3E50; }
            QPushButton { background-color: #3498DB; color: white; border: none; padding: 10px; border-radius: 5px; font-weight: bold; }
            QPushButton:hover { background-color: #2980B9; }
            QLineEdit { border: 1px solid #BDC3C7; padding: 8px; border-radius: 5px; }
        """)

        main_widget = QWidget()
        layout = QHBoxLayout(main_widget)
        self.setCentralWidget(main_widget)

        # --- 左栏：档案录入 ---
        left_panel = QVBoxLayout()
        reg_box = QGroupBox("1. 档案管理")
        reg_layout = QVBoxLayout()
        self.name_input = QLineEdit();
        self.name_input.setPlaceholderText("请输入识别姓名...")
        reg_layout.addWidget(self.name_input)
        btn_reg = QPushButton("📸 录入照片到库")
        btn_reg.clicked.connect(self.register_face)
        reg_layout.addWidget(btn_reg)
        self.db_info = QLabel(f"库中已登记: {len(self.known_names)} 人")
        reg_layout.addWidget(self.db_info)
        reg_layout.addStretch()
        reg_box.setLayout(reg_layout)
        left_panel.addWidget(reg_box)
        layout.addLayout(left_panel, 1)

        # --- 中栏：实时监控 ---
        mid_panel = QVBoxLayout()
        stream_box = QGroupBox("2. 实时视频监控 (RTSP 流)")
        stream_layout = QVBoxLayout()
        self.cam_label = QLabel("正在连接视频源...")
        self.cam_label.setFixedSize(550, 450)
        self.cam_label.setStyleSheet("background-color: #2C3E50; border-radius: 5px;")
        self.cam_label.setAlignment(Qt.AlignCenter)
        stream_layout.addWidget(self.cam_label)
        self.rt_res_label = QLabel("实时识别: 等待中")
        self.rt_res_label.setFont(QFont("微软雅黑", 14, QFont.Bold))
        stream_layout.addWidget(self.rt_res_label)
        self.rt_dist_label = QLabel("距离数值: --")
        stream_layout.addWidget(self.rt_dist_label)
        stream_box.setLayout(stream_layout)
        mid_panel.addWidget(stream_box)
        layout.addLayout(mid_panel, 2)

        # --- 右栏：静态结果 ---
        right_panel = QVBoxLayout()
        static_box = QGroupBox("3. 静态图片识别 (分析对比)")
        static_layout = QVBoxLayout()
        self.static_label = QLabel("未加载图片")
        self.static_label.setFixedSize(550, 450)
        self.static_label.setStyleSheet("border: 2px dashed #BDC3C7; background: #FBFCFC;")
        self.static_label.setAlignment(Qt.AlignCenter)
        static_layout.addWidget(self.static_label)

        btn_upload = QPushButton("📁 选择本地图片识别")
        btn_upload.setStyleSheet("background-color: #27AE60;")
        btn_upload.clicked.connect(self.manual_verify)
        static_layout.addWidget(btn_upload)

        self.st_res_label = QLabel("分析结果: --")
        self.st_res_label.setFont(QFont("微软雅黑", 14, QFont.Bold))
        static_layout.addWidget(self.st_res_label)
        self.st_dist_label = QLabel("距离数值: --")
        static_layout.addWidget(self.st_dist_label)

        static_layout.addStretch()
        static_box.setLayout(static_layout)
        right_panel.addWidget(static_box)
        layout.addLayout(right_panel, 2)

    # --- 逻辑引擎 ---

    def load_database(self):
        if not os.path.exists(self.db_dir): os.makedirs(self.db_dir)
        self.known_embeddings, self.known_names = [], []
        files = [x for x in os.listdir(self.db_dir) if x.lower().endswith(('.jpg', '.png'))]
        for f in files:
            img = cv2.imread(os.path.join(self.db_dir, f))
            emb = self.get_embedding(img)
            if emb is not None:
                self.known_embeddings.append(emb);
                self.known_names.append(os.path.splitext(f)[0])
        if self.known_embeddings: self.known_embeddings = np.array(self.known_embeddings)

    def get_embedding(self, img_bgr):
        if img_bgr is None or img_bgr.size == 0: return None
        try:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(img_rgb).resize((160, 160))
            face_tensor = fixed_image_standardization(torch.tensor(np.array(face_pil)).permute(2, 0, 1).float()).to(
                self.device).unsqueeze(0)
            with torch.no_grad():
                return self.recognizer(face_tensor).cpu().numpy()[0]
        except:
            return None

    def core_recognize(self, frame):
        """核心识别函数：返回标注后的图、姓名、最短距离"""
        results = self.detector(frame, verbose=False, conf=0.5)[0]
        final_name, final_dist = "未知", 0.0

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            face = frame[max(0, y1):y2, max(0, x1):x2]
            emb = self.get_embedding(face)

            if emb is not None and len(self.known_embeddings) > 0:
                # 计算欧氏距离
                dists = np.linalg.norm(self.known_embeddings - emb, axis=1)
                idx = np.argmin(dists)
                final_dist = dists[idx]
                if final_dist < SamePersonThreshold:
                    final_name = self.known_names[idx]

            # 画框和数据
            color = (46, 204, 113) if final_name != "未知" else (231, 76, 60)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{final_name} ({final_dist:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            break  # 仅演示处理一张脸

        return frame, final_name, final_dist

    def update_frame(self):
        """实时视频流处理"""
        if not self.cap.isOpened(): return
        for _ in range(4): self.cap.grab()  # 跳帧
        ret, frame = self.cap.retrieve()
        if ret:
            frame = cv2.resize(frame, (640, 480))
            processed, name, dist = self.core_recognize(frame)
            self.rt_res_label.setText(f"实时识别: {name}")
            self.rt_dist_label.setText(f"距离数值 (Dist): {dist:.4f}")
            self.display_to_label(processed, self.cam_label)

    def manual_verify(self):
        """静态图片处理"""
        path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.jpg *.png)")
        if path:
            img = cv2.imread(path)
            processed, name, dist = self.core_recognize(img)
            self.st_res_label.setText(f"分析结果: {name}")
            self.st_dist_label.setText(f"距离数值 (Dist): {dist:.4f}")
            self.display_to_label(processed, self.static_label)

    def register_face(self):
        name = self.name_input.text().strip()
        if not name: QMessageBox.warning(self, "错误", "请先输入姓名"); return
        path, _ = QFileDialog.getOpenFileName(self, "选择录入照片", "", "Images (*.jpg *.png)")
        if path:
            img = cv2.imread(path)
            cv2.imwrite(os.path.join(self.db_dir, f"{name}.jpg"), img)
            self.load_database()
            self.db_info.setText(f"已录入: {len(self.known_names)} 人")
            QMessageBox.information(self, "库更新", f"{name} 的特征已提取并入库")

    def display_to_label(self, img_bgr, label):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        q_img = QImage(img_rgb.data, img_rgb.shape[1], img_rgb.shape[0], img_rgb.shape[1] * 3, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(q_img).scaled(label.size(), Qt.KeepAspectRatio))

    def closeEvent(self, event):
        self.cap.release();
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaceRecognitionSystem()
    window.show()
    sys.exit(app.exec_())