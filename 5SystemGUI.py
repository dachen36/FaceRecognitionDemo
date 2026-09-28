#基础版 文件数据库型... 录入新图片 比对 ... 预留了摄像头 仅facenet对比

import sys
import os
import cv2
import torch
import numpy as np
from PIL import Image
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel,
                             QPushButton, QVBoxLayout, QHBoxLayout, QGroupBox,
                             QLineEdit, QMessageBox, QFileDialog, QFrame)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization

bgcolor="#f5f5f5"
SamePersonThreshold=1.14 # 测试LFW数据集的结论 <此阈值定义为同一个人(理想情况 纯理论值)


class FaceRecognitionSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI 人脸识别系统")
        self.resize(1300, 750)
        # 浅色简约主题样式
        self.setStyleSheet("""
            /* 主窗口：浅灰色背景，不刺眼 */
            QMainWindow { background-color: #F9F9F9; }

            /* 分组框：深灰色文字，淡蓝色边框 */
            QGroupBox { 
                color: #333333; 
                font-weight: bold; 
                border: 1px solid #D1D1D1; 
                margin-top: 15px; 
                padding: 10px; 
                border-radius: 8px; 
            }

            /* 标签：标准黑色文字 */
            QLabel { color: #2D2D2D; }

            /* 按钮：纯白色背景，悬浮时变浅灰 */
            QPushButton { 
                background-color: #FFFFFF; 
                color: #333333; 
                border-radius: 5px; 
                padding: 8px 15px; 
                border: 1px solid #CCCCCC; 
            }
            QPushButton:hover { 
                background-color: #F0F0F0; 
                border: 1px solid #BBBBBB;
            }

            /* 输入框：白色背景，聚焦感强 */
            QLineEdit { 
                background-color: #FFFFFF; 
                color: #333333; 
                border: 1px solid #DCDCDC; 
                padding: 6px; 
                border-radius: 4px;
            }
            QLineEdit:focus {
                border: 1px solid #0078D7; /* 聚焦时显示蓝色边框 */
            }
        """)

        # --- 1. 初始化人脸识别引擎 (大脑) ---
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"📡 正在初始化计算设备: {self.device}")

        try:
            self.recognizer = InceptionResnetV1(pretrained=None).to(self.device)
            self.recognizer.load_state_dict(torch.load("20180402-114759-vggface2.pt", map_location=self.device),
                                            strict=False) # 加载你验证过的权重---FaceNet
            self.recognizer.eval()
            print("✅ FaceNet 加载成功")
        except Exception as e:
            print(f"❌ 模型加载失败，请检查权重文件路径: {e}")

        self.threshold = SamePersonThreshold  # 你跑出的黄金阈值
        self.db_dir = "database"        # 数据库文件夹
        self.known_embeddings = []
        self.known_names = []
        self.load_database()

        # --- 2. 界面初始化 ---
        self.init_ui()

        # --- 3. 摄像头硬件自检 ---
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        if not self.cap.isOpened():
            self.cam_label.setText("🚫 摄像头硬件不可用\n(Hardware Unavailable)\n\n请使用下方“上传图片”模式")
            self.cam_label.setStyleSheet(
                "color: #FF5555; background-color: #000; font-size: 18px; border: 2px dashed #444;")
        else:
            self.timer.timeout.connect(self.update_frame)
            self.timer.start(30)

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        self.setCentralWidget(main_widget)

        # --- 左侧：录入与管理面板 ---
        left_panel = QVBoxLayout()
        reg_box = QGroupBox("人脸档案录入")
        reg_layout = QVBoxLayout()

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("请输入姓名(ID号) 再点击下方上传照片")
        reg_layout.addWidget(self.name_input)

        btn_reg = QPushButton("📤 选择并录入照片")
        btn_reg.setStyleSheet("background-color: #2E7D32;")
        btn_reg.clicked.connect(self.register_face)
        reg_layout.addWidget(btn_reg)

        reg_layout.addStretch()
        reg_box.setLayout(reg_layout)
        left_panel.addWidget(reg_box)
        main_layout.addLayout(left_panel, 1)

        # --- 中间：实时/上传预览区 ---
        mid_panel = QVBoxLayout()
        preview_box = QGroupBox("视觉监控/预览区域")
        preview_layout = QVBoxLayout()

        self.cam_label = QLabel("正在启动...")
        self.cam_label.setFixedSize(640, 480)
        self.cam_label.setAlignment(Qt.AlignCenter)
        self.cam_label.setStyleSheet("background-color: #000; border: 2px solid #333;")
        preview_layout.addWidget(self.cam_label)

        btn_manual = QPushButton("📁 无摄像头模式：上传本地图片识别")
        btn_manual.clicked.connect(self.manual_verify)
        preview_layout.addWidget(btn_manual)

        preview_box.setLayout(preview_layout)
        mid_panel.addWidget(preview_box)
        main_layout.addLayout(mid_panel, 2)

        # --- 右侧：识别状态报告 ---
        right_panel = QVBoxLayout()
        status_box = QGroupBox("识别分析报告")
        status_layout = QVBoxLayout()

        self.status_title = QLabel("系统待机中")
        self.status_title.setFont(QFont("Arial", 22, QFont.Bold))
        self.status_title.setAlignment(Qt.AlignCenter)
        status_layout.addWidget(self.status_title)

        self.dist_label = QLabel("欧氏距离: --")
        self.dist_label.setAlignment(Qt.AlignCenter)
        status_layout.addWidget(self.dist_label)

        self.db_info = QLabel(f"库中已登记: {len(self.known_names)} 人")
        self.db_info.setStyleSheet("color: #888;")
        status_layout.addWidget(self.db_info)

        status_layout.addStretch()
        status_box.setLayout(status_layout)
        right_panel.addWidget(status_box)
        main_layout.addLayout(right_panel, 1)

    # ================= 核心业务逻辑 =================

    def load_database(self):
        """文件系统作为数据库：读取 database 文件夹下所有图片名及向量"""
        if not os.path.exists(self.db_dir): os.makedirs(self.db_dir)
        self.known_embeddings = []
        self.known_names = []

        #最简单的for循环读取 后期再改为向量数据库？？？
        files = [f for f in os.listdir(self.db_dir) if f.lower().endswith(('.jpg', '.png'))]
        for f in files:
            img = cv2.imread(os.path.join(self.db_dir, f))
            emb = self.get_embedding(img)
            if emb is not None:
                self.known_embeddings.append(emb)
                self.known_names.append(os.path.splitext(f)[0])

        if self.known_embeddings:
            self.known_embeddings = np.array(self.known_embeddings)
        print(f"📊 数据库加载完成，已登记人员: {self.known_names}")

    def get_embedding(self, img_bgr):
        """核心处理：BGR -> RGB -> 160x160 -> 标准化 -> 特征向量"""
        if img_bgr is None: return None
        try:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(img_rgb).resize((160, 160))
            face_tensor = fixed_image_standardization(
                torch.tensor(np.array(face_pil)).permute(2, 0, 1).float()
            ).to(self.device).unsqueeze(0)

            with torch.no_grad():
                return self.recognizer(face_tensor).cpu().numpy()[0]
        except Exception as e:
            print(f"提取特征出错: {e}")
            return None

    def register_face(self):
        """录入流程：保存图片到文件夹并更新内存向量库"""
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "提示", "请先输入人员姓名！")
            return

        file_path, _ = QFileDialog.getOpenFileName(self, "选择人脸照片", "", "Images (*.jpg *.png)")
        if file_path:
            img = cv2.imread(file_path)
            # 文件数据库：文件名即 ID
            save_path = os.path.join(self.db_dir, f"{name}.jpg")
            cv2.imwrite(save_path, img)
            self.load_database()  # 重新加载
            self.db_info.setText(f"库中已登记: {len(self.known_names)} 人")
            QMessageBox.information(self, "成功", f"人员【{name}】已成功入库！")

    def manual_verify(self):
        """静态识别：上传一张图片进行对比分析"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择待识别人脸", "", "Images (*.jpg *.png)")
        if file_path:
            img = cv2.imread(file_path)
            self.display_image(img)
            emb = self.get_embedding(img)
            self.compare_and_report(emb)

    def compare_and_report(self, live_emb):
        """比对逻辑并更新 UI 报告"""
        if live_emb is None: return
        if len(self.known_embeddings) == 0:
            self.status_title.setText("数据库空")
            return

        # 计算欧氏距离向量
        dists = np.linalg.norm(self.known_embeddings - live_emb, axis=1)
        idx = np.argmin(dists)
        min_dist = dists[idx]

        self.dist_label.setText(f"欧氏距离: {min_dist:.4f}")

        if min_dist < self.threshold:
            self.status_title.setText(f"✅ {self.known_names[idx]}")
            self.status_title.setStyleSheet("color: #DEFF9A;")
        else:
            self.status_title.setText("❌ 未知人员")
            self.status_title.setStyleSheet("color: #FF5555;")

    def update_frame(self):
        """摄像头刷新逻辑"""
        ret, frame = self.cap.read()
        if ret:
            self.display_image(frame)
            # 为了性能，此处可以添加逻辑：每隔 15 帧自动调用一次 self.compare_and_report()

    def display_image(self, img_bgr):
        """将 OpenCV 图片渲染到 QLabel"""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        q_img = QImage(img_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.cam_label.setPixmap(QPixmap.fromImage(q_img).scaled(self.cam_label.size(), Qt.KeepAspectRatio))

    def closeEvent(self, event):
        if self.cap.isOpened(): self.cap.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaceRecognitionSystem()
    window.show()
    sys.exit(app.exec_())