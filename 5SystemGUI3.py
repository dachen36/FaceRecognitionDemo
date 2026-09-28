# 距离标在框上+ 英文界面 纯静态图片检测

import sys
import os
import cv2
import torch
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel,
                             QPushButton, QVBoxLayout, QHBoxLayout, QGroupBox,
                             QLineEdit, QMessageBox, QFileDialog)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
from ultralytics import YOLO

# Threshold for FaceNet (Euclidean distance)
SamePersonThreshold = 1.14


class FaceRecognitionSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Face Recognition System (YOLO + FaceNet)")
        self.resize(1300, 750)

        # Style Sheet
        self.setStyleSheet("""
            QMainWindow { background-color: #F9F9F9; }
            QGroupBox { color: #333333; font-weight: bold; border: 1px solid #D1D1D1; margin-top: 15px; padding: 10px; border-radius: 8px; }
            QPushButton { background-color: #FFFFFF; color: #333333; border-radius: 5px; padding: 8px 15px; border: 1px solid #CCCCCC; }
            QPushButton:hover { background-color: #F0F0F0; border: 1px solid #BBBBBB; }
            QLineEdit { background-color: #FFFFFF; padding: 6px; border-radius: 4px; border: 1px solid #DCDCDC; }
        """)

        # --- 1. Model Initialization ---
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"📡 Device: {self.device}")

        # Load YOLOv8-face
        self.detector = YOLO("yolov8n-face-lindevs.pt")

        try:
            self.recognizer = InceptionResnetV1(pretrained=None).to(self.device)
            # Load FaceNet weights
            self.recognizer.load_state_dict(torch.load("20180402-114759-vggface2.pt", map_location=self.device),
                                            strict=False)
            self.recognizer.eval()
            print("✅ FaceNet loaded successfully")
        except Exception as e:
            print(f"❌ Model load error: {e}")

        self.threshold = SamePersonThreshold
        self.db_dir = "database"
        self.known_embeddings = []
        self.known_names = []
        self.load_database()

        # --- 2. UI Layout ---
        self.init_ui()

        # --- 3. Camera ---
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        if not self.cap.isOpened():
            self.cam_label.setText("🚫 Camera Unavailable")
        else:
            self.timer.timeout.connect(self.update_frame)
            self.timer.start(30)

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        self.setCentralWidget(main_widget)

        # Left Panel
        left_panel = QVBoxLayout()
        reg_box = QGroupBox("Registration")
        reg_layout = QVBoxLayout()
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter Name/ID...")
        btn_reg = QPushButton("📤 Register Photo")
        btn_reg.clicked.connect(self.register_face)
        reg_layout.addWidget(self.name_input)
        reg_layout.addWidget(btn_reg)
        reg_layout.addStretch()
        reg_box.setLayout(reg_layout)
        left_panel.addWidget(reg_box)

        # Middle Panel
        mid_panel = QVBoxLayout()
        preview_box = QGroupBox("Live Preview")
        preview_layout = QVBoxLayout()
        self.cam_label = QLabel("Loading...")
        self.cam_label.setFixedSize(640, 480)
        self.cam_label.setAlignment(Qt.AlignCenter)
        self.cam_label.setStyleSheet("background-color: #000; border: 2px solid #333;")
        btn_manual = QPushButton("📁 Upload Image for Verification")
        btn_manual.clicked.connect(self.manual_verify)
        preview_layout.addWidget(self.cam_label)
        preview_layout.addWidget(btn_manual)
        preview_box.setLayout(preview_layout)
        mid_panel.addWidget(preview_box)

        # Right Panel
        right_panel = QVBoxLayout()
        status_box = QGroupBox("Analysis Report")
        status_layout = QVBoxLayout()
        self.status_title = QLabel("Standby")
        self.status_title.setFont(QFont("Arial", 22, QFont.Bold))
        self.status_title.setAlignment(Qt.AlignCenter)
        self.dist_label = QLabel("Distance: --")
        self.dist_label.setAlignment(Qt.AlignCenter)
        self.db_info = QLabel(f"Registered: {len(self.known_names)}")
        status_layout.addWidget(self.status_title)
        status_layout.addWidget(self.dist_label)
        status_layout.addWidget(self.db_info)
        status_layout.addStretch()
        status_box.setLayout(status_layout)
        right_panel.addWidget(status_box)

        main_layout.addLayout(left_panel, 1)
        main_layout.addLayout(mid_panel, 2)
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
            img_resized = cv2.resize(img_rgb, (160, 160))
            face_tensor = fixed_image_standardization(
                torch.tensor(img_resized).permute(2, 0, 1).float()
            ).to(self.device).unsqueeze(0)
            with torch.no_grad():
                return self.recognizer(face_tensor).cpu().numpy()[0]
        except:
            return None

    def identify_face(self, face_img):
        emb = self.get_embedding(face_img)
        if emb is None or len(self.known_embeddings) == 0:
            return "Unknown", 0.0

        dists = np.linalg.norm(self.known_embeddings - emb, axis=1)
        idx = np.argmin(dists)
        min_dist = dists[idx]

        if min_dist < self.threshold:
            return self.known_names[idx], min_dist
        else:
            return "Unknown", min_dist

    def draw_info(self, frame, x1, y1, x2, y2, name, dist, color):
        """Draw rectangle and text (English only)"""
        # Draw Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Prepare text: "Name (Distance)"
        label = f"{name} ({dist:.2f})"

        # Draw text background
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return frame

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret: return

        results = self.detector(frame, verbose=False)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size > 0:
                name, dist = self.identify_face(face_crop)
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

                # Draw on frame
                frame = self.draw_info(frame, x1, y1, x2, y2, name, dist, color)

                # Update UI
                self.status_title.setText(name)
                self.dist_label.setText(f"Distance: {dist:.4f}")

        self.display_image(frame)

    def manual_verify(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.jpg *.png)")
        if file_path:
            img = cv2.imread(file_path)
            results = self.detector(img, verbose=False)[0]
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                name, dist = self.identify_face(img[y1:y2, x1:x2])
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                img = self.draw_info(img, x1, y1, x2, y2, name, dist, color)
            self.display_image(img)

    def register_face(self):
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Warning", "Please enter a name first")
            return

        file_path, _ = QFileDialog.getOpenFileName(self, "Select Face Photo", "", "Images (*.jpg *.png)")
        if file_path:
            img = cv2.imread(file_path)
            save_path = os.path.join(self.db_dir, f"{name}.jpg")
            cv2.imwrite(save_path, img)
            self.load_database()
            self.db_info.setText(f"Registered: {len(self.known_names)}")
            QMessageBox.information(self, "Success", f"Face recorded as: {name}")

    def display_image(self, img_bgr):
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