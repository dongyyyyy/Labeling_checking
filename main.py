import sys
import os
import json
import pandas as pd
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QTextEdit, QSizePolicy
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QImage, QKeyEvent
from PyQt5.QtCore import Qt
import cv2
import random

class ImageViewer(QWidget):
    def __init__(self, image_dir, annotation_df):
        super().__init__()
        self.setWindowTitle("Bounding Box Viewer")
        self.image_dir = image_dir
        self.annotation_df = annotation_df
        self.image_list = annotation_df['filename'].unique().tolist()
        self.current_index = 0

        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.info_box = QTextEdit(self)
        self.info_box.setReadOnly(True)
        self.info_box.setFixedWidth(300)

        self.prev_button = QPushButton('Previous')
        self.next_button = QPushButton('Next')
        self.save_button = QPushButton('Save Annotations')

        self.prev_button.clicked.connect(self.show_prev_image)
        self.next_button.clicked.connect(self.show_next_image)
        self.save_button.clicked.connect(self.save_annotations)

        hbox_buttons = QHBoxLayout()
        hbox_buttons.addWidget(self.prev_button)
        hbox_buttons.addWidget(self.next_button)
        hbox_buttons.addWidget(self.save_button)

        hbox_main = QHBoxLayout()
        hbox_main.addWidget(self.label)
        hbox_main.addWidget(self.info_box)

        vbox = QVBoxLayout()
        vbox.addLayout(hbox_main)
        vbox.addLayout(hbox_buttons)
        self.setLayout(vbox)

        self.class_colors = self.generate_class_colors()
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()
        self.showFullScreen()
        self.show_image()

    def generate_class_colors(self):
        unique_classes = self.annotation_df['class_name'].unique()
        colors = {}
        for cls in unique_classes:
            colors[cls] = tuple(random.randint(0, 255) for _ in range(3))
        return colors

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Right:
            self.show_next_image()
        elif event.key() == Qt.Key_Left:
            self.show_prev_image()
        elif event.key() == Qt.Key_Escape:
            self.close()

    def show_image(self):
        filename = self.image_list[self.current_index]
        image_path = os.path.join(self.image_dir, filename)
        image = cv2.imread(image_path)
        if image is None:
            print(f"이미지를 불러올 수 없습니다: {image_path}")
            return

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = self.annotation_df[self.annotation_df['filename'] == filename]

        info_text = f"[ {filename} 에 포함된 객체 정보 ]\n\n"
        for idx, (_, row) in enumerate(bboxes.iterrows()):
            x, y, w, h = row['x'], row['y'], row['width'], row['height']
            class_name = row['class_name']
            region_id = row.get('region_id', idx)
            extra_info = []
            for col in ['crack_severity', 'crack_shape', 'garbage_type', 'manhole_type']:
                val = row.get(col, '')
                if val and val != 'none':
                    extra_info.append(f"{col}: {val}")

            label_text = f"ID:{region_id} - {class_name}"
            color = self.class_colors.get(class_name, (0, 255, 0))
            image = self.draw_bbox(image, x, y, w, h, label_text, color)

            info_text += f"[ID {region_id}] Class: {class_name}\n"
            for e in extra_info:
                info_text += f"    {e}\n"
            info_text += "\n"

        self.info_box.setText(info_text)

        height, width, _ = image.shape
        bytes_per_line = 3 * width
        qimage = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(qimage))

    def draw_bbox(self, image, x, y, w, h, label, color):
        image = image.copy()
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text_x, text_y = x, y - 10 if y - 10 > 10 else y + 20
        cv2.putText(image, label, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, label, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        return image

    def show_next_image(self):
        if self.current_index < len(self.image_list) - 1:
            self.current_index += 1
            self.show_image()

    def show_prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.show_image()

    def save_annotations(self):
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Annotations", "annotations_saved.csv", "CSV Files (*.csv)")
        if save_path:
            self.annotation_df.to_csv(save_path, index=False)
            print(f"✅ 저장 완료: {save_path}")

if __name__ == '__main__':
    from PyQt5.QtGui import QImage

    app = QApplication(sys.argv)

    # 데이터 로드
    target_file='04_1'
    # 경로 설정
    csv_path = f'./labels/{target_file}_revised.csv'
    image_dir = f'./images/{target_file}/'  # 실제 이미지들이 저장된 폴더
    # output_dir = 'output_images'
    

    
    df = pd.read_csv(csv_path)
    df = df[df['region_shape_attributes'].str.strip() != '{}'].copy()
    df['shape'] = df['region_shape_attributes'].apply(json.loads)
    df['attrs'] = df['region_attributes'].apply(json.loads)

    # 각 필드 추출
    rows = []
    for _, row in df.iterrows():
        shape = row['shape']
        attrs = row['attrs']
        if shape.get("name") != "rect":
            continue
        base = {
            'filename': row['filename'],
            'x': int(shape['x']),
            'y': int(shape['y']),
            'width': int(shape['width']),
            'height': int(shape['height']),
            'class_name': attrs.get('class_name', 'unknown'),
            'crack_severity': attrs.get('crack_severity', ''),
            'crack_shape': attrs.get('crack_shape', ''),
            'garbage_type': attrs.get('garbage_type', ''),
            'manhole_type': attrs.get('manhole_type', '')
        }
        rows.append(base)

    annotation_df = pd.DataFrame(rows)

    viewer = ImageViewer(image_dir=image_dir, annotation_df=annotation_df)
    viewer.show()

    # viewer = ImageViewer(image_dir=image_dir, annotation_df=annotation_df)
    # viewer.resize(1024, 768)
    # viewer.show()

    sys.exit(app.exec_())



