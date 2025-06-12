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
    def __init__(self, image_dir, annotation_df_1, annotation_df_2):
        super().__init__()
        self.setWindowTitle("Bounding Box Viewer")
        self.image_dir = image_dir
        self.annotation_df_1 = annotation_df_1
        self.annotation_df_2 = annotation_df_2
        self.image_list = sorted(set(annotation_df_1['filename']).union(annotation_df_2['filename']))
        self.current_index = 0

        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.info_box_1 = QTextEdit(self)
        self.info_box_1.setReadOnly(True)
        self.info_box_1.setFixedWidth(400)
        self.info_box_1.setPlaceholderText("Revised CSV 라벨 정보")

        self.info_box_2 = QTextEdit(self)
        self.info_box_2.setReadOnly(True)
        self.info_box_2.setFixedWidth(400)
        self.info_box_2.setPlaceholderText("Original CSV 라벨 정보")

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

        vbox_info = QVBoxLayout()
        vbox_info.addWidget(self.info_box_1)
        vbox_info.addWidget(self.info_box_2)

        hbox_main = QHBoxLayout()
        hbox_main.addWidget(self.label)
        hbox_main.addLayout(vbox_info)

        vbox = QVBoxLayout()
        vbox.addLayout(hbox_main)
        vbox.addLayout(hbox_buttons)
        self.setLayout(vbox)


        self.info_box_1.cursorPositionChanged.connect(self.highlight_selected_bbox_1)
        self.highlighted_region_id = None
        self.highlighted_source = None

        self.info_box_2.cursorPositionChanged.connect(self.highlight_selected_bbox_2)
        self.highlighted_region_id_2 = None
        self.highlighted_source_2 = None


        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()
        self.showFullScreen()
        self.show_image()

    def highlight_selected_bbox_1(self):
        line_num = self.info_box_1.textCursor().blockNumber()
        region_id = self.region_map_1.get(line_num)
        if region_id is not None:
            self.highlighted_region_id = region_id
            self.highlighted_source = 1
            self.show_image()

    def highlight_selected_bbox_2(self):
        line_num = self.info_box_2.textCursor().blockNumber()
        region_id = self.region_map_2.get(line_num)
        if region_id is not None:
            self.highlighted_region_id_2 = region_id
            self.highlighted_source_2 = 1
            self.show_image()

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
        bboxes_1 = self.annotation_df_1[self.annotation_df_1['filename'] == filename]
        bboxes_2 = self.annotation_df_2[self.annotation_df_2['filename'] == filename]

        info_text_1 = f"[ {filename} - Revised ]\n\n"
        info_text_2 = f"[ {filename} - Original ]\n\n"

        self.region_map_1 = {}
        self.region_map_2 = {}
        drawn_labels = set()

        for line_idx, (_, row) in enumerate(bboxes_1.iterrows()):
            x, y, w, h = row['x'], row['y'], row['width'], row['height']
            region_id = row['region_id']
            label = f"V1: ID:{region_id: <3} - {row['class_name']: <12}"
            is_highlight = (self.highlighted_source == 1 and self.highlighted_region_id == region_id)
            image = self.draw_bbox(image, x, y, w, h, label, (0, 0, 255), offset=0, highlight=is_highlight)
            info_text_1 += f"{label: <30} {row['crack_severity']: <10} {row['crack_shape']: <10} {row['garbage_type']: <10} {row['manhole_type']: <10}\n"
            self.region_map_1[line_idx + 2] = region_id  # +2 to skip header lines
            drawn_labels.add((x, y, w, h, row['class_name']))

        for idx, (_, row) in enumerate(bboxes_2.iterrows()):
            x, y, w, h = row['x'], row['y'], row['width'], row['height']
            region_id = row['region_id']
            label = f"V2: ID:{row['region_id']: <3} - {row['class_name']: <12}"
            offset = -15 if (x, y, w, h, row['class_name']) in drawn_labels else 0
            is_highlight = (self.highlighted_source_2 == 1 and self.highlighted_region_id_2 == region_id)
            image = self.draw_bbox(image, x, y, w, h, label, (0, 255, 0), offset=offset, highlight=is_highlight)
            info_text_2 += f"{label: <30} {row['crack_severity']: <10} {row['crack_shape']: <10} {row['garbage_type']: <10} {row['manhole_type']: <10}\n"
            self.region_map_2[idx + 2] = region_id  # +2 to skip header lines

        self.info_box_1.setText(info_text_1)
        self.info_box_2.setText(info_text_2)

        height, width, _ = image.shape
        bytes_per_line = 3 * width
        qimage = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(qimage))

    def draw_bbox(self, image, x, y, w, h, label, color, offset=0, highlight=False):
        image = image.copy()
        pen_color = (255, 0, 0) if highlight else color
        cv2.rectangle(image, (x, y), (x + w, y + h), pen_color, 2)
        text_x, text_y = x, y - 10 + offset if y - 10 + offset > 10 else y + 20
        cv2.putText(image, label, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, label, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, pen_color, 1, cv2.LINE_AA)
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
            combined_df = pd.concat([self.annotation_df_1, self.annotation_df_2])
            combined_df.to_csv(save_path, index=False)
            print(f"✅ 저장 완료: {save_path}")

if __name__ == '__main__':
    app = QApplication(sys.argv)

    # 데이터 로드
    target_file='04_2'
    # 경로 설정
    csv_path_1 = f'./labels/{target_file}_revised.csv'
    csv_path_2 = f'./labels/{target_file}_csv.csv'
    image_dir = f'./images/{target_file}/'  # 실제 이미지들이 저장된 폴더
    # output_dir = 'output_images'


    def preprocess(csv_path):
        df = pd.read_csv(csv_path)
        df = df[df['region_shape_attributes'].str.strip() != '{}'].copy()
        df['shape'] = df['region_shape_attributes'].apply(json.loads)
        df['attrs'] = df['region_attributes'].apply(json.loads)
        rows = []
        for idx, row in df.iterrows():
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
                'manhole_type': attrs.get('manhole_type', ''),
                'region_id': row.get('region_id', idx)
            }
            rows.append(base)
        return pd.DataFrame(rows)

    df1 = preprocess(csv_path_1)
    df2 = preprocess(csv_path_2)

    differing_files = []
    for filename in set(df1['filename']).union(df2['filename']):
        a1 = df1[df1['filename'] == filename].sort_values(by=['x','y','width','height','class_name']).reset_index(drop=True)
        a2 = df2[df2['filename'] == filename].sort_values(by=['x','y','width','height','class_name']).reset_index(drop=True)
        if not a1.equals(a2):
            differing_files.append(filename)

    df1 = df1[df1['filename'].isin(differing_files)]
    df2 = df2[df2['filename'].isin(differing_files)]

    viewer = ImageViewer(image_dir=image_dir, annotation_df_1=df1, annotation_df_2=df2)
    viewer.show()

    sys.exit(app.exec_())
