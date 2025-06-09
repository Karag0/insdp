import sys
import os
import cv2
import numpy as np
from PIL import Image
import torch
from diffusers import StableDiffusionInpaintPipeline
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLineEdit, QLabel, QFileDialog, QTextEdit, 
    QProgressBar, QMessageBox, QGridLayout, QSpinBox
)
from PyQt6.QtCore import QThread, pyqtSignal


class VideoProcessor(QThread):
    progress_updated = pyqtSignal(int)
    finished_processing = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, video_path, save_dir, prompt, negative_prompt, width, height):
        super().__init__()
        self.video_path = video_path
        self.save_dir = save_dir
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.target_width = width
        self.target_height = height
        self.stopped = False

    def run(self):
        try:
            # Загрузка модели
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32
            
            self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=dtype,
                safety_checker=None,  # ОТКЛЮЧАЕМ NSFW-ФИЛЬТР
            ).to(device)
            
            # Оптимизации для ускорения
            if device == "cuda":
                self.pipeline.enable_xformers_memory_efficient_attention()
                self.pipeline.unet.to(memory_format=torch.channels_last)
                torch.cuda.empty_cache()
            
            # Открытие видеофайла
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.error_occurred.emit("Ошибка открытия видеофайла")
                return

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            current_frame = 0

            while cap.isOpened() and not self.stopped:
                ret, frame = cap.read()
                if not ret:
                    break

                # Изменение размера кадра до целевого разрешения
                resized_frame = cv2.resize(
                    frame, 
                    (self.target_width, self.target_height),
                    interpolation=cv2.INTER_AREA
                )
                
                # Конвертация в RGB
                rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                
                # Создание маски (черные области)
                gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)
                mask_pil = Image.fromarray(mask).convert("L")

                # Коррекция размеров под требования модели (кратность 64)
                def adjust_size(size):
                    return size - size % 64
                
                gen_width = adjust_size(self.target_width)
                gen_height = adjust_size(self.target_height)
                
                # Обработка кадра
                result = self.pipeline(
                    prompt=self.prompt,
                    negative_prompt=self.negative_prompt,
                    image=pil_image,
                    mask_image=mask_pil,
                    height=gen_height,
                    width=gen_width,
                    num_inference_steps=16,
                    guidance_scale=7.5,
                ).images[0]
                
                # Возврат к целевому разрешению
                final_result = result.resize(
                    (self.target_width, self.target_height),
                    Image.LANCZOS
                )

                # Сохранение кадра
                frame_path = os.path.join(self.save_dir, f"frame_{current_frame:05d}.png")
                final_result.save(frame_path)

                current_frame += 1
                progress = int((current_frame / total_frames) * 100)
                self.progress_updated.emit(progress)

            cap.release()
            if not self.stopped:
                self.finished_processing.emit()

        except Exception as e:
            self.error_occurred.emit(str(e))

    def stop(self):
        self.stopped = True


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Inpainting Tool")
        self.setGeometry(100, 100, 600, 450)  # Увеличили высоту
        
        # Основные виджеты
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        
        # Видео файл
        self.video_layout = QHBoxLayout()
        self.video_label = QLabel("Видео файл:")
        self.video_path_edit = QLineEdit()
        self.video_path_edit.setReadOnly(True)
        self.browse_video_btn = QPushButton("Обзор...")
        self.video_layout.addWidget(self.video_label)
        self.video_layout.addWidget(self.video_path_edit)
        self.video_layout.addWidget(self.browse_video_btn)
        
        # Разрешение
        self.resolution_layout = QGridLayout()
        self.resolution_label = QLabel("Разрешение генерации:")
        self.width_label = QLabel("Ширина:")
        self.width_spin = QSpinBox()
        self.width_spin.setRange(64, 2048)
        self.width_spin.setValue(512)
        self.height_label = QLabel("Высота:")
        self.height_spin = QSpinBox()
        self.height_spin.setRange(64, 2048)
        self.height_spin.setValue(512)
        
        self.resolution_layout.addWidget(self.resolution_label, 0, 0)
        self.resolution_layout.addWidget(self.width_label, 1, 0)
        self.resolution_layout.addWidget(self.width_spin, 1, 1)
        self.resolution_layout.addWidget(self.height_label, 2, 0)
        self.resolution_layout.addWidget(self.height_spin, 2, 1)
        
        # Промпты
        self.prompt_label = QLabel("Позитивный промпт:")
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setPlaceholderText("high quality, detailed")
        self.negative_prompt_label = QLabel("Негативный промпт:")
        self.negative_prompt_edit = QTextEdit()
        self.negative_prompt_edit.setPlaceholderText("blurry, low quality")
        
        # Кнопка обработки
        self.process_btn = QPushButton("Обработать видео")
        self.process_btn.setEnabled(False)
        
        # Прогресс бар
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        # Сборка интерфейса
        self.layout.addLayout(self.video_layout)
        self.layout.addLayout(self.resolution_layout)  # Добавляем блок разрешения
        self.layout.addWidget(self.prompt_label)
        self.layout.addWidget(self.prompt_edit)
        self.layout.addWidget(self.negative_prompt_label)
        self.layout.addWidget(self.negative_prompt_edit)
        self.layout.addWidget(self.process_btn)
        self.layout.addWidget(self.progress_bar)
        
        # Обработчики событий
        self.browse_video_btn.clicked.connect(self.browse_video)
        self.process_btn.clicked.connect(self.process_video)
        self.video_path_edit.textChanged.connect(self.check_inputs)

    def browse_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите видео файл",
            "",
            "Video Files (*.mp4 *.avi *.mov)"
        )
        if file_path:
            self.video_path_edit.setText(file_path)

    def check_inputs(self):
        has_video = bool(self.video_path_edit.text())
        self.process_btn.setEnabled(has_video)

    def process_video(self):
        video_path = self.video_path_edit.text()
        save_dir = QFileDialog.getExistingDirectory(
            self,
            "Выберите папку для сохранения кадров"
        )
        
        if not save_dir:
            return
        
        prompt = self.prompt_edit.toPlainText().strip() or "high quality, detailed"
        negative_prompt = self.negative_prompt_edit.toPlainText().strip() or "blurry, low quality"
        width = self.width_spin.value()
        height = self.height_spin.value()
        
        # Проверка минимального размера
        if width < 64 or height < 64:
            QMessageBox.warning(self, "Ошибка", "Минимальный размер изображения: 64x64 пикселей")
            return
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.process_btn.setEnabled(False)
        self.browse_video_btn.setEnabled(False)
        
        # Предупреждение о времени обработки
        QMessageBox.information(
            self, 
            "Информация", 
            f"Обработка видео с разрешением {width}x{height}...\n"
            "Для достижения лучших результатов используйте разрешения, кратные 64.\n"
            "Процесс может занять значительное время."
        )
        
        self.processor = VideoProcessor(
            video_path,
            save_dir,
            prompt,
            negative_prompt,
            width,
            height
        )
        
        self.processor.progress_updated.connect(self.progress_bar.setValue)
        self.processor.finished_processing.connect(self.processing_finished)
        self.processor.error_occurred.connect(self.handle_error)
        self.processor.start()

    def processing_finished(self):
        QMessageBox.information(self, "Готово", "Обработка видео завершена!")
        self.reset_ui()

    def handle_error(self, message):
        QMessageBox.critical(self, "Ошибка", message)
        self.reset_ui()

    def reset_ui(self):
        self.progress_bar.setVisible(False)
        self.process_btn.setEnabled(True)
        self.browse_video_btn.setEnabled(True)
        self.processor = None

    def closeEvent(self, event):
        if hasattr(self, 'processor') and self.processor and self.processor.isRunning():
            reply = QMessageBox.question(
                self,
                "Подтверждение",
                "Обработка все еще выполняется. Вы уверены, что хотите прервать?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.processor.stop()
                self.processor.wait()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
            
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())