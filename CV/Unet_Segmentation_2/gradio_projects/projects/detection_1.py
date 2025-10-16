
import onnxruntime as ort
import numpy as np
from PIL import Image
import os
import gradio as gr
import sys
import cv2


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# print("ПУТЬ:  ", BASE_DIR)

# ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..'))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
# Добавляем корень проекта в sys.path, если его там нет
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# print("ПУТЬ:  ", ROOT_DIR)
# импортируем модуль как абсолютный из корня проекта
from projects.files.utils import gradio_video_processing, onnx_inference
from projects.common.session import ort_session, get_device


image_path = os.path.join(BASE_DIR, "files/3.jpg")
video_path = os.path.join(BASE_DIR, "files/1.mp4")
model_path = os.path.join(BASE_DIR, "files/retinaface_resnet50.onnx")



def get_detection_tab_1():

        gr.Markdown("## Детекция лиц с помощью ONNX (Промежуточная аттестация)")
        gr.Markdown("---")
        with gr.Row():
            # Левая колонка со слайдерами (общие для обеих вкладок)
            with gr.Column(scale=1):
                # Добавляем метку устройства
                device_label = gr.Label(value=get_device(), label="Работаем на устройстве")
                confidence_slider = gr.Slider(0, 1, value=0.7, label="Порог уверенности")
                nms_slider = gr.Slider(0, 1, value=0.4, label="Порог NMS")
                max_size_slider = gr.Slider(256, 2048, value=1200, step=64, label="Максимальный размер изображения")


            # Правая колонка с вкладками для изображения и видео
            with gr.Column(scale=3):
                with gr.Tabs():
                    with gr.TabItem("Изображение"):
                        input_image = gr.Image(type="numpy", value=image_path, label="Загрузите изображение", height=320)
                        output_image = gr.Image(label="Результат с детекцией")
                        btn_img = gr.Button("Запустить детекцию")

                        btn_img.click(
                            onnx_inference,
                            inputs=[input_image, confidence_slider, nms_slider, max_size_slider],
                            outputs=output_image,
                        )

                    with gr.TabItem("Видео"):
                        video_io = gr.Video(
                            label="Загрузите видео",
                            sources=["upload", "webcam"],  # позволяет выбрать загрузку или веб-камеру
                            value=video_path,
                            height=512
                        )
                        frame_skip_slider = gr.Slider(1, 10, value=4, step=1, label="Обрабатывать каждый n-й кадр")
                        btn_vid = gr.Button("Запустить обработку")

                        btn_vid.click(
                            gradio_video_processing,
                            inputs=[video_io, confidence_slider, nms_slider, max_size_slider, frame_skip_slider],
                            outputs=video_io,
                        )









