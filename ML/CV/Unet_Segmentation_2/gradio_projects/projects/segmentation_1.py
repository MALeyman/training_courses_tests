
import onnxruntime as ort
import numpy as np
from PIL import Image
import os
import gradio as gr
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# print("ПУТЬ:  ", BASE_DIR)

ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
# Добавляем корень проекта в sys.path, если его там нет
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


from projects.common.session import ort_session, get_device



image_path = os.path.join(BASE_DIR, "files/999.png")
model_path = os.path.join(BASE_DIR, "files/unetpp_model.onnx")

session = ort.InferenceSession(model_path)

def preprocess_image(img, size=(512, 256)):
    img = img.convert("RGB").resize(size)  # PIL: (width, height)
    img_np = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_np = (img_np - mean) / std
    img_np = img_np.transpose(2, 0, 1)  # HWC -> CHW
    img_np = np.expand_dims(img_np, axis=0).astype(np.float32)
    return img_np


def segment_and_overlay(input_img, selected_classes):
    orig_width, orig_height = input_img.size  # оригинальный размер

    # Предобработка и инференс 
    input_tensor = preprocess_image(input_img)  
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})
    pred_mask = np.argmax(outputs[0], axis=1)[0]  # размер (256, 512)

    # Палитра и имена классов
    palette = [
        [0, 0, 0],      # ФОН
        [128, 64,128],  # road
        [244, 35,232],  # sidewalk
        [70, 70, 70],   # building
        [102,102,156],  # wall
        [190,153,153],  # fence
        [153,153,153],  # pole
        [250,170, 30],  # traffic light
        [220,220,  0],  # traffic sign
        [107,142, 35],  # vegetation
        [152,251,152],  # terrain
        [70,130,180],   # sky
        [220, 20, 60],  # person
        [255,  0,  0],  # rider
        [0,  0,142],    # car
        [0,  0, 70],    # truck
        [0, 60,100],    # bus
        [0, 80,100],    # train
        [0,  0,230],    # motorcycle
        [119, 11, 32],  # bicycle
    ]

    class_names = [
        "ФОН", "road", "sidewalk", "building", "wall", "fence", "pole",
        "traffic light", "traffic sign", "vegetation", "terrain", "sky",
        "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"
    ]

    selected_indices = [class_names.index(cls) for cls in selected_classes]

    color_mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
    for cls_idx in selected_indices:
        color_mask[pred_mask == cls_idx] = palette[cls_idx]

    mask_img = Image.fromarray(color_mask)

    # Растягиваем маску до оригинального размера изображения
    mask_img = mask_img.resize((orig_width, orig_height), resample=Image.NEAREST)

    # Наложение маски на исходное изображение с прозрачностью
    base = input_img.convert("RGBA")
    overlay = mask_img.convert("RGBA")
    blended = Image.blend(base, overlay, alpha=0.3)

    return blended

# Палитра и имена классов
palette = [
    [0, 0, 0],      # ФОН
    [128, 64,128],  # road
    [244, 35,232],  # sidewalk
    [70, 70, 70],   # building
    [102,102,156],  # wall
    [190,153,153],  # fence
    [153,153,153],  # pole
    [250,170, 30],  # traffic light
    [220,220,  0],  # traffic sign
    [107,142, 35],  # vegetation
    [152,251,152],  # terrain
    [70,130,180],   # sky
    [220, 20, 60],  # person
    [255,  0,  0],  # rider
    [0,  0,142],    # car
    [0,  0, 70],    # truck
    [0, 60,100],    # bus
    [0, 80,100],    # train
    [0,  0,230],    # motorcycle
    [119, 11, 32],  # bicycle
]

class_names = [
    "ФОН", "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"
]



def get_segmentation_tab():

        gr.Markdown("## Сегментация изображений (Итоговая аттестация)")
        gr.Markdown("---")
        with gr.Row():
            with gr.Column(scale=1):
                # Добавляем метку устройства
                device_label = gr.Label(value=get_device(), label="Работаем на устройстве")
                selected = gr.CheckboxGroup(label="Выберите классы для отображения", choices=class_names, value=class_names[:3])
                btn = gr.Button("Сегментировать")
            with gr.Column(scale=3):
                input_image = gr.Image(type="pil", value=image_path, label="Исходное изображение", height=320)
                output_image = gr.Image(label="Результат сегментации")
        btn.click(fn=segment_and_overlay, inputs=[input_image, selected], outputs=output_image)
  



