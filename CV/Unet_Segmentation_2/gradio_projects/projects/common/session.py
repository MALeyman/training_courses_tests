# projects/common/session.py

import onnxruntime
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# print("ПУТЬ:  ", BASE_DIR)

ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
# Добавляем корень проекта в sys.path, если его там нет
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# print("ПУТЬ:  ", ROOT_DIR)
model_path = os.path.join(BASE_DIR, "retinaface_resnet50.onnx")
model_path_2 = os.path.join(BASE_DIR, "yolo.onnx")


providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
ort_session = onnxruntime.InferenceSession(model_path, providers=providers)
ort_session_2 = onnxruntime.InferenceSession(model_path_2, providers=providers)

def get_device():
    providers = onnxruntime.get_available_providers()
    if 'CUDAExecutionProvider' in providers:
        return "Устройство: GPU (CUDA)"
    else:
        return "Устройство: CPU"
