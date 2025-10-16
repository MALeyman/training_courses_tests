

import numpy as np
import os
import shutil
from PIL import Image
from torch.utils.data import Dataset
import torch
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
import cv2
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy
import gc

import torchvision.transforms as T
import random

import random, numpy as np, torch
from PIL import Image, ImageEnhance
import torchvision.transforms.functional as TF




# Предсказание модели
def prediction_mask(path_img, model):
    """ 
        Сегментация изображения
        path_img: путь к изображению
        model: модель
    """
    img = Image.open(path_img).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((256, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
 
    img_tensor = preprocess(img).unsqueeze(0) 

    with torch.no_grad():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        img_tensor = img_tensor.to(device) 
        output = model(img_tensor)
        prediction = torch.argmax(output, dim=1) 
        return img, prediction




# Препроцесс для ONNX 
def preprocess_image_onnx(path_img, input_size=(512, 256)):
    """ 
    Загрузка изображения и препроцессинг
    """

    img = Image.open(path_img).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((256, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
 
    img_tensor = preprocess(img).unsqueeze(0) 

    return img, img_tensor

# Предсказание ONNX
def prediction_mask_onnx(path_img, onnx_session):
    img, img_np = preprocess_image_onnx(path_img)

    
    input_name = onnx_session.get_inputs()[0].name
    img_np = img_np.cpu().numpy()
    outputs = onnx_session.run(None, {input_name: img_np})

    # outputs[0] — выход модели, shape (1, num_classes, H, W)
    pred_mask = np.argmax(outputs[0], axis=1)[0]  # (H, W)

    return img, pred_mask


