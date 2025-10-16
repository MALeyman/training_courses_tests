"""  
    Автор: Лейман Максим
    Дата создания: 20.06.2025
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from shutil import move
import xml.etree.ElementTree as ET
import pandas as pd
import torch
from skimage import io
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2



class FaceKeypointsDataset2(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, img_size=(224, 224)):
        self.keypoints_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.keypoints_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.keypoints_frame.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        image_np = np.array(image)
        orig_h, orig_w = image_np.shape[:2]

        keypoints = self.keypoints_frame.iloc[idx, 1:].values.astype('float').reshape(-1, 2)

        if self.transform:
            augmented = self.transform(image=image_np, keypoints=keypoints)
            image_np = augmented['image']
            keypoints = np.array(augmented['keypoints'])
        else:
            image_np = cv2.resize(image_np, self.img_size)
            scale_x = self.img_size[0] / orig_w
            scale_y = self.img_size[1] / orig_h
            keypoints[:, 0] *= scale_x
            keypoints[:, 1] *= scale_y

        image_tensor = transforms.ToTensor()(image_np)

        sample = {'image': image_tensor, 'keypoints': keypoints}
        return sample



# Датасет ключевых точек лица
class FaceKeypointsDataset(Dataset):
    """ 
        Датасет детекции ключевых точек лица
    """
    def __init__(self, csv_file, root_dir, transform=None, img_size=(224, 224)):
        self.keypoints_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.keypoints_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.keypoints_frame.iloc[idx, 0])
        try:
            image = Image.open(img_name).convert('RGB')
        except Exception as e:
            print(f"Ошибка при открытии {img_name}: {e}")
            
        orig_w, orig_h = image.size
        keypoints = self.keypoints_frame.iloc[idx, 1:].values.astype('float').reshape(-1, 2)
        # Масштабируем точки после ресайза
        if self.transform:
            image = self.transform(image)  
            new_w, new_h = self.img_size
        else:
            image = image.resize(self.img_size)
            new_w, new_h = self.img_size
            image = transforms.ToTensor()(image)
        
        # print(image.shape)
        keypoints[:, 0] = keypoints[:, 0] * (new_w / orig_w)
        keypoints[:, 1] = keypoints[:, 1] * (new_h / orig_h)
        sample = {'image': image, 'keypoints': keypoints}
        return sample



# Трансформация аннотаций
def annotation_transform(patch_in='dataset/dataset_2/annotations.xml', path_out='dataset/dataset_2/training.csv'):
    """ 
         Преобразование аннотаций разных датасетов к одному виду
    """
    rows = []
    tree = ET.parse(patch_in)
    root = tree.getroot()

    for image_elem in root.findall('image'):
        image_name = image_elem.attrib['name']
        keypoints = []
        for point_elem in image_elem.findall('points'):
            coords = point_elem.attrib['points'].split(',')
            keypoints.extend([float(coords[0]), float(coords[1])])
        row = [image_name] + keypoints
        rows.append(row)

    # Определяем максимальное количество точек (чтобы все строки были одинаковой длины)
    max_points = max(len(r) for r in rows) - 1  # -1

    # Формируем имена столбцов
    columns = ['image_name']
    for i in range(max_points // 2):
        columns += [f'keypoint_{i+1}_x', f'keypoint_{i+1}_y']

    # Дополняем строки пустыми значениями, если точек меньше максимума
    for row in rows:
        while len(row) < len(columns):
            row.append('')

    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(path_out, index=False)


# Переименовывает изображения с уникальным смещением 
def convert_xml_to_csv_with_renaming(
    image_folder,
    xml_path,
    csv_out_path,
    image_prefix=8000,
    image_ext='.jpg'
):
    """
    Переименовывает изображения с уникальным смещением, парсит аннотации из XML и сохраняет их в CSV.
    
    :param image_folder: Папка с изображениями
    :param xml_path: Путь к XML-файлу аннотаций
    :param csv_out_path: Куда сохранить итоговый CSV
    :param image_prefix: С какого номера начинать новые имена файлов (например, 8000)
    :param image_ext: Расширение файлов изображений (по умолчанию .jpg)
    """
    # 1. Переименование изображений
    image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(image_ext)])
    old_to_new = {}
    for idx, old_name in enumerate(image_files):
        new_name = f"{idx + image_prefix}{image_ext}"
        old_path = os.path.join(image_folder, old_name)
        new_path = os.path.join(image_folder, new_name)
        move(old_path, new_path)
        old_to_new[old_name] = new_name

    # 2. Преобразование XML в CSV
    tree = ET.parse(xml_path)
    root = tree.getroot()
    rows = []

    for image_elem in root.findall('image'):
        old_image_name = os.path.basename(image_elem.attrib['name'])
        new_image_name = old_to_new.get(old_image_name)
        keypoints = []
        for point_elem in image_elem.findall('points'):
            coords = point_elem.attrib['points'].split(',')
            keypoints.extend([float(coords[0]), float(coords[1])])
        row = [new_image_name] + keypoints
        rows.append(row)

    # Определяем максимальное количество точек
    max_points = max(len(r) for r in rows) - 1
    columns = ['image_name']
    for i in range(max_points // 2):
        columns += [f'keypoint_{i+1}_x', f'keypoint_{i+1}_y']

    # Дополняем строки пустыми значениями
    for row in rows:
        while len(row) < len(columns):
            row.append('')

    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(csv_out_path, index=False)



def rename_images_and_update_csv(
    image_folder,
    csv_path,
    csv_out_path,
    start_index=9000,
    image_ext='.jpg'
):
    """ 
        переименовывает изображения и обновляет файл аннотаций
    """
    # 1. Загрузка CSV с сохранением первой строки 
    df = pd.read_csv(csv_path, sep=',', header=0)  
    old_names = df.iloc[:, 0].tolist()
  
    # 2. Переименование изображений и формирование новых имён
    new_names = []
    for idx, old_name in enumerate(old_names):
      
        new_name = f"{idx + start_index}{image_ext}"
        old_path = os.path.join(image_folder, old_name)
        new_path = os.path.join(image_folder, new_name)
        if os.path.exists(old_path):
            move(old_path, new_path)
        new_names.append(new_name)
     
    print(new_names)
    # 3. Замена первого столбца на новые имена
    df.iloc[:, 0] = new_names

    # 4. Сохранение обновленного CSV
    df.to_csv(csv_out_path, sep='\t', header=True, index=False)





