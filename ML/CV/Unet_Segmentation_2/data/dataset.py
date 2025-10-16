""" 
Автор: Лейман М.А.
Дата создания: 24.06.2025
"""

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import numpy as np
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
from torch.utils.data import Subset



# #########################   Датасет
# загрузка с PIL
class CityscapesFlatDataset(Dataset):
    """  
        Датасет: загрузка с PIL
    """
    def __init__(self, root_dir, transform=None, target_transform=None):
        self.images_dir = os.path.join(root_dir, 'images')
        self.targets_dir = os.path.join(root_dir, 'targets')
        self.transform = transform
        self.target_transform = target_transform

        self.images = sorted(os.listdir(self.images_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)

        # Формируем имя маски, заменяя суффикс
        target_name = img_name.replace('leftImg8bit.png', 'gtFine_labelTrainIds.png')
        target_path = os.path.join(self.targets_dir, target_name)

        image = Image.open(img_path).convert('RGB')
        target = Image.open(target_path)
        target_np = np.array(target)
        target_np = target_np + 1
        target_np[target_np == 256] = 0
        target = Image.fromarray(target_np)
        # print(image.size)
        # print(target.size)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)

        return image, target



# загрузка с openCV
class CityscapesFlatDataset2(Dataset):
    """ 
        Датасет: загрузка с openCV
    """
    def __init__(self, root_dir, transform=None, val_transform=None):
        self.images_dir = os.path.join(root_dir, 'images')
        self.targets_dir = os.path.join(root_dir, 'targets')
        self.transform = transform
        self.val_transform = val_transform

        self.images = sorted(os.listdir(self.images_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)

        # Формируем имя маски, заменяя суффикс
        target_name = img_name.replace('leftImg8bit.png', 'gtFine_labelTrainIds.png')
        target_path = os.path.join(self.targets_dir, target_name)

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image not found or corrupted: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        target = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
        if target is None:
            raise FileNotFoundError(f"Mask not found or corrupted: {target_path}")

        # Смещаем классы на +1, заменяем 256 на 0 (фон)
        target = target.astype(np.int32) + 1
        target[target == 256] = 0
        target = target.astype(np.uint8)

        image = image.transpose(2, 0, 1)
        if self.transform:
            image, target = self.transform(image, target)
        else:
            image = cv2.resize(
                image.transpose(1, 2, 0), 
                (512, 256), 
                interpolation=cv2.INTER_LINEAR
            ).transpose(2, 0, 1)
            target = cv2.resize(
                target, 
                (512, 256), 
                interpolation=cv2.INTER_NEAREST
            )
        image = torch.from_numpy(image).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        image = (image - mean) / std
        
        target = torch.from_numpy(target).long()
        return image, target



class CityscapesFlatDataset3(Dataset):
    """
    Датасет для изображений и масок с одинаковыми именами файлов.
    """

    def __init__(self, root_dir, transform=None, img_size=(512, 256)):
        self.images_dir = os.path.join(root_dir, 'images')
        self.targets_dir = os.path.join(root_dir, 'targets')
        self.transform = transform
        self.img_size = img_size

        # Список файлов изображений
        self.images = sorted(os.listdir(self.images_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        target_path = os.path.join(self.targets_dir, img_name)  # маска с тем же именем

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image not found or corrupted: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        target = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
        if target is None:
            raise FileNotFoundError(f"Mask not found or corrupted: {target_path}")

       
        # target = target.astype(np.int32) + 1
        target[target == 20] = 0
        # target = target.astype(np.uint8)

        # Применяем resize к изображению и маске
        image = cv2.resize(image, (self.img_size[0], self.img_size[1]), interpolation=cv2.INTER_LINEAR)
        target = cv2.resize(target, (self.img_size[0], self.img_size[1]), interpolation=cv2.INTER_NEAREST)

        # Преобразуем в формат CHW для PyTorch
        image = image.transpose(2, 0, 1).astype(np.float32) / 255.0


        if self.transform:
            image, target = self.transform(image, target)
        else:
            image = cv2.resize(
                image.transpose(1, 2, 0), 
                (512, 256), 
                interpolation=cv2.INTER_LINEAR
            ).transpose(2, 0, 1)
            target = cv2.resize(
                target, 
                (512, 256), 
                interpolation=cv2.INTER_NEAREST
            )

        # Нормализация
        mean = np.array([0.485, 0.456, 0.406]).reshape(3,1,1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3,1,1)
        image = (image - mean) / std


        image = torch.from_numpy(image).float()
        target = torch.from_numpy(target).long()

        return image, target



# Аугментации
class CustomAugmentationsNumPy:
    def __init__(
        self,
        img_size=(256, 512),
        p_flip=0.0,            # Вертикальный флип (не используется)
        p_hflip=0.0,           # Горизонтальный флип
        p_brightness=0.0,      # Изменение яркости
        p_noise=0.0,           # Шум
        p_swap_channels=0.0,   # Смена каналов
        p_contrast=0.0,        # Изменение контраста
        p_saturation=0.0,      # Изменение насыщенности
        p_random_crop=0.0,     # вероятность применения random crop
        crop_scale=(0.6, 0.8), # масштаб вырезаемой области относительно исходного размера
        crop_ratio=(0.4, 0.6)  # соотношение сторон вырезаемой области (h/w)

    ):
        """ 
        Аугментации
        """
        self.img_size = img_size
        self.p_flip = p_flip
        self.p_brightness = p_brightness 
        self.p_noise = p_noise
        self.p_swap_channels = p_swap_channels  
        self.p_contrast = p_contrast
        self.p_saturation = p_saturation
        self.p_hflip = p_hflip
        self.p_random_crop = p_random_crop
        self.crop_scale = crop_scale
        self.crop_ratio = crop_ratio


    def _random_resized_crop(self, img_np, mask_np):
        h, w = img_np.shape[1:]  # CHW -> (H, W)

        for attempt in range(10):
            target_area = random.uniform(*self.crop_scale) * h * w
            aspect_ratio = random.uniform(*self.crop_ratio)

            new_w = int(round(np.sqrt(target_area * aspect_ratio)))
            new_h = int(round(np.sqrt(target_area / aspect_ratio)))

            if new_w <= w and new_h <= h:
                x1 = random.randint(0, w - new_w)
                y1 = random.randint(0, h - new_h)

                # Вырезаем область
                img_crop = img_np[:, y1:y1+new_h, x1:x1+new_w]
                mask_crop = mask_np[y1:y1+new_h, x1:x1+new_w]

                # Масштабируем до нужного размера
                img_resized = cv2.resize(
                    img_crop.transpose(1, 2, 0), 
                    (self.img_size[1], self.img_size[0]), 
                    interpolation=cv2.INTER_LINEAR
                ).transpose(2, 0, 1)

                mask_resized = cv2.resize(
                    mask_crop,
                    (self.img_size[1], self.img_size[0]),
                    interpolation=cv2.INTER_NEAREST
                )

                return img_resized, mask_resized

        # Если не удалось подобрать crop, возвращаем исходные с ресайзом
        img_resized = cv2.resize(
            img_np.transpose(1, 2, 0), 
            (self.img_size[1], self.img_size[0]), 
            interpolation=cv2.INTER_LINEAR
        ).transpose(2, 0, 1)

        mask_resized = cv2.resize(
            mask_np,
            (self.img_size[1], self.img_size[0]),
            interpolation=cv2.INTER_NEAREST
        )
        return img_resized, mask_resized



    def _add_noise(self, img_np, std=0.01):
        noise = np.random.normal(0, std, img_np.shape).astype(np.float32)
        img_np = np.clip(img_np + noise, 0, 1)
        return img_np


    def adjust_contrast(self, img_np, factor):
        mean = img_np.mean(axis=(1, 2), keepdims=True)
        img_adj = np.clip((img_np - mean) * factor + mean, 0, 1)
        return img_adj


    def adjust_saturation(self, img_np, factor):
        # img_np: CHW, float32, диапазон [0, 1]
        img_hwc = np.transpose(img_np, (1, 2, 0))
        img_hwc = np.clip(img_hwc, 0, 1)
        img_hsv = cv2.cvtColor((img_hwc * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        img_hsv[..., 1] = np.clip(img_hsv[..., 1] * factor, 0, 255)
        img_hsv = img_hsv.astype(np.uint8)
        img_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
        return np.transpose(img_rgb, (2, 0, 1))


    def __call__(self, img_np, mask_np):
        # Проверка типов и размеров
        assert isinstance(img_np, np.ndarray), f"Image must be numpy array, got {type(img_np)}"
        assert isinstance(mask_np, np.ndarray), f"Mask must be numpy array, got {type(mask_np)}"
        

        if random.random() < self.p_hflip:  # вероятность горизонтального флипа
            img_np = img_np[:, :, ::-1]
            mask_np = mask_np[:, ::-1]

        if random.random() < self.p_noise:  # Добавление шума
            img_np = self._add_noise(img_np)

        if random.random() < self.p_contrast:  # Изменение контраста
            factor = random.uniform(0.5, 3)
            img_np = self.adjust_contrast(img_np, factor)

        if random.random() < self.p_saturation:  # Изменение насыщенности
            factor = random.uniform(0.3, 4)
            img_np = self.adjust_saturation(img_np, factor)
        

        if random.random() < self.p_brightness:   # Изменение яркости
            factor = random.uniform(0.5, 1.5)
            img_np = np.clip(img_np * factor, 0, 1)

        if random.random() < self.p_swap_channels:   # Смена каналов RGB <-> BGR 
            # Для формата CHW (каналы, высота, ширина)
            img_np = img_np[::-1, :, :].copy()  # Инвертирование порядка каналов
 
        if random.random() < self.p_random_crop:    # Применяем random crop
            img_np, mask_np = self._random_resized_crop(img_np, mask_np)


        # Ресайз
        try:
            # Для изображения: CHW -> HWC для OpenCV
            img_resized = cv2.resize(
                img_np.transpose(1, 2, 0), 
                (self.img_size[1], self.img_size[0]),  # (width, height)
                interpolation=cv2.INTER_LINEAR
            ).transpose(2, 0, 1)  # Обратно в CHW
            
            # Для маски: HW формат
            mask_resized = cv2.resize(
                mask_np, 
                (self.img_size[1], self.img_size[0]), 
                interpolation=cv2.INTER_NEAREST
            )
        except Exception as e:
            print(f"Error during resize: {e}")
            print(f"Image shape: {img_np.shape}, Target shape: {mask_np.shape}")
            raise

        return img_resized, mask_resized



# #########################  Создание даталоадеров
def prepare_cityscapes_loaders(dataset_class, root_dir,
                               size,
                               batch_size,
                               num_workers=0,
                                p_hflip=0.2,
                                p_brightness=0.1,
                                p_noise=0.1,
                                p_swap_channels=0.005, 
                                p_contrast=0.1,
                                p_saturation=0.1,
                                p_random_crop=0.2):
    """ 
    Содание даталоадеров 
    """
                            
    # Трансформации для train (с аугментациями)
    train_transforms = CustomAugmentationsNumPy(
        img_size=(256, 512),
        p_flip=0.0,
        p_hflip=p_hflip,
        p_brightness=p_brightness,
        p_noise=p_noise,
        p_swap_channels=p_swap_channels, 
        p_contrast=p_contrast,
        p_saturation=p_saturation,
        p_random_crop=p_random_crop
    )

    # Трансформации для val (без аугментаций, только resize)
    val_transform =  CustomAugmentationsNumPy(
        img_size=(256, 512),
        p_flip=0.0,
        p_hflip=0.0,
        p_brightness=0.0,
        p_noise=0.0,
        p_swap_channels=0.0, 
        p_contrast=0.0,
        p_saturation=0.0,
        p_random_crop=0.0
    )

    # Датасет разделённый на train и Val
    train_dir = os.path.join(root_dir, 'train')
    val_dir = os.path.join(root_dir, 'val')

    train_dataset = dataset_class(root_dir=train_dir, transform=train_transforms)
    val_dataset = dataset_class(root_dir=val_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=num_workers)

    print(f'Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}')
    return train_loader, val_loader, train_dataset, val_dataset




# Разделение датасета
def split_dataset(dataset_dir):
    """ 
    Разделение датасета и сохранение
    """

    images_dir = os.path.join(dataset_dir, 'images')
    masks_dir = os.path.join(dataset_dir, 'targets')
    # Пути к новым папкам для train и val
    output_dir = 'dataset_split'
    train_images_dir = os.path.join(output_dir, 'train', 'images')
    train_masks_dir = os.path.join(output_dir, 'train', 'targets')
    val_images_dir = os.path.join(output_dir, 'val', 'images')
    val_masks_dir = os.path.join(output_dir, 'val', 'targets')

    # Создаем папки, если не существуют
    for folder in [train_images_dir, train_masks_dir, val_images_dir, val_masks_dir]:
        os.makedirs(folder, exist_ok=True)

    # Получаем список всех файлов изображений
    all_images = sorted(os.listdir(images_dir))

    # Фиксируем seed для воспроизводимости
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(all_images))

    # Задаем долю для обучения
    train_ratio = 0.85
    train_size = int(len(all_images) * train_ratio)

    train_indices = shuffled_indices[:train_size]
    val_indices = shuffled_indices[train_size:]

    # Функция копирования пары (изображение + маска)
    def copy_pair(idx, src_img_list):
        img_name = src_img_list[idx]
        mask_name = img_name  # имена масок совпадают с именами изображений

        # Полные пути к исходным файлам
        src_img_path = os.path.join(images_dir, img_name)
        src_mask_path = os.path.join(masks_dir, mask_name)

        # Копируем в train или val в зависимости от индекса
        if idx in train_indices:
            dst_img_path = os.path.join(train_images_dir, img_name)
            dst_mask_path = os.path.join(train_masks_dir, mask_name)
        else:
            dst_img_path = os.path.join(val_images_dir, img_name)
            dst_mask_path = os.path.join(val_masks_dir, mask_name)

        shutil.copy2(src_img_path, dst_img_path)
        shutil.copy2(src_mask_path, dst_mask_path)

    # Копируем все пары
    for i in range(len(all_images)):
        copy_pair(i, all_images)

    print(f"Данные успешно разделены: {train_size} файлов в train, {len(all_images) - train_size} в val.")