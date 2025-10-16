
import pandas as pd
from IPython.display import display
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
import onnxruntime as ort
import numpy as np
from PIL import Image




######### Очистка памяти
def emty_cache():
    """ 
        Очистка памяти
    """
    gc.collect()  
    torch.cuda.empty_cache()


# Преобразует маску в цветное изображение
def mask_to_rgb(mask, class_palette):
    """Преобразует маску в цветное изображение по палитре Cityscapes."""
    if hasattr(mask, 'cpu'):
        mask = mask.cpu().numpy()
    return class_palette[mask]


# денормализация изображения
def denormalize(image):
    """Денормализует изображение."""
    mean = np.array([0.485, 0.456, 0.406]).reshape(3,1,1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3,1,1)
    return np.clip((image * std) + mean, 0, 1)


def show_images_and_masks(images, masks, class_palette, num=2):
    """
    Выводит num: нормализованное изображение, денормализованное изображение,
    цветная маска и маска в оттенках серого.
    """
    for i in range(min(num, len(images))):
        image = images[i].cpu().numpy() if hasattr(images[i], 'cpu') else images[i]
        mask = masks[i].cpu().numpy() if hasattr(masks[i], 'cpu') else masks[i]
        denorm_image = denormalize(image)
        mask_rgb = mask_to_rgb(mask, class_palette)

        fig, axes = plt.subplots(1, 4, figsize=(22, 7))
        axes[0].imshow(np.transpose(image, (1, 2, 0)))
        axes[0].set_title('Normalized Image')
        axes[0].axis('off')

        axes[1].imshow(np.transpose(denorm_image, (1, 2, 0)))
        axes[1].set_title('Denormalized Image')
        axes[1].axis('off')

        axes[2].imshow(mask_rgb)
        axes[2].set_title('Mask (colored)')
        axes[2].axis('off')

        axes[3].imshow(mask, cmap='gray', vmin=0, vmax=19)
        axes[3].set_title('Mask (gray)')
        axes[3].axis('off')

        plt.tight_layout()
        plt.show()



# #########################  Визуализация изображений
def visualize_image_and_mask(image, mask, class_palette):
    """
    Универсальная визуализация изображения и маски.
    Поддерживает входные изображения из OpenCV (NumPy), PIL, PyTorch Tensor.
    """
    import numpy as np
    import torch
    import matplotlib.pyplot as plt

    #   Обработка изображения 
    if isinstance(image, torch.Tensor):
        # PyTorch Tensor: [C, H, W] -> [H, W, C] + денормализация
        mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(3,1,1)
        image = image * std + mean
        image = image.clamp(0, 1)
        image = image.permute(1, 2, 0).cpu().numpy()
        image = (image * 255).astype(np.uint8)
    elif 'PIL' in str(type(image)):
        # PIL Image -> NumPy
        image = np.array(image)
    elif isinstance(image, np.ndarray):
        # OpenCV: BGR -> RGB
        if image.ndim == 3 and image.shape[2] == 3:
            image = image[..., ::-1]
    else:
        raise TypeError(f"Неподдерживаемый тип изображения: {type(image)}")

    #    Обработка маски 
    if isinstance(mask, torch.Tensor):
        mask_np = mask.cpu().numpy()
    elif isinstance(mask, np.ndarray):
        mask_np = mask
    else:
        # PIL Image
        mask_np = np.array(mask)

    mask_vis = mask_np.copy()
    mask_vis[mask_vis == 255] = 19  
    color_mask = class_palette[mask_vis.squeeze()]

    #     Визуализация 
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(image)
    axs[0].set_title('Image')
    axs[0].axis('off')

    axs[1].imshow(color_mask)
    axs[1].set_title('Mask')
    axs[1].axis('off')

    plt.show()


# #################  Визуализация 
def decode_segmap(mask, colormap):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id in range(len(colormap)):
        color_mask[mask == cls_id] = colormap[cls_id]
    return color_mask


# Функция визуализации
def visualize_segmentation(image_pil, pred_mask, colormap, alpha=0.5, title=None):
    """
    image_pil: PIL.Image — исходное изображение
    pred_mask: torch.Tensor или numpy.ndarray (H, W) с классами
    colormap: dict с цветами классов
    alpha: прозрачность наложения маски
    title: str — общий заголовок для всей фигуры
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Преобразуем PIL Image в numpy (H, W, 3)
    image_np = np.array(image_pil.convert("RGB"))

    # Если pred_mask — тензор, преобразуем в numpy
    if hasattr(pred_mask, 'cpu'):
        pred_mask = pred_mask.cpu().numpy()

    # Убираем лишние размерности, если есть
    if pred_mask.ndim == 3:
        pred_mask = pred_mask.squeeze(0)
    if pred_mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {pred_mask.shape}")

    # Получаем цветную маску
    color_mask = decode_segmap(pred_mask, colormap)
    image_resized = image_pil.resize((512, 256), resample=Image.BILINEAR)
    image_np = np.array(image_resized)
    # Накладываем маску на изображение с прозрачностью
    overlay = (image_np * (1 - alpha) + color_mask * alpha).astype(np.uint8)

    # Визуализация
    plt.figure(figsize=(20, 4))
    if title is not None:
        plt.suptitle(title, fontsize=16)  # Добавляем общий заголовок

    plt.subplot(1, 3, 1)
    plt.title("Исходное изображение")
    plt.imshow(image_np)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Сегментированные маски")
    plt.imshow(color_mask)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Наложение")
    plt.imshow(overlay)
    plt.axis('off')

    plt.show()



# Объединение датасета
def merge_folders(src_root, dst_folder):
    """ 
       Объединение исходного датасета в общий каталог 
    """
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    for split in ['train', 'val']:
        split_path = os.path.join(src_root, split)
        for city in os.listdir(split_path):
            city_path = os.path.join(split_path, city)
            if os.path.isdir(city_path):
                for file_name in os.listdir(city_path):
                    src_file = os.path.join(city_path, file_name)
                    dst_file = os.path.join(dst_folder, file_name)
                    if os.path.exists(dst_file):
                        base, ext = os.path.splitext(file_name)
                        dst_file = os.path.join(dst_folder, f"{split}_{city}_{base}{ext}")
                    shutil.copy2(src_file, dst_file)



# вывод результатов
def load_and_display_metrics_with_highlight(filename='results.csv'):
    """
    Загружает таблицу метрик из CSV и выводит её с выделением максимального avg_iou.

    Args:
        filename (str): путь к CSV-файлу с метриками (по умолчанию 'results.csv')

    Returns:
        pd.DataFrame или None: загруженный DataFrame, либо None если файл не найден
    """
    if os.path.exists(filename):
        df = pd.read_csv(filename, encoding='utf-8')
        # Выделяем максимальное значение в столбце 'avg_iou'
        styled_df = df.style.highlight_max(subset=['avg_iou'], color='green')
        display(styled_df)
        return df
    else:
        print(f"Файл '{filename}' не найден.")
        return None


# Заполнение таблицы результатами 
def save_metrics_to_csv(comment, model_name, avg_loss, avg_acc, avg_iou, filename='results.csv'):
    """
    Загружает существующий CSV (если есть), добавляет новую строку с метриками и сохраняет обратно.

    Args:
        comment (str): комментарий к эксперименту
        model_name (str): название модели
        avg_loss (float): среднее значение loss
        avg_acc (float): средняя точность (accuracy)
        avg_iou (float): среднее значение IoU
        filename (str): имя файла для сохранения (по умолчанию 'results.csv')
    """
    # Создаем новую запись в виде DataFrame
    new_row = pd.DataFrame([{
        'Комментарий': comment,
        'Модель': model_name,
        'avg_loss': avg_loss,
        'avg_acc': avg_acc,
        'avg_iou': avg_iou
    }])

    # Проверяем, существует ли файл
    if os.path.exists(filename):
        # Загружаем существующий файл
        df = pd.read_csv(filename, encoding='utf-8')
        # Добавляем новую строку
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        # Если файла нет, создаём новый DataFrame
        df = new_row

    # Сохраняем DataFrame обратно в CSV
    df.to_csv(filename, index=False, encoding='utf-8')

