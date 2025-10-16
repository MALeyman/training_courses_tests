""" 
Автор: Лейман Максим  

Дата создания: 18.06.2025
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
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import gc
import cv2
import torchvision.transforms as T
import matplotlib.patches as patches
from ultralytics import YOLO
from IPython.display import clear_output



######### Очистка памяти
def empty_cache():
    """ 
        Очистка памяти
    """
    gc.collect()  
    torch.cuda.empty_cache()


# Загружает изображение, модель и  выводит изображение с боксами
def detect_and_visualize(image_path, label_path=None, model=None, class_names=None):
    """
    Универсальная функция для загрузки изображения, 
    отображения таргетов (если label_path), 
    предсказаний (если model) и их совмещения.

    Args:
        image_path (str): путь к изображению
        label_path (str, optional): путь к аннотациям в формате YOLO
        model (ultralytics.YOLO, optional): загруженная модель YOLO
        class_names (list[str], optional): имена классов (если None, берутся из модели или default)
    """
    # Загрузка и подготовка изображения
    img_cv = cv2.imread(image_path)
    if img_cv is None:
        raise FileNotFoundError(f"Изображение не найдено: {image_path}")
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    transform = T.ToTensor()
    image_tensor = transform(img_rgb)

    # Загрузка таргетов из label_path (если есть)
    target = None
    if label_path is not None and os.path.exists(label_path):
        h, w, _ = img_rgb.shape
        boxes = []
        labels = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_id = int(parts[0])
                x_center, y_center, bw, bh = map(float, parts[1:5])
                x1 = (x_center - bw / 2) * w
                y1 = (y_center - bh / 2) * h
                x2 = (x_center + bw / 2) * w
                y2 = (y_center + bh / 2) * h
                boxes.append([x1, y1, x2, y2])
                labels.append(class_id)
        target = {'boxes': boxes, 'labels': labels}

    # Получение предсказаний из модели (если есть)
    pred = None
    if model is not None:
        results = model(image_path)
        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy().tolist()
        labels = result.boxes.cls.cpu().numpy().astype(int).tolist()
        pred = {'boxes': boxes, 'labels': labels}
        # Если class_names не переданы, берем из модели
        if class_names is None and hasattr(model, 'names'):
            class_names = model.names

    # Если class_names не заданы, ставим дефолт
    if class_names is None:
        class_names = ['class_' + str(i) for i in range(100)]

    # Визуализация
    visualize_image_with_boxes(image_tensor, target=target, pred=pred, class_names=class_names)



#  Просмотр предсказанных точек (изображение загружается по пути)
def show_image_with_predictions(path_image, model, device):
    """
    Загружает изображение предсказывает точки и Визуализирует изображение и предсказанные ключевые точки.
    img_tensor: torch.Tensor [C, H, W] или [H, W, C], значения 0...1 или 0...255
    predicted_keypoints: np.array shape [N, 2] или torch.Tensor [N, 2]
    """

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
    transforms.Resize((224, 224)),
        transforms.ToTensor(),    # [0,1]
        # transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    # Загрузка и подготовка изображения
    orig_img = Image.open(path_image).convert('RGB')

    img_tensor = transform(orig_img)  # [C, H, W]
    img_tensor = img_tensor.to(device) 
    print(img_tensor.shape)
    # Предсказание
    model.eval()
    with torch.no_grad():
        output = model(img_tensor.unsqueeze(0))  # [1, N*2]
    predicted_keypoints = output.cpu().numpy().reshape(-1, 2)  # [[x1, y1], [x2, y2], ...]

    # Приводим изображение к numpy и нужному формату
    if isinstance(img_tensor, torch.Tensor):
        if img_tensor.ndim == 3 and img_tensor.shape[0] in [1, 3]:
            img = img_tensor.permute(1, 2, 0).cpu().numpy()
        else:
            img = img_tensor.cpu().numpy()
    else:
        img = img_tensor

    # Если изображение нормализовано (0...1), переводим в 0...255 для корректного отображения
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)
   
    # Приводим точки к numpy
    if isinstance(predicted_keypoints, torch.Tensor):
        predicted_keypoints = predicted_keypoints.cpu().numpy()

    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.scatter(predicted_keypoints[:, 0], predicted_keypoints[:, 1], c='lime', s=10, label='Predicted')
    plt.axis('off')
    plt.title('Predicted Keypoints')
    plt.legend()
    plt.show()


# Просмотр изображения из батча
def show_batch_with_keypoints(batch, batch_idx=0):
    """
    Визуализирует изображение и ключевые точки из батча DataLoader.
    batch: батч, полученный из DataLoader (например, next(iter(dataloader)))
    batch_idx: индекс изображения в батче для отображения
    """
    images = batch['image']  # тензор [B, C, H, W]
    keypoints = batch['keypoints']  # список numpy массивов или тензоров разной длины

    # Преобразуем изображение в numpy и поменяем порядок осей для matplotlib
    if images.ndim == 4 and images.shape[1] in [1, 3]:
        images = images.permute(0, 2, 3, 1).cpu().numpy()
    else:
        images = images.cpu().numpy()

    img = images[batch_idx]

    # Ключевые точки могут быть numpy массивом или тензором
    kps = keypoints[batch_idx]
    if hasattr(kps, 'numpy'):
        kps = kps.numpy()

    # Если изображение одноканальное, убираем последний размер
    if img.shape[-1] == 1:
        img = img.squeeze(-1)

    plt.figure(figsize=(6, 6))
    plt.imshow(img.astype(np.uint8) if img.max() > 1.5 else img, cmap='gray')
    plt.scatter(kps[:, 0], kps[:, 1], c='r', s=20)
    plt.axis('off')
    plt.title('Изображение с Keypoints')
    plt.show()


# Визуализирует изображение и ключевые точки из батча DataLoader
def show_batch_with_keypoints2(batch, batch_idx=0):
    """
    Визуализирует изображение и ключевые точки из батча DataLoader.
    batch: батч, полученный из DataLoader (например, next(iter(dataloader)))
    batch_idx: индекс изображения в батче для отображения
    """
    # Получаем изображение и ключевые точки
    images = batch['image']  # shape: [B, H, W, C] или [B, C, H, W]
    keypoints = batch['keypoints']  # shape: [B, N, 2]

    # Если изображение в формате [B, C, H, W], переводим в [B, H, W, C]
    if images.ndim == 4 and images.shape[1] in [1, 3]:
        images = images.permute(0, 2, 3, 1).numpy()
    else:
        images = images.numpy()

    img = images[batch_idx]
    kps = keypoints[batch_idx].numpy()

    # Если изображение одноканальное, убираем последний размер
    if img.shape[-1] == 1:
        img = img.squeeze(-1)
    # print(img)
    plt.figure(figsize=(6, 6))
    plt.imshow(img.astype(np.uint8) if img.max() > 1.5 else img, cmap='gray')
    plt.scatter(kps[:, 0], kps[:, 1], c='r', s=20)
    plt.axis('off')
    plt.title('Изображение с Keypoints')
    plt.show()


# Функция просмотра датасета с аннотациями
def img_show(path_image='dataset/dataset_1/images', path_annotation='dataset/dataset_1/training_1.csv', idx=0):
    """ 
         Функция просмотра датасета с аннотациями

    """

    # Загрузка аннотаций
    df = pd.read_csv(path_annotation)
    # Получаем имя файла изображения из первого столбца
    image_name = df.loc[idx, 'image_name']
    image_path = os.path.join(path_image, image_name)

    # Загружаем изображение через PIL
    image = np.array(Image.open(image_path).convert('RGB'))
    # Получаем координаты ключевых точек (все столбцы, кроме image_name)
    keypoint_cols = [col for col in df.columns if col != 'image_name']
    keypoints = df.loc[idx, keypoint_cols].values.astype(float).reshape(-1, 2)

    # Создаем фигуру с двумя подграфиками
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    # Слева — оригинальное изображение
    axs[0].imshow(image)
    axs[0].set_title('Оригинал')
    axs[0].axis('off')

    # Справа — изображение с наложенными ключевыми точками
    axs[1].imshow(image)
    axs[1].scatter(keypoints[:, 0], keypoints[:, 1], c='r', s=10)
    axs[1].set_title('С ключевыми точками')
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()


#  Вывод изображения в реальных  пропорциях
def show_original_image_with_predictions(path_image, model, device, model_input_size=(224, 224)):
    """
    Вывод изображения в реальных  пропорциях
    orig_img: np.ndarray или PIL.Image (исходное изображение, [H, W, C])
    predicted_keypoints: np.ndarray [N, 2] — координаты в масштабе модели (например, 224x224)
    model_input_size: tuple (ширина, высота), в каком размере работала модель
    """

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
    transforms.Resize((224, 224)),
        transforms.ToTensor(),    # [0,1]
        # transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    # # Загрузка и подготовка изображения
    orig_img = Image.open(path_image).convert('RGB')

    img_tensor = transform(orig_img)  # [C, H, W]
    img_tensor = img_tensor.to(device) 
    print(img_tensor.shape)
    # Предсказание
    model.eval()
    with torch.no_grad():
        print(img_tensor.unsqueeze(0).shape)
        output = model(img_tensor.unsqueeze(0))  # [1, N*2]
    predicted_keypoints = output.cpu().numpy().reshape(-1, 2)  # [[x1, y1], [x2, y2], ...]


    # Получаем размер исходного изображения
    if hasattr(orig_img, 'size'):  # PIL.Image
        orig_w, orig_h = orig_img.size
        img = np.array(orig_img)
    else:  # np.ndarray
        orig_h, orig_w = orig_img.shape[:2]
        img = orig_img

    # пересчитаем  в исходный размер
    model_w, model_h = model_input_size
    scale_x = orig_w / model_w
    scale_y = orig_h / model_h
    keypoints_orig = predicted_keypoints.copy()
    keypoints_orig[:, 0] *= scale_x
    keypoints_orig[:, 1] *= scale_y

    plt.figure(figsize=(8,  8))  # масштаб
    plt.imshow(img)
    plt.scatter(keypoints_orig[:, 0], keypoints_orig[:, 1], c='lime', s=10, label='Predicted')
    plt.axis('off')
    plt.gca().set_aspect('auto')  # пропорции исходного изображения
    plt.tight_layout(pad=0)
    plt.show()


# Просмотр видео с боксами
#  Ресайзит кадр для вывода и сохранения с сохранением пропорций
def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    """ 
         Ресайзит кадр для вывода и сохранения с сохранением пропорций
    """
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


# просмотр и сохранение видео с предсказанными боксами
def show_and_save_video_with_boxes(video_path, model, output_path, frame_step=1, target_width=None, target_height=None, window_name='Video with Boxes'):
    """  
        Просмотр и сохранение видео с предсказанными боксами
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Не удалось открыть видео {video_path}")

    # Получаем исходные размеры и FPS
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Вычисляем размеры для записи с сохранением пропорций
    if target_width is not None and target_height is None:
        r = target_width / orig_width
        out_dim = (target_width, int(orig_height * r))
    elif target_height is not None and target_width is None:
        r = target_height / orig_height
        out_dim = (int(orig_width * r), target_height)
    elif target_width is not None and target_height is not None:
        out_dim = (target_width, target_height)
    else:
        out_dim = (orig_width, orig_height)

    # Создаем VideoWriter для сохранения видео
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    out = cv2.VideoWriter(output_path, fourcc, fps, out_dim)

    class_names = model.names if hasattr(model, 'names') else None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            # Ресайз кадра для модели 
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Инференс модели (без вывода в консоль)
            results = model(img_rgb, verbose=False)
            result = results[0]

            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            labels = result.boxes.cls.cpu().numpy().astype(int)

            # Рисуем боксы на исходном кадре
            for box, label in zip(boxes, labels):
                x1, y1, x2, y2 = box
                color = (0, 255, 0)  # зеленый
                color_text = (0, 0, 255)  # красный для текста
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                if class_names is not None and label < len(class_names):
                    text = class_names[label]
                else:
                    text = str(label)
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_text, 2)

            # Ресайзим кадр для вывода и сохранения с сохранением пропорций
            frame_resized = resize_with_aspect_ratio(frame, width=out_dim[0], height=out_dim[1])

            # Показываем кадр
            cv2.imshow(window_name, frame_resized)

            # Записываем кадр в файл
            out.write(frame_resized)

            # Выход по нажатию 'q'
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

            fps = cap.get(cv2.CAP_PROP_FPS)
            delay = int(1000 / fps) if fps > 0 else 30  # задержка в мс
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break

        frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Видео сохранено в {output_path}")


def visualize_image_with_boxes(image_tensor, target=None, pred=None, class_names=None):
    """ 
        Визуализация изображения с боксами
    """
    if image_tensor.ndim == 4:
        image_tensor = image_tensor[0]
    image_np = image_tensor.detach().cpu().numpy()
    if image_np.shape[0] == 1:
        image_np = np.repeat(image_np, 3, axis=0)
    image_np = np.transpose(image_np, (1, 2, 0))
    if image_np.max() <= 1.0:
        image_np = (image_np * 255).astype(np.uint8)
    else:
        image_np = image_np.astype(np.uint8)

    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image_np)


    def draw_boxes(boxes, labels=None, color='g'):
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            if labels is not None:
                label = labels[i]
                if class_names is not None and label < len(class_names):
                    label_text = ""
                else:
                    label_text = str(label)
                ax.text(x1, y1 - 5, label_text, color=color, fontsize=12, weight='bold')

    if target is not None:
        boxes = target.get('boxes', [])
        labels = target.get('labels', None)
        if boxes:
            draw_boxes(boxes, labels, color='g')
    if pred is not None:
        boxes = pred.get('boxes', [])
        labels = pred.get('labels', None)
        if boxes:
            draw_boxes(boxes, labels, color='r')

    plt.axis('off')
    plt.show()


def crop_video(video_path, model, frame_step=1, visualize=False):
    """
    Читает видео, подаёт кадры на модель, выводит/сохраняет предсказания.

    Args:
        video_path (str): путь к видеофайлу
        model (ultralytics.YOLO): загруженная модель YOLOv8
        frame_step (int): обрабатывать каждый n-й кадр
        visualize (bool): показывать кадры с предсказаниями
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Не удалось открыть видео {video_path}")

    transform = T.ToTensor()
    class_names = model.names if hasattr(model, 'names') else None

    image_tensors = []
    preds = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            # OpenCV читает в BGR, перевод в RGB
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_tensor = transform(img_rgb)
            print(img_rgb.shape)
            # Инференс модели
            results = model(img_rgb)
            result = results[0]

            boxes = result.boxes.xyxy.cpu().numpy().tolist()
            labels = result.boxes.cls.cpu().numpy().astype(int).tolist()
            pred = {'boxes': boxes, 'labels': labels}

            if visualize:
                visualize_image_with_boxes(image_tensor, pred=pred, class_names=class_names)

            image_tensors.append(image_tensor)
            preds.append(pred)
        frame_idx += 1

    cap.release()
    print("Обработка видео завершена.")

    return image_tensors, preds


# Преобразование в тензор для подачи в модель
def preprocess_face_roi(face_roi, target_size=(224, 224), device='cpu'):
    """ 
        Преобразование в тензор для подачи в модель
    """
    face_resized = cv2.resize(face_roi, target_size)
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    face_tensor = torch.from_numpy(face_rgb).permute(2, 0, 1).float() / 255.0 
    face_tensor = face_tensor.unsqueeze(0) 
    face_tensor = face_tensor.to(device)    
    return face_tensor


# Детектит ключевые точки на изображении
def detect_keypoints_on_faces(frame, boxes, model_keypoint, target_size=(224, 224), device='cpu'):
    """ 
        Детектит ключевые точки на изображении
    """
    keypoints_all = []
    h_frame, w_frame = frame.shape[:2]
  

    for box in boxes:
        x1, y1, x2, y2 = box.astype(int)
        print("координаты бокса 0:", x1, y1, x2, y2)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w_frame - 1, x2)
        y2 = min(h_frame - 1, y2)

        print("координаты бокса", x1, y1, x2, y2)

        face_roi = frame[y1:y2, x1:x2]
        if face_roi.size == 0:
            keypoints_all.append(None)
            continue

        face_tensor = preprocess_face_roi(face_roi, target_size, device)
        # print(face_tensor.shape)
        with torch.no_grad():
            preds = model_keypoint(face_tensor)
            # print(preds)
        preds = preds.squeeze(0).cpu().numpy().reshape(-1, 2)
        # print(preds)
        h_roi, w_roi = face_roi.shape[:2]
        # print("До масштабирования:", preds.min(), preds.max())

        # preds[:, 0] *= w_roi
        # preds[:, 1] *= h_roi

        # preds[:, 0] += x1
        # preds[:, 1] += y1
        # print(preds)
        # print("После масштабирования:", preds.min(), preds.max())

        keypoints_all.append(preds)

    return keypoints_all



# Просмотр изображения с точками (ориг. пропорции)
def show_keypoints_on_original(image_np, keypoints_pred, model_input_size=(224, 224)):
    """ 
    Просмотр изображения с ключевыми точками с оригинальными пропорциями
        Args:
        image_np (numpy.ndarray): изображение 
        keypoints_pred (numpy.ndarray):  предсказанные точки
    """
    orig_h, orig_w = image_np.shape[:2]
    model_w, model_h = model_input_size

    scale_x = orig_w / model_w
    scale_y = orig_h / model_h

    keypoints_orig = keypoints_pred.copy()
    keypoints_orig[:, 0] *= scale_x
    keypoints_orig[:, 1] *= scale_y

    plt.imshow(image_np)
    plt.axis('off')
    plt.scatter(keypoints_orig[:, 0], keypoints_orig[:, 1], c='r', s=10)
    plt.show()



def rescale_keypoints(keypoints, orig_size, new_size):
    H_orig, W_orig = orig_size
    H_new, W_new = new_size
    keypoints = np.array(keypoints)
    keypoints[:, 0] = keypoints[:, 0] * (W_orig / W_new)  # x
    keypoints[:, 1] = keypoints[:, 1] * (H_orig / H_new)  # y
    return keypoints

# Просмотр изображения с точками 
def show_image_with_keypoints_rescaled(image_np, keypoints, new_size=(224, 224)):
    """ 
    Просмотр изображения с ключевыми точками 
        Args:
        image_np (numpy.ndarray): изображение 
        keypoints_pred (numpy.ndarray):  предсказанные точки
    """
    print("размер изображения", image_np.shape)
    print("максимум значений точек", keypoints.max())
    
    orig_size = image_np.shape[:2]  # (H, W)
    keypoints_rescaled = rescale_keypoints(keypoints, orig_size, new_size)
    print("новый размер изображения", image_np.shape)
    print("максимум значений точек", keypoints_rescaled.max())
    print(orig_size, new_size)
    plt.imshow(image_np)
    plt.axis('off')
    plt.scatter(keypoints_rescaled[:, 0], keypoints_rescaled[:, 1], c='r', s=10)
    plt.show()



# Просмотр изображения с ключевыми точками
def show_image_with_keypoints_tensor(image_tensor, keypoints):
    """
    Отобразить изображение с ключевыми точками.

    Args:
        image_tensor (torch.Tensor): тензор изображения с формой (1, 3, H, W)
        keypoints (array-like): координаты точек в формате [(x1, y1), (x2, y2), ...]
    """
    print("размер тензора", image_tensor.shape)
    
    # Убираем batch размерность
    img = image_tensor.squeeze(0)  # (3, H, W)
    
    # Переставляем оси в (H, W, C)
    img = img.permute(1, 2, 0).cpu().numpy()
    
    # Если значения не в диапазоне [0,1], нормализуем (пример)
    if img.max() > 1.0:
        img = img / 255.0
    
    print("максимум значений точек", np.max(keypoints))
    
    plt.imshow(img)
    plt.axis('off')

    x_coords = [pt[0] for pt in keypoints]
    y_coords = [pt[1] for pt in keypoints]

    plt.scatter(x_coords, y_coords, c='r', s=10)
    plt.show()


def prepare_image_for_model(image_np, transform):
    if image_np.dtype != np.uint8:
        image_np = (image_np * 255).astype(np.uint8)
    img_pil = Image.fromarray(image_np)
    image_tensor = transform(img_pil)  
    image_tensor = image_tensor.unsqueeze(0)  
    return image_tensor


# Вырезает боксы из изображения с отступами
def boxes_intersect(box1, box2):
    # box = (xmin, ymin, xmax, ymax)
    return not (box1[2] <= box2[0] or box1[0] >= box2[2] or box1[3] <= box2[1] or box1[1] >= box2[3])


def crop_boxes_on_frame_with_margin_no_overlap(image_tensors, preds, margin=10):
    """ 
    Вырезает боксы из изображения с отступами
    """
    out_tensor = []

    for idx in range(len(image_tensors)):
        clear_output(wait=True)
        image_tensor, pred = image_tensors[idx], preds[idx]
        visualize_image_with_boxes(image_tensor, pred=pred, class_names=None)
        print(image_tensor.shape)

        image_tensor_list = []
        boxes_list = []
        cropped_image_list = []
        print(type(image_tensor))
        _, H, W = image_tensor.shape

        # Получаем список исходных боксов (без расширения)
        original_boxes = [tuple(map(int, box)) for box in pred['boxes']]

        for idy, box in enumerate(original_boxes):
            xmin, ymin, xmax, ymax = box

            # Пробуем расширить бокс с запасом
            xmin_exp = max(xmin - margin, 0)
            ymin_exp = max(ymin - margin, 0)
            xmax_exp = min(xmax + margin, W)
            ymax_exp = min(ymax + margin, H)

            expanded_box = (xmin_exp, ymin_exp, xmax_exp, ymax_exp)

            # Проверяем пересечение расширенного бокса с другими исходными боксами (кроме текущего)
            overlap = False
            for j, other_box in enumerate(original_boxes):
                if j == idy:
                    continue
                if boxes_intersect(expanded_box, other_box):
                    overlap = True
                    break

            # Если есть пересечение — не расширяем бокс
            if overlap:
                xmin_exp, ymin_exp, xmax_exp, ymax_exp = xmin, ymin, xmax, ymax

            cropped_image = image_tensor[:, ymin_exp:ymax_exp, xmin_exp:xmax_exp]

            image_tensor_list.append(image_tensor)
            boxes_list.append((xmin_exp, ymin_exp, xmax_exp, ymax_exp))
            cropped_image_list.append(cropped_image)

        out_tensor.append({
            'image': image_tensor_list,
            'boxes': boxes_list,
            'crop_images': cropped_image_list
        })

    return out_tensor


def boxes_intersect(box1, box2):
    # box = (xmin, ymin, xmax, ymax)
    return not (box1[2] <= box2[0] or box1[0] >= box2[2] or box1[3] <= box2[1] or box1[1] >= box2[3])

def crop_boxes_on_frame_with_margin_no_overlap_2(image_tensor, original_boxes, margin=10):
    """ 
    Вырезает боксы из изображения с отступами
    """

    out_tensor = []
    image_tensor_list = []
    boxes_list = []
    cropped_image_list = []

    _, H, W = image_tensor.shape

    for idy, box in enumerate(original_boxes):
        xmin, ymin, xmax, ymax = map(int, box)

        # Пробуем расширить бокс с запасом
        xmin_exp = int(max(xmin - margin, 0))
        ymin_exp = int(max(ymin - margin, 0))
        xmax_exp = int(min(xmax + margin, W))
        ymax_exp = int(min(ymax + margin, H))

        expanded_box = (xmin_exp, ymin_exp, xmax_exp, ymax_exp)

        # Проверяем пересечение расширенного бокса с другими исходными боксами (кроме текущего)
        overlap = False
        for j, other_box in enumerate(original_boxes):
            if j == idy:
                continue
            if boxes_intersect(expanded_box, other_box):
                overlap = True
                break

        # Если есть пересечение — не расширяем бокс
        if overlap:
            xmin_exp, ymin_exp, xmax_exp, ymax_exp = xmin, ymin, xmax, ymax


        cropped_image = image_tensor[:, int(ymin_exp):int(ymax_exp), int(xmin_exp):int(xmax_exp)]


        image_tensor_list.append(image_tensor)
        boxes_list.append((xmin_exp, ymin_exp, xmax_exp, ymax_exp))
        cropped_image_list.append(cropped_image)


    return cropped_image_list




def process_video_with_face_keypoints(video_path, model_detect, model_keypoint, output_path,
                                     frame_step=1, target_size=(224, 224), window_name='Face Keypoints', device='cpu'):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Не удалось открыть видео {video_path}")

    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (orig_width, orig_height))

    frame_idx = 0
    class_names = model_detect.names if hasattr(model_detect, 'names') else None

    # Переносим модели на устройство
    model_detect.to(device)
    model_keypoint.to(device)
    model_detect.eval()
    model_keypoint.eval()

    transform_1 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),    # [0,1]
    ])

    transform = transforms.ToTensor()
    transform_1 = transforms.Resize(target_size)
    while True:
        ret, frame = cap.read() # считытвает текущий кадр: frame - кадр, ret - результат выполнения
        if not ret:
            break

        if frame_idx % frame_step == 0:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # transform = T.ToTensor()

            image_tensor = transform(img_rgb)
            # Детекция лиц
            results = model_detect(img_rgb, verbose=False)
            result = results[0]                     # Результат
            boxes = result.boxes.xyxy.cpu().numpy() # координаты боксов в xyxy
            pred = {'boxes': boxes}


            # Вырежем боксы из изображения (кадра)
            out_tensor = crop_boxes_on_frame_with_margin_no_overlap_2(image_tensor, boxes)

            # print(out_tensor)
            keypoints_all = []
            for idy in range(len(boxes)):
                # clear_output(wait=True)

                cropped_img = out_tensor[idy]

                input_tensor = cropped_img.to(device).unsqueeze(0)
                input_tensor = transform_1(input_tensor)
              
                model_keypoint = model_keypoint.to(device=device)
                with torch.no_grad():
                    result_point = model_keypoint(input_tensor)
                predicted_keypoints = result_point.cpu().numpy().reshape(-1, 2)
                    
                # Преобразование координат точек в систему исходного изображения
                x1, y1, x2, y2 = boxes[idy].astype(int)
                w, h = x2 - x1, y2 - y1

                # Масштабируем точки к размеру исходного бокса

                scaled_points = predicted_keypoints.copy()
                scaled_points[:, 0] = scaled_points[:, 0] * (w / 224) + x1
                scaled_points[:, 1] = scaled_points[:, 1] * (h / 224) + y1

                keypoints_all.append(scaled_points)     
       
            # Рисуем боксы и ключевые точки
            for box, keypoints in zip(boxes, keypoints_all):
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

                if keypoints is not None:
                    for (x, y) in keypoints.astype(int):
                        cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

            cv2.imshow(window_name, frame)
            out.write(frame)

            if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                break

        frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Видео сохранено в {output_path}")



# Вырезает боксы из изображение без отступов
def crop_boxes_on_frame(image_tensors, preds):
    """ 
    Вырезает боксы из изображение  без отступов
    """
    out_tensor = []
    for idx in range(len(image_tensors)):
        # Проходим по всем кадрам из видео 
        # idx = 1
        clear_output(wait=True)
        image_tensor, pred = image_tensors[idx], preds[idx]
        visualize_image_with_boxes(image_tensor, pred=pred, class_names=None)
        print(image_tensor.shape)
        image_tensor_list = []
        boxes_list = []
        cropped_image_list = []
        for idy in range(len(pred['boxes'])):
            # Вырезаем из текущего кадра  лицо по боксу
            box = pred['boxes'][idy]  # если pred - словарь с ключом 'boxes', или просто pred если это тензор
            print(box)

            xmin, ymin, xmax, ymax = map(int, box)  # переводим в int и на CPU

            # Вырезаем область из тензора (C, H, W)
            cropped_image = image_tensor[:, ymin:ymax, xmin:xmax]
            
            print(cropped_image.shape)

            image_tensor_list.append(image_tensor)
            boxes_list.append(box)
            cropped_image_list.append(cropped_image)
       

        out_tensor.append({
            'image': image_tensor_list,
            'boxes': boxes_list,
            'crop_images': cropped_image_list
        })

    return out_tensor











