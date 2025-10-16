import os
import cv2
import matplotlib.pyplot as plt
import time
import IPython.display as display
import matplotlib.patches as patches
import numpy as np
import torch
import torchvision.transforms.functional as F1
from torchvision import transforms
from torchvision.ops import nms
import torchvision.ops as ops
from ultralytics import YOLO
from PIL import ImageFont, ImageDraw, Image






# ===================================================================
# Функции просмотра изображений с датасета для проверки боксов

# Функция для загрузки аннотаций
def load_annotations(label_file):
    """ 
        Функция для загрузки аннотаций
    """
    with open(label_file, "r") as file:
        lines = file.readlines()
    
    bboxes = []
    for line in lines:
        data = line.strip().split()
        if len(data) != 5:
            continue
        class_id = int(data[0])  # Класс объекта
        x_center, y_center, width, height = map(float, data[1:])
        bboxes.append((class_id, x_center, y_center, width, height))
    return bboxes


# Функция для отображения изображения с боксами
def show_image_with_boxes(image_file, label_file, size_x=10, size_y=10):
    """ 
        Функция для отображения изображения с боксами
    """
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
    
    h, w, _ = image.shape                       # размеры изображения
    
    # аннотации
    bboxes = load_annotations(label_file)
    
    # Рисуем боксы
    for class_id, x_center, y_center, width, height in bboxes:
        x1 = int((x_center - width / 2) * w)
        y1 = int((y_center - height / 2) * h)
        x2 = int((x_center + width / 2) * w)
        y2 = int((y_center + height / 2) * h)

        color = (255, 0, 0)  # цвет боксов
        thickness = 2
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(image, str(class_id), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, color, thickness)

    plt.figure(figsize=(size_x, size_y))
    plt.imshow(image)
    plt.axis("off")
    plt.show()


# Просмотр изображения с боксами
def visualize_img(image_name, images_path, labels_path, size_x=10, size_y=10):
    """ 
        Просмотр изображения с боксами
    """
    # пути к файлам
    image_file = os.path.join(images_path, image_name)
    label_file = os.path.join(labels_path, image_name.replace('.jpg', '.txt').replace('.png', '.txt'))

    print(image_file)
    print(label_file)
    # Проверяем, существует ли файл и аннотация
    if os.path.exists(image_file) and os.path.exists(label_file):
        show_image_with_boxes(image_file, label_file, size_x = size_x, size_y=size_y)
    else:
        print("Файл изображения или аннотации не найден!")


# Просмотр всех изображений с боксами
def visualize_img_full(images_path="dataset/datasets_full/images/train", labels_path="dataset/datasets_full/labels/train", size_x = 10, size_y=10):
    """  
        Просмотр всех изображений с боксами
    """
    # Получаем список файлов изображений
    image_files = sorted([f for f in os.listdir(images_path) if f.endswith(('.jpg', '.png'))])
    
    if image_files:
        for image_name in image_files:
            display.clear_output(wait=True)  # Очистка вывода
            try:
                print(image_name)
                visualize_img(image_name, images_path, labels_path, size_x = size_x, size_y=size_y)                
                time.sleep(4)  # Задержка 4 секунды                
            except FileNotFoundError:
                print(f"Файл {image_name} не найден, пропускаем...")
            # break
    print("Все изображения проверены!")

# =====================================================================

# изуализация предсказаний или разметки heatmap'а формата (предсказание ячеек)
def visualize_sample_heatmap(image, target, threshold=0.5, class_names=None, num_anchors=3, num_classes=8):
    """
    Визуализация предсказаний или разметки heatmap'а формата [B*C, S, S].
    
    image: Tensor [3, H, W]
    target: Tensor [B*C, S, S]
    """
    image = image.permute(1, 2, 0).cpu().numpy()
    H, W = image.shape[:2]
    
    BxC, S, _ = target.shape
    assert BxC == num_anchors * num_classes, "target shape должен быть [B*C, S, S]"

    # Восстановим форму [B, C, S, S]
    target = target.view(num_anchors, num_classes, S, S)

    grid_y = np.linspace(0, H, S + 1, dtype=int)
    grid_x = np.linspace(0, W, S + 1, dtype=int)

    class_cells = []
    for gy in range(S):
        for gx in range(S):
            for a in range(num_anchors):
                for c in range(num_classes):
                    if target[a, c, gy, gx] > threshold:
                        x1 = grid_x[gx]
                        y1 = grid_y[gy]
                        x2 = grid_x[gx + 1]
                        y2 = grid_y[gy + 1]
                        class_cells.append((c, gy, gx, x1, y1, x2 - x1, y2 - y1))

    # 1) Первая фигура: изображение с сеткой и ячейками 
    fig_img, ax_img = plt.subplots(figsize=(12, 12))
    ax_img.imshow(image)
    ax_img.set_title('Изображение с ячейками')

    for x in grid_x:
        ax_img.axvline(x=x, color='white', linestyle='--', linewidth=0.2)
    for y in grid_y:
        ax_img.axhline(y=y, color='white', linestyle='--', linewidth=0.2)

    for c, gy, gx, x1, y1, w, h in class_cells:
        ax_img.add_patch(patches.Rectangle((x1, y1), w, h, edgecolor='lime', linewidth=1, fill=False))
        ax_img.text(x1 + 2, y1 + 12, f"{c if class_names is None else class_names[c]}", color='red', fontsize=7)

    plt.tight_layout()
    plt.show()


        # 2) Вторая фигура: список объектов и ячеек 
    fig_text, ax_text = plt.subplots(figsize=(6, max(3, len(class_cells) * 0.25)))
    ax_text.axis('off')
    ax_text.set_title('Обнаруженные классы и ячейки', fontsize=12, pad=10)

    for idx, (c, gy, gx, *_ ) in enumerate(class_cells):
        name = class_names[c] if class_names else f"Класс {c}"
        ax_text.text(0.01, 1 - idx * 0.05, f"{name} -> Ячейка (Y={gy}, X={gx})",
                     fontsize=10, verticalalignment='top', transform=ax_text.transAxes)


    plt.tight_layout()
    plt.show()


# Отображает изображение с боксами
def visualize_sample(img, target, class_names=None):
    """
    Отображает изображение с боксами.

    img: тензор изображения (C, H, W) или PIL.Image
    target: словарь с 'boxes' и 'labels'
    class_names: список имён классов 
    """
    # тензор, переводим в numpy
    if isinstance(img, torch.Tensor):
        img = F1.to_pil_image(img)

    fig, ax = plt.subplots(1, figsize=(16, 12))
    ax.imshow(img)

    boxes = target['boxes']
    labels = target['labels']

    for box_idx, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = box
        width = xmax - xmin
        height = ymax - ymin

        # Нарисовать боксы
        rect = patches.Rectangle(
            (xmin, ymin),
            width,
            height,
            linewidth=2,
            edgecolor='r',
            facecolor='none'
        )
        ax.add_patch(rect)

        # Класс
        label = labels[box_idx].item()
        if class_names:
            caption = class_names[label]
        else:
            caption = str(label)

        ax.text(xmin, ymin - 5, caption, color='red', fontsize=12, weight='bold')

    plt.axis('off')
    plt.show()





# Отрисовка изображения с предсказанными и истинными боксами.
def plot_image_with_boxes(img_tensor, pred_boxes=None, pred_labels=None, pred_scores=None,
                           target_boxes=None, target_labels=None,
                           class_names=None, score_threshold=0.5, apply_nms=True, iou_threshold=0.4):
    """
    Отрисовка изображения с предсказанными и истинными боксами.

    img_tensor: Tensor изображения [3, H, W]
    pred_boxes: Tensor[N_pred, 4]
    pred_labels: Tensor[N_pred]
    pred_scores: Tensor[N_pred]
    target_boxes: Tensor[N_gt, 4]
    target_labels: Tensor[N_gt]
    class_names: список имён классов
    apply_nms: нужно ли применять NMS к предсказаниям
    iou_threshold: IoU-порог для подавления боксов
    """
    
    img = img_tensor.permute(1, 2, 0).cpu().numpy()  

    if img.max() <= 1.0:
        img = (img * 255).astype('uint8')

    fig, ax = plt.subplots(1, figsize=(16, 12))
    ax.imshow(img)

    # Обработаем предсказания (красные)
    if pred_boxes is not None and len(pred_boxes) > 0:
        # Оставляем только те боксы, у которых score > threshold
        mask = pred_scores >= score_threshold
        pred_boxes = pred_boxes[mask]
        pred_scores = pred_scores[mask]
        if pred_labels is not None:
            pred_labels = pred_labels[mask]

        # Применяем NMS
        if apply_nms and len(pred_boxes) > 0:
            keep = ops.nms(pred_boxes, pred_scores, iou_threshold=iou_threshold)
            pred_boxes = pred_boxes[keep]
            pred_scores = pred_scores[keep]
            if pred_labels is not None:
                pred_labels = pred_labels[keep]

        for i in range(len(pred_boxes)):
            box = pred_boxes[i]
            x_min, y_min, x_max, y_max = box

            rect = patches.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=2,
                edgecolor='red',
                facecolor='none'
            )
            ax.add_patch(rect)

            label = ""
            if pred_labels is not None:
                label_idx = pred_labels[i].item()
                if class_names is not None:
                    label += class_names[label_idx]
                else:
                    label += str(label_idx) + ": "

            if pred_scores is not None:
                label += f" {pred_scores[i]:.2f}"

            if label:
                ax.text(x_min, y_min -5, label, color='red', fontsize=12)

    # таргеты (зелёные)
    if target_boxes is not None and len(target_boxes) > 0:
        for i in range(len(target_boxes)):
            box = target_boxes[i]
            x_min, y_min, x_max, y_max = box

            rect = patches.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=2,
                edgecolor='green',
                facecolor='none',
                linestyle='--'
            )
            ax.add_patch(rect)

            # if target_labels is not None:
            #     label_idx = target_labels[i].item()
            #     label = class_names[label_idx] if class_names is not None else str(label_idx)
            #     ax.text(x_min, y_min - 10, label, color='green', fontsize=12)

    plt.axis('off')
    plt.show()



def inference_model_faster(model, image_path, device, S=16, B=2, C=1, threshold=0.3, iou_thresh=0.3, size = 736):
    model.eval()

 
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(img_tensor)[0]

        pred_boxes = predictions['boxes'].cpu()
        pred_labels = predictions['labels'].cpu()
        pred_scores = predictions['scores'].cpu()

        # Рисуем
        plot_image_with_boxes(
            img_tensor[0].cpu(),
            pred_boxes=pred_boxes,
            pred_labels=pred_labels,
            pred_scores=pred_scores,
            target_boxes=None,
            target_labels=None,
            class_names=None,
            score_threshold=threshold,
            apply_nms=True, 
            iou_threshold=iou_thresh
        )



def video_detection(model_path, video_path, output_video_path, target_classes=None):
    model = YOLO(model_path)

    screen_width = 1000
    screen_height = 1000
    conf = 0.1
    iou = 0.4

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 30, (screen_width, screen_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (screen_width, screen_height))
        results = model.track(frame_resized, persist=True, conf=conf, iou=iou, agnostic_nms=True, verbose=False)

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            labels = result.boxes.cls.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else [None] * len(boxes)

            for box, label, score in zip(boxes, labels, scores):
                class_name = model.names[int(label)]
                
                # Фильтрация по нужным классам
                if target_classes and class_name not in target_classes:
                    continue

                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(frame_resized, f'{class_name} {score:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        out.write(frame_resized)
        cv2.imshow('Frame', frame_resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()