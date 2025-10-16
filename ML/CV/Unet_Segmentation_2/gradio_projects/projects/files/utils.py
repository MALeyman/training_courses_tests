"""  

"""

from torchvision.ops import nms

from typing import Union, Optional, Any, Dict, List, Tuple
import albumentations as A
from itertools import product
from math import ceil
import numpy as np
import os
import torch
from torch.nn import functional as F
import cv2
import onnxruntime
from projects.common.session import ort_session, ort_session_2
import subprocess
import tempfile


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_device():
    providers = onnxruntime.get_available_providers()
    if 'CUDAExecutionProvider' in providers:
        return "Устройство: GPU (CUDA)"
    else:
        return "Устройство: CPU"




def decode_landm(
    pre: torch.Tensor, priors: torch.Tensor, variances: Union[List[float], Tuple[float, float]]
) -> torch.Tensor:
    """Decode landmarks from predictions using priors to undo the encoding we did for offset regression at train time.
    Args:
        pre: landmark predictions for loc layers,
            Shape: [num_priors, 10]
        priors: Prior boxes in center-offset form.
            Shape: [num_priors, 4].
        variances: Variances of priorboxes
    Return:
        decoded landmark predictions
    """
    return torch.cat(
        (
            priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
        ),
        dim=1,
    )


def decode(
    loc: torch.Tensor, priors: torch.Tensor, variances: Union[List[float], Tuple[float, float]]
) -> torch.Tensor:
    """Decode locations from predictions using priors to undo the encoding we did for offset regression at train time.
    Args:
        loc: location predictions for loc layers,
            Shape: [num_priors, 4]
        priors: Prior boxes in center-offset form.
            Shape: [num_priors, 4].
        variances: Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat(
        (
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1]),
        ),
        1,
    )
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def convert_to_h264(input_path, output_path):
    command = [
        'ffmpeg',
        '-y',  # перезаписать выходной файл
        '-i', input_path,
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        output_path
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)


def prior_box1(min_sizes, steps, clip, image_size):
    feature_maps = [[ceil(image_size[0] / step), ceil(image_size[1] / step)] for step in steps]

    anchors = []
    for k, f in enumerate(feature_maps):
        t_min_sizes = min_sizes[k]
        for i, j in product(range(f[0]), range(f[1])):
            for min_size in t_min_sizes:
                s_kx = min_size / image_size[1]
                s_ky = min_size / image_size[0]
                dense_cx = [x * steps[k] / image_size[1] for x in [j + 0.5]]
                dense_cy = [y * steps[k] / image_size[0] for y in [i + 0.5]]
                for cy, cx in product(dense_cy, dense_cx):
                    anchors += [cx, cy, s_kx, s_ky]

    output = torch.Tensor(anchors).view(-1, 4)
    if clip:
        output.clamp_(max=1, min=0)
    return output


def pad_to_size(
    target_size: Tuple[int, int],
    image: np.array,
    bboxes: Optional[np.ndarray] = None,
    keypoints: Optional[np.ndarray] = None,
) -> Dict[str, Union[np.ndarray, Tuple[int, int, int, int]]]:
    """Pads the image on the sides to the target_size

    Args:
        target_size: (target_height, target_width)
        image:
        bboxes: np.array with shape (num_boxes, 4). Each row: [x_min, y_min, x_max, y_max]
        keypoints: np.array with shape (num_keypoints, 2), each row: [x, y]

    Returns:
        {
            "image": padded_image,
            "pads": (x_min_pad, y_min_pad, x_max_pad, y_max_pad),
            "bboxes": shifted_boxes,
            "keypoints": shifted_keypoints
        }

    """
    target_height, target_width = target_size

    image_height, image_width = image.shape[:2]

    if target_width < image_width:
        raise ValueError(f"Target width should bigger than image_width" f"We got {target_width} {image_width}")

    if target_height < image_height:
        raise ValueError(f"Target height should bigger than image_height" f"We got {target_height} {image_height}")

    if image_height == target_height:
        y_min_pad = 0
        y_max_pad = 0
    else:
        y_pad = target_height - image_height
        y_min_pad = y_pad // 2
        y_max_pad = y_pad - y_min_pad

    if image_width == target_width:
        x_min_pad = 0
        x_max_pad = 0
    else:
        x_pad = target_width - image_width
        x_min_pad = x_pad // 2
        x_max_pad = x_pad - x_min_pad

    result = {
        "pads": (x_min_pad, y_min_pad, x_max_pad, y_max_pad),
        "image": cv2.copyMakeBorder(image, y_min_pad, y_max_pad, x_min_pad, x_max_pad, cv2.BORDER_CONSTANT),
    }

    if bboxes is not None:
        bboxes[:, 0] += x_min_pad
        bboxes[:, 1] += y_min_pad
        bboxes[:, 2] += x_min_pad
        bboxes[:, 3] += y_min_pad

        result["bboxes"] = bboxes

    if keypoints is not None:
        keypoints[:, 0] += x_min_pad
        keypoints[:, 1] += y_min_pad

        result["keypoints"] = keypoints

    return result


def unpad_from_size(
    pads: Tuple[int, int, int, int],
    image: Optional[np.array] = None,
    bboxes: Optional[np.ndarray] = None,
    keypoints: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Crops patch from the center so that sides are equal to pads.

    Args:
        image:
        pads: (x_min_pad, y_min_pad, x_max_pad, y_max_pad)
        bboxes: np.array with shape (num_boxes, 4). Each row: [x_min, y_min, x_max, y_max]
        keypoints: np.array with shape (num_keypoints, 2), each row: [x, y]

    Returns: cropped image

    {
            "image": cropped_image,
            "bboxes": shifted_boxes,
            "keypoints": shifted_keypoints
        }

    """
    x_min_pad, y_min_pad, x_max_pad, y_max_pad = pads

    result = {}

    if image is not None:
        height, width = image.shape[:2]
        result["image"] = image[y_min_pad : height - y_max_pad, x_min_pad : width - x_max_pad]

    if bboxes is not None:
        bboxes[:, 0] -= x_min_pad
        bboxes[:, 1] -= y_min_pad
        bboxes[:, 2] -= x_min_pad
        bboxes[:, 3] -= y_min_pad

        result["bboxes"] = bboxes

    if keypoints is not None:
        keypoints[:, 0] -= x_min_pad
        keypoints[:, 1] -= y_min_pad

        result["keypoints"] = keypoints

    return result


def convert_to_h264(input_path, output_path):
    command = [
        'ffmpeg',
        '-y',  # перезаписать выходной файл
        '-i', input_path,
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        output_path
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)


def convert_to_h264(input_path, output_path):
    command = [
        'ffmpeg',
        '-y',  # перезаписать выходной файл
        '-i', input_path,
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        output_path
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)



def tensor_from_rgb_image(image: np.ndarray) -> torch.Tensor:
    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(image)





def predict_jsons1(ort_session, image, confidence_threshold=0.7, nms_threshold=0.4, max_size=1200) -> List[Dict[str, Union[List, float]]]:
    with torch.no_grad():
        original_height, original_width = image.shape[:2]

        scale_landmarks = torch.from_numpy(np.tile([max_size, max_size], 5)).to(device)
        scale_bboxes = torch.from_numpy(np.tile([max_size, max_size], 2)).to(device)
        transform = A.Compose([A.LongestMaxSize(max_size=max_size, p=1), A.Normalize(p=1)])
        transformed_image = transform(image=image)["image"]
        variance = [0.1, 0.2]
        paded = pad_to_size(target_size=(max_size, max_size), image=transformed_image)

        pads = paded["pads"]
        prior_box = prior_box1(
                min_sizes=[[16, 32], [64, 128], [256, 512]],
                steps=[8, 16, 32],
                clip=False,
                image_size=(max_size, max_size),
            ).to(device)

        torched_image = tensor_from_rgb_image(paded["image"]).to(device)

        torched_image = torched_image.unsqueeze(0)
        input_numpy = torched_image.cpu().numpy().astype(np.float32)
        
        input_name = ort_session.get_inputs()[0].name
        loc, conf, land  = ort_session.run(None, {input_name: input_numpy})



        # loc, conf, land — numpy-массивы, полученные из ONNX Runtime
        loc = torch.from_numpy(loc).to(device)
        conf = torch.from_numpy(conf).to(device)
        land = torch.from_numpy(land).to(device)


        # print("loc: ", type(loc), loc.shape)
        # print("conf: ", type(conf), conf.shape)
        # print("land: ", type(land), land.shape)

        # loc, conf, land = model.model(torched_image.unsqueeze(0))

        # conf = softmax(conf, axis=-1)
        conf = F.softmax(conf, dim=-1)
        
        annotations: List[Dict[str, Union[List, float]]] = []

        boxes = decode(loc.data[0], prior_box, variance)

        boxes *= scale_bboxes
        scores = conf[0][:, 1]

        landmarks = decode_landm(land.data[0], prior_box, variance)
        landmarks *= scale_landmarks

        # ignore low scores
        valid_index = torch.where(scores > confidence_threshold)[0]
        boxes = boxes[valid_index]
        landmarks = landmarks[valid_index]
        scores = scores[valid_index]

        # Sort from high to low
        order = scores.argsort(descending=True)
        boxes = boxes[order]
        landmarks = landmarks[order]
        scores = scores[order]

        # do NMS
        keep = nms(boxes, scores, nms_threshold)
        boxes = boxes[keep, :].int()

        if boxes.shape[0] == 0:
            return [{"bbox": [], "score": -1, "landmarks": []}]

        landmarks = landmarks[keep]

        scores = scores[keep].cpu().numpy().astype(np.float64)
        boxes = boxes.cpu().numpy()
        landmarks = landmarks.cpu().numpy()
        landmarks = landmarks.reshape([-1, 2])

        unpadded = unpad_from_size(pads, bboxes=boxes, keypoints=landmarks)

        resize_coeff = max(original_height, original_width) / max_size

        boxes = (unpadded["bboxes"] * resize_coeff).astype(int)
        landmarks = (unpadded["keypoints"].reshape(-1, 10) * resize_coeff).astype(int)

        for box_id, bbox in enumerate(boxes):
            x_min, y_min, x_max, y_max = bbox

            x_min = np.clip(x_min, 0, original_width - 1)
            x_max = np.clip(x_max, x_min + 1, original_width - 1)

            if x_min >= x_max:
                continue

            y_min = np.clip(y_min, 0, original_height - 1)
            y_max = np.clip(y_max, y_min + 1, original_height - 1)

            if y_min >= y_max:
                continue

            annotations += [
                {
                    "bbox": bbox.tolist(),
                    "score": scores[box_id],
                    "landmarks": landmarks[box_id].reshape(-1, 2).tolist(),
                }
            ]

        return annotations



def vis_annotations(image: np.ndarray, annotations: List[Dict[str, Any]]) -> np.ndarray:
    vis_image = image.copy()

    for annotation in annotations:
        landmarks = annotation["landmarks"]

        colors = [(255, 0, 0), (128, 255, 0), (255, 178, 102), (102, 128, 255), (0, 255, 255)]

        for landmark_id, (x, y) in enumerate(landmarks):
            vis_image = cv2.circle(vis_image, (x, y), radius=3, color=colors[landmark_id], thickness=3)

        bbox = annotation.get("bbox", [])
        if len(bbox) != 4:
            # Пропускаем, если bbox пустой или некорректный
            continue

        x_min, y_min, x_max, y_max = annotation["bbox"]

        x_min = np.clip(x_min, 0, x_max - 1)
        y_min = np.clip(y_min, 0, y_max - 1)

        vis_image = cv2.rectangle(vis_image, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2)
    return vis_image



def gradio_video_processing(
    video_file,
    confidence_threshold=0.7,
    nms_threshold=0.4,
    max_size=1200,
    frame_skip=1,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    global ort_session  # ort_session инициализирован глобально

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file {video_file}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width == 0 or height == 0:
        raise RuntimeError(f"Invalid video dimensions: width={width}, height={height}")

    width = (width // 2) * 2
    height = (height // 2) * 2

    # Путь для временного хранения видео внутри проекта
    temp_dir = "temp_videos"
    os.makedirs(temp_dir, exist_ok=True)

    temp_path = os.path.join(temp_dir, "temp_video.mp4")
    final_path = os.path.join(temp_dir, "final_video.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))

    frame_idx = 0
    last_annotation = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            annotation = predict_jsons1(
                ort_session,
                img_rgb,
                confidence_threshold=confidence_threshold,
                nms_threshold=nms_threshold,
                max_size=max_size,
            )
            last_annotation = annotation
            img_vis = vis_annotations(img_rgb, annotation)
            img_bgr = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
        else:
            if last_annotation is not None:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_vis = vis_annotations(img_rgb, last_annotation)
                img_bgr = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = frame

        out.write(img_bgr)
        frame_idx += 1

    cap.release()
    out.release()

    # Перекодировка в H264 для совместимости с браузерами
    convert_to_h264(temp_path, final_path)
    # Удаляем временный файл с исходным видео
    if os.path.exists(temp_path):
        os.remove(temp_path)

    return final_path




def onnx_inference(image: np.ndarray, confidence_threshold=0.7, nms_threshold=0.4, max_size=1200):
    """
    Функция для инференса ONNX модели на входном изображении.
    image: RGB numpy array
    Возвращает изображение с аннотациями (BGR numpy array)
    """
    # Конвертируем BGR (OpenCV) в RGB, если нужно
    if image.shape[2] == 3:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = image

    # Запуск предсказания 
    annotation = predict_jsons1(
        ort_session,
        img_rgb,
        confidence_threshold=confidence_threshold,
        nms_threshold=nms_threshold,
        max_size=max_size,
    )

    # Визуализация аннотаций
    img_vis = vis_annotations(img_rgb, annotation)

    # Конвертируем обратно в BGR для отображения в OpenCV/Gradio
    img_bgr = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)

    return img_bgr




def onnx_inference2(image: np.ndarray, confidence_threshold=0.7, nms_threshold=0.4, imgsz=736):
    """
    Функция для инференса ONNX модели на входном изображении.
    image: RGB numpy array
    Возвращает изображение с аннотациями (BGR numpy array)
    """
    # Конвертируем BGR (OpenCV) в RGB, если нужно
    if image.shape[2] == 3:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = image

    # Запуск предсказания
    output, h0, w0, imgsz = predict_jsons2(
        ort_session_2,
        img_rgb,
        confidence_threshold=confidence_threshold,
        nms_threshold=nms_threshold,
        imgsz=imgsz,
    )

    # Визуализация аннотаций
    img_vis = vis_annotations2(img_rgb, output, h0, w0, imgsz, confidence_threshold=confidence_threshold, nms_threshold=nms_threshold)

    # Конвертируем обратно в BGR для отображения в OpenCV/Gradio
    img_bgr = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)

    return img_bgr




def vis_annotations2(image: np.ndarray, output, h0, w0, imgsz, confidence_threshold=0.7, nms_threshold=0.4) -> np.ndarray:
    vis_image = image.copy()
    scale_x = w0 / imgsz
    scale_y = h0 / imgsz

    CLASS_NAMES = ['building', 'plane', 'fish net', 'harbor', 'well', 'helicopter', 'vehicle', 'ship']
    CLASS_COLORS = [
        (0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 0, 255),
        (255, 255, 0), (128, 0, 128), (0, 128, 255), 
    ]

    # Фильтруем по confidence_threshold
    filtered_dets = [det for det in output[0] if det[4] >= confidence_threshold]
    if not filtered_dets:
        return vis_image  # Нет детекций выше порога

    # Конвертируем в тензоры для NMS
    boxes = torch.tensor([det[:4] for det in filtered_dets], dtype=torch.float32)
    scores = torch.tensor([det[4] for det in filtered_dets])
    classes = [int(det[5]) for det in filtered_dets]

    # Применяем NMS
    keep_indices = nms(boxes, scores, nms_threshold)

    for idx in keep_indices:
        det = filtered_dets[idx]
        x1, y1, x2, y2, conf_score, cls = det

        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)

        class_id = int(cls)
        label = f"{CLASS_NAMES[class_id]} {conf_score:.2f}"
        color = CLASS_COLORS[class_id % len(CLASS_COLORS)]

        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis_image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return vis_image



def predict_jsons2(ort_session_2, original_image, confidence_threshold=0.7, nms_threshold=0.4, imgsz=736):
    with torch.no_grad():

        h0, w0 = original_image.shape[:2]  # оригинальные размеры

        # Ресайз под модель
        image_resized = cv2.resize(original_image, (imgsz, imgsz))
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        input_tensor = image_rgb.transpose(2, 0, 1) / 255.0
        input_tensor = np.expand_dims(input_tensor.astype(np.float32), axis=0)


        # ======          ONNX Inference 
        input_name = ort_session_2.get_inputs()[0].name
        output_name = ort_session_2.get_outputs()[0].name
        output = ort_session_2.run([output_name], {input_name: input_tensor})[0]
       
        return output, h0, w0, imgsz



def gradio_video_processing2(
    video_file,
    confidence_threshold=0.5,
    nms_threshold=0.6,
    imgsz=736,
    frame_skip=1,
):
    global ort_session_2  # инициализирован глобально


    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file {video_file}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("Размеры", width, height)
    if width == 0 or height == 0:
        raise RuntimeError(f"Invalid video dimensions: width={width}, height={height}")

    width = (width // 2) * 2
    height = (height // 2) * 2

    # Путь для временного хранения видео внутри проекта
    temp_dir = "temp_videos"
    os.makedirs(temp_dir, exist_ok=True)

    temp_path = os.path.join(temp_dir, "temp_video.mp4")
    final_path = os.path.join(temp_dir, "final_video.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))

    frame_idx = 0
    last_annotation = None


    while True:
        ret, frame = cap.read()
        if not ret:
            break


        if frame_idx % frame_skip == 0:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            annotation, h0, w0, imgsz = predict_jsons2(
                ort_session_2,
                img_rgb,
                confidence_threshold=confidence_threshold,
                nms_threshold=nms_threshold,
                imgsz=imgsz,
            )
            last_annotation = annotation

            img_vis = vis_annotations2(img_rgb, annotation, h0, w0, imgsz, confidence_threshold=confidence_threshold, nms_threshold=nms_threshold)
            img_bgr = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
        # else:
        #     if last_annotation is not None:
        #         img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #         img_vis = vis_annotations2(img_rgb, last_annotation, h0, w0, imgsz)
        #         img_bgr = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
        #     else:
        #         img_bgr = frame

        # img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


        # annotation, h0, w0, imgsz = predict_jsons2(
        #     ort_session_2,
        #     img_rgb,
        #     confidence_threshold=confidence_threshold,
        #     nms_threshold=nms_threshold,
        #     imgsz=imgsz,
        # )
        # last_annotation = annotation

        # img_vis = vis_annotations2(img_rgb, annotation, h0, w0, imgsz, confidence_threshold=0.7, nms_threshold=0.4)
        # img_bgr = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)


        out.write(img_bgr)
        frame_idx += 1

    cap.release()
    out.release()

    # Перекодировка в H264 для совместимости с браузерами
    convert_to_h264(temp_path, final_path)
    # Удаляем временный файл с исходным видео
    if os.path.exists(temp_path):
        os.remove(temp_path)

    return final_path


