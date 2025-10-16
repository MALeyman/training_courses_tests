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
import torch.nn as nn




class LossManager(nn.Module):
    def __init__(self, loss_config,  num_classes=19):
        super().__init__()
        self.losses = {}
        self.weights = {}
        self.num_classes = num_classes

        for name, params in loss_config.items():
            self.weights[name] = params['weight']
            if name == 'cross_entropy':
                self.losses[name] = nn.CrossEntropyLoss()
            elif name == 'dice':
                self.losses[name] = lambda outputs, masks: dice_loss(
                    masks, outputs, num_classes=self.num_classes
                )
            elif name == 'focal':
                self.losses[name] = FocalLoss(gamma=2.0, alpha=0.25)
            elif name == 'lovasz':
                self.losses[name] = lovasz_hinge
    
    def forward(self, outputs, masks):
        total_loss = 0
        for name, loss_fn in self.losses.items():
            loss_val = loss_fn(outputs, masks)
            total_loss += self.weights[name] * loss_val
        return total_loss


def bce_loss_torch(y_real, y_pred):
    y_pred = y_pred.view(-1)
    y_real = y_real.view(-1)
    y_pred = y_pred.type(torch.float32)
    y_real = y_real.type(torch.float32)

    loss = F.relu(y_pred) - y_real * y_pred + torch.log(1. + torch.exp(-abs(y_pred)))
    return torch.mean(loss)


def dice_loss(y_real, y_pred, num_classes=19, smooth=1e-8):
    """
    Многоклассовая Dice Loss с поддержкой пропуска отсутствующих классов.
    
    Args:
        y_real: [B, H, W] Ground truth маска (целые числа 0..num_classes-1)
        y_pred: [B, C, H, W] Логиты модели
        num_classes: Количество классов
        smooth: Сглаживающий коэффициент
        
    Returns:
        Среднее значение Dice Loss по присутствующим классам
    """
    # Применяем softmax к предсказаниям
    y_pred_softmax = torch.softmax(y_pred, dim=1)
    
    # Конвертируем маски в one-hot представление
    y_real_onehot = F.one_hot(y_real, num_classes=num_classes).permute(0, 3, 1, 2).float()
    
    # Вычисляем intersection и union для всех классов
    intersection = torch.sum(y_pred_softmax * y_real_onehot, dim=(0, 2, 3))
    union = torch.sum(y_pred_softmax, dim=(0, 2, 3)) + torch.sum(y_real_onehot, dim=(0, 2, 3))
    
    # Определяем присутствующие классы
    class_present = torch.sum(y_real_onehot, dim=(0, 2, 3)) > 0
    
    # Вычисляем Dice для каждого класса
    dice_per_class = (2. * intersection + smooth) / (union + smooth)
    
    # Усредняем только по присутствующим классам
    if torch.any(class_present):
        dice_avg = torch.mean(dice_per_class[class_present])
        return 1 - dice_avg
    return torch.tensor(1.0, device=y_pred.device, requires_grad=True)



class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(reduction='none', ignore_index=self.ignore_index)
    def forward(self, inputs, targets):
        """
        inputs: логиты модели, shape [B, C, ...]
        targets: целочисленные метки классов, shape [B, ...]
        """
        logpt = -self.ce_loss(inputs, targets)  
        pt = torch.exp(logpt)  # вероятности правильных классов

        # Фокусируемся на сложных примерах
        focal_term = (1 - pt) ** self.gamma

        # Балансировка классов
        if self.alpha is not None:
            alpha_t = self.alpha
            loss = -alpha_t * focal_term * logpt
        else:
            loss = -focal_term * logpt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss



# https://arxiv.org/pdf/1705.08790.pdf
def grad_lovasz(sorted_errors):

    err_cumm = sorted_errors.float().cumsum(0)
    opp_err_cumm = (1 - sorted_errors).float().cumsum(0)
    p = len(sorted_errors)
    err = sorted_errors.sum()
    intersection = err - err_cumm
    union = err + opp_err_cumm
    grad = 1 - intersection / union
    if p > 1:
        grad[1:p] = grad[1:p] - grad[0:p-1]
    return grad


def lovasz_hinge_binary(logits, labels):
    logits = logits.view(-1)
    labels = labels.view(-1)
    sgn = 2 * labels.float() - 1
    errs = 1 - logits * sgn
    sorted_errors, indices = torch.sort(errs, descending=True)
    errs_for_grad = labels[indices]
    grad = grad_lovasz(errs_for_grad)
    loss = torch.dot(F.relu(sorted_errors), grad)
    return loss


# https://arxiv.org/pdf/1512.07797.pdf. 
def lovasz_hinge(probas, labels, classes='present'):
    """
    probas: [B, C, H, W] — вероятности после softmax
    labels: [B, H, W] — метки классов
    classes: какие классы учитывать

    Возвращает средний lovasz loss по классам.
    """  
    B, C, H, W = probas.size()
    loss = 0.
    for c in range(C):
        fg = (labels == c).float()
        if classes == 'present' and fg.sum() == 0:
            continue
        class_pred = probas[:, c, :, :]
        class_pred_flat = class_pred.reshape(-1)  
        fg_flat = fg.reshape(-1)
        loss += lovasz_hinge_binary(class_pred_flat, fg_flat)
    return loss / C




# ###################  Тренировка
def train_model(model, train_loader, val_loader, optimizer, device, num_epochs=20, scheduler=None, save_path='best_model.pth', num_classes=20, loss_config={'cross_entropy': {'weight': 1.0}} ):
    model.to(device)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    train_ious, val_ious = [], []
    best_val_iou = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    loss_manager = LossManager(loss_config, num_classes=num_classes)
    def iou_score(pred, target, n_classes=19):
        ious = []
        pred = pred.view(-1)
        target = target.view(-1)
        for cls in range(n_classes):
            pred_inds = (pred == cls)
            target_inds = (target == cls)
            intersection = (pred_inds & target_inds).sum().item()
            union = (pred_inds | target_inds).sum().item()
            if union == 0:
                ious.append(float('nan'))  # класс отсутствует в выборке
            else:
                ious.append(intersection / union)
        # Средний IoU по классам, игнорируя nan
        ious = [iou for iou in ious if not np.isnan(iou)]
        if len(ious) == 0:
            return 0
        return np.mean(ious)

    for epoch in range(1, num_epochs+1):
        model.train()
        running_loss = 0
        running_corrects = 0
        running_total = 0
        running_iou = 0
        corrects1 = 0
        corrects2 = 0
        train_batches = len(train_loader)
        val_batches = len(val_loader)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} Train")
        for images, masks in pbar:
            images = images.to(device)
           
            masks = masks.to(device).long()
            optimizer.zero_grad()
            masks = masks.squeeze(1).long()
            outputs = model(images)  # [B, C, H, W]
            # loss = F.cross_entropy(outputs, masks, ignore_index=0)
            loss = loss_manager(outputs, masks)
            loss.backward()
            # loss = F.cross_entropy(outputs, masks)
            # loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss.item() 

            preds = torch.argmax(outputs, dim=1)
            corrects1 += (preds == masks).sum().item()
            running_total += masks.numel()

            batch_iou = iou_score(preds.cpu(), masks.cpu())
            running_iou += batch_iou 
            acc=corrects1/running_total
            running_corrects += acc
            pbar.set_postfix(loss=loss.item(), acc=acc, iou=batch_iou)



        epoch_loss = running_loss / train_batches
        epoch_acc = running_corrects / train_batches
        epoch_iou = running_iou / train_batches

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        train_ious.append(epoch_iou)

        # Валидация
        model.eval()
        val_loss = 0
        val_corrects = 0
        val_total = 0
        val_iou_sum = 0

        with torch.no_grad():
            pbar2 = tqdm(val_loader, desc=f"Epoch {epoch} Val")
            for images, masks in pbar2:
                images = images.to(device)
               
                masks = masks.to(device).long()
                masks = masks.squeeze(1).long()

                outputs = model(images)
                loss = loss_manager(outputs, masks)
    
                # loss = F.cross_entropy(outputs, masks)

                val_loss += loss.item() 

                preds = torch.argmax(outputs, dim=1)
                corrects2 += (preds == masks).sum().item()
                val_total += masks.numel()

                batch_iou = iou_score(preds.cpu(), masks.cpu())
                val_iou_sum += batch_iou 
                acc=corrects2/val_total
                val_corrects += acc

                pbar2.set_postfix(loss=loss.item(), acc=acc, iou=batch_iou)

        val_epoch_loss = val_loss / val_batches
        val_epoch_acc = val_corrects / val_batches
        val_epoch_iou = val_iou_sum / val_batches

        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_acc)
        val_ious.append(val_epoch_iou)

        print(f"Epoch {epoch} summary: Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, IoU: {epoch_iou:.4f} | Val Loss: {val_epoch_loss:.4f}, Acc: {val_epoch_acc:.4f}, IoU: {val_epoch_iou:.4f}")
        # Сохраняем модель
        if val_epoch_iou > best_val_iou:
            best_val_iou = val_epoch_iou
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), save_path)
            print(f"Сохранена лучшая модель {epoch} с val IoU: {best_val_iou:.4f}")
                  
        if scheduler is not None:
            scheduler.step()
            
    # Восстанавливаем лучшие веса
    model.load_state_dict(best_model_wts)

    epochs = range(1, num_epochs+1)

    plt.figure(figsize=(18,5))
    plt.subplot(1,3,1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over epochs')

    plt.subplot(1,3,2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy over epochs')

    plt.subplot(1,3,3)
    plt.plot(epochs, train_ious, label='Train IoU')
    plt.plot(epochs, val_ious, label='Val IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.title('IoU over epochs')

    plt.show()


# тестирование модели
def test_model(model, test_loader, device, num_classes=20, loss_config={'cross_entropy': {'weight': 1.0}}):
    model.to(device)
    model.eval()

    loss_manager = LossManager(loss_config, num_classes=num_classes)

    test_loss = 0
    test_corrects = 0
    test_total = 0
    test_iou_sum = 0

    def iou_score(pred, target, n_classes=num_classes):
        ious = []
        pred = pred.view(-1)
        target = target.view(-1)
        for cls in range(n_classes):
            pred_inds = (pred == cls)
            target_inds = (target == cls)
            intersection = (pred_inds & target_inds).sum().item()
            union = (pred_inds | target_inds).sum().item()
            if union == 0:
                ious.append(float('nan'))
            else:
                ious.append(intersection / union)
        ious = [iou for iou in ious if not np.isnan(iou)]
        if len(ious) == 0:
            return 0
        return np.mean(ious)

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device).long()
            masks = masks.squeeze(1).long()

            outputs = model(images)  # [B, C, H, W]
            loss = loss_manager(outputs, masks)

            test_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            test_corrects += (preds == masks).sum().item()
            test_total += masks.numel()

            batch_iou = iou_score(preds.cpu(), masks.cpu())
            test_iou_sum += batch_iou

            acc = test_corrects / test_total
            pbar.set_postfix(loss=loss.item(), acc=acc, iou=batch_iou)

    avg_loss = test_loss / len(test_loader)
    avg_acc = test_corrects / test_total
    avg_iou = test_iou_sum / len(test_loader)

    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}, IoU: {avg_iou:.4f}")

    return avg_loss, avg_acc, avg_iou

