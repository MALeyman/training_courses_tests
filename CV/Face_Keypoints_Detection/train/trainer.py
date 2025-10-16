""" 
Автор: Лейман Максим  

Дата создания: 18.06.2025
"""

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy


def train(
    model, 
    train_loader, 
    val_loader, 
    optimizer, 
    criterion, 
    scheduler, 
    device, 
    num_epochs=20, 
    save_path='best_model.pth'
):
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in pbar:
            images = batch['image'].float().to(device)
            keypoints = batch['keypoints'].float().to(device)
            if keypoints.ndim == 3:
                keypoints = keypoints.view(keypoints.size(0), -1)

            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, keypoints)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            pbar.set_postfix(loss=loss.item())

        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # Валидация
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):

                images = batch['image'].float().to(device)

                keypoints = batch['keypoints'].float().to(device)
                if keypoints.ndim == 3:
                    keypoints = keypoints.view(keypoints.size(0), -1)


                outputs = model(images)
                loss = criterion(outputs, keypoints)
                val_running_loss += loss.item() * images.size(0)

        epoch_val_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

        # Шаг Scheduler 
        if scheduler is not None:
            scheduler.step(epoch_val_loss)

        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), save_path)
            print(f"Сохранена лучшая модель на эпохе  {epoch+1} с val loss {best_val_loss:.4f}")

    # Загрузка весов
    model.load_state_dict(best_model_wts)

    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()

    return model, train_losses, val_losses

