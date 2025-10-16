import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import KFold


# Определение класса модели
class SentimentClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SentimentClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm1 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(p=0.2)  # Добавление слоя Dropout
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm1(embedded)
        output = self.dropout(output)
        output, _ = self.lstm2(output)
        output = self.dropout(output)  # Применение Dropout к выходу LSTM
        output = self.fc(output[:, -1, :])  # Используется только последний выход LSTM
        return output
    


# Определение класса модели для инференса
class SentimentClassifierInferens(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SentimentClassifierInferens, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm1 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)       
        self.dropout = nn.Dropout(p=0.2)  # Добавление слоя Dropout
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm1(embedded)
        output, _ = self.lstm2(output)
        output = output[-1, :]  # Получение последнего выхода LSTM
        output = self.fc(output)  # Применение полносвязного слоя
        return output


def train_model(model,  optimizer, criterion, train_loader, val_loader, num_epochs=5, save_model_path = 'models/model_1.pth', device='cpu'):
    # Обучение модели
    train_losses = []  # Список для сохранения значений функции потерь на тренировочном наборе
    val_losses = []  # Список для сохранения значений функции потерь на валидационном наборе
    train_accuracies = []  # Список для сохранения значений точности на тренировочном наборе
    val_accuracies = []  # Список для сохранения значений точности на валидационном наборе

    
    best_val_accuracy = 0  # Инициализация переменной для отслеживания лучшего значения функции потерь на валидационном наборе
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training", leave=False)
        for inputs, labels in train_loader_tqdm:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            train_loader_tqdm.set_postfix(loss=loss.item(), accuracy=total_correct/total_samples)


        train_accuracy = total_correct / total_samples
        train_loss = total_loss / len(train_loader)


        # Валидация модели
        val_loss = float('inf')
        model.eval()
        val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation", leave=False)
        with torch.no_grad():
            total_val_loss = 0
            total_val_correct = 0
            total_val_samples = 0

            for inputs, labels in val_loader_tqdm:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                val_loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                total_val_correct += (predicted == labels).sum().item()
                total_val_samples += labels.size(0)

                total_val_loss += val_loss.item()
                val_loader_tqdm.set_postfix(loss=val_loss.item(), accuracy=total_val_correct/total_val_samples)


            val_accuracy = total_val_correct / total_val_samples
            val_loss = total_val_loss / len(val_loader)

            ###########################################
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)    
            ###############################################
            
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")

         # Проверка, является ли текущая модель лучшей
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            print("Сохранена лучшая модель")
            # Сохранение модели
            torch.save(model.state_dict(), save_model_path)
            
    # Построение графика потерь
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()

    # Построение графика точности
    plt.figure()
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.show()
        



####     КРОС ВАЛИДАЦИЯ
def train_model_cross(model, optimizer, criterion, train_data_tensor, train_labels_tensor, k_folds = 5, num_epochs=5, save_model_path = 'models/model_2.pth', batch_size=24, device='cpu'):
   
    X = train_data_tensor
    y = train_labels_tensor

    # Создание объекта K-fold cross-validation
    kfold = KFold(n_splits=k_folds)

    # Обучение модели
    best_val_accuracy = 0  # Инициализация переменной для отслеживания лучшего значения функции потерь на валидационном наборе

    # Итерация по фолдам
    for fold, (train_indices, val_indices) in enumerate(kfold.split(X)):

        # Разделение данных на тренировочный и валидационный наборы
        train_X, train_y = X[train_indices], y[train_indices]
        val_X, val_y = X[val_indices], y[val_indices]

        # Создание DataLoader для тренировочного и валидационного наборов
        train_dataset = torch.utils.data.TensorDataset(train_X, train_y)
        val_dataset = torch.utils.data.TensorDataset(val_X, val_y)



        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Обучение модели
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            total_correct = 0
            total_samples = 0

            train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training", leave=False)
            for inputs, labels in train_loader_tqdm:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                train_loader_tqdm.set_postfix(loss=loss.item(), accuracy=total_correct/total_samples)
                
            train_accuracy = total_correct / total_samples
            train_loss = total_loss / len(train_loader)

            # Валидация модели
            val_loss = float('inf')
            model.eval()
            val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation", leave=False)
            with torch.no_grad():
                total_val_loss = 0
                total_val_correct = 0
                total_val_samples = 0

                for inputs, labels in val_loader_tqdm:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs)
                    val_loss = criterion(outputs, labels)

                    _, predicted = torch.max(outputs.data, 1)
                    total_val_correct += (predicted == labels).sum().item()
                    total_val_samples += labels.size(0)

                    total_val_loss += val_loss.item()
                    val_loader_tqdm.set_postfix(loss=val_loss.item(), accuracy=total_val_correct/total_val_samples)
                    
                val_accuracy = total_val_correct / total_val_samples
                val_loss = total_val_loss / len(val_loader)

            print(f"Fold {fold+1} | Epoch {epoch+1}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")

             # Проверка, является ли текущая модель лучшей
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                print("Модель сохранена")
                # Сохранение модели
                torch.save(model.state_dict(), save_model_path)



# Тестировани е модели
def evaluation_model(model, test_loader, device='cpu'):
    ''' 
    Тестировани е модели
    '''
    # Оценка модели
    model.eval()
    with torch.no_grad():
        total_test_correct = 0
        total_test_samples = 0

        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total_test_correct += (predicted == labels).sum().item()
            total_test_samples += labels.size(0)

        test_accuracy = total_test_correct / total_test_samples
    text = f" точность: {test_accuracy:.4f}"
    return text  

