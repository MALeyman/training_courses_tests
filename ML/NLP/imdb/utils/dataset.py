import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import random



class ReviewDataset(Dataset):
    ''' 
    класс датасета
    '''
    def __init__(self, data, labels):
        self.data = data  # список списков чисел (векторы)
        self.labels = labels  # список меток

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Преобразуем элементы в тензоры PyTorch
        x = torch.tensor(self.data[idx], dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y





def augment_random_replace(tensor, replace_prob=0.1):
    # tensor - 1D список или тензор чисел
    # с вероятностью replace_prob заменяем элементы != 0 на 1
    augmented = []
    for x in tensor:
        if x != 0 and random.random() < replace_prob:
            augmented.append(1)
        else:
            augmented.append(x)
    return augmented

class ReviewDatasetWithAug(Dataset):
    ''' 
    класс датасета
    '''
    def __init__(self, data, labels, augment=False, replace_prob=0.05):
        self.data = data
        self.labels = labels
        self.augment = augment
        self.replace_prob = replace_prob

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]

        if self.augment:
            x = augment_random_replace(x, self.replace_prob)

        x_tensor = torch.tensor(x, dtype=torch.long)
        y_tensor = torch.tensor(y, dtype=torch.long)

        return x_tensor, y_tensor
    



    