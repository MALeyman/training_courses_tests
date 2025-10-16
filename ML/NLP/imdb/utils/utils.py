import os 
import pathlib 
import numpy as np
from glob import glob
import pickle
import shutil
import torch


##### функция преобразования текста в числа для инференса
def transform_text_inferens(text1, path_vocab, len_text=280):
    fl = False 
    with open(path_vocab, "r") as f1:
        vocab = f1.read().splitlines()
    vocab = vocab[:39998]  
    mass = [] 
    str1 =''    
    for ch in text1:
        if len(mass)>len_text-1: # если слов больше чем нужно, выходим
            data_tensor = torch.tensor(mass)
            return data_tensor
        if ch !=' ':
            str1 = str1 + ch
        if ch ==' ':
            if str1 !='':                      
                fl = False                                                            
                for i in range(len(vocab)):
                    if str1.lower() == vocab[i].lower():
                        fl = True
                        mass.append(i+2)
                        str1 =''
                        break                                                 
                if fl == False: #  если слово не найдено, заменяем нулями
                    mass.append(0)
                    str1 =''        
    for i in range(len(vocab)): # Проверяем последнее слово
        if str1.lower() == vocab[i].lower():
            fl = True
            mass.append(i+2)
            str1 =''
            break                                                 
    if fl == False: #  если слово не найдено, заменяем нулями
        mass.append(0)
        str1 =''

    if len(mass)<len_text:  #  если слов меньше чем нужно, добавляем нулями.
        while len(mass)<len_text:
            mass.insert(0, 0)
            str1 =''       
    data_tensor = torch.tensor(mass)    
    return data_tensor



# функция преобразования файла текста в массив с числами (Токенами)
def transform_text_1(file1, path_vocab):
    ''' 
    функция преобразования файла текста в массив с числами
    '''
    # Читаем словарь и создаём маппинг слова -> индекс
    with open(path_vocab, "r") as f_vocab:
        lines = f_vocab.read().splitlines()
    vocab_dict = {word.lower(): idx+2 for idx, word in enumerate(lines)}

    mass = []
    with open(file1, "r") as f_text:
        text = f_text.read()

    # Разбиваем весь текст на слова по пробелу
    words = text.split()

    for word in words:
        word_lower = word.lower()
        if word_lower in vocab_dict:
            mass.append(vocab_dict[word_lower])
        else:
            mass.append(0)  # если слова нет в словаре

    return mass



import os
import random
import pickle

def transform_text_2(text1, vocab, len_text, rand=False):
    mass = [] 
    str1 =''    
    fl = False
    list_1 = [0, 1]

    for ch in text1:
        if len(mass) > len_text - 1:
            return mass
        if ch != ' ':
            str1 = str1 + ch
        if ch == ' ':
            if str1 != '':
                fl = False
                for i in range(len(vocab)):
                    if str1.lower() == vocab[i].lower():
                        fl = True
                        random_choice = random.choice(list_1)
                        if rand and random_choice == 0:
                            mass.append(1)
                            break
                        mass.append(i + 2)
                        str1 = ''
                        break
                if not fl:
                    mass.append(0)
                    str1 = ''
    # Проверка последнего слова
    if str1 != '':
        fl = False
        for i in range(len(vocab)):
            if str1.lower() == vocab[i].lower():
                fl = True
                mass.append(i + 2)
                break
        if not fl:
            mass.append(0)

    while len(mass) < len_text:
        mass.insert(0, 0)
    return mass





import os
import random
# Преобразование датасета разделение на данные и метки
def create_dataset_no_split(base_path, vocab, len_text, rand=False):
    ''' 
    Преобразование датасета разделение на данные и метки
    '''
    classes = [folder for folder in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, folder))]
    label_mapping = {1:0, 2:1, 3:2, 4:3, 7:4, 8:5, 9:6, 10:7}

    all_data = []
    all_labels = []

    for cl in classes:
        folder_path = os.path.join(base_path, cl)
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        random.shuffle(files)  # если нужна случайность в порядке

        for file_name in files:
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            vec = transform_text_2(text, vocab, len_text, rand=rand)
            all_data.append(vec)
            all_labels.append(label_mapping[int(cl)])

    return all_data, all_labels






# Разделение датасета на сбалансированный train и test
def create_balanced_and_test_datasets(base_path, vocab, len_text, rand=False):
    ''' 
    Разделение датасета на сбалансированный train и test из остатков
    '''
    classes = [folder for folder in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, folder))]
    class_counts = {}

    # Подсчёт количества файлов в каждом классе
    for cl in classes:
        folder_path = os.path.join(base_path, cl)
        count = len([name for name in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, name))])
        class_counts[cl] = count

    min_count = min(class_counts.values())
    print(f"Минимальное количество примеров на класс: {min_count}")

    balanced_data = []
    balanced_labels = []
    test_data = []
    test_labels = []

    for cl in classes:
        folder_path = os.path.join(base_path, cl)
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        random.shuffle(files)
        label_mapping = {1:0, 2:1, 3:2, 4:3, 7:4, 8:5, 9:6, 10:7}

        # Выбираем min_count файлов для сбалансированного датасета
        balanced_files = files[:min_count]
        # Остальные — для тестового датасета
        test_files = files[min_count:]

        # Обработка balanced
        for file_name in balanced_files:
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            vec = transform_text_2(text, vocab, len_text, rand=rand)
            balanced_data.append(vec)
            balanced_labels.append(label_mapping[int(cl)])


        # Обработка test
        for file_name in test_files:
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            vec = transform_text_2(text, vocab, len_text, rand=rand)
            test_data.append(vec)
            test_labels.append(label_mapping[int(cl)])

    return balanced_data, balanced_labels, test_data, test_labels



# функция получения имени файла из пути
def str_name(str1):
    ''' 
    функция получения имени файла из пути
    '''
    p = os.path.basename(str1)
    return os.path.splitext(p)[0]

# функция получения рейтинга отзыва и номера отзыва по имени файла
def inv1(filename):
    '''
    Функция для получения рейтинга отзыва и номера (названия) файла по имени файла.
    Ожидается, что имя файла имеет формат "номер_рейтинг.txt", например "2429_1.txt".
    Возвращает кортеж: (рейтинг как строка, имя файла без расширения)
    '''
    # Убираем расширение файла
    base_name = filename.split('.')[0]
    # Разделяем по подчёркиванию
    parts = base_name.split('_')
    if len(parts) < 2:
        return None, None  # Если формат некорректный
    
    rating = parts[-1]  # Рейтинг — часть после последнего подчёркивания
    file_number = '_'.join(parts[:-1])  # Всё до рейтинга — номер файла (может содержать подчёркивания)
    
    return file_number, rating


# функция вывода текста  отзыва
def read_file(file1):
    ''' 
    функция вывода текста  отзыва
    '''
    with open(file1, "r") as f1:
        text = f1.read()
        print(text)
        f1.close
    return text


# функция разбивает исходный датасет по классам (по рейтингу)
def preprocess_text(mas, cls):   
    ''' 
    функция разбивает исходный датасет по классам (по рейтингу)
    '''
    for i in range(len(mas)):
        str_n = mas[i]
        str_1 = str_name(str_n)
        
        name_faile, num_class = inv1(str_1)
        name_faile1 = name_faile + '.txt'
        
        if num_class == '10':
            path_new = cls[0]
        elif num_class == '1':
            path_new = cls[1]          
        elif num_class == '2':
            path_new = cls[2]
        elif num_class == '3':
            path_new = cls[3]
        elif num_class == '4':
            path_new = cls[4]
        elif num_class == '7':
            path_new = cls[5]  
        elif num_class == '8':
            path_new = cls[6]
        elif num_class == '9':
            path_new = cls[7]                    

                  
        shutil.copy2(str_n, os.path.join(path_new, name_faile1))


# функция загрузки словаря для кодировки слов из файла
def vocab_read(path_vocab):
    ''' 
    функция загрузки словаря для кодировки слов из файла
    '''
    with open(path_vocab, "r") as f1:
        line = f1.read().splitlines()
    return line




