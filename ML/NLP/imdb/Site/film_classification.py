import io
import streamlit as st

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

path_1 = os.getcwd()
path_vocab = os.path.join(path_1, 'imdb.vocab')
path_model = os.path.join(path_1, 'model_3.pth')


# гиперпараметры модели
input_size = 40000 
hidden_size = 256
num_classes = 8


#Определение класса модели
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
        output, _ = self.lstm2(output)
        #output = self.dropout(output)  # Применение Dropout к выходу LSTM
        #output = self.fc(output[:, -1, :])  # Используется только последний выход LSTM
        
        output = output[-1, :]  # Получение последнего выхода LSTM
        output = self.fc(output)  # Применение полносвязного слоя
        return output


##### функция преобразования текста в числа

def transform_text(text1, path_vocab=path_vocab, len_text=280):
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






# Создание и загрузка модели
model = SentimentClassifier(input_size, hidden_size, num_classes).to('cpu')


#@st.cache(allow_output_mutation=True)
model.load_state_dict(torch.load(path_model, map_location=torch.device('cpu')))
model.eval()

st.title('Классификации отзывов к фильмам')
st.title('Загрузите отзыв на английском языке')
st.title('Выполнил Лейман М.А')


#st.text_area(label='текст отзыва', value="", height=None, max_chars=None, key=None, help=None, on_change=None, args=None, kwargs=None, *, placeholder=None, disabled=False, label_visibility="visible")

text_input = st.text_input("Введите текст отзыва для предсказания")


if st.button("Предсказать"):
	text2 = transform_text(text_input)
	output = model(text2)
	probs = F.softmax(output, dim=0)
	predicted_class = torch.argmax(probs, dim=0)
	prediction = predicted_class.item()
	
	if prediction < 4:
		prediction += 1
		st.write("Отзыв негативный")
	if prediction > 3:
		prediction += 3
		st.write("Отзыв положительный")
	st.write("Предсказанная метка:", prediction)
















