# Семантическая сегментация с помощью U-Net на уличных сценах  

###  Итоговая аттестация курса «Профессия ML-инженер»   
Реализация моделей сегментации изображений уличных сцен.  


-----------------------------------------

## Содержание  

- [Технологии](#технологии)  
- [Использование](#Использование)  
- [Обучение моделей](#Обучение-моделей)  
- [Запуск проекта на Gradio](#Запуск-проекта-на-Gradio)  
- [Веб-приложения модели](#Веб-приложения-модели)  
- [Приложение на Gradio](#Приложение-на-Gradio)  
- [Приложение на PythonAnywhere](#Приложение-на-PythonAnywhere)  
------------------------  

## Технологии  
- [Python](https://www.python.org/)  
- [Pytorch](https://pytorch.org/)
- [Onnx](https://onnx.ai/)
- [OpenCV](https://opencv.org/)
- [Gradio](https://www.gradio.app/)  
- [SMP](https://smp.readthedocs.io/en/latest/models.html)
- [pythonanywhere](https://www.pythonanywhere.com/)
-------------------------------------------

## Использование  

### Обучение моделей

В каталоге проекта выполнить команду  

```sh
python -m venv .myenv   
source .myenv/bin/activate   
pip install -r requirements.txt  
```


### Запуск проекта на Gradio:  

В каталоге проекта выполнить команду

```sh
python -m venv .myenv   
source .myenv/bin/activate   
cd gradio_projects  
pip install -r requirements.txt
```

Запуск Gradio  

```sh
source .myenv/bin/activate 
cd gradio_projects 
python app.py
```


-----------------------------  

## Веб-приложения модели  


### [<u>Приложение на Gradio</u>](https://huggingface.co/spaces/makc-mon173/projects)

![image](https://github.com/user-attachments/assets/4b901054-cb5a-4d20-b029-7cdc3c4c8fd5)  


### [<u>Приложение на PythonAnywhere</u>](https://leimansite.pythonanywhere.com/segmentation/) 

![image](https://github.com/user-attachments/assets/231067c8-4436-4cae-bd0d-fccadff70565)  












