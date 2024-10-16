# Тестовое задание


https://github.com/user-attachments/assets/f0c79b87-c8f5-41a7-9270-8ea90f134377


## Описание задания
Дан видеофайл crowd.mp4. Необходимо написать программу на языке Python, которая будет выполнять детекцию людей и их отрисовку на этом видео. Также нужно проанализировать полученный результат и сформулировать шаги по дальнейшему улучшению качества распознавания.

## Требования
- Язык программирования: Python.
- Оформление: проект должен быть размещён в git-репозитории (либо оформлен в стиле git-репозитория) и содержать:
    - Точку входа в программу.
    - Docstrings для функций и классов.
    - Оформление в соответствии с PEP8.
    - Файл requirements.txt.
    - Файл README.md с описанием проекта, установкой и запуском.
- Вывод: на выходе программы должно получиться видео с отрисованными людьми. Отрисовка не должна значительно перекрывать исходный кадр, детектируемые объекты должны быть различимы. Каждая отрисовка должна содержать имя класса и уверенность в распознавании.
- Кросс-платформенность: программа должна одинаково работать на Linux, MacOS и Windows.
- Допустимые инструменты: допускается использование Jupyter Notebook для инференса и предобученных весов.
- Ограничения: решение в 10 строк не будет приниматься. Чтение видео, загрузка весов, отрисовка и сохранение видео должны быть прописаны в явном виде.

## Главные функции
- **Обнаружение людей**: используется модель Yolov5 для обнаружения людей на видео.
- **Отслеживание людей**: каждый обнаруженный человек отслеживается с помощью OpenCV.
- **Оценка уверенности**: для каждого обнаруженного человека указывается степень уверенности модели в правильности ответа.

## Зависимости
- **YOLOv5**: семейство моделей YOLO (You Only Look Once), предназначенных для обнаружения объектов. Yolov5 известен своей скоростью и точностью обнаружения объектов.
- **OpenCV**: библиотека компьютерного зрения с открытым исходным кодом, которая предоставляет инструменты для обработки изображений и видео. В этом проекте OpenCV используется для отслеживания обнаруженных лиц и отрисовки ограничивающих рамок на видео.

## Обучение модели
Модель обучалась на датасете из открытых источников, содержащем более 15000 фотографий с размеченными на них людьми. Обучение производилось на 30 эпохах. Полученные веса содержатся в файле [best.pt](./best.pt).

## Перспективы развития
В дальнейшем возможно дообучение весов на большем количестве эпох. Также при б*о*льших ресурсах могут быть использованы более новые версии моделей YOLO и более сложные их архитектуры.

## Пример работы 
Пример работы программы представлен в [sample](./sample/) в виде исходного видеофайла [crowd.mp4](./sample/crowd.mp4) и выходного видеофайла [crowd_changed_compressed.mp4](./sample/crowd_changed_compressed.mp4)

## Установка
1) Клонирование репозитория:
```
git clone https://github.com/HolyGasCan/testHumanDetection.git
cd testHumanDetection
```
2) Установка необходимых библиотек:
```
pip install -r requirements.txt
```

## Запуск
Для запуска необходимо запустить скрипт [main.py](./main.py):
```
python main.py -i "путь/к/исходному/файлу" -o "путь/к/выходному/файлу/без/расширения"
```
