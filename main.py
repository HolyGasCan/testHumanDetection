import cv2
import numpy as np
import torch
import argparse

import platform
if platform.system() == "Windows":
    import pathlib
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath


DEFAULT_MODEL_PATH = './best.pt'
GREEN_COLOUR = (0, 255, 0)
LINE_THICKNESS = 2
FONT_SCALE = 1
CONFIDENCE_ROUND = 2


def createModel(path=None):
    """
    Возвращает созданную модель по переданному пути,
    иначе использует стандартный путь.
    """
    model_path = DEFAULT_MODEL_PATH if path is None else path
    torch.hub.set_dir('./torch_cache')
    model = torch.hub.load('ultralytics/yolov5',
                           'custom',
                           model_path,
                           trust_repo=True)

    return model


def argsParser():
    """Обрабатывает аргументы при запуске из консоли."""
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("-i",
                           "--input",
                           type=str,
                           default=None,
                           help="Путь к исходному видеофайлу.")
    arg_parse.add_argument("-o",
                           "--output",
                           type=str,
                           default=None,
                           help="Путь к выходному видеофайлу.")
    args = vars(arg_parse.parse_args())

    return args


def checkPaths(input_path, output_path):
    """Проверяет, введены ли пути к исходному и выходному видеофайлам."""
    paths_correct = True

    if input_path is None:
        print("Не введён путь к исходному видеофайлу.")
        paths_correct = False
    elif output_path is None:
        print("Не введён путь к выходному видеофайлу.")
        paths_correct = False

    if not paths_correct:
        return 1

    return 0


def main():
    """Главная функция."""
    args = argsParser()
    input_path = args["input"]
    output_path = args["output"]

    if checkPaths(input_path, output_path):
        return 1

    torch_model = createModel()
    capture = cv2.VideoCapture(input_path)    # Создание потока видео из файла

    # Получение высоты, ширины и FPS исходного видеофайла
    # для дальнейшего сохранения полученного видеофайла
    video_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_fps = capture.get(cv2.CAP_PROP_FPS)

    # Указание кодека и создание объекта для сохранения видеофайла
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output = cv2.VideoWriter(f'{output_path}.mp4',
                             fourcc,
                             video_fps,
                             (video_width, video_height))

    # Получение первого кадра исходного видеофайла и проверка потока видео
    retval, frame = capture.read()
    if not retval:
        capture.release()
        print("Исходный видеофайл не найден.")
        return 2

    print("Происходит преобразование видеофайла. Подождите...")

    while True:
        # Основной цикл по каждому кадру с детекцией
        # и отрисовкой найденных людей
        results = torch_model(frame)

        # Преобразование результатов работы в датафрейм
        results_pandas = results.pandas().xyxy[0]

        human_results = results_pandas.loc[results_pandas['name'] == 'person']

        for index, row in human_results.iterrows():
            x1 = int(row['xmin'])
            y1 = int(row['ymin'])
            x2 = int(row['xmax'])
            y2 = int(row['ymax'])
            name = str(row['name'])
            confidence = np.round(float(row['confidence']), CONFIDENCE_ROUND)

            cv2.rectangle(frame,
                          (x1, y1),
                          (x2, y2),
                          GREEN_COLOUR,
                          LINE_THICKNESS)
            cv2.putText(frame,
                        f'{name} {str(confidence)}',
                        (x1, y1),
                        cv2.FONT_HERSHEY_COMPLEX,
                        FONT_SCALE,
                        GREEN_COLOUR,
                        LINE_THICKNESS,
                        cv2.LINE_AA)

        output.write(frame)

        retval, frame = capture.read()
        if not retval:
            break

    capture.release()
    print("Видеофайл преобразован!")

    return 0


if __name__ == "__main__":
    main()
