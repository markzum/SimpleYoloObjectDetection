import cv2
import time
from ultralytics import YOLO

# Загрузка модели YOLOv8
model = YOLO('yolov8s.pt')

# Открытие веб-камеры (или видеофайла)
cap = cv2.VideoCapture('video.mp4')

# Счетчик FPS
fps = 0
fps_counter = 0
fps_start_time = time.time()

while True:
    # Чтение кадра
    ret, frame = cap.read()

    # Прекращение работы, если кадр не получен
    if not ret:
        break

    # Scale frame to 640 with ratio save
    scale_percent = 640 / frame.shape[1]
    width = int(frame.shape[1] * scale_percent)
    height = int(frame.shape[0] * scale_percent)
    frame = cv2.resize(frame, (width, height))
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Детекция объектов
    results = model.predict(frame)

    annotated_frame = results[0].plot()

    # Обновление счетчика FPS
    fps_counter += 1
    if fps_counter >= 10:
        fps = fps_counter / (time.time() - fps_start_time)
        fps_start_time = time.time()
        fps_counter = 0

        # Вывод FPS на экран
    cv2.putText(annotated_frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Отображение результата
    cv2.imshow('Object Detection with YOLOv8', annotated_frame)

    # Завершение работы по нажатию клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов и закрытие окон
cap.release()
cv2.destroyAllWindows()
