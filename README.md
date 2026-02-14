# CV
---

Этот репозиторий содержит проекты по компьютерному зрению, демонстрирующие навыки работы с классификацией, регрессией (Object Localization) и современными системами детекции и трекинга объектов.

---

## Основные проекты

### 1. Object Detection & Tracking (YOLOv8 + ByteTrack)

Система для обработки видеопотоков, выполняющая детекцию и стабильное отслеживание объектов.

* **Файлы:** `main.py`, `transform.py`
* **Стек:** YOLOv8, Supervision, ByteTrack, OpenCV.
* **Особенности:**
* Реализован пайплайн конвертации набора кадров (MOT17) в видео (`transform.py`).
* Трекинг объектов с использованием ByteTrack для сохранения ID объектов между кадрами.
* Оптимизация под CUDA: использование Half Precision (FP16) и Fuse слоев нейросети для ускорения инференса.

### 2. Sun Localization (CNN Regression)

Задача определения координат объекта (Солнца) на синтетическом датасете.

* **Файлы:** `sun_cnn.py`, `sun_gen.py`, `sun_model_cnn.py`
* **Технологии:** Сверточные нейросети (CNN), Pygame (для генерации данных), MSE Loss.
* **Особенности:**
* Самостоятельная генерация датасета с помощью Pygame (`sun_gen.py`).
* Архитектура CNN для предсказания непрерывных величин (координат X, Y).
* Скрипт для тестирования весов обученной модели на новых изображениях.



### 3. MNIST & Deep Learning Fundamentals

Базовые эксперименты с классификацией и оптимизацией нейросетей.

* **Файлы:** `batch_norm.py`, `class_mnist.py`, `class_auto_load.py`
* **Ключевые навыки:**
* **Batch Normalization:** Реализация и исследование влияния на стабильность обучения.
* **Custom Datasets:** Написание собственных классов `Dataset` для эффективной загрузки изображений.
* **Model Management:** Сохранение и автоматическая загрузка состояний модели и оптимизатора (`state_dict`).



---

## Стек

* **Deep Learning:** PyTorch (nn.Module, Autograd, Functional API).
* **Computer Vision:** Ultralytics YOLOv8, OpenCV (обработка видео, отрисовка аннотаций).
* **Data Processing:** Torchvision Transforms, PIL, Supervision.
* **Инструменты:** CUDA для GPU-ускорения, TQDM для визуализации процесса обучения, Pygame для синтеза данных.

---

## Архитектура 

### Детекция и Трекинг (`main.py`)

```python
# Пример использования YOLOv8 + ByteTrack
yolo = YOLO("yolov8n.pt").to(dev)
trk = sv.ByteTrack()
det = trk.update_with_detections(det)

```

### Сверточная сеть для нахождения координат объекта (`sun_cnn.py`)

```python
model = nn.Sequential(
  nn.Conv2d(3, 32, 3, padding='same'),
  nn.ReLU(),
  nn.MaxPool2d(2),
  nn.Conv2d(32, 8, 3, padding='same'),
  nn.ReLU(),
  nn.MaxPool2d(2),
  nn.Conv2d(8, 4, 3, padding='same'),
  nn.ReLU(),
  nn.MaxPool2d(2),
  nn.Flatten(),
  nn.Linear(4096, 128),
  nn.ReLU(),
  nn.Linear(128, 2)
)


```
