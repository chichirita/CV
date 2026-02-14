# CV
---

Этот репозиторий содержит проекты по компьютерному зрению, демонстрирующие навыки работы с классификацией, регрессией и современными системами детекции и трекинга объектов.

---

## Основные проекты

### 1. Детекция и Трекинг (YOLOv8 + ByteTrack)

Система для обработки видеопотоков, выполняющая детекцию и стабильное отслеживание объектов.

* **Файлы:** `main.py`, `transform.py`
* **Стек:** YOLOv8, Supervision, ByteTrack, OpenCV.
* **Особенности:**
* Реализован пайплайн конвертации набора кадров (MOT17) в видео (`transform.py`).
* Трекинг объектов с использованием ByteTrack для сохранения ID объектов между кадрами.
* Оптимизация под CUDA: использование Half Precision и fuse слоев нейросети для ускорения инференса.

### 2. Нахождение координат солнца (регрессия CNN)

Задача определения координат объекта на синтетическом датасете.

* **Файлы:** `sun_cnn.py`, `sun_gen.py`, `sun_model_cnn.py`
* **Технологии:** CNN, Pygame (для генерации данных), torchvision.
* **Особенности:**
* Самостоятельная генерация датасета с помощью Pygame (`sun_gen.py`).
* Архитектура CNN для предсказания непрерывных величин (координат X, Y).
* Скрипт для тестирования весов обученной модели на новых изображениях.

### 3. Базовые модели глубокого обучения

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

### Сверточная сеть для распознавания сторон кости (`dice_cnn.py`)

```python
model = keras.Sequential([
  layers.Conv2D(32, (3, 3), activation='relu',
  input_shape=(self.image_size, self.image_size, 3)),
  layers.MaxPooling2D((2, 2)),
  layers.BatchNormalization(),

  layers.Conv2D(64, (3, 3), activation='relu'),
  layers.MaxPooling2D((2, 2)),
  layers.BatchNormalization(),

  layers.Conv2D(128, (3, 3), activation='relu'),
  layers.MaxPooling2D((2, 2)),
  layers.BatchNormalization(),

  layers.Conv2D(256, (3, 3), activation='relu'),
  layers.MaxPooling2D((2, 2)),
  layers.BatchNormalization(),

  layers.Flatten(),
  layers.Dense(512, activation='relu'),
  layers.Dropout(0.5),
  layers.Dense(256, activation='relu'),
  layers.Dropout(0.3),
  layers.Dense(6, activation='softmax')
])

```

Визуализация данных:
<img width="1489" height="274" alt="image" src="https://github.com/user-attachments/assets/93f751b2-ac74-4649-bc3f-fc5ddde58d5b" />
Результаты обучения:
<img width="1189" height="390" alt="image" src="https://github.com/user-attachments/assets/96ef4b31-5aff-4a11-9a14-891f5aeb6ab0" />
