# Сервис сегментации новообразований на основе PyTorch(U-NET) + Streamlit

Простой веб-сервис для сегментации новообразований с использованием модели U-NET, реализованной на `PyTorch`, и интерфейса на `Streamlit`. Позволяет загружать одно или несколько изображений (включая формат `.bmp`), применять модель и отображать результаты с нанесённой маской.

## Структура проекта

```
│
├── train.ipynb            # Jupyter с обучением модели U-NET
├── unet.pth               # Веса модели U-NET
├── app.py                 # Основное приложение Streamlit
├── model.py               # Логика загрузки и инференса модели
├── utils.py               # Функции предобработки и постобработки
├── requirements.txt       # Список зависимостей
└── README.md              # Документация
```

## Установка

1. Клонировать репозиторий
```
git clone https://github.com/alexysxeightn/unet_neoplasms_segmentation.git
cd unet_neoplasms_segmentation
```

2. Установить зависимости
```
pip install -r requirements.txt
```

3. Запустить приложение
```
streamlit run app.py
```

## Как использовать

1. Загрузите одно или несколько изображений (можно зажать Ctrl/Cmd для множественного выбора).
2. Нажмите кнопку "Выполнить сегментацию".
3. Результаты будут показаны в виде галереи:
- Оригинал
- Изображение с маской (синяя полупрозрачная область)

## Архитектура модели

В проекте используется модель типа U-Net — популярная архитектура для задач семантической сегментации изображений. Она состоит из энкодера (`downsampler`) и декодера (`upsampler`) с механизмом skip-соединений, что позволяет сохранять детали изображения на разных уровнях абстракции.

![Снимок экрана 2025-05-18 184906](https://github.com/user-attachments/assets/3acd4d2e-946d-419f-a88f-e682ee982579)

Для выбора лучшей модели в качестве метрики использовалось среднее `IoU` - степень пересечения между предсказанной маской и истинной маской (ground truth): $IoU = \frac{Intersection (TP)}{Union (TP + FP + FN)}$
- `TP (True Positive)` — пиксели, правильно классифицированные как объект.
- `FP (False Positive)` — пиксели, ошибочно отнесённые к объекту.
- `FN (False Negative)` — пиксели объекта, ошибочно отнесённые к фону.

![IoU](https://github.com/user-attachments/assets/d653fd39-ab09-44a9-96e5-d81043b8fea3)

### Общая структура:
- `DoubleConv`: два последовательных блока свёртки + `BatchNorm` + `ReLU`
- `MaxPool2d`: уменьшает размерность пространственного представления в 2 раза
- `ConvTranspose2d`: увеличивает пространственные размерности (transposed convolution / deconvolution)
- `Skip-connections`: соединяют соответствующие слои энкодера и декодера
- `Bottleneck`: самый глубокий слой между энкодером и декодером
- `Final Convolution`: преобразует фичи в выходной канал маски

### Параметры модели:
```
UNET(
  in_channels=3,
  out_channels=1,
  features=[64, 128, 256, 512]
)
```
- `in_channels`: 3 (RGB)
- `out_channels`: 1 (бинарная маска)
- `features`: количество фильтров на каждом уровне энкодера

Подробно архитектура модели разобрана в статье [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

Модель обучалась на датасете [PH2Dataset](https://www.kaggle.com/datasets/kanametov/ph2dataset) c использованием аугментаций:
```
Resize(256, 256)
HorizontalFlip(p=0.5)
VerticalFlip(p=0.5)
RandomRotate90(p=0.5)
ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.2)
ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.2)
CLAHE(p=0.5)
GridDistortion(p=0.5)
Normalize(mean=[0.0], std=[1.0])
```
