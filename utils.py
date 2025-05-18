from PIL import Image
import numpy as np
import torch
import cv2


def preprocess_image(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB")
    original_size = image.size  # (width, height)
    image = image.resize((256, 256))  # ресайз под вход модели
    img_np = np.array(image).transpose(2, 0, 1) / 255.0  # CHW, нормализация
    return torch.tensor(img_np, dtype=torch.float32).unsqueeze(0), original_size


def apply_mask(
    image: Image.Image, mask: np.ndarray, original_size: tuple
) -> Image.Image:
    # Маска сейчас (256, 256), нужно привести к original_size (W, H)
    mask_resized = cv2.resize(
        mask.astype(np.uint8), original_size, interpolation=cv2.INTER_NEAREST
    )

    # Создаем RGBA слой для маски
    mask_rgba = np.zeros((*mask_resized.shape, 4), dtype=np.uint8)
    mask_rgba[mask_resized == 1] = [0, 0, 255, 100]  # синий полупрозрачный цвет

    # Преобразуем изображение в массив
    background = np.array(image)
    overlay = Image.fromarray(mask_rgba, "RGBA")

    # Накладываем маску
    result = Image.new("RGBA", image.size)
    result.paste(image, (0, 0))
    result.paste(overlay, (0, 0), overlay)
    return result.convert("RGB")
