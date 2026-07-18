"""Модуль предобработки изображений для улучшения качества распознавания рукописного текста"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional, Tuple, Union
import cv2
import numpy as np

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


def load_image(image_path: PathLike, as_gray: bool = False) -> np.ndarray:
    """Загружает изображение с диска.
    Args:
        image_path: путь к файлу изображения.
        as_gray: если True, сразу конвертирует в оттенки серого

    Raises:
        FileNotFoundError: если файл не найден
        ValueError: если OpenCV не смог декодировать файл
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Изображение не найдено: {path}")

    flag = cv2.IMREAD_GRAYSCALE if as_gray else cv2.IMREAD_COLOR
    img = cv2.imread(str(path), flag)
    if img is None:
        raise ValueError(f"Не удалось прочитать изображение: {path}")

    logger.debug("Загружено изображение %s (%dx%d)", path.name, img.shape[1], img.shape[0])
    return img


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Конвертирует BGR-изображение в оттенки серого (или возвращает как есть)"""
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Неожиданная размерность изображения: {image.shape}")


def resize_image(image: np.ndarray, target_width: Optional[int] = None,max_side: int = 2000, interpolation: int = cv2.INTER_AREA) -> np.ndarray:
    """Масштабирует изображение, сохраняя пропорции

    Приоритет:
      1. Если задан `target_width` — ширина приводится к нему
      2. Иначе, если большая сторона превышает `max_side` — выполняется downscale

    Args:
        image: входное изображение
        target_width: целевая ширина (опционально)
        max_side: максимальный размер большей стороны
        interpolation: метод интерполяции (по умолчанию INTER_AREA)
    """

    h, w = image.shape[:2]
    if target_width is not None:
        scale = target_width / w
    elif max(h, w) > max_side:
        scale = max_side / max(h, w)
    else:
        return image 

    if abs(scale - 1.0) < 1e-3:
        return image

    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    logger.debug("Resize: %dx%d -> %dx%d (scale=%.3f)", w, h, new_w, new_h, scale)
    return resized


def denoise(image: np.ndarray, strength: int = 10) -> np.ndarray:
    """Мягкое шумоподавление bilateral filter """
    return cv2.bilateralFilter(image, d=5, sigmaColor=strength, sigmaSpace=strength)


def enhance_contrast(image: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """Усиливает локальный контраст через CLAHE — критично для рукописного текста"""
    gray = to_grayscale(image)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(gray)


def adaptive_binarization(image: np.ndarray, block_size: int = 31, c: int = 10, method: str = "gaussian") -> np.ndarray:
    """Адаптивная бинаризация — устойчива к неравномерной освещённости скана/фото.

    Args:
        image: входное изображение 
        block_size: размер окна адаптивного порога (должен быть нечётным)
        c: константа, вычитаемая из взвешенного среднего
        method: 'gaussian' или 'mean'

    Returns:
        Бинарное изображение (dtype=uint8, значения 0/255)
    """
    gray = to_grayscale(image)

    if block_size % 2 == 0:
        block_size += 1  

    method_flag = cv2.ADAPTIVE_THRESH_GAUSSIAN_C if method == "gaussian" else cv2.ADAPTIVE_THRESH_MEAN_C

    binary = cv2.adaptiveThreshold(
        gray,
        maxValue=255,
        adaptiveMethod=method_flag,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=block_size,
        C=c,
    )
    logger.debug("Адаптивная бинаризация: block=%d, C=%d, method=%s", block_size, c, method)
    return binary


def preprocess_image(
    image_path: PathLike,
    *,
    target_width: Optional[int] = None,
    max_side: int = 2000,
    do_denoise: bool = True,
    do_contrast: bool = True,
    do_binarize: bool = False,
    binarization_block_size: int = 31,
    binarization_c: int = 10,
) -> np.ndarray:
    """Полная предобработки изображения для OCR: загрузка, масштабирование, шумоподавление, контраст и бинаризация"""
    img = load_image(image_path, as_gray=True)

    img = resize_image(img, target_width=target_width, max_side=max_side)

    if do_denoise:
        img = denoise(img)

    if do_binarize:
        img = adaptive_binarization(
            img,
            block_size=binarization_block_size,
            c=binarization_c,
        )
    elif do_contrast:
        img = enhance_contrast(img)

    return img


def save_image(image: np.ndarray, output_path: PathLike) -> Path:
    """Сохраняет обработанное изображение на диск."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), image):
        raise IOError(f"Не удалось сохранить изображение: {out}")
    logger.debug("Сохранено изображение: %s", out)
    return out