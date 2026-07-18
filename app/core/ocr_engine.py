"""OCR-движок на базе EasyOCR
Поддерживает русский и английский языки"""

from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
from app.utils import preprocess_image

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

@dataclass
class OCRResult:
    """Результат распознавания одного изображения."""

    text: str
    blocks: List[dict] = field(default_factory=list)
    elapsed_sec: float = 0.0
    image_size: tuple = (0, 0)

    @property
    def avg_confidence(self) -> float:
        if not self.blocks:
            return 0.0
        return float(np.mean([b["confidence"] for b in self.blocks]))


class OCREngine:
    """Обёртка над EasyOCR """

    def __init__(
        self,
        languages: Optional[List[str]] = None,
        use_gpu: bool = False,
        model_storage_directory: Optional[str] = None,
        min_confidence: float = 0.3,
        preprocess: bool = True,
    ) -> None:
        self.languages = languages or ["en", "ru"]
        self.use_gpu = use_gpu
        self.model_storage_directory = model_storage_directory
        self.min_confidence = min_confidence
        self.preprocess = preprocess

        self._reader = None  
        logger.info(
            "OCREngine создан: langs=%s, gpu=%s, min_conf=%.2f, preprocess=%s",
            self.languages,
            self.use_gpu,
            self.min_confidence,
            self.preprocess,
        )


    def _ensure_loaded(self) -> None:
        if self._reader is not None:
            return

        try:
            import easyocr  
        except ImportError as e:
            raise RuntimeError(
                "EasyOCR не установлен. Выполните: pip install easyocr"
            ) from e

        logger.info("Загрузка моделей EasyOCR для языков %s ...", self.languages)
        t0 = time.perf_counter()
        try:
            self._reader = easyocr.Reader(
                self.languages,
                gpu=self.use_gpu,
                model_storage_directory=self.model_storage_directory,
                verbose=False,
            )
        except Exception as e:
            logger.exception("Ошибка инициализации EasyOCR")
            raise RuntimeError(f"Не удалось инициализировать EasyOCR: {e}") from e

        logger.info("Модели EasyOCR загружены за %.2f сек", time.perf_counter() - t0)

    def predict(
        self,
        image_source: Union[PathLike, np.ndarray],
        *,
        min_confidence: Optional[float] = None,
    ) -> OCRResult:
        """Распознаёт текст на изображении
        Args:
            image_source: путь к изображению или numpy-массив 
            min_confidence: переопределяет порог уверенности для этого вызова

        Returns:
            OCRResult с текстом, блоками и метаданными.

        Raises:
            FileNotFoundError: если файл не найден.
            RuntimeError: при ошибке распознавания.
        """
        self._ensure_loaded()
        threshold = min_confidence if min_confidence is not None else self.min_confidence

        t0 = time.perf_counter()
        try:
            image = self._load_and_preprocess(image_source)
        except FileNotFoundError:
            raise
        except Exception as e:
            logger.exception("Ошибка предобработки изображения")
            raise RuntimeError(f"Ошибка предобработки: {e}") from e

        try:
            raw_results = self._reader.readtext(image)
        except Exception as e:
            logger.exception("Ошибка OCR")
            raise RuntimeError(f"EasyOCR вернул ошибку: {e}") from e

        blocks = self._filter_and_format(raw_results, threshold)
        text = self._join_blocks(blocks)
        elapsed = time.perf_counter() - t0

        logger.info(
            "OCR завершён за %.2f сек: блоков=%d, avg_conf=%.2f, текст_len=%d",
            elapsed,
            len(blocks),
            np.mean([b["confidence"] for b in blocks]) if blocks else 0.0,
            len(text),
        )

        return OCRResult(
            text=text,
            blocks=blocks,
            elapsed_sec=elapsed,
            image_size=(image.shape[1], image.shape[0]),
        )

    def _load_and_preprocess(self, source: Union[PathLike, np.ndarray]) -> np.ndarray:
        if isinstance(source, np.ndarray):
            image = source
        else:
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"Файл не найден: {path}")
            if self.preprocess:
                image = preprocess_image(path)
            else:
                from app.utils.image_processing import load_image
                image = load_image(path, as_gray=True)
        return image

    @staticmethod
    def _filter_and_format(raw_results: list, threshold: float) -> List[dict]:
        """Фильтрует результаты EasyOCR по уверенности и приводит к единому формату."""
        blocks: List[dict] = []
        for bbox, text, confidence in raw_results:
            if confidence < threshold:
                continue
            blocks.append(
                {
                    "text": text.strip(),
                    "confidence": float(confidence),
                    "bbox": [[int(p[0]), int(p[1])] for p in bbox],
                }
            )
        return blocks

    @staticmethod
    def _join_blocks(blocks: List[dict]) -> str:
        if not blocks:
            return ""
        return " ".join(b["text"] for b in blocks if b["text"])

    def reload(self) -> None:
        """Принудительно перезагружает модели"""
        self._reader = None
        self._ensure_loaded()