"""Легковесный менеджер локальной БД для хранения истории распознаваний."""

from __future__ import annotations
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]
DEFAULT_DB_PATH = Path("data/ocr_records.db")


@dataclass(frozen=True)
class OCRRecord:
    """Структура одной записи из таблицы `records`"""
    id: int
    datetime: str
    file_name: str
    text: str


class DBManager:
    """Менеджер SQLite БД для хранения истории OCR-распознаваний"""
    def __init__(self, db_path: Optional[PathLike] = None) -> None:
        self._db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self._conn: Optional[sqlite3.Connection] = None
        self._closed = False

        self._init_db()


    def _init_db(self) -> None:
        """Создаёт директорию, файл БД и таблицу `records`"""
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(
                str(self._db_path),
                check_same_thread=False, 
                timeout=10.0,
            )
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")

            self._create_table()
            logger.info("БД инициализирована: %s", self._db_path)
        except sqlite3.Error as e:
            logger.exception("Ошибка инициализации БД")
            raise RuntimeError(f"Не удалось инициализировать БД: {e}") from e

    def _create_table(self) -> None:
        """Создаёт таблицу `records`, если она не существует"""
        assert self._conn is not None
        schema = """
        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            datetime TEXT NOT NULL,
            file_name TEXT NOT NULL,
            text TEXT NOT NULL
        )
        """
        self._conn.execute(schema)
        self._conn.commit()
        logger.debug("Таблица `records` проверена/создана")


    def __enter__(self) -> DBManager:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        """Fallback: закрываем соединение при сборке мусора"""
        if not self._closed:
            self.close()

    def save_record(self, file_name: str, text: str) -> int:
        """Сохраняет результат распознавания в БД
        Args:
            file_name: имя исходного файла изображения
            text: распознанный текст
        Returns:
            ID созданной записи.
        Raises:
            RuntimeError: если БД закрыта или произошла ошибка записи
        """
        self._ensure_open()
        assert self._conn is not None

        now_iso = datetime.now().isoformat(timespec="seconds")

        try:
            cursor = self._conn.execute(
                "INSERT INTO records (datetime, file_name, text) VALUES (?, ?, ?)",
                (now_iso, file_name, text),
            )
            self._conn.commit()
            record_id = cursor.lastrowid
            logger.info("Запись сохранена: id=%d, file=%s", record_id, file_name)
            return record_id
        except sqlite3.Error as e:
            logger.exception("Ошибка сохранения записи")
            raise RuntimeError(f"Не удалось сохранить запись: {e}") from e

    def get_all_records(self) -> List[OCRRecord]:
        """Возвращает все записи из БД, отсортированные по убыванию ID"""
        self._ensure_open()
        assert self._conn is not None

        try:
            cursor = self._conn.execute(
                "SELECT id, datetime, file_name, text FROM records ORDER BY id DESC"
            )
            rows = cursor.fetchall()
            records = [OCRRecord(*row) for row in rows]
            logger.debug("Загружено записей: %d", len(records))
            return records
        except sqlite3.Error as e:
            logger.exception("Ошибка чтения записей")
            raise RuntimeError(f"Не удалось прочитать записи: {e}") from e

    def get_record_by_id(self, record_id: int) -> Optional[OCRRecord]:
        """Возвращает запись по ID или None, если не найдена"""
        self._ensure_open()
        assert self._conn is not None
        try:
            cursor = self._conn.execute(
                "SELECT id, datetime, file_name, text FROM records WHERE id = ?",
                (record_id,),
            )
            row = cursor.fetchone()
            return OCRRecord(*row) if row else None
        except sqlite3.Error as e:
            logger.exception("Ошибка чтения записи по ID")
            raise RuntimeError(f"Не удалось прочитать запись: {e}") from e

    def delete_record(self, record_id: int) -> bool:
        """Удаляет запись по ID и возвращает True, если запись была удалена"""
        self._ensure_open()
        assert self._conn is not None

        try:
            cursor = self._conn.execute("DELETE FROM records WHERE id = ?", (record_id,))
            self._conn.commit()
            deleted = cursor.rowcount > 0
            if deleted:
                logger.info("Запись удалена: id=%d", record_id)
            else:
                logger.warning("Запись не найдена для удаления: id=%d", record_id)
            return deleted
        except sqlite3.Error as e:
            logger.exception("Ошибка удаления записи")
            raise RuntimeError(f"Не удалось удалить запись: {e}") from e

    def clear_all(self) -> int:
        """Удаляет все записи из БД и возвращает количество удалённых записей"""
        self._ensure_open()
        assert self._conn is not None

        try:
            cursor = self._conn.execute("SELECT COUNT(*) FROM records")
            count = cursor.fetchone()[0]

            self._conn.execute("DELETE FROM records")
            self._conn.commit()
            logger.info("Удалено всех записей: %d", count)
            return count
        except sqlite3.Error as e:
            logger.exception("Ошибка очистки БД")
            raise RuntimeError(f"Не удалось очистить БД: {e}") from e

    def close(self) -> None:
        """Закрывает соединение с БД"""
        if self._closed or self._conn is None:
            return
        try:
            self._conn.close()
            self._closed = True
            logger.debug("Соединение с БД закрыто")
        except sqlite3.Error as e:
            logger.exception("Ошибка закрытия соединения")

    def _ensure_open(self) -> None:
        """Проверяет, что соединение открыто."""
        if self._closed or self._conn is None:
            raise RuntimeError("Соединение с БД закрыто. Создайте новый инстанс DBManager.")

    @property
    def db_path(self) -> Path:
        """Путь к файлу БД"""
        return self._db_path

    def __repr__(self) -> str:
        status = "closed" if self._closed else "open"
        return f"<DBManager path={self._db_path} status={status}>"