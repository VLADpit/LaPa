"""Streamlit-интерфейс для приложения распознавания рукописного текста"""

from __future__ import annotations
import sys
import logging
from pathlib import Path
from typing import Optional
import numpy as np
import streamlit as st
from PIL import Image
from core import DBManager, OCRRecord
from core import OCREngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


PAGE_TITLE = "Рукописный OCR"
SUPPORTED_FORMATS = ("png", "jpg", "jpeg")
DB_PATH = Path("data/ocr_records.db")


@st.cache_resource(show_spinner="Загрузка OCR-модели...")
def get_ocr_engine() -> OCREngine:
    """Возвращает singleton-инстанс OCREngine"""
    logger.info("Инициализация OCREngine")
    return OCREngine(languages=["en", "ru"], use_gpu=False, min_confidence=0.3)


@st.cache_resource(show_spinner="Подключение к БД...")
def get_db_manager() -> DBManager:
    """Возвращает singleton-инстанс DBManager"""
    logger.info("Инициализация DBManager: %s", DB_PATH)
    return DBManager(db_path=DB_PATH)


st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon="🖋️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_session_state() -> None:
    """Инициализирует ключи session_state, если их ещё нет."""
    defaults = {
        "uploaded_image": None,        
        "image_name": None,            
        "recognized_text": "",         
        "is_recognized": False,        
        "selected_record_id": None,   
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


def render_sidebar() -> None:
    """Отрисовывает боковую панель с историей"""
    with st.sidebar:
        st.header("История распознаваний")

        db = get_db_manager()
        try:
            records = db.get_all_records()
        except Exception as e:
            logger.exception("Ошибка чтения истории")
            st.error(f"Не удалось загрузить историю: {e}")
            records = []

        if not records:
            st.info("История пуста. Распознайте первое изображение!")
            return

        st.caption(f"Всего записей: **{len(records)}**")
        st.divider()

        options = {f"{r.id}. {r.file_name} ({r.datetime[:10]})": r for r in records}
        selected_label = st.selectbox(
            "Выберите запись",
            options=list(options.keys()),
            key="sidebar_select",
        )

        if selected_label:
            selected_record: OCRRecord = options[selected_label]
            st.session_state.selected_record_id = selected_record.id

            with st.expander("Детали записи", expanded=True):
                st.markdown(f"**Файл:** `{selected_record.file_name}`")
                st.markdown(f"**Дата:** {selected_record.datetime}")
                st.markdown("**Текст:**")
                st.text(selected_record.text)

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Загрузить в редактор", use_container_width=True):
                        st.session_state.recognized_text = selected_record.text
                        st.session_state.image_name = selected_record.file_name
                        st.session_state.uploaded_image = None  
                        st.session_state.is_recognized = True
                        st.rerun()
                with col2:
                    if st.button("Удалить", use_container_width=True):
                        try:
                            db.delete_record(selected_record.id)
                            st.success("Запись удалена")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Ошибка удаления: {e}")

        st.divider()
        if st.button("🧹 Очистить всю историю", use_container_width=True):
            try:
                count = db.clear_all()
                st.success(f"Удалено записей: {count}")
                st.rerun()
            except Exception as e:
                st.error(f"Ошибка очистки: {e}")


def render_main_area() -> None:
    """Отрисовывает главную зону приложения."""
    st.title(PAGE_TITLE)
    st.caption("Распознавание рукописного текста (RU/EN) на базе EasyOCR")

    uploaded_file = st.file_uploader(
        "Загрузите изображение с рукописным текстом",
        type=list(SUPPORTED_FORMATS),
        help="Поддерживаются форматы PNG, JPG, JPEG",
    )

    if uploaded_file is not None:
        st.session_state.uploaded_image = uploaded_file.getvalue()
        st.session_state.image_name = uploaded_file.name
        if not st.session_state.is_recognized:
            st.session_state.recognized_text = ""

    if st.session_state.uploaded_image is not None:
        image = Image.open(
            __import__("io").BytesIO(st.session_state.uploaded_image)
        )
        st.image(
            image,
            caption=f"{st.session_state.image_name}",
             use_column_width=True,
        )

    col_btn, col_spacer = st.columns([1, 4])
    with col_btn:
        recognize_clicked = st.button(
            "Распознать",
            type="primary",
        )

    if recognize_clicked and st.session_state.uploaded_image is not None:
        with st.spinner("Распознавание текста..."):
            try:
                engine = get_ocr_engine()
                image_bytes = st.session_state.uploaded_image
                nparr = np.frombuffer(image_bytes, dtype=np.uint8)
                image_cv = Image.open(__import__("io").BytesIO(image_bytes)).convert("RGB")
                image_np = np.array(image_cv)[:, :, ::-1] 

                result = engine.predict(image_np)
                st.session_state.recognized_text = result.text
                st.session_state.is_recognized = True

                st.success(
                    f"Распознано за {result.elapsed_sec:.2f} сек "
                    f"(средняя уверенность: {result.avg_confidence:.2f})"
                )
                logger.info(
                    "Распознавание завершено: file=%s, blocks=%d, text_len=%d",
                    st.session_state.image_name,
                    len(result.blocks),
                    len(result.text),
                )
            except Exception as e:
                logger.exception("Ошибка распознавания")
                st.error(f"Ошибка распознавания: {e}")

    st.subheader("Результат распознавания")
    edited_text = st.text_area(
        "Отредактируйте текст при необходимости:",
        value=st.session_state.recognized_text,
        height=250,
        placeholder="Здесь появится распознанный текст...",
        label_visibility="collapsed",
    )
    if edited_text != st.session_state.recognized_text:
        st.session_state.recognized_text = edited_text

    col_save, col_download, col_spacer = st.columns([1, 1, 3])

    with col_save:
        save_clicked = st.button(
            "Сохранить в БД",
            use_container_width=True,
            disabled=not st.session_state.is_recognized
            or not st.session_state.recognized_text.strip()
            or st.session_state.image_name is None,
        )

    if save_clicked:
        try:
            db = get_db_manager()
            record_id = db.save_record(
                file_name=st.session_state.image_name,
                text=st.session_state.recognized_text)
            st.success(f"Сохранено в БД (ID: {record_id})")
            logger.info("Запись сохранена: id=%d, file=%s", record_id, st.session_state.image_name)
            st.rerun()  
        except Exception as e:
            logger.exception("Ошибка сохранения в БД")
            st.error(f"Ошибка сохранения: {e}")

    with col_download:
        if st.session_state.recognized_text.strip():
            file_name_txt = (
                Path(st.session_state.image_name).stem + ".txt"
                if st.session_state.image_name
                else "recognized_text.txt"
            )
            st.download_button(
                label="Скачать как .txt",
                data=st.session_state.recognized_text.encode("utf-8"),
                file_name=file_name_txt,
                mime="text/plain",
                use_container_width=True,
            )
        else:
            st.button(
                "Скачать как .txt",
                use_container_width=True,
                disabled=True,
            )

def main() -> None:
    render_sidebar()
    render_main_area()

if __name__ == "__main__":
    main()