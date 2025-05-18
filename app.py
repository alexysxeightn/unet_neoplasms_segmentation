import streamlit as st
from PIL import Image
import numpy as np

from model import load_model, predict
from utils import preprocess_image, apply_mask

st.set_page_config(page_title="Сегментация новообразований", layout="wide")
st.title("Сегментация новообразований")


# Загрузка модели
@st.cache_resource
def get_model(path="unet.pth"):
    return load_model(path)


model = get_model()

uploaded_files = st.file_uploader(
    "Загрузите одно или несколько изображений",
    type=["bmp", "jpg", "png", "jpeg"],
    accept_multiple_files=True,
)

if uploaded_files:
    if st.button("Выполнить сегментацию"):
        results = []

        with st.spinner("Обработка изображений..."):
            for uploaded_file in uploaded_files:
                image = Image.open(uploaded_file).convert("RGB")
                tensor, original_size = preprocess_image(image)
                mask = predict(model, tensor)
                segmented_image = apply_mask(image, mask, original_size)

                results.append({"original": image, "segmented": segmented_image})

        st.subheader("Результаты сегментации:")
        cols_per_row = 3  # количество пар изображений на строку
        for i in range(0, len(results), cols_per_row):
            cols = st.columns(cols_per_row * 2)  # одна колонка на оригинал + маска
            batch = results[i : i + cols_per_row]
            idx = 0
            for res in batch:
                with cols[idx]:
                    st.image(res["original"], caption="Оригинал", width=256)
                with cols[idx + 1]:
                    st.image(res["segmented"], caption="С маской", width=256)
                idx += 2
