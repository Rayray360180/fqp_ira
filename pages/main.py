import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler

import streamlit_app
import streamlit_app2
import streamlit_app3


# Функция для построения 3D графика
def plot_3d(df):
    # Убираем строки, где не хватает данных для осей
    df_plot = df.dropna(subset=['Prediction', 'RMSE', 'R2'])
    fig = px.scatter_3d(df_plot, x='Prediction', y='RMSE', z='R2', color='Model', title='3D-график производительности ML-модели по всем разделам')

    # Устанавливаем размер графика
    fig.update_layout(
        width=800,  # ширина графика в пикселях
        height=600  # высота графика в пикселях
    )

    st.plotly_chart(fig, use_container_width=False)  # Используем False, чтобы применить указанные размеры



# Функция для подготовки и нормализации данных
def prepare_data():
    data = {
        "Model": ["Линейная регрессия", "Случайный лес", "Модель ARIMA", "Градиентный бустинг", "CatBoost",
                  "Линейная регрессия (Ridge)", "Линейная регрессия (Lasso)", "Случайный лес (R2-only)",
                  "Стохастический градиентный спуск", "CatBoost (R2-only)",
                  "Линейная регрессия (large)", "Случайный лес (large)", "CatBoost (large)"],
        "Prediction": [80.38, 70.77, 53.47, 54.89, 66.94, None, None, None, None, None, 23193.00, 28674.88, 25918.61],
        "RMSE": [16.89, 8.30, 4.64, 31.61, 0.27, None, None, None, None, None, None, None, None],
        "R2": [0.71, 0.93, None, -0.19, 1.00, 0.35, 0.32, 0.41, 0.35, 0.43, None, None, None]
    }
    df = pd.DataFrame(data)
    scaler = MinMaxScaler()
    df[['Prediction', 'RMSE', 'R2']] = scaler.fit_transform(df[['Prediction', 'RMSE', 'R2']].astype(float))
    return df


# Навигация и страницы
PAGES = {
    "Главная": "main",
    "Раздел 1": streamlit_app,
    "Раздел 2": streamlit_app2,
    "Раздел 3": streamlit_app3
}


def main():
    st.sidebar.title('Навигация')
    selection = st.sidebar.radio("Перейти к странице:", list(PAGES.keys()))

    if selection == "Главная":
        show_main_page()
    else:
        page = PAGES[selection]
        with st.spinner(f"Загрузка {selection} ..."):
            page.run()


def show_main_page():
    st.title('Главная страница')
    st.write("Добро пожаловать в приложение!")
    st.write("Это главная страница, отсюда вы можете перейти к другим интересующим вас разделам.")

    st.write("Реализация приложения по ВКР Ибраева Рустама")

    # Подготовка данных и визуализация
    df = prepare_data()
    plot_3d(df)


if __name__ == "__main__":
    main()
