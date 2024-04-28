def run():
    import streamlit as st
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    from sklearn.ensemble import RandomForestRegressor

    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.tsa.stattools import adfuller

    from sklearn.ensemble import GradientBoostingRegressor
    from catboost import CatBoostRegressor

    from sklearn.preprocessing import OneHotEncoder
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import RidgeCV, LassoCV

    # Функция для установки пользовательских стилей
    def set_css():
        st.markdown("""
        <style>
        .markdown-font {
            font-size:16px;
            font-family: 'Arial';
            line-height: 1.6;
        }
        </style>
        """, unsafe_allow_html=True)

    # Установка пользовательских CSS стилей
    set_css()

    # Основной заголовок
    st.markdown('<p class="big-font title-font">ВКР Ибраева Рустама</p>', unsafe_allow_html=True)

    # Подзаголовок
    st.subheader('Раздел 2: Прогнозирование ожидаемой заработной платы на основе информации из резюме кандидатов')

    # Установка пользовательских стилей для Markdown
    def set_custom_styles():
        st.markdown("""
        <style>
        .big-font {
            font-size:20px !important;
            font-weight: bold;
        }
        .data-font {
            font-size:16px !important;
            font-family: 'Courier New', Courier, monospace;
        }
        </style>
        """, unsafe_allow_html=True)

    set_custom_styles()

    # Основной заголовок приложения
    st.markdown('<p class="big-font">Исходные данные и предобработка набора данных</p>', unsafe_allow_html=True)

    # Раскрывающийся блок с Markdown текстом
    expander = st.expander("Подробнее о наборе данных")
    markdown_text = """
    ## Описание набора данных
    В проекте используется база резюме, выгруженная с сайта поиска вакансий hh.ru.
    Датасет (до предобработки) содержит 44744 обезличенные записи по 12 признакам (после от некоторых признаков откажусь в процессе создания моделей):
    * **Пол, возраст** - пол и возраст соискателя;
    * **ЗП** - ожидаемая заработная плата;
    * **Ищет работу на должность:** - сведенья о желаемой должности;
    * **Город, переезд, командировки** - город проживания соискателя, его готовность к переезду и командировкам;
    * **Занятость - желаемая занятость в виде одной из категорий:** полная занятость, частичная занятость, проектная работа, волонтерство, стажировка;
    * **График** - желаемый график работы в виде одной из категорий: полный день, сменный график, гибкий график, удаленная работа, вахтовый метод;
    * **Опыт работы** - сведенья об опыте работы соискателя;
    * **Последнее/нынешнее место работы** - сведенья последнем/нынешнем месте работы;
    * **Последняя/нынешняя должность** - сведенья о последней/нынешней должности;
    * **Образование и ВУЗ** - уровень образования соискателя и наименование законченного учебного заведения;
    * **Обновление резюме** - дата и время последнего обновления резюме соискателем;
    * **Авто** - наличие у соискателя автомобиля.
    Также дополнительно берутся данные для датасета из файла который содержит сведенья о курсах валют.
    
    
    ## Предобработка данных
    * **Предобработка данных вынесена за рамки приложения, но освещается в ВКР.**
    В данной задаче была проведена работа по исследованию и очистке данных на примере датасета содержащего резюме соискателей с сайта поиска вакансий hh.ru.
    Было проведено преобразование данных путем формирования новых информативных признаков и удаления исходных, не несущих полезной информации. 
    Выполнено исследование зависимостей в данных. Проведена очистка данных: удалены дублированные записи, проведена обработка пропусков в данных, ликвидированы выбросы.
    
    
    ## Главная цель
    Построить модель, которая бы автоматически определяла примерный уровень заработной платы, подходящей пользователю, исходя из информации, которую он указал о себе.
    Для достижения цели были использованы разные методы машинного обучения.
    """
    expander.markdown(markdown_text, unsafe_allow_html=True)

    st.write("Обзор исходного набора данных")

    st.write("Прогнозирование методом (Ridge): Линейная регрессия с L2-регуляризацией (штраф за большие веса).")

    st.write(
        "Прогнозирование методом (Lasso): Линейная регрессия с L1-регуляризацией, которая может приводить к обнулению некоторых коэффициентов, тем самым выполняя отбор признаков.")

    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.linear_model import Ridge, Lasso, LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

    # Загрузка данных
    df = pd.read_csv(r"D:\DataScience\dst_new.csv")

    # Список столбцов, которые нужно удалить
    columns_to_drop = ["Авто", "Последнее/нынешнее место работы", "Ищет работу на должность:",
                       "Последняя/нынешняя должность"]

    # Удаление столбцов
    df_reduced = df.drop(columns=columns_to_drop)

    # Кодирование категориальных переменных
    encoder = OneHotEncoder()
    categorical_columns = df_reduced.select_dtypes(include=['object']).columns
    df_encoded = pd.DataFrame(encoder.fit_transform(df_reduced[categorical_columns]).toarray(),
                              columns=encoder.get_feature_names_out(categorical_columns))
    df_reduced = df_reduced.drop(categorical_columns, axis=1)
    df_reduced = pd.concat([df_reduced, df_encoded], axis=1)

    # Разделение данных на обучающую и тестовую выборки
    X = df_reduced.drop('ЗП (руб)', axis=1)
    y = df_reduced['ЗП (руб)']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Масштабирование признаков
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Обучение моделей
    ridge_reg = Ridge(alpha=1.0)
    lasso_reg = Lasso(alpha=1.0)
    ridge_reg.fit(X_train_scaled, y_train)
    lasso_reg.fit(X_train_scaled, y_train)

    # Предсказание
    y_pred_ridge = ridge_reg.predict(X_test_scaled)
    y_pred_lasso = lasso_reg.predict(X_test_scaled)

    # Вычисление метрик
    mse_ridge = mean_squared_error(y_test, y_pred_ridge)
    mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
    r2_ridge = r2_score(y_test, y_pred_ridge)
    mse_lasso = mean_squared_error(y_test, y_pred_lasso)
    mae_lasso = mean_absolute_error(y_test, y_pred_lasso)
    r2_lasso = r2_score(y_test, y_pred_lasso)

    # Вывод результатов
    st.write("Ridge Regression: MSE =", mse_ridge, ", MAE =", mae_ridge, ", R2 =", r2_ridge)
    st.write("Lasso Regression: MSE =", mse_lasso, ", MAE =", mae_lasso, ", R2 =", r2_lasso)

    # График для Ridge Regression
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred_ridge, alpha=0.3)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    ax.set_title('Ridge Regression Predictions')
    st.pyplot(fig)

    # График для Lasso Regression
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred_lasso, alpha=0.3)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    ax.set_title('Lasso Regression Predictions')
    st.pyplot(fig)

    # Установка пользовательских стилей для Markdown
    st.markdown("""
    <style>
    .detail {
        font-size: 16px;
    }
    .subheader {
        color: #0c4b33;
        font-weight: bold;
    }
    .highlight {
        background-color: #e6f7ff;
        padding: 5px 10px;
        border-radius: 5px;
        border-left: 5px solid #2196f3;
    }
    </style>
    """, unsafe_allow_html=True)

    # Создание раскрывающегося блока для анализа
    expander = st.expander("Подробный анализ результатов моделей")
    with expander:
        st.markdown("""
    <div class="detail">
        <p class="subheader">Общий обзор метрик:</p>
        <ul>
            <li><span class="highlight">MSE (Среднеквадратичная Ошибка)</span>: Высокие значения для обеих моделей указывают на большие ошибки в предсказаниях.</li>
            <li><span class="highlight">MAE (Средняя Абсолютная Ошибка)</span>: Средние ошибки около 32,592 у обеих моделей подтверждают значительные отклонения от фактических значений.</li>
            <li><span class="highlight">R² (Коэффициент детерминации)</span>: Значения около 0.329 указывают на недостаточное объяснение вариативности данных моделями.</li>
        </ul>
        <p class="subheader">Сравнение моделей:</p>
        <ul>
            <li>Обе модели показывают <span class="highlight">похожие результаты</span>, что свидетельствует о минимальном влиянии L1-регуляризации в Lasso по сравнению с L2 в Ridge.</li>
            <li>Незначительное улучшение в R² у Lasso модели по сравнению с Ridge <span class="highlight">не приводит к значительным изменениям</span> в эффективности моделей.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # df_reduced уже содержит данные, масштабированные и готовые к дальнейшей фильтрации
    # Фильтрация данных по возрасту, опыту работы и уровню заработной платы
    df_filtered = df_reduced[(df_reduced['Возраст'] <= 60) &
                             (df_reduced['Опыт работы (месяц)'] <= 450) &
                             (df_reduced['ЗП (руб)'] <= 400000)]

    # Проверка размера нового DataFrame после фильтрации
    st.write("Размер отфильтрованного DataFrame:", df_filtered.shape)

    from sklearn.preprocessing import StandardScaler

    # Выделение признаков и целевой переменной
    X = df_filtered.drop('ЗП (руб)', axis=1)
    y = df_filtered['ЗП (руб)']

    # Масштабирование признаков
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Обучение модели Ridge с регуляризацией
    ridge_reg = Ridge(alpha=1000)  # Используем значение alpha, найденное ранее
    ridge_reg.fit(X_train, y_train)
    y_pred = ridge_reg.predict(X_test)

    # Оценка модели
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Вывод результатов
    st.write("MSE после очистки и масштабирования данных (Ridge регрессия):", mse)
    st.write("R2 после очистки и масштабирования данных (Ridge регрессия):", r2)

    st.title("Прогнозирование методом случайного леса")

    st.write("Ввиду большого объема данных, выведу только результаты:")

    # Вставка текста на Markdown
    markdown_text = """
    ##### Улучшение модели случайного леса, т.к. изначальные результаты были на уровне Ridge регресии:
    Были сделан ряд шагов для поиска лучшей модели (данные шаги отображены в ВКР):
    1. Случайный поиск лучших параметров для модели
    2. Определение лучшей модели исходя из полученных параметров
    3. Определение базовых моделей
    4. Создание стекинг-регрессора, используя линейную регрессию как мета-модель
    5. Оценка стекинг-модели
    6. Разделение данных для блендинга
    7. Усреднение предсказаний
    8. Оценка комбинированной модели
    """

    # Отображение Markdown
    st.markdown(markdown_text, unsafe_allow_html=True)

    st.write("Best Random Forest MSE:1939639988.55627")
    st.write("Best Random Forest R2:0.41718721660213876")

    st.title("Прогнозирование методом CatBoost (продвинутая библиотека градиентного бустинга на деревьях решений)")

    # Вставка текста на Markdown
    markdown_text = """
    ##### Лучшие метрики (результат работы отображен в ВКР):
    Best parameters found:  {'learning_rate': 0.021544346900318832, 'l2_leaf_reg': 7, 'iterations': 1500, 'depth': 10}
    ***
    Best CatBoost MSE: 1883910539.7647233
    ***
    Best CatBoost R2: 0.43393250715041487
    """

    # Отображение Markdown
    st.markdown(markdown_text, unsafe_allow_html=True)

    markdown_text = """
    Детальные выводы предоставлены в ВКР.
    """

    # Отображение Markdown
    st.markdown(markdown_text, unsafe_allow_html=True)

    # Создание данных для таблицы в формате DataFrame
    data = {
        "Модель": [
            "Ridge (Линейная регрессия с L2-регуляризацией)",
            "Lasso (Линейная регрессия с L1-регуляризацией)",
            "Случайный лес",
            "Стохастический градиентный спуск",
            "CatBoost (продвинутая библиотека градиентного бустинга на деревьях решений)"
        ],
        "R²": [
            0.35,
            0.32,
            0.41,
            0.35,
            0.43
        ]
    }

    df = pd.DataFrame(data)

    # Преобразование DataFrame в HTML и добавление пользовательских стилей
    def generate_table(dataframe, max_rows=10):
        st.write(
            f"<style> .dataframe th, .dataframe td {{ border: 1px solid #FFFFFF; padding: 10px; }} .dataframe thead th {{ background-color: #516572; color: #FFFFFF; }} .dataframe tbody tr:nth-child(odd) {{ background-color: #E8F4F8; }} .dataframe tbody tr:nth-child(even) {{ background-color: #F4FAFD; }} </style>",
            unsafe_allow_html=True)
        return st.write(dataframe.to_html(escape=False, index=False), unsafe_allow_html=True)

    # Вывод стилизованной таблицы
    st.write("### Результаты моделей по коэффициенту R²", unsafe_allow_html=True)
    generate_table(df)

    st.write("Стохастический градиентный спуск отражен в ВКР")


if __name__ == "__main__":
    run()
