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
    st.subheader(
        'Раздел 1: Прогнозирование  численности безработных в Республике Татарстан (раздел занятости и безработица)')

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
    st.markdown('<p class="big-font">Анализ данных о безработице</p>', unsafe_allow_html=True)

    # Раскрывающийся блок с Markdown текстом
    expander = st.expander("Подробнее о наборе данных")
    markdown_text = """
    ## Описание набора данных
    * **Year** - год
    * **Unemployment_Thousands** - значение показателя численность безработных, тыс.чел.
    * **GRP_Million_Rubles** - производство валового регионального продукта по Республике Татарстан
    * **Crisis** - финансово-экономические кризисы в России
    
    ## Источники данных
    - Год; Значение показателя численность безработных, тыс.чел. - "Социально-экономические показатели регионов России" (Росстат)
    - Производство валового регионального продукта по Республике Татарстан - Производство валового регионального продукта по Республике Татарстан (Росстат)
    - Финансово-экономические кризисы в России - Финансово-экономические кризисы в России (Википедия)
    
    ## Релевантность данных
    Исторические данные: Данные за период с 2000 по 2022 годы релевантны для построения модели, поскольку они позволяют анализировать долгосрочные тренды и сезонные колебания безработицы.
    (Документ обновлен 27.03.2024г. добавлены данные за 2022 год)
    Изменение методики подсчета: Смена методики подсчета в 2017 году (с возрастной категории 15-72 лет на категорию 15 лет и старше) может внести некоторую погрешность в модель, поскольку данные до и после изменения могут быть не полностью сопоставимы.
    Учет кризисов: Информация о кризисных периодах является важным признаком, поскольку экономические кризисы часто сопровождаются ростом безработицы. Их учет в виде бинарного признака улучшает точность модели.
    """
    expander.markdown(markdown_text, unsafe_allow_html=True)

    # Исходные данные
    data = {
        "Year": list(range(2000, 2023)),
        "Unemployment_Thousands": [159, 116, 100, 125, 138, 125, 105, 108, 96, 168, 126, 95, 85, 81, 81, 82, 77, 71, 68,
                                   66, 74, 54, 46],
        "GRP_Million_Rubles": [186154.4, 213740, 250596, 305086.1, 391116, 482759.2, 605911.5, 757401.4, 926056.7,
                               885064, 1001622.8, 1305947, 1437001, 1551472.1, 1661413.8, 1867258.7, 2058139.9,
                               2264655.8, 2622773.9, 2808753.3, 2631286.8, 3533272.5, 4179258.6],
        "Crisis": [0] * 7 + [1] * 3 + [0] * 4 + [1] * 2 + [0] * 6 + [1]
    }

    # Преобразование года
    data["Year"] = [year - 2000 for year in data["Year"]]

    # Создание DataFrame
    df = pd.DataFrame(data)

    # Нормализация данных
    scaler = MinMaxScaler()
    df[["Year", "Unemployment_Thousands", "GRP_Million_Rubles"]] = scaler.fit_transform(
        df[["Year", "Unemployment_Thousands", "GRP_Million_Rubles"]])

    df.head(24)

    # Подготовка данных
    X = df[["Year", "GRP_Million_Rubles", "Crisis"]]
    y = df["Unemployment_Thousands"]

    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Создание и обучение модели линейной регрессии
    model = LinearRegression()
    model.fit(X, y)

    # Интерфейс для пользователя
    st.title("Прогнозирование методом линейной регрессии.")

    # Прогнозирование для 2023 года (23 года после 2000 года)
    year_2023 = scaler.transform([[23, 0, 0]])[0][0]  # Преобразование года с учетом нормализации
    grp_2023 = X["GRP_Million_Rubles"].mean()  # Среднее значение ВРП
    crisis_2023 = 0  # Предположение отсутствия кризиса в 2023

    # Делаем предсказание
    unemployment_pred_2023 = model.predict([[year_2023, grp_2023, crisis_2023]])

    # Обратное преобразование предсказания к исходному масштабу
    unemployment_pred_2023 = scaler.inverse_transform([[0, unemployment_pred_2023[0], 0]])[0][1]

    # Вывод результата
    st.write(f"Прогнозируемое значение для 2023 года: {unemployment_pred_2023:.2f} тыс. человек")

    # Добавление прогноза для 2023 года к исходным данным
    years_original = list(range(2000, 2023)) + [2023]  # Добавляем 2023 год к списку лет
    unemployment_original = data['Unemployment_Thousands'] + [80.37]  # Добавляем прогнозируемое значение безработицы

    # Визуализация исходных данных
    plt.figure(figsize=(10, 6))
    plt.scatter(years_original[:-1], data['Unemployment_Thousands'], color='blue', label='Исходные данные')

    # Добавление прогноза на график
    plt.scatter([2023], [80.37], color='green', label='Прогноз на 2023 год', zorder=5)

    # Построение линии тренда (линии линейной регрессии) на основе исходных данных и прогноза
    z = np.polyfit(years_original, unemployment_original, 1)  # Используем полином 1-й степени для линии тренда
    p = np.poly1d(z)
    plt.plot(years_original, p(years_original), "r--", label='Линия прогноза')

    plt.xlabel('Год')
    plt.ylabel('Количество безработных, тыс.')
    plt.title('Динамика безработицы и прогноз на 2023 год')
    plt.legend()
    plt.grid(True)
    # Показываем график в Streamlit
    st.pyplot(plt)

    # Получаем предсказания модели на обучающем наборе данных
    y_pred_scaled = model.predict(X)

    # Обратное преобразование предсказаний к исходному масштабу
    y_pred = scaler.inverse_transform([[0, pred, 0] for pred in y_pred_scaled])[:, 1]

    # Истинные значения (уже в исходном масштабе)
    y_true = data['Unemployment_Thousands']

    # Расчет метрик ошибки
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_true, y_pred)

    st.title("Расчет метрик ошибки")

    st.write("MAE(Средняя абсолютная ошибка), "
             "MSE(Среднеквадратическая ошибка), RMSE(Среднеквадратическое отклонение), R^2(Коэффициент детерминации)")

    st.write(f"MAE: {mae:.2f}")
    st.write(f"MSE: {mse:.2f}")
    st.write(f"RMSE: {rmse:.2f}")
    st.write(f"R^2: {r2:.2f}")

    import streamlit as st

    # Установка пользовательских стилей для Markdown
    def set_custom_styles():
        st.markdown("""
        <style>
        .big-font {
            font-size:18px !important;
            font-weight: bold;
        }
        .info-font {
            font-size:16px !important;
            font-family: 'Courier New', Courier, monospace;
            line-height: 1.6;
        }
        .highlight {
            background-color: #f0f2f6;
            border-left: 5px solid #0078D4;
            padding: 4px;
        }
        </style>
        """, unsafe_allow_html=True)

    set_custom_styles()

    # Раскрывающийся блок с информацией о метриках
    expander = st.expander("Подробнее о метриках модели")
    markdown_text = """
    <div class="info-font">
    <ul>
        <li class="highlight">Средняя абсолютная ошибка (MAE) составляет 12.19.<br>
        Это значит, что в среднем модель ошибается на 12.19 тысяч безработных при предсказании.</li>
        <li class="highlight">Среднеквадратичная ошибка (MSE) равна 285.14, указывая на то, что квадраты ошибок предсказаний в среднем составляют 285.14.<br>
        Эта метрика более чувствительна к большим ошибкам, чем MAE.</li>
        <li class="highlight">Корень из среднеквадратичной ошибки (RMSE) составляет 16.89, обеспечивая представление о величине ошибки в тех же единицах, что и целевая переменная (тысячи безработных).</li>
        <li class="highlight">Коэффициент детерминации (R²) равен 0.71, что означает, что примерно 71% вариации целевой переменной (количество безработных) может быть объяснено используемыми предикторами в модели.</li>
    </ul>
    <p>Коэффициент детерминации 0.71 свидетельствует о том, что модель достаточно хорошо справляется с задачей предсказания, но все же оставляет место для улучшения. Возможно, добавление дополнительных или более релевантных признаков, а также оптимизация текущих параметров могли бы улучшить качество модели.</p>
    </div>
    """
    expander.markdown(markdown_text, unsafe_allow_html=True)

    # Разница между истинными и предсказанными значениями
    errors = y_true - y_pred

    # Визуализация ошибок
    plt.figure(figsize=(10, 6))
    plt.bar(years_original[:-1], errors, color='orange', label='Ошибка предсказания')
    plt.axhline(0, color='red', linestyle='--', label='Нулевая ошибка')
    plt.xlabel('Год')
    plt.ylabel('Ошибка')
    plt.title('Ошибка предсказания модели линейной регрессии')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

    st.title("Прогнозирование методом случайного леса (Random Forest)")

    # Random Forest model на тех же данных

    # Создание модели случайного леса
    random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Обучение модели случайного леса на тех же данных
    random_forest_model.fit(X, y)

    # Предсказание для 2023 года, используя те же данные
    # не нужно изменять входные данные, так как они уже подготовлены
    unemployment_pred_2023_rf = random_forest_model.predict([[year_2023, grp_2023, crisis_2023]])

    # Обратное преобразование предсказания к исходному масштабу для случайного леса
    unemployment_pred_2023_rf_original = scaler.inverse_transform([[0, unemployment_pred_2023_rf[0], 0]])[0][1]

    st.write(f"Прогнозируемое значение для 2023 года: {unemployment_pred_2023_rf_original:.2f} тыс.чел.")

    # Получаю предсказания модели на обучающем наборе данных
    y_pred_rf_scaled = random_forest_model.predict(X)

    # Обратное преобразование предсказаний к исходному масштабу
    y_pred_rf = scaler.inverse_transform([[0, pred, 0] for pred in y_pred_rf_scaled])[:, 1]

    # Расчет метрик ошибки для модели случайного леса
    mae_rf = mean_absolute_error(y_true, y_pred_rf)
    mse_rf = mean_squared_error(y_true, y_pred_rf)
    rmse_rf = mse_rf ** 0.5
    r2_rf = r2_score(y_true, y_pred_rf)

    st.title("Расчет метрик ошибки")

    st.write(f"MAE (Средняя абсолютная ошибка): {mae_rf:.2f}")
    st.write(f"MSE (Среднеквадратическая ошибка): {mse_rf:.2f}")
    st.write(f"RMSE (Среднеквадратическое отклонение): {rmse_rf:.2f}")
    st.write(f"R^2 (Коэффициент детерминации): {r2_rf:.2f}")

    # Определяем пользовательские стили для улучшения внешнего вида текста
    def set_custom_styles():
        st.markdown("""
        <style>
        .info-font {
            font-size:16px !important;
            font-family: 'Courier New', Courier, monospace;
            line-height: 1.6;
        }
        .highlight {
            background-color: #f0f9ff;
            border-left: 5px solid #4a8ec7;
            padding: 0.5rem;
            margin-top: 0.5rem;
            margin-bottom: 0.5rem;
        }
        </style>
        """, unsafe_allow_html=True)

    set_custom_styles()

    # Заголовок приложения
    st.title("Сравнение полученных моделей машинного обучения (линейная регрессия/случайный лес)")

    # Раскрывающийся блок для детального сравнения моделей
    expander = st.expander("Подробное сравнение моделей")
    markdown_text = """
    <div class="info-font">
    <h4>Вывод по сравнению двух моделей:</h4>
    <p class="highlight">
    Результаты по метрикам ошибок для модели случайного леса (Random Forest) показывают значительное улучшение 
    в точности предсказаний по сравнению с моделью линейной регрессии. 
    </p>
    
    <h5>Средняя Абсолютная Ошибка (MAE)</h5>
    <p class="highlight">
    MAE равна 5.59, что означает, что в среднем абсолютное отклонение предсказаний модели от истинных значений составляет 5.59 тысяч. 
    Это показывает, что модель достаточно хорошо способна предсказывать количество безработных, с относительно небольшой средней ошибкой.
    </p>
    
    <h5>Среднеквадратичная Ошибка (MSE)</h5>
    <p class="highlight">
    MSE составляет 68.84, представляя собой среднее квадратов ошибок между предсказанными и истинными значениями. 
    Поскольку MSE возводит ошибки в квадрат перед их усреднением, большие ошибки наказываются более сильно, 
    что делает эту метрику особенно полезной при желании минимизировать большие отклонения в предсказаниях.
    </p>
    
    <h5>Корень из Среднеквадратичной Ошибки (RMSE)</h5>
    <p class="highlight">
    RMSE, равный 8.30, является квадратным корнем из MSE и представляет собой стандартное отклонение ошибок предсказания.
    Эта метрика полезна, поскольку она возвращается к исходным единицам измерения целевой переменной и предоставляет понимание о величине ошибок в тех же единицах,
    что и предсказываемая переменная.
    </p>
    
    <h5>Коэффициент Детерминации (R²)</h5>
    <p class="highlight">
    R², или коэффициент детерминации, равный 0.93, показывает, насколько хорошо будущие выборки, вероятно, будут предсказаны моделью.
    Значение 0.93 указывает на то, что примерно 93% вариации уровня безработицы могут быть объяснены используемыми признаками в модели.
    Это указывает на высокую эффективность модели в предсказании уровня безработицы.
    </p>
    
    <h5>Общий Анализ</h5>
    <p class="highlight">
    Мои расчеты указывают на высокую эффективность модели случайного леса в задаче прогнозирования численности безработных в РТ.
    Низкие значения MAE и RMSE говорят о том, что модель в среднем делает точные предсказания с небольшими ошибками.
    Высокий коэффициент детерминации R² свидетельствует о том, что модель хорошо справляется с объяснением вариации численности безработных,
    что делает её надежным инструментом для прогнозирования. Это может быть особенно полезно для планирования экономической политики и разработки программ поддержки занятости.
    </p>
    </div>
    """
    expander.markdown(markdown_text, unsafe_allow_html=True)

    st.title("Визуализация модели случайного леса")

    # Подготовка данных для визуализации
    years = np.array(years_original[:-1])  # Годы до 2023
    true_values = np.array(y_true)  # Истинные значения безработицы
    predicted_values_rf = y_pred_rf  # Предсказанные значения модели случайного леса

    # Визуализация исходных данных и предсказаний модели
    plt.figure(figsize=(12, 6))
    plt.plot(years, true_values, label='Истинные значения', color='blue', marker='o')
    plt.plot(years, predicted_values_rf, label='Предсказания случайного леса', color='red', linestyle='--', marker='x')

    plt.title('Прогноз уровня безработицы: Истинные значения vs Предсказания случайного леса')
    plt.xlabel('Год')
    plt.ylabel('Количество безработных, тыс.')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

    # Вычисление ошибок предсказания
    errors_rf = true_values - predicted_values_rf

    # Визуализация ошибок
    plt.figure(figsize=(12, 6))
    plt.bar(years, errors_rf, color='orange', label='Ошибка предсказания')

    plt.axhline(0, color='green', linestyle='--', label='Нулевая ошибка')
    plt.title('Ошибка предсказания модели случайного леса по годам')
    plt.xlabel('Год')
    plt.ylabel('Ошибка')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

    st.title("Прогнозирование методом ARIMA (интегрированное скользящее среднее с авторегрессией)")

    # Использую только колонку безработицы как одномерный временной ряд
    unemployment_series = pd.Series(data['Unemployment_Thousands'],
                                    index=pd.date_range(start='2000', periods=len(data['Unemployment_Thousands']),
                                                        freq='A'))

    # Визуализируем временной ряд
    unemployment_series.plot(title='Тысячи безработных с течением времени')
    st.pyplot(plt)

    # определил параметры модели ARIMA
    # Здесь использую параметры
    p, d, q = 5, 1, 0

    # Создание и обучение модели ARIMA
    arima_model = ARIMA(unemployment_series, order=(p, d, q))
    arima_result = arima_model.fit()

    # Прогнозирование
    forecast = arima_result.forecast(steps=1)  # Прогнозируем на один шаг вперед, то есть на 2023 год
    st.write(f"Прогнозируемое количество безработных в 2023 году: {forecast.iloc[0]:.2f} тыс.чел.")

    # Вставка текста на Markdown
    markdown_text = """
    ##### Определение параметров 
    p, d, и q для модели ARIMA является ключевым этапом в процессе моделирования временных рядов.
    Эти параметры помогают определить структуру модели, включая порядок авторегрессии (AR), степень интегрирования (I),
    и порядок скользящего среднего (MA). Выполнив анализ автокорреляционной функции (ACF) и частичной автокорреляционной функции (PACF)
    для временного ряда безработицы, чтобы определить подходящие значения для этих параметров.
    """

    # Отображение Markdown
    st.markdown(markdown_text, unsafe_allow_html=True)

    # Предварительно, данные о безработице уже загружены в unemployment_series

    st.title("Визуализация ACF, PACF (использую первое дифференцирование)")

    # Визуализация ACF
    plt.figure(figsize=(14, 7))
    plt.subplot(211)
    plot_acf(unemployment_series.diff().dropna(), ax=plt.gca(), lags=10)  # Используем первое дифференцирование

    # Визуализация PACF
    plt.subplot(212)
    plot_pacf(unemployment_series.diff().dropna(), ax=plt.gca(), lags=10)  # Используем первое дифференцирование

    st.pyplot(plt)

    # Установка пользовательских стилей
    st.markdown("""
    <style>
    .data-info {
        font-size: 16px;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        line-height: 1.5;
    }
    .markdown-section {
        background-color: #f1f3f5;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # Раскрывающийся блок для детальной информации
    expander = st.expander("Чем руководствовался при определении параметров ARIMA модели (p, d, q)")

    # Детальное описание в Markdown
    markdown_text = """
    <div class="data-info">
    <h4>Вывод по определению параметров модели:</h4>
    <p><strong>Как Определить d:</strong><br>
    Параметр <em>d</em> отражает, сколько раз нужно дифференцировать временной ряд, чтобы сделать его стационарным.
    Стационарность — это свойство временного ряда, при котором его статистические характеристики, такие как среднее и дисперсия, не изменяются со временем.
    </p>
    <p><strong>Проверка на стационарность:</strong><br>
    Сначала временной ряд проверяется на стационарность, например, с помощью теста Дики-Фуллера. Если ряд не стационарен, мы применяем дифференцирование.
    Дифференцирование: Применяем первое дифференцирование к временному ряду и снова проверяем на стационарность. Если после первого дифференцирования ряд стал стационарным, <em>d=1</em>.
    </p>
    <p><strong>Как Определить p и q:</strong><br>
    После того, как определили, что <em>d=1</em>, анализируем ACF и PACF для дифференцированного ряда, чтобы определить <em>p</em> и <em>q</em>.
    </p>
    <div class="markdown-section">
    <strong>PACF (для определения p):</strong> Смотрим на график частичной автокорреляционной функции (PACF) дифференцированного ряда.
    Значение <em>p</em> соответствует лагу, после которого PACF резко падает к нулю или затухает. 
    </div>
    <div class="markdown-section">
    <strong>ACF (для определения q):</strong> Аналогично, анализируя график автокорреляционной функции (ACF) дифференцированного ряда,
    мы определяем <em>q</em> как лаг, после которого ACF резко падает к нулю или затухает.
    </div>
    <p><strong>Заключение:</strong><br>
    Значения <em>p=5</em>, <em>d=1</em>, <em>q=0</em> были выбраны в результате всех перечисленных шагов при анализе. 
    Точные значения этих параметров определены на основе анализа ACF и PACF конкретного временного ряда, 
    а также могут потребовать дополнительной настройки и проверки через процедуру кросс-валидации
    или сравнения различных моделей с использованием информационных критериев, 
    таких как AIC или BIC.
    </p>
    </div>
    """
    expander.markdown(markdown_text, unsafe_allow_html=True)

    # Исходные данные
    data = {
        "Year": list(range(2000, 2023)),
        "Unemployment_Thousands": [159, 116, 100, 125, 138, 125, 105, 108, 96, 168, 126, 95, 85, 81, 81, 82, 77, 71, 68,
                                   66, 74, 54, 46],
    }

    # Создание DataFrame
    df = pd.DataFrame(data)
    df['Year'] = pd.to_datetime(df['Year'], format='%Y')
    df.set_index('Year', inplace=True)

    st.write("Проверка на стационарность")
    # Проверка на стационарность
    result = adfuller(df['Unemployment_Thousands'])
    st.write(f'ADF Statistic: {result[0]}')
    st.write(f'p-value: {result[1]}')

    # Разделяем данные на обучающую и тестовую выборки
    train, test = df['Unemployment_Thousands'][:-1], df['Unemployment_Thousands'][-1:]

    # Обучение модели ARIMA
    model = ARIMA(train, order=(5, 1, 0))
    model_fit = model.fit()

    # Предсказания
    forecast = model_fit.forecast(steps=1)

    # Метрики ошибок
    # Для вычисления метрик использую все кроме последнего значения, которое использовалось для теста
    y_true = train[-1:].values  # Истинные значения
    y_pred = forecast.values  # Предсказанные значения

    # Расчет метрик
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    # Заголовок
    st.title("Метрики ошибок")

    # Вывод метрик
    st.write(f"MAE (Средняя абсолютная ошибка): {mae:.2f}")
    st.write(f"MSE (Среднеквадратическая ошибка): {mse:.2f}")
    st.write(f"RMSE (Среднеквадратическое отклонение): {rmse:.2f}")

    # Вывод метрик
    # st.write(f"MAE:{mae}, MSE:{mse}, RMSE:{rmse}")

    # Установка пользовательских стилей для Markdown
    def set_custom_styles():
        st.markdown("""
        <style>
        .info-text {
            font-size: 16px;
            font-family: 'Helvetica', sans-serif;
        }
        .highlight {
            background-color: #f4f4f8;
            border-left: 5px solid #007BFF;
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
        }
        </style>
        """, unsafe_allow_html=True)

    set_custom_styles()

    # Заголовок для раздела
    st.title("Анализ эффективности модели ARIMA")

    # Раскрывающийся блок для подробной информации
    expander = st.expander("Подробные результаты и анализ модели ARIMA")
    markdown_text = """
    <div class="info-text">
    <p><strong>После выполнения модели ARIMA для временного ряда безработицы и расчёта метрик ошибок получились следующие результаты:</strong></p>
    
    <div class="highlight">
    <p><strong>ADF (Augmented Dickey-Fuller) Тест:</strong></p>
    <p>ADF Statistic: -2.10<br>
    p-value: 0.244<br>
    ADF тест показывает, что на уровне значимости 0.05 мы не можем отвергнуть нулевую гипотезу о наличии единичного корня,
    что указывает на то, что временной ряд возможно может быть нестационарным.
    Однако для целей нашей задачи прогнозирования и определения точности среди всех построенных моделей продолжим с предположением, что дифференцирование первого порядка делает ряд стационарным.</p>
    </div>
    
    <div class="highlight">
    <p><strong>Метрики ошибок для последнего предсказанного значения:</strong></p>
    <p>Средняя Абсолютная Ошибка (MAE): 4.65<br>
    Среднеквадратичная Ошибка (MSE): 21.62<br>
    Корень из Среднеквадратичной Ошибки (RMSE): 4.65<br>
    Эти метрики показывают, насколько близко предсказания модели ARIMA к реальным значениям временного ряда. В данном случае мы предсказывали только одно значение в будущее,
    так что метрики основаны на этом одном предсказании.</p>
    <p>MAE и RMSE около 4.65 указывают на то, что в среднем модель ошибается примерно на 4.65 тысячи в предсказании численности безработных.
    В контексте исходных данных и величины изменений численности безработных эти значения ошибок могут быть считаться приемлемыми.</p>
    </div>
    </div>
    """
    expander.markdown(markdown_text, unsafe_allow_html=True)

    st.title("Прогнозирование методом градиентный бустинг")

    # Использование исходных данных
    data = {
        "Year": list(range(2000, 2023)),
        "Unemployment_Thousands": [159, 116, 100, 125, 138, 125, 105, 108, 96, 168, 126, 95, 85, 81, 81, 82, 77, 71, 68,
                                   66, 74, 54, 46],
        "GRP_Million_Rubles": [186154.4, 213740, 250596, 305086.1, 391116, 482759.2, 605911.5, 757401.4, 926056.7,
                               885064, 1001622.8, 1305947, 1437001, 1551472.1, 1661413.8, 1867258.7, 2058139.9,
                               2264655.8, 2622773.9, 2808753.3, 2631286.8, 3533272.5, 4179258.6],
        "Crisis": [0] * 7 + [1] * 3 + [0] * 4 + [1] * 2 + [0] * 6 + [1]
    }

    # Преобразование в DataFrame
    df = pd.DataFrame(data)

    # Вектор признаков и целевая переменная
    X = df.drop(columns=['Unemployment_Thousands'])  # Все колонки кроме целевой переменной
    y = df['Unemployment_Thousands']  # Целевая переменная

    # Для прогноза на 2023 год использую все данные до этого года как обучающие
    # Здесь имитирую прогноз, так как не можем разделить данные на обучающие и тестовые в обычном смысле
    X_train = X[:-1]
    y_train = y[:-1]
    X_test = X[-1:]  # Предполагаемый вектор признаков для 2023 года

    # Создание модели градиентного бустинга
    gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

    # Обучение модели
    gb_model.fit(X_train, y_train)

    # Предсказание для 2023 года
    unemployment_pred_2023_gb = gb_model.predict(X_test)

    st.write(
        f"Прогнозируемое значение численности безработных в 2023 году: {unemployment_pred_2023_gb[0]:.2f} тыс.чел.")

    # Вставка текста на Markdown
    markdown_text = """
    Чтобы рассчитать метрики ошибок для модели градиентного бустинга, нужны как предсказанные значения,
    так и истинные значения. Поскольку у нас нет доступа к истинному значению безработицы за 2023 год,
    не имеется возможным на данном этапе рассчитать метрики ошибок для моего прогноза на 2023 год. 
    Однако, можно оценить производительность модели на имеющихся данных до 2022 года, используя часть этих данных для обучения, а другую часть — для тестирования.
    """

    # Отображение Markdown
    st.markdown(markdown_text, unsafe_allow_html=True)

    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X[:-1], y[:-1], test_size=0.2, random_state=42)

    # Создание и обучение модели градиентного бустинга
    gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gb_model.fit(X_train, y_train)

    # Предсказания на тестовой выборке
    y_pred_gb = gb_model.predict(X_test)

    # Расчет метрик ошибок
    mae_gb = mean_absolute_error(y_test, y_pred_gb)
    mse_gb = mean_squared_error(y_test, y_pred_gb)
    rmse_gb = np.sqrt(mse_gb)
    r2_gb = r2_score(y_test, y_pred_gb)

    st.title("Расчет метрик ошибок")

    st.write(f"MAE (Средняя абсолютная ошибка): {mae_gb:.2f}")
    st.write(f"MSE (Среднеквадратическая ошибка): {mse_gb:.2f}")
    st.write(f"RMSE (Среднеквадратическое отклонение): {rmse_gb:.2f}")
    st.write(f"R^2 (Коэффициент детерминации): {r2_gb:.2f}")

    # Определение пользовательских стилей CSS
    def set_custom_styles():
        st.markdown("""
        <style>
        .info-text {
            font-size: 16px;
            font-family: 'Arial', sans-serif;
        }
        .highlight {
            background-color: #f8f9fa;
            border-left: 5px solid #007BFF;
            padding: 10px;
            margin-top: 10px;
            margin-bottom: 10px;
            border-radius: 5px;
        }
        </style>
        """, unsafe_allow_html=True)

    set_custom_styles()

    # Заголовок приложения
    st.title("Анализ эффективности модели градиентного бустинга")

    # Раскрывающийся блок для детального анализа
    expander = st.expander("Подробные результаты модели градиентного бустинга")
    markdown_text = """
    <div class="info-text">
    <p><strong>Анализируя полученные метрики ошибок для модели градиентного бустинга, мы можем сделать несколько ключевых выводов о её производительности на имеющихся данных о безработице:</strong></p>
    
    <div class="highlight">
    <strong>Средняя Абсолютная Ошибка (MAE):</strong> 22.84<br>
    Это означает, что в среднем модель ошибается на 22.84 тысячи при предсказании количества безработных.
    Учитывая масштаб данных о безработице, это может быть значительной ошибкой, особенно если средние значения в ряду близки к этим цифрам.
    </div>
    
    <div class="highlight">
    <strong>Среднеквадратичная Ошибка (MSE):</strong> 999.43<br>
    Эта метрика показывает среднее квадратичное отклонение предсказаний от фактических значений.
    Значительная величина MSE указывает на наличие больших ошибок в некоторых предсказаниях модели.
    </div>
    
    <div class="highlight">
    <strong>Корень из Среднеквадратичной Ошибки (RMSE):</strong> 31.61<br>
    RMSE возвращает ошибку к исходным единицам измерения данных и позволяет лучше оценить величину ошибок. Значение в 31.61 тысячу говорит о том, что у модели есть проблемы с точностью предсказаний для данного набора данных.
    </div>
    
    <div class="highlight">
    <strong>Коэффициент детерминации (R²):</strong> -0.19<br>
    R² показывает, какая доля вариации зависимой переменной объясняется независимыми переменными в модели.
    Отрицательный R² говорит о том, что выбранная модель не подходит для данных или некорректно настроена.
    </div>
    
    <p><strong>Возможные причины:</strong><br>
    Недостаточно данных: Для эффективного обучения моделей градиентного бустинга может потребоваться больше данных.<br>
    Некорректные параметры модели: Эксперименты с настройками параметров могут улучшить результаты.<br>
    Недостаточно признаков: Рассмотрение дополнительных данных или создание новых признаков может помочь.<br>
    Проблемы с данными: Проверка данных на наличие выбросов и пропущенных значений может улучшить производительность модели.</p>
    </div>
    """
    expander.markdown(markdown_text, unsafe_allow_html=True)

    # Настройка интерфейса
    st.title("Прогнозирование методом CatBoost (продвинутая библиотека градиентного бустинга на деревьях решений)")

    # Подготовка данных
    X = df.drop(columns=['Unemployment_Thousands'])  # Признаки
    y = df['Unemployment_Thousands']  # Целевая переменная
    X_train, X_test = X[:-1], X[-1:]
    y_train = y[:-1]

    # Создание и обучение модели CatBoost
    cb_model = CatBoostRegressor(iterations=500, depth=4, learning_rate=0.1, l2_leaf_reg=3, silent=True,
                                 random_state=42)
    cb_model.fit(X_train, y_train)

    # Предсказание и визуализация результатов
    y_pred_train = cb_model.predict(X_train)
    y_pred_cb_2023 = cb_model.predict(X_test)

    # Расчет метрик
    mae = mean_absolute_error(y_train, y_pred_train)
    mse = mean_squared_error(y_train, y_pred_train)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_train, y_pred_train)

    # Вывод метрик и предсказания
    st.write(f"Прогнозируемое значение численности безработных в 2023 г.: {y_pred_cb_2023[0]:.2f} тыс.чел.")
    st.title("Расчет метрик ошибок")
    st.write(f"MAE (Средняя абсолютная ошибка): {mae:.2f}")
    st.write(f"MSE (Среднеквадратическая ошибка): {mse:.2f}")
    st.write(f"RMSE (Среднеквадратическое отклонение): {rmse:.2f}")
    st.write(f"R^2 (Коэффициент детерминации): {r2:.2f}")

    # График сравнения фактических данных и предсказаний
    plt.figure(figsize=(10, 5))
    plt.plot(df['Year'][:-1], y_train, label='Фактическая', marker='o')
    plt.plot(df['Year'][:-1], y_pred_train, label='Прогнозируемая', marker='x')
    plt.scatter([df['Year'].iloc[-1]], [y_pred_cb_2023[0]], color='red', label='прогнозируемое значение (2023г.)',
                zorder=5)
    plt.title('Фактическая и прогнозируемая безработица')
    plt.xlabel('Год')
    plt.ylabel('Безработица тыс.')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

    # Установка пользовательских стилей для Markdown
    def set_custom_styles():
        st.markdown("""
        <style>
        .info-text {
            font-size: 16px;
            font-family: 'Arial', sans-serif;
        }
        .highlight {
            background-color: #f8f9fa;
            border-left: 5px solid #2c7be5;
            padding: 10px;
            margin-top: 10px;
            margin-bottom: 10px;
            border-radius: 5px;
        }
        </style>
        """, unsafe_allow_html=True)

    set_custom_styles()

    # Заголовок приложения
    st.title("Анализ эффективности модели CatBoost")

    # Раскрывающийся блок для подробного анализа
    expander = st.expander("Подробные результаты и анализ модели CatBoost")
    markdown_text = """
    <div class="info-text">
    <p><strong>Результаты метрик ошибок для модели, обученной с использованием CatBoost, показывают выдающуюся производительность на обучающем наборе данных:</strong></p>
    
    <div class="highlight">
    <strong>MAE (Средняя Абсолютная Ошибка):</strong> 0.23<br>
    Это означает, что в среднем модель ошибается на 0.23 тысячи (или 230 человек), когда делает предсказания по обучающему набору данных. Для задач прогнозирования численности безработных это указывает на высокую точность модели.
    </div>
    
    <div class="highlight">
    <strong>MSE (Среднеквадратичная Ошибка):</strong> 0.07<br>
    Такой низкий показатель MSE свидетельствует о том, что модель очень точно предсказывает значения по обучающему набору данных.
    </div>
    
    <div class="highlight">
    <strong>RMSE (Корень из Среднеквадратичной Ошибки):</strong> 0.27<br>
    Такое низкое значение указывает на высокую точность предсказаний модели.
    </div>
    
    <div class="highlight">
    <strong>R² (Коэффициент детерминации):</strong> 1.00<br>
    Это означает, что модель объясняет около 100% вариации целевой переменной на обучающем наборе данных.
    </div>
    
    <p><strong>Анализ результатов:</strong><br>
    Хотя результаты выглядят идеально для обучающего набора данных, они вызывают опасения относительно переобучения модели. Важно проверить её производительность на отдельном тестовом наборе данных или использовать методы кросс-валидации. Что будет показано в следующих разделах.</p>
    </div>
    """
    expander.markdown(markdown_text, unsafe_allow_html=True)

    st.title("Итоговые показатели всех моделей и сравнительная аналитика")

    # Данные для таблицы
    data = {
        "Модель": ["Линейная регрессия", "Случайный лес", "Модель ARIMA", "Градиентный бустинг", "CatBoost"],
        "Прогноз": ["80.38 тыс. человек", "70.77 тыс. человек", "53.47 тыс. человек", "54.89 тыс. человек",
                    "66.94 тыс. человек"],
        "MAE": [12.19, 5.59, 4.64, 22.84, 0.23],
        "MSE": [285.14, 68.84, 21.62, 999.43, 0.07],
        "RMSE": [16.89, 8.30, 4.64, 31.61, 0.27],
        "R²": [0.71, 0.93, "-", -0.19, 1.00]
    }

    df = pd.DataFrame(data)

    # Преобразование DataFrame в HTML
    html = df.to_html(index=False, escape=False)

    # Замена стандартных стилей таблицы
    custom_html = html.replace('<table border="1" class="dataframe">',
                               '<table style="width:100%;border-collapse:collapse;border:1px solid #ccc;font-size:16px;">') \
        .replace('<th>', '<th style="background-color:#4CAF50;color:white;padding:8px;text-align:left;">') \
        .replace('<td>', '<td style="padding:8px;text-align:left;border-bottom:1px solid #ddd;">')

    # Пользовательские стили для Streamlit
    st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #009688;
        color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)

    # Отображение таблицы
    # st.markdown(custom_html, unsafe_allow_html=True)

    # Кнопка для переключения видимости таблицы
    if st.button('Показать/скрыть результаты'):
        st.markdown(custom_html, unsafe_allow_html=True)

    # Далее можете добавлять другие элементы UI с помощью стандартных функций Streamlit, такие как:
    # st.write('Текстовый блок')
    # st.text_input('Ввод текста')
    # st.button('Кнопка')


if __name__ == "__main__":
    run()
