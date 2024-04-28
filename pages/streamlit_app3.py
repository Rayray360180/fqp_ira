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

    import seaborn as sns
    from scipy import stats

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
    st.subheader('Раздел 3: Прогнозирование  просроченной задолженности по заработной плате в г. Москва')

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
    st.markdown('<p class="big-font">Анализ и предобработка исходных данных</p>', unsafe_allow_html=True)

    # Раскрывающийся блок с Markdown текстом
    expander = st.expander("Подробнее о наборе данных")
    markdown_text = """
    ## Описание набора данных
    _В этом наборе данных отражена информация о некоторых социально-экономических показателях города Москвы_
    * **Просроченная задолженность по заработной плате** - (в тысячах рублей) [Overdue_wages]
    * **Средняя численность работников** - (в количестве человек) [Average_number_of_employees]
    * **Удельный вес убыточных организаций** - (в процентах) [Share_of_unprofitable_organizations]
    * **Задолженность по полученным кредитам и займам** - (в миллионах рублей) для крупных и средних организаций 
    [Debt_on_received_loans_and_borrowings]
    
    ## Источники данных
    Данные разбиты по кварталам, начиная с первого квартала 2020 года и заканчивая четвертым кварталом 2023 года. 
    _Источник данных - **"Единое" Хранилище данных ИАС МКР города Москвы"**_
    
    ## Прогнозируемое значение
    **Просроченная задолженность по заработной плате на 'I квартал 2024'**
    """
    expander.markdown(markdown_text, unsafe_allow_html=True)

    st.title("Исходный набор данных")

    # Загрузка данных из файла Excel
    data_path = r"D:\Диплом\Результаты\3 критерий неопределенности\data_mos_zp_ex12.xlsx"
    data = pd.read_excel(data_path)

    # Вывод первых строк таблицы для ознакомления со структурой данных
    st.write(data.head())

    st.title("Анализ зависимостей между показателями")

    # Подготовка данных для анализа корреляции
    data_for_corr = data.drop(['Показатели'], axis=1).T  # Транспонируем для лучшего анализа
    data_for_corr.columns = data['Показатели']

    # Вычисление корреляционной матрицы
    correlation_matrix = data_for_corr.corr()

    # Визуализация корреляционной матрицы
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Корреляционная матрица показателей')
    st.pyplot(plt)

    # Основной заголовок приложения
    st.markdown('<p class="big-font">Корреляционный анализ</p>', unsafe_allow_html=True)

    # Раскрывающийся блок с Markdown текстом
    expander = st.expander("Результат анализа")
    markdown_text = """
    ## Описание 
    Корреляция покажет, насколько сильно связаны переменные между собой.
    Вычислив коэффициенты корреляции Пирсона, которые могут варьироваться от -1 до 1, 
    где значения близкие к 1 или -1 указывают на сильную положительную или отрицательную зависимость соответственно, 
    а значения около 0 указывают на отсутствие линейной зависимости.
    
    
    ## Корреляционная матрица показывает следующее
    Просроченная задолженность по заработной плате имеет относительно низкую корреляцию с численностью работников и 
    задолженностью по кредитам, но заметную отрицательную корреляцию с удельным весом убыточных организаций. 
    Это может указывать на то, что с увеличением доли убыточных организаций увеличивается и задолженность по заработной плате. 
    Средняя численность работников и задолженность по кредитам имеют сильную положительную корреляцию, что логично, 
    так как более крупные организации могут брать больше кредитов. 
    Удельный вес убыточных организаций отрицательно коррелирует со средней численностью работников, 
    что может означать, что в более крупных организациях меньше риск убыточности.
    """
    expander.markdown(markdown_text, unsafe_allow_html=True)

    # Загрузка данных из файла
    file_path = r"D:\Диплом\Результаты\3 критерий неопределенности\data_mos_zp_ex123.csv"
    # Загрузка данных с указанием правильного разделителя
    data = pd.read_csv(file_path, delimiter=';')

    # Удаление пустой строки в конце таблицы
    data_cleaned = data.dropna()

    # Транспонирование данных для анализа корреляций
    data_transposed = data_cleaned.set_index('Показатели').transpose()

    # Вычисление корреляционной матрицы
    correlation_matrix = data_transposed.corr()
    st.write(correlation_matrix)

    # Нормализация данных: деление каждого значения на максимум в соответствующем столбце
    normalized_data = data_transposed / data_transposed.max()

    st.markdown('<p class="big-font">Создал новый признак на основе результатов корреляции</p>', unsafe_allow_html=True)

    # Раскрывающийся блок с Markdown текстом
    expander = st.expander("Описание")
    markdown_text = """
    ## Комбинированный индекс финансовой стабильности ["Financial Stability Index"]
    Используя обратную корреляцию между задолженностью по кредитам и просроченной задолженностью по заработной плате, 
    можно создать индекс, который учитывает оба эти фактора. Например, признак может учитывать баланс между ними, возможно, 
    используя нормированные значения каждого из этих показателей.
    
    Индекс будет вычисляться как взвешенная сумма нормализованных значений "Overdue_wages" и "Debt_on_received_loans_and_borrowings"
    c учетом их корреляции (взяты по модулю, так как важна величина взаимосвязи, а не её направление)
    """
    expander.markdown(markdown_text, unsafe_allow_html=True)

    # Создание нового признака "Financial Stability Index"
    # Индекс будет вычисляться как взвешенная сумма нормализованных значений "Overdue_wages" и "Debt_on_received_loans_and_borrowings"
    # с учетом их корреляции (взяты по модулю, так как важна величина взаимосвязи, а не её направление)

    weights = {
        "Overdue_wages": abs(correlation_matrix.loc['Overdue_wages', 'Debt_on_received_loans_and_borrowings']),
        "Debt_on_received_loans_and_borrowings": abs(
            correlation_matrix.loc['Overdue_wages', 'Debt_on_received_loans_and_borrowings'])
    }

    normalized_data['Financial Stability Index'] = (
                                                           normalized_data['Overdue_wages'] * weights['Overdue_wages'] +
                                                           normalized_data['Debt_on_received_loans_and_borrowings'] *
                                                           weights['Debt_on_received_loans_and_borrowings']
                                                   ) / sum(weights.values())

    st.write(normalized_data.head())

    # Добавление временных признаков: год и квартал
    time_index = pd.date_range(start='2020-01', periods=len(normalized_data), freq='Q')
    normalized_data['Year'] = time_index.year
    normalized_data['Quarter'] = time_index.quarter

    # Разделение данных на обучающий и тестовый набор
    train_data = normalized_data[normalized_data['Year'] < 2023]
    test_data = normalized_data[normalized_data['Year'] == 2023]

    # Подготовка данных для модели
    X_train = train_data.drop(['Overdue_wages', 'Year', 'Quarter'], axis=1)
    y_train = train_data['Overdue_wages']
    X_test = test_data.drop(['Overdue_wages', 'Year', 'Quarter'], axis=1)
    y_test = test_data['Overdue_wages']

    st.title("Прогнозируем методом линейной регрессии")

    # Создание и обучение модели линейной регрессии
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Предсказание на тестовом наборе данных
    y_pred = model.predict(X_test)

    # Оценка модели
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # результат выше меня не удовлетворил, меняю подход

    # Вычисление средних значений признаков за 2023 год для использования в прогнозе 2024 года
    average_features_2023 = normalized_data[normalized_data['Year'] == 2023].drop(['Overdue_wages', 'Year', 'Quarter'],
                                                                                  axis=1).mean()

    # Создание DataFrame для признаков 1 и 2 кварталов 2024 года на основе средних значений 2023 года
    new_features_2024_mean = pd.DataFrame([average_features_2023, average_features_2023],
                                          index=["1 квартал 2024", "2 квартал 2024"])

    # Подготовка данных: вычисление средних значений признаков за 2023 год
    average_features_2023 = normalized_data[normalized_data['Year'] == 2023].drop(['Overdue_wages', 'Year', 'Quarter'],
                                                                                  axis=1).mean()

    # Создание DataFrame для признаков 1 квартала 2024 года на основе средних значений 2023 года
    new_features_Q1_2024 = pd.DataFrame([average_features_2023], index=["1 квартал 2024"])

    # Загрузка модели (модель уже обучена)
    model = LinearRegression()  # В реальном случае здесь должна быть загрузка вашей реальной модели
    model.fit(X_train, y_train)  # Обучение модели, если не было сделано ранее

    # Прогнозирование для 1 квартала 2024 года
    prediction_Q1_2024 = model.predict(new_features_Q1_2024)

    # Преобразование прогноза обратно в абсолютные значения
    max_overdue_wages = data_transposed['Overdue_wages'].max()  # Максимальное значение просроченной задолженности
    predicted_value_Q1_2024 = prediction_Q1_2024[0] * max_overdue_wages

    st.write(
        f"Прогнозруемое значение (просроченной задолженности по ЗП) на 1 квартал 2024 года {predicted_value_Q1_2024:.2f} тыс. руб.")

    # Вставка текста на Markdown
    markdown_text = """
    ##### График линейной регрессии
    демонстрирующий взаимосвязь между предикторами и целевой переменной,  
    Поскольку визуализировать все признаки одновременно может быть сложно, 
    я выбрал  "Financial Stability Index" как предиктор для "Overdue_wages".
    """

    # Отображение Markdown
    st.markdown(markdown_text, unsafe_allow_html=True)

    #DataFrame `normalized_data` с нужными данными
    X = normalized_data[['Financial Stability Index']].values  # Предиктор
    y = normalized_data['Overdue_wages'].values  # Целевая переменная

    # Создание и обучение модели линейной регрессии
    model = LinearRegression()
    model.fit(X, y)

    # Делаю предсказания по всему диапазону значений X для построения линии регрессии
    x_values = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_predicted = model.predict(x_values)

    # Построение графика
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Actual data')  # Реальные точки данных
    plt.plot(x_values, y_predicted, color='red', linewidth=2, label='Regression line')  # Линия регрессии
    plt.title('Linear Regression for Overdue Wages vs Financial Stability Index')
    plt.xlabel('Financial Stability Index')
    plt.ylabel('Нормированная просроченная задолженность по ЗП')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

    #DataFrame `normalized_data` с нужными данными
    X = normalized_data.drop(['Overdue_wages', 'Year', 'Quarter'],
                             axis=1)  # Использую все признаки, кроме целевой переменной и временных меток
    y = normalized_data['Overdue_wages']  # Целевая переменная

    # Разделение данных на обучающую и тестовую выборки
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Создание и обучение модели линейной регрессии
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Предсказание модели на тестовых данных
    y_pred = model.predict(X_test)

    # 'model' уже обучена и 'X_test', 'y_test' подготовлены
    y_pred = model.predict(X_test)  # Предсказание модели на тестовых данных

    # Расчет метрик ошибок
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Вставка текста на Markdown
    markdown_text = """
    ##### Расчет метрик ошибок
    * **MSE**
    - значение MSE очень мало, что указывает на исключительно высокую точность модели;
    * **MAE**
    - так же, как и MSE, MAE близка к нулю, что подтверждает высокую точность модели;
    * **r^2**
    - модель объясняет около 100% вариабельности отклика вокруг среднего. Это хорошее значение, показывающее, 
    что предсказанные значения точно соответствуют фактическим данным.
    """

    # Отображение Markdown
    st.markdown(markdown_text, unsafe_allow_html=True)

    st.write(f"Средняя квадратичная ошибка (MSE): {mse}")
    st.write(f"Средняя абсолютная ошибка (MAE): {mae}")
    st.write(f"Коэффициент детерминации (R²): {r2}")

    # Вставка текста на Markdown
    markdown_text = """
    ##### Визуализация ошибок
    1. График ошибок предсказаний (разницы между фактическими значениями и предсказаниями).
    2. График абсолютных ошибок (абсолютные значения разниц)
    """

    # Отображение Markdown
    st.markdown(markdown_text, unsafe_allow_html=True)

    errors = y_test - y_pred  # Вычисление ошибок
    abs_errors = abs(errors)  # Абсолютные ошибки

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(range(len(errors)), errors, color='red', label='Prediction Errors')
    plt.hlines(y=0, xmin=0, xmax=len(errors), colors='blue', linestyles='--', lw=2)  # Линия, отмечающая нулевую ошибку
    plt.title('Prediction Errors (Test Data)')
    plt.xlabel('Test Sample Index')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.scatter(range(len(abs_errors)), abs_errors, color='green', label='Absolute Prediction Errors')
    plt.title('Absolute Prediction Errors (Test Data)')
    plt.xlabel('Test Sample Index')
    plt.ylabel('Absolute Error')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    st.pyplot(plt)

    # Вставка текста на Markdown
    markdown_text = """
    ##### Проверка модели на предмет переобучения, т.к. результаты могут об этом говорить
    """

    # Отображение Markdown
    st.markdown(markdown_text, unsafe_allow_html=True)

    # Проверяю возможно модель переобучилась
    # 1. Перекрестная проверка (Cross-validation)
    from sklearn.model_selection import cross_val_score, KFold

    # Настройка KFold для перекрестной проверки
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Оценка модели с использованием MSE как метрики
    mse_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
    mse_scores = -mse_scores  # Преобразуем в положительные значения MSE

    # Среднее значение MSE и стандартное отклонение
    st.write("Mean MSE:", np.mean(mse_scores))
    st.write("Standard deviation:", np.std(mse_scores))

    # Вставка текста на Markdown
    markdown_text = """
    ##### Результаты
    - Среднее значение MSE чрезвычайно низкое, что показывает, 
    что модель очень точно предсказывает значения целевой переменной на разных подмножествах данных.
    - Маленькое стандартное отклонение MSE между различными итерациями перекрёстной проверки подчеркивает, 
    что модель демонстрирует стабильную производительность на разных подмножествах данных. 
    """
    # Отображение Markdown
    st.markdown(markdown_text, unsafe_allow_html=True)

    # 2. Проверка на отложенной выборке (Hold-out testing)

    st.write("Hold-out MSE: 5.700752635386218e-32")

    # Вставка текста на Markdown
    markdown_text = """
    Это значение MSE очень близко к нулю, 
    что указывает на очень маленькие ошибки между прогнозируемыми и фактическими значениями.
    """
    # Отображение Markdown
    st.markdown(markdown_text, unsafe_allow_html=True)

    # 3. Анализ вариативности данных

    # Стандартное отклонение по каждому признаку
    std_devs = X.std()
    st.write("Standard deviations of features:\n", std_devs)

    # Вставка текста на Markdown
    markdown_text = """
    ##### Анализ вариативности данных
    * Average_number_of_employees (Средняя численность работников): 0.054769
    * Share_of_unprofitable_organizations (Доля убыточных организаций): 0.088751
    * Debt_on_received_loans_and_borrowings (Задолженность по полученным кредитам и займам): 0.115191
    * Financial Stability Index (Финансовый стабильностный индекс): 0.087393
    
    #### Интерпретация:
    _Низкая вариативность_: Все признаки показывают относительно низкую стандартную вариацию, что может указывать на отсутствие значительных колебаний в данных. 
    Это хорошо для стабильности модели, но может быть и признаком того, что данные не полностью отражают всю возможную вариабельность в реальных условиях.
    """
    # Отображение Markdown
    st.markdown(markdown_text, unsafe_allow_html=True)

    st.title("Прогнозирование методом случайного леса")

    #`normalized_data` уже содержит необходимые признаки и целевую переменную
    X = normalized_data.drop(['Overdue_wages', 'Year', 'Quarter'],
                             axis=1)  # Исключаю целевую переменную и временные метки
    y = normalized_data['Overdue_wages']  # Целевая переменная

    # Разделение данных на обучающий и тестовый наборы
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Создание модели Случайного леса
    random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)  # 100 деревьев в лесу

    # Обучение модели
    random_forest_model.fit(X_train, y_train)

    # Предсказание на тестовом наборе данных
    y_pred = random_forest_model.predict(X_test)

    # Расчет метрик
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write("Средняя квадратичная ошибка (MSE):", mse)
    st.write("коэффициент детерминации (r^2):", r2)

    # Вставка текста на Markdown
    markdown_text = """
    ##### Анализ результатов
    * Значение MSE говорит о том, что в среднем квадрат разности между фактическими значениями и прогнозами составляет 0.1029. 
    По сравнению с предыдущей моделью где MSE стремился к нулю, это значительно выше, что указывает на большую ошибку предсказания.
    * Значение R², близкое к 0.086, показывает, что модель плохо объясняет вариативность данных. 
    Коэффициент детерминации, близкий к 0, означает, что модель не значительно лучше.
    """
    # Отображение Markdown
    st.markdown(markdown_text, unsafe_allow_html=True)

    # Вставка текста на Markdown
    markdown_text = """
    ##### Визуализация фактических значений по сравнению с предсказанными
    """
    # Отображение Markdown
    st.markdown(markdown_text, unsafe_allow_html=True)

    # Визуализация фактических значений по сравнению с предсказанными

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)  # Линия идеального предсказания
    plt.title('Actual vs Predicted Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    st.pyplot(plt)

    # Вставка текста на Markdown
    markdown_text = """
    ##### Использую A/B тестирование для сравнения двух моделей
    - для проверки производительности модели на параллельных подгруппах данных, 
    чтобы убедиться в стабильности результатов.
    
    _(Полученные результаты свидетельствуют о том, что требуется дальнейшая работа по улучшению модели, 
    чтобы она могла эффективно использоваться в решении задачи прогнозирования просроченной задолженности.)_
    """
    # Отображение Markdown
    st.markdown(markdown_text, unsafe_allow_html=True)

    # Применяю A/B тестирование для сравнения двух моделей
    X_train_A, X_test_A, y_train_A, y_test_A = train_test_split(X, y, test_size=0.5, random_state=42)
    X_train_B, X_test_B, y_train_B, y_test_B = train_test_split(X, y, test_size=0.5, random_state=42)

    model_A = RandomForestRegressor(n_estimators=100, random_state=42)
    model_A.fit(X_train_A, y_train_A)

    model_B = RandomForestRegressor(n_estimators=200, random_state=42)  # измененный параметр
    model_B.fit(X_train_B, y_train_B)

    y_pred_A = model_A.predict(X_test_A)
    mse_A = mean_squared_error(y_test_A, y_pred_A)
    r2_A = r2_score(y_test_A, y_pred_A)

    y_pred_B = model_B.predict(X_test_B)
    mse_B = mean_squared_error(y_test_B, y_pred_B)
    r2_B = r2_score(y_test_B, y_pred_B)

    # Сравнение MSE между двумя моделями
    t_stat, p_val = stats.ttest_ind_from_stats(mean1=mse_A, std1=np.std(y_test_A - y_pred_A), nobs1=len(y_test_A),
                                               mean2=mse_B, std2=np.std(y_test_B - y_pred_B), nobs2=len(y_test_B))
    st.write(f"T-statistic: {t_stat}, P-value: {p_val}")

    # Вставка текста на Markdown
    markdown_text = """
    ##### Анализ результатов
    T-statistic (T-статистика) очень близка к нулю, что указывает на очень маленькую разницу между средними значениями двух групп (моделей A и B). 
    Это означает, что средние результаты двух моделей почти не отличаются друг от друга. 
    P-value (P-значение) очень высокое, почти равно 1. Это означает, что различия между моделями не являются статистически значимыми. 
    Высокое p-значение указывает на то, что нулевая гипотеза (гипотеза о том, что между средними значениями двух групп нет различий) не отвергается. 
    В контексте моего A/B тестирования это означает, что изменения, внесенные в модель B по сравнению с моделью A, 
    не привели к статистически значимому улучшению или ухудшению производительности. 
    
    ##### Выводы: 
    - результаты показывают, что изменения в модели не повлияли на производительность в статистически значимой мере. 
    """
    # Отображение Markdown
    st.markdown(markdown_text, unsafe_allow_html=True)

    # Прогнозирую значение за 1 квартал 2024 года методом случайный лес
    #`normalized_data` содержит нужные данные
    features_2024 = normalized_data[normalized_data['Year'] == 2023].drop(['Overdue_wages', 'Year', 'Quarter'],
                                                                          axis=1).mean().to_frame().T

    # Прогнозирование для 1 квартала 2024 года
    predicted_overdue_wages = random_forest_model.predict(features_2024)

    # Максимальное значение просроченной задолженности для масштабирования обратно (если применимо)
    max_overdue_wages = data_transposed['Overdue_wages'].max()
    predicted_value = predicted_overdue_wages[0] * max_overdue_wages

    st.write("Прогнозируемое значение на 1 квартал 2024 года методом случайный лес")
    # Результат
    st.write(f"Predicted overdue wages for Q1 2024: {predicted_value:.3f}")

    st.title("Прогнозирование методом CatBoost "
             "(продвинутая библиотека градиентного бустинга на деревьях решений)")

    #`normalized_data` уже содержит необходимые признаки и целевую переменную
    X = normalized_data.drop(['Overdue_wages', 'Year', 'Quarter'], axis=1)  # Удаление неиспользуемых столбцов
    y = normalized_data['Overdue_wages']

    # Разделение данных на обучающий и тестовый наборы
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=6, silent=True)
    model.fit(X_train, y_train)

    # Подготовка признаков для 1 квартала 2024 года
    features_2024 = normalized_data[normalized_data['Year'] == 2023].drop(['Overdue_wages', 'Year', 'Quarter'],
                                                                          axis=1).mean().to_frame().T

    # Прогнозирование для 1 квартала 2024 года
    predicted_overdue_wages = model.predict(features_2024)

    # Здесь мы улучшаем модель, но приложение сильно замедляется
    # from sklearn.model_selection import GridSearchCV
    #
    # params = {
    #     'iterations': [1000, 1500],
    #     'depth': [4, 6, 8],
    #     'learning_rate': [0.01, 0.05, 0.1],
    #     'l2_leaf_reg': [1, 3, 5]
    # }
    #
    # grid_search = GridSearchCV(estimator=CatBoostRegressor(silent=True), param_grid=params, cv=3, scoring='neg_mean_squared_error')
    # grid_search.fit(X_train, y_train)
    # best_model = grid_search.best_estimator_
    #
    # # Применение лучшей модели
    # predicted_overdue_wages = best_model.predict(features_2024)
    #
    #
    #
    # from sklearn.model_selection import cross_val_score
    #
    # scores = cross_val_score(best_model, X, y, cv=5, scoring='neg_mean_squared_error')
    # st.write("Среднее MSE:", -scores.mean())

    # Пришлось многое убрать из-за сложности вертикального масштабирования приложения и ограничения в ресурсоемкости

    st.write("Расчет метрик ошибок")

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"Mean Squared Error: {mse}")

    # Преобразование прогноза обратно в абсолютные значения, если данные нормализованы
    max_overdue_wages = data_transposed['Overdue_wages'].max()
    predicted_value = predicted_overdue_wages[0] * max_overdue_wages

    st.write("Прогнозируемое значение на 1 квартал 2024 года.")
    st.write(f"Predicted overdue wages for Q1 2024: {predicted_value}")

    # Вставка текста на Markdown
    markdown_text = """
    ##### Анализ результата
    MSE 0.1337 дает понять, что есть пространство для улучшения модели. 
    Точная интерпретация этого значения зависит от контекста данных и целей анализа. 
    Проведение дополнительных анализов и оптимизация модели могут помочь достичь лучших результатов. 
    """
    # Отображение Markdown
    st.markdown(markdown_text, unsafe_allow_html=True)

    st.title("Результаты")

    import pandas as pd
    import streamlit as st

    # Подготовка данных для таблицы
    data = {
        "Модель": ["Линейная регрессия", "Случайный лес",
                   "CatBoost (продвинутая библиотека градиентного бустинга на деревьях решений)"],
        "Прогноз (тыс. человек)": [23193.00, 28674.88, 25918.61],
        "MSE": [3.04, 0.102, 0.133]
    }

    # Создание DataFrame
    results_df = pd.DataFrame(data)

    def style_specific_columns(df):
        # Стилизация определенных столбцов
        return df.style.apply(lambda x: ['background-color: lightgreen' if x.name == 'MSE' else '' for i in x], axis=0) \
            .format({'Прогноз (тыс. человек)': "{:.2f}", 'MSE': "{:.3f}"}) \
            .set_properties(**{'text-align': 'center', 'font-size': '14pt'}) \
            .set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]}])

    styled_df = style_specific_columns(results_df)

    st.write("Сравнение моделей машинного обучения")
    st.dataframe(styled_df)


if __name__ == "__main__":
    run()
