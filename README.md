# Проект «Разработка Информационной Системы прогнозирования рынка труда России методами машинного обучения».
 
## Формирование проблемы
Проблема, которую предстоит решить в рамках данной работы, за-ключается в необходимости разработки такой информационной системы, 
которая была бы способна анализировать большие объемы данных, учи-тывая множество параметров, и предоставлять точные прогнозы по разви-тию рынка труда России. 
Для решения этой задачи необходимо сформу-лировать и исследовать ряд вопросов, связанных с выбором методов ма-шинного обучения, определением наиболее значимых переменных для прогнозирования, 
разработкой алгоритмов обработки данных и оценки точности получаемых прогнозов.

## Используемые методы исследования
В процессе работы предусматривается использование современных про-граммных средств и библиотек машинного обучения, 
таких как scikit-learn, streamlit, pandas, numpy, matplotlib, statsmodels, catboost, scipy, seaborn, plotly 
что позволит эффективно обрабатывать большие объемы данных и реализовывать сложные алгоритмы. 

## Степень освещенности темы в литературе
Тема прогнозирования рынка труда с использованием методов ма-шинного обучения занимает важное место в научных исследованиях по-следних лет. 
Это область на пересечении экономики труда, информатики и прикладной статистики, где ключевое внимание уделяется разработке 
и применению алгоритмов для анализа больших данных и выявления тен-денций на рынке труда.

## 3 подзадачи в рамках системы
Эта система нацелена на решение трех основных задач: 1. **прогнозирование численности безработных в Республике Татарстан**,
2. **прогнозирование ожидаемой заработной платы на основе информации из резюме кандидатов** 
и 3. **прогнозирование просроченной задолженности по заработной плате в г. Москва**. 

## Общая архитетура
Для реализации информационной системы прогнозирования рынка труда России методами машинного обучения, 
было выбрано приложение на Streamlit. Данное приложение представляет собой эффективное решение, 
сочетающее в себе как серверные, так и клиентские аспекты. Это поз-воляет обеспечивать быстрый доступ к машинному обучению 
и аналитиче-ским инструментам через удобный веб-интерфейс. 
Streamlit — это открытая библиотека для Python, разработанная для быстрого создания интерактивных веб-приложений для машинного обуче-ния и анализа данных. 
Она позволяет разработчикам с минимальными за-тратами времени создавать визуально привлекательные, 
интерактивные веб-интерфейсы для научных и аналитических приложений. 

## Разработка алгоритмов
Разработка алгоритмов машинного обучения, предназначенных для точного прогнозирования рынка труда в России, представляет собой сложный процесс, требующий тщательного подхода к выбору методов и инструментов. 
В контексте создания информационной системы с использованием Streamlit, основное внимание уделяется разработке алгоритмов, 
которые могут эффективно работать в реальном времени и обладать высокой производительностью. 
Это диктует ряд единых требований к разработке алгоритмов и построению моделей.
**Одним из ключевых требований является использование нормализованных исходных данных для всех алгоритмов**. У нормализации данных есть альтернативы, но именно касаемо наших задач было принято решение привести все данные к единой шкале, 
чтобы обеспечить сопоставимость результатов и максимальную эффективность обработки.
**Следующим значимым аспектом является выбор алгоритмов низкой и средней сложности**. 
**Выбор обусловлен необходимостью поддержания высокой производительности приложения**. 
В условиях Streamlit, где приложение должно оперативно реагировать на запросы пользователя и обновляться в реальном времени, использование сложных и трудоемких алгоритмов может привести к неприемлемым задержкам. 
**Поэтому в работе основное предпочтение отдается моделям, которые могут быстро обрабатывать данные и выдавать результаты, не ухудшая пользовательский опыт**.
**Также было установлено требование о применении одинакового количества времени, затраченном на разработку каждого алгоритма**. 
Это условие введено для того, чтобы обеспечить справедливые и одинаковые условия для сравнения различных методов машинного обучения. 
**Такой подход позволяет оценить эффективность каждой модели в чистом виде, без предвзятости, связанной с различным уровнем вложений в разработку и оптимизацию**.
**Важно отметить, что все эти требования к разработке алгоритмов были выдвинуты не только для достижения технической эффективности, 
но и для обеспечения исследовательской целостности исследования. Создание такого набора стандартов позволяет нам систематически подходить к анализу данных и обеспечивает повторяемость 
и проверяемость результатов исследовательских экспериментов**. Это подход, который я применяю в данной работе, 
позволяет максимально объективно оценить потенциал различных алгоритмов машинного обучения в решении актуальных задач прогнозирования рынка труда.

_____________________________________________
## Дополнительно:
Запуск приложения: streamlit run main.py
Также, перед началом следует загрузить и поменять путь используемых наборов данных.
![Main](https://github.com/Rayray360180/fqp_ira/assets/68148073/6c0c1948-ac75-4f59-9e75-311fe7fa214a)
![one1](https://github.com/Rayray360180/fqp_ira/assets/68148073/71422151-9b32-419f-80e4-9305a3f17427)
![three3](https://github.com/Rayray360180/fqp_ira/assets/68148073/7d88531a-c765-419f-b817-0ab324ead1f6)



