# Тестовое Задание

* Используя исходные данные представленные по ссылке Train_Data_200k создать модель, нацеленную на прогнозирование значений параметров Target_1...4 по значениям Tag_1...79.
* Применить функцию к тестовой выборке представленной в файле test_data_100k и на основании значений Tag_1...79 получить прогнозы для параметров Target_1...4.
* После получения модели написать pipeline, который берёт данные из базы (SQLite), подготавливает данные, делает предсказания при помощи обученной модели и сохраняет результат предсказаний в новую таблицу.
* Полученные прогнозы, топ-10 значимых тэгов, рабочую тетрадь и pipeline, направить в качестве результата в ответном письме.
