from __future__ import annotations
import sqlite3 as sq
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from datetime import datetime
import time
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from abc import ABC, abstractmethod


def normalise_data(data):
    """
    Функция преобразует пустые строки в NaN значения
    """
    for col in data.columns:
        data[col] = data[col].replace('', np.nan)
    return data

def import_data_sql(database='dataset.db'):
    '''
    Получение тренировочного и тестового датасета из БД SQLite
    '''

    sqlite_connection = sq.connect(database)

    data_or = pd.read_sql_query("SELECT * FROM train_200;", sqlite_connection)

    data_test = pd.read_sql_query("SELECT * FROM test_100;", sqlite_connection)

    if sqlite_connection:
        sqlite_connection.close()

    data_or = normalise_data(data_or)
    data_test = normalise_data(data_test)

    return data_or, data_test


class Dataset:
    '''
    Класс для предварительной обработки датасета
    '''
    def __init__(self, data, data_test,  target_cols=None, fill_cols_null=None, cols_drop=None, cols_normalize=None, target=None, ):
        self.data = data
        self.data_test = data_test
        self.target_cols = target_cols
        self.fill_cols_null = fill_cols_null
        self.cols_drop = cols_drop
        self.cols_normalize = cols_normalize
        self.target = target

    def del_duplicates(self):

        # Метод для  удаления дубликатов
        self.data.drop_duplicates(inplace=True)

    def del_target_nan(self):

        # Метод для удаления строк, которые содержат nan значения в таргетах
        for target in self.target_cols:
            self.data = self.data.loc[~self.data[target].isna()]

        return self.data

    def get_target(self):

        # Метод для получения таргета
        self.target = self.data[self.target_cols]
        return self.target

    def fill_nan_dataset(self):

        # Метод, который заполняет колонки из списка fill_cols_null нулями,
        # остальные колонки заполняются с помощью линейной интерполяции
        self.data[self.fill_cols_null] = self.data[self.fill_cols_null].fillna(0)
        self.data_test[self.fill_cols_null] = self.data_test[self.fill_cols_null].fillna(0)

        self.data = self.data.interpolate(method='linear')
        self.data = self.data.fillna(self.data.median())
        self.data_test = self.data_test.fillna(self.data.median())

        return self.data, self.data_test

    def drop_corr_cols(self, th=0.85):

        # Метод, который удаляет столбцы, коэфициент корреляции в котором выше порога th.
        a = self.data.corr().round(2).abs().unstack().sort_values()
        list_corr_col = []
        for index, val in a.items():
            if val >= th and index[0] != index[1] and index[1] not in list_corr_col:
                list_corr_col.append(index[0])
        set_list_corr_col = list(set(list_corr_col))
        self.data.drop(set_list_corr_col, inplace=True, axis=1)

        return self.data



    def drop_cols(self):

        # print(self.data)
        # Метод, для удаление колонок таргета из датасета и колонок из списка cols_drop
        self.data.drop(self.target_cols, inplace=True, axis=1)

        if self.cols_drop:
            self.data.drop(self.cols_drop, inplace=True, axis=1)

        return self.data


    def drop_cols_test(self):

        # Метод, который оставляет в тестовой выборке те же столбцы, что и в тренировочной
        col_drop_test = []
        for el in self.data_test.columns:
            if el not in self.data.columns:
                col_drop_test.append(el)
        self.data_test.drop(col_drop_test, inplace=True, axis=1)

        return self.data_test


    def normalization_train_test(self):

        # Метод, для нормализации данных

        full_data = pd.concat([self.data, self.data_test])
        # Преобразует столбец с временем в числовой
        if self.cols_normalize:

            def time_2ms(str_time):
                dt_end = datetime.strptime(str_time, "%Y-%m-%d %H:%M:%S")
                dt_start = datetime.strptime('1970-01-01 00:00:00', "%Y-%m-%d %H:%M:%S")
                dt_epoch = (dt_end - dt_start).total_seconds()
                return dt_epoch

            full_data[self.cols_normalize] = full_data[self.cols_normalize].apply(time_2ms)

        scaler = MinMaxScaler()
        norm_data = pd.DataFrame(scaler.fit_transform(full_data),
                                 columns=full_data.columns, index=full_data.index)

        self.data = norm_data.loc[:199999, :]
        self.data_test = norm_data.loc[0:, :]

        return self.data, self.data_test



class Handler(ABC):

    @abstractmethod
    def set_next(self, handler):
        pass

    @abstractmethod
    def handle(self, request):
        pass


class AbstractHandler(Handler):

    def __init__(self, data, data_test, target):
        self.data = data
        self.data_test = data_test
        self.target = target

    _next_handler = None

    def set_next(self, handler):
        self._next_handler = handler
        return handler

    @abstractmethod
    def handle(self, request):
        if self._next_handler:
            return self._next_handler.handle(request)

        return None

    def train_test_ds(self,data, target):

        X_train, X_val, y_train, y_val = train_test_split(data, target, test_size=0.2, random_state=0)
        return X_train, X_val, y_train, y_val

    def pred_to_csv(self, predict):

        predict.to_csv(r'test_predict.csv')



class MultiRegr_Model(AbstractHandler):

    def __init__(self, data, data_test, target):
        self.data = data
        self.data_test = data_test
        self.target = target

    def handle(self, request):
        if request == "Multi_regr_model":

            X_train, X_val, y_train, y_val = super().train_test_ds(self.data, self.target)

            # обучение модели
            model = MultiOutputRegressor(Ridge(random_state=123))
            model_fit = model.fit(X_train, y_train)

            # предсказание на выборке валидации
            val_pred = model_fit.predict(X_val)
            score = model_fit.score(X_val, y_val)
            mse = mean_squared_error(y_val, val_pred)
            print(f'MultiOutputRegressor score: {score}\n'
                  f'MultiOutputRegressor mse: {mse}\n')

            # предсказание на тестовой выборке
            predict = pd.DataFrame(model_fit.predict(self.data_test), columns=self.target.columns)

            super().pred_to_csv(predict)

            return True

        else:
            return super().handle(request)


class RidgePipeline_Model(AbstractHandler):

    def __init__(self, data, data_test, target):
        self.data = data
        self.data_test = data_test
        self.target = target


    def handle(self, request):
        if request == "RidgePipeline_Model":

            X_train, X_val, y_train, y_val = super().train_test_ds(self.data, self.target)

            dict_predict = {}

            for tgt in self.target.columns:

                model = make_pipeline(PolynomialFeatures(degree=1), Ridge(alpha=0.2))
                model_fit = model.fit(X_train, y_train[tgt])

                # предсказание на выборке валидации
                val_pred = model_fit.predict(X_val)
                score = model_fit.score(X_val, y_val[tgt])
                mse = mean_squared_error(y_val[tgt], val_pred)
                print(f'Таргет {tgt}\n'
                      f'Pipeline model score: {score}\n'
                      f'Pipeline model mse: {mse}\n')

                # предсказание на тестовом наборе
                predict_test = model_fit.predict(self.data_test)
                dict_predict[tgt] = predict_test


            predict = pd.DataFrame(dict_predict, columns=self.target.columns)
            super().pred_to_csv(predict)

            return True
        else:
            return super().handle(request)



def model_code(handler, model_name):

    result = handler.handle(model_name)
    if result:
        pass
    else:
        print(f"Выберете модель из списка {list_models}", end="")



if __name__ == "__main__":

    data, data_test = import_data_sql()
    target_cols = ['target1', 'target2', 'target3', 'target4']
    fill_cols_null = ['tag19', 'tag36', 'tag37', 'tag38', 'tag39',
                      'tag44', 'tag45', 'tag46', 'tag48', 'tag52',
                      'tag53', 'tag54', 'tag61', 'tag62', 'tag69', 'tag73']
    cols_drop = ['tag46']
    cols_normalize = 'time'

    list_models = ['Multi_regr_model', 'RidgePipeline_Model']

    dataset = Dataset(data, data_test, target_cols, fill_cols_null, cols_drop, cols_normalize)

    # Препроцессинг датасета
    dataset.del_duplicates()
    dataset.del_target_nan()
    dataset.get_target()
    dataset.fill_nan_dataset()
    dataset.drop_cols()
    dataset.drop_corr_cols()
    dataset.drop_cols_test()
    dataset.normalization_train_test()

    # Создание экземпляров классов
    multi_regr = MultiRegr_Model(dataset.data, dataset.data_test, dataset.target)
    RidgePipeline_Model = RidgePipeline_Model(dataset.data, dataset.data_test, dataset.target)

    # Последовательность исполнения в "цепочке обязанностей"
    multi_regr.set_next(RidgePipeline_Model)

    # Вызов обучения модели
    model_name = 'RidgePipeline_Model'
    model_code(multi_regr, model_name)



