import logging
import json
from sqlalchemy import create_engine
import numpy as np
from io import BytesIO
import pymysql
from collections import deque

class DailyTradingEnv():

    def __init__(self, seq_length, test_date_rate, data_type, file_path):
        self._seq_length = seq_length
        self._test_date_rate = test_date_rate
        self._data_type = data_type
        self._file_path = file_path
        # Open, High, Low, Volume, Close
        self.train_size = None
        self.trainX = None
        self.trainY = None

        self.test_size = None
        self.testX = None
        self.testY = None

        #self.mydb = Mydb()
        self.reset()

    # Standardization
    def _data_standardization(self, x):
        x_np = np.asarray(x)
        return (x_np - x_np.mean()) / x_np.std()

    # 너무 작거나 너무 큰 값이 학습을 방해하는 것을 방지하고자 정규화한다
    # x가 양수라는 가정하에 최소값과 최대값을 이용하여 0~1사이의 값으로 변환
    # Min-Max scaling
    def _min_max_scaling(self, x):
        x_np = np.asarray(x)
        return (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-7)  # 1e-7은 0으로 나누는 오류 예방차원

    # 정규화된 값을 원래의 값으로 되돌린다
    # 정규화하기 이전의 org_x값과 되돌리고 싶은 x를 입력하면 역정규화된 값을 리턴한다
    def reverse_min_max_scaling(self, org_x, x):
        org_x_np = np.asarray(org_x)
        x_np = np.asarray(x)
        return (x_np * (org_x_np.max() - org_x_np.min() + 1e-7)) + org_x_np.min()

    def _build_data_set_200(self, xy):
        # stock, date, Open, High, Low, Volume, Close
        # 데이터의위치변경진행
        self.volume = xy[:, [-2]]  # volume copy
        temp = np.delete(xy, 4, 1)
        self.price = temp[:, 1:5]
        #self.price = np.concatenate((temp, close), axis=1)  # axis=1, 세로로 합친다

        self.norm_price = self._min_max_scaling(self.price)  # 가격형태 데이터 정규화 처리
        print("price.shape: ", self.price.shape)
        print("price[0]: ", self.price[0])
        print("norm_price[0]: ", self.norm_price[0])
        print("=" * 100)  # 화면상 구분용

        self.norm_volume = self._min_max_scaling(self.volume)  # 거래량형태 데이터 정규화 처리
        print("volume.shape: ", self.volume.shape)
        print("volume[0]: ", self.volume[0])
        print("norm_volume[0]: ", self.norm_volume[0])
        print("=" * 100)  # 화면상 구분용

        norm_xy = np.concatenate((self.norm_price, self.norm_volume), axis=1)  # axis=1, 세로로 합친다
        # Open, High, Low, Close, Volume
        x = self.xy = norm_xy
        #y = self.xy = xy
        y = norm_xy[:, [-2]]  # Close as label

        # build a dataset
        dataX = []
        dataY = []

        self.index_init = []
        index = 0
        change_stock = False
        for i in range(0,len(y) - self._seq_length):
            if change_stock == True and index != i:
                continue

            stock = xy[i][0]
            append_tf = True
            for j in (i+1, i+self._seq_length):
                if stock != xy[j][0]:
                    index = j
                    append_tf = False
                    change_stock = True
                    break
            if append_tf == True:
                if change_stock == True:
                    self.index_init.append(1)
                    change_stock = False
                else:
                    self.index_init.append(0)
                _x = x[i:i + self._seq_length]
                _y = y[i + self._seq_length]  # Next close price
                print(_x, "->", _y)
                dataX.append(_x)
                dataY.append(_y)
                index = index + 1

        # train/test split
        self.train_size = int(len(dataY) * self._test_date_rate)
        self.test_size = len(dataY) - self.train_size
        self.train_index, self.test_index = np.array(self.index_init[0:self.train_size]), np.array(
            self.index_init[self.train_size:len(self.index_init)])
        self.trainX, self.testX = np.array(dataX[0:self.train_size]), np.array(
            dataX[self.train_size:len(dataX)])
        self.trainY, self.testY = np.array(dataY[0:self.train_size]), np.array(
            dataY[self.train_size:len(dataY)])


    def _build_data_set(self, xy):
        # 데이터의위치변경진행
        close = xy[:, [-1]]  # close copy
        temp = np.delete(xy, 4, 1)
        self.volume = temp[:, [-1]]  # volume copy
        self.price = np.delete(xy, 3, 1)  # delete volume
        #self.price = np.concatenate((temp, close), axis=1)  # axis=1, 세로로 합친다

        self.norm_price = self._min_max_scaling(self.price)  # 가격형태 데이터 정규화 처리
        print("price.shape: ", self.price.shape)
        print("price[0]: ", self.price[0])
        print("norm_price[0]: ", self.norm_price[0])
        print("=" * 100)  # 화면상 구분용

        self.norm_volume = self._min_max_scaling(self.volume)  # 거래량형태 데이터 정규화 처리
        print("volume.shape: ", self.volume.shape)
        print("volume[0]: ", self.volume[0])
        print("norm_volume[0]: ", self.norm_volume[0])
        print("=" * 100)  # 화면상 구분용

        xy = np.concatenate((self.norm_price, self.norm_volume), axis=1)  # axis=1, 세로로 합친다
        # Open, High, Low, Close, Volume
        x = self.xy = xy
        #y = self.xy = xy
        y = xy[:, [-2]]  # Close as label

        # build a dataset
        dataX = []
        dataY = []
        for i in range(0, len(y) - self._seq_length):
            _x = x[i:i + self._seq_length]
            _y = y[i + self._seq_length]  # Next close price
            print(_x, "->", _y)
            dataX.append(_x)
            dataY.append(_y)

        # train/test split
        self.train_size = int(len(dataY) * self._test_date_rate)
        self.test_size = len(dataY) - self.train_size
        self.trainX, self.testX = np.array(dataX[0:self.train_size]), np.array(
            dataX[self.train_size:len(dataX)])
        self.trainY, self.testY = np.array(dataY[0:self.train_size]), np.array(
            dataY[self.train_size:len(dataY)])

    def _get_state_file_data(self):
        if self._file_path == './20100101_sample.txt':
            # stock, date, Open, High, Low, Volume, Close
            xy = np.loadtxt(self._file_path, delimiter=',', usecols=(0, 2, 3, 4, 5, 6))
        else:
            # Open, High, Low, Volume, Close
            xy = np.loadtxt(self._file_path, delimiter=',')
            xy = xy[::-1]  # reverse order (chronically ordered)

        return xy

    def get_recent_data(self):
        return np.array([self.xy[len(self.xy) - self._seq_length:]])

    def _get_state_data(self):

        #sql = "SELECT a.stock_code, a.date, a.datetime, a.open, a.high, a.low, a.close, a.volume, a.firstbuy \
        sql = "SELECT a.stock_code, a.date, a.open, a.high, a.low, a.volume, a.close \
               FROM tb_daily_trans a, \
                    ( \
                     SELECT a.stock_code \
                       FROM (SELECT stock_code FROM TB_Stock_Master m WHERE (m.kospi200 = 'Y' OR m.kosdakP = 'Y')      AND m.Use_YN = 'Y') a \
        ORDER BY RAND() \
                LIMIT 1 \
                ) b \
                WHERE a.stock_code = b.stock_code \
                AND a.date >= '20100101' \
                ORDER BY a.date "
        #list to np.array
        results = self.mydb.select_sql_excute(sql)
        results_as_list = [ [i[2],i[3],i[4],i[5],i[6]] for i in results]
        #y = np.array(x, dtype=np.float16)
        array = np.asarray(results_as_list, dtype=np.float32)
        print(array)

        return array

    def reset(self):
        if self._data_type == 'File':
            xy = self._get_state_file_data()
        else:
            xy = self._get_state_data()

        #if self._file_path ==  './20100101_sample.txt':
        self._build_data_set_200(xy)
        #else:
        #    self._build_data_set(xy)


class Mydb():

    def __init__(self):
        with open('db_account.json') as data_file:
            self.data_loaded = json.load(data_file)
            self.conn = pymysql.connect(host=self.data_loaded["host"], port=self.data_loaded["port"],
            user=self.data_loaded["user"],
            passwd=self.data_loaded["passwd"],
            db=self.data_loaded["db"], charset=self.data_loaded["charset"])

    def select_sql_excute(self, sql):
        cur = self.conn.cursor()
        cur.execute(sql)
        result = cur.fetchall()
        self.conn.close()
        return (result)

    def select_sql_excute_stock(self, sql, stock_code, stock_date):
        cur = self.conn.cursor()
        cur.execute(sql, (stock_code, stock_date))
        result = cur.fetchall()
        self.conn.close()
        return (result)

    def save_table(self, df, table):
        DB_URI = "mysql+mysqlconnector://{user}:{password}@{host}:{port}/{db}"

        try:
            engine = create_engine(DB_URI.format(
                user=self.data_loaded["user"],
                password=self.data_loaded["passwd"],
                host=self.data_loaded["host"],
                port=self.data_loaded["port"],
                db=self.data_loaded["db"]),
                connect_args={'time_zone': '+09:00'}
            )

            self.conn = engine.connect()

            df.to_sql(name=table, con=engine, index='false', if_exists='append')
            print("Saved successfully!!")

        except:
            traceback.print_exc()

        finally:
            self.conn.close()

'''
if __name__ == "__main__":
    mydb = Mydb()
    env = DailyTradingEnv()
'''
