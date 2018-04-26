import logging
import json
from sqlalchemy import create_engine
import numpy as np
from io import BytesIO
import pymysql
import gym
from gym import spaces
from collections import deque

CODE_MARK_MAP = {0: ' ', 1: 'O', 2: 'X'}
NUM_LOC = 9

CONTROL_STOCK = 3
# 0 hold, 1 ~ NUM_STOCK buy, NUM_STOCK+1 ~ NUM_STOCK*2 sell
NUM_ACTION = CONTROL_STOCK * 2 + 1
NUM_STATE_INFO = 11

START_AMT = 10000000
END_TIME = '145500'
#tic 20180402153030

O_REWARD = 1
X_REWARD = -1
NO_REWARD = -0.001

LEFT_PAD = ' '
LOG_FMT = logging.Formatter('%(levelname)s '
'[%(filename)s:%(lineno)d] %(message)s',
'%Y-%m-%d %H:%M:%S')

'''
def tomark(code):
    return CODE_MARK_MAP[code]


def tocode(mark):
    return 1 if mark == 'O' else 2


def next_mark(mark):
    return 'X' if mark == 'O' else 'O'


def agent_by_mark(agents, mark):
    for agent in agents:
        if agent.mark == mark:
    return agent


def after_action_state(state, action):
    """Execute an action and returns resulted state.

    Args:
    state (tuple): Board status + mark
    action (int): Action to run

    Returns:
    tuple: New state
    """

    board, mark = state
    nboard = list(board[:])
    nboard[action] = tocode(mark)
    nboard = tuple(nboard)
    return nboard, next_mark(mark)


def check_game_status(board):
    """Return game status by current board status.

    Args:
    board (list): Current board state

    Returns:
    int:
    -1: game in progress
    0: draw game,
    1 or 2 for finished game(winner mark code).
    """
    for t in [1, 2]:
    for j in range(0, 9, 3):
    if [t] * 3 == [board[i] for i in range(j, j+3)]:
    return t
    for j in range(0, 3):
    if board[j] == t and board[j+3] == t and board[j+6] == t:
    return t
    if board[0] == t and board[4] == t and board[8] == t:
    return t
    if board[2] == t and board[4] == t and board[6] == t:
    return t

    for i in range(9):
    if board[i] == 0:
    # still playing
    return -1

    # draw game
    return 0

'''

class DailyTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, alpha=0.02, show_number=False):
        self.action_space = spaces.Discrete(NUM_ACTION)
        self.observation_space = spaces.Discrete(NUM_STATE_INFO)
        self.alpha = alpha
        self.set_start_mark('O')
        self.show_number = show_number

        # 잔고, 평균매입가, 매입수량, 매입가치, 매입가치포함 잔고, 수익률
        self.balance = START_AMT
        self.buying_avramt = 0
        self.buying_volume = 0
        self.buying_value = 0
        self.return_value = 0
        self.return_rate = 0

        self.tictime = '090000'
        #self.next_stat = null
        #self._seed()
        self.mydb = Mydb()
        self.reset()

    def set_start_mark(self, mark):
        self.start_mark = mark

    def _firtstbuy(self, amt):
        if amt < 5000 :
            unit = 5
        elif amt < 10000 :
            unit = 10
        elif amt < 50000 :
            unit = 50
        elif amt < 100000 :
            unit = 100
        elif amt < 500000 :
            unit = 500
        else :
            unit = 1000

        amt += unit
        return amt

    def _add_tic(self):
        hour = int(self.tictime[:2])
        min = int(self.tictime[2:4])
        sec = int(self.tictime[4:6])

        if sec == 59:
            sec = 0
            if min == 59:
                min = 0
                hour += 1
            else:
                min += 1
        else:
            sec += 1

        if sec < 10:
            sec_c = '0' + str(sec)
        else:
            sec_c = str(sec)

        if min < 10:
            min_c = '0' + str(min)
        else:
            min_c = str(min)

        if hour < 10:
            hour_c = '0' + str(hour)
        else:
            hour_c = str(hour)


        self.tictime = hour_c + min_c + sec_c

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
        array = np.asarray(results_as_list, dtype=np.float32)
        print(array)

        return array

    def _get_return_value(self, close, balance, buying_volume):
        sell_value = (close - close * 0.315 / 100) * buying_volume if buying_volume > 0 else 0
        return(int(balance + sell_value))

    def _get_return_rate(self, return_value):
        return(int((return_value - START_AMT) / START_AMT * 100))

    def reset(self):
        self.board = [0] * NUM_STATE_INFO
        self.tic_que = self._get_state_data()
        #self.done = False
        #return self._get_obs()

    def step(self, action):
        """Step environment by action.

        Args:
        action (int): Location

        Returns:
        list: Obeservation
        int: Reward
        bool: Done
        dict: Additional information
        행동, next 환경, 결과, 보상
        """
        assert self.action_space.contains(action)

        #loc = action
        #if self.done:
        #    return self._get_obs(), 0, True, None

        reward = NO_REWARD
        # place
        temp = self.tic_que.popleft()
        self.board = _toboard(temp, action)

        reward, status = _check_game_status()
        logging.debug("check_game_status board {} reward '{}'"
        " status {}".format(self.board, reward, status))
        if status > 0:
            self.done = True

        # switch turn
        return self._get_obs(), reward, self.done, None


    def _get_obs(self):
        return tuple(self.board)

    def render(self, mode='human', close=False):
        if close:
            return
        if mode == 'human':
            print(self.board)
        else:
             self._show_board(logging.info)
             logging.info('')
    def _show_result(self, showfn, reward):
        status = check_game_status(self.board)
        assert status > 0
        msg = "ReturnRate '{}'!".format(self.return_rate)
        showfn("==== Finished: {} ====".format(msg))
        showfn('')



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
