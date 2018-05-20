'''
The purpose of this module is to provide tools for
converting OHLCV, market, and SMA data into inputs
for a word vector embedding and a 1D CNN

NOTE there are some instances in which data is
assumed to be small enough to use uint32,
and other forced dtypes
'''

import numpy as np
import pandas as pd
import sqlite3
import h5py


BINS_CHG = np.array([0.96, 0.975, 0.9875, 0.995, 1.0, 1.005, 1.0125, 1.025, 1.04])

MULT_BARSIZE = 10
BINS_BARSIZE = np.array([0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07])

MULT_CL = 100
BINS_CL = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1])

MULT_MKT = 1000
BINS_MKT = np.array([0.97, 0.98, 0.99, 0.997, 1.0, 1.003, 1.01, 1.02, 1.03])

MULT_VOLSMA50 = 10000
BINS_VOLSMA50 = np.array([0.33, 0.67, 1.0, 1.5, 2.0, 4.0, 7.0])

MULT_PRSMA50 = 100000
BINS_PRSMA50 = np.array([0.90, 0.95, 0.98, 1.0, 1.02, 1.05, 1.1])

MULT_PRSMA200 = 1000000
BINS_PRSMA200 = np.array([0.70, 0.90, 0.95, 1.05, 1.1, 1.3])


def get_data(sym_list, upper_lim):
    price_hist_conn = sqlite3.connect("/home/carl/Trading/KaggleSecuritiesData/1Day/StockData.db")
    cur = price_hist_conn.cursor()
    query = "SELECT Date, Close FROM 'SP500IDX' WHERE Date < '{}' ORDER BY Date ASC".format(upper_lim)
    res = cur.execute(query)
    bulk = np.array(res.fetchall())
    mkt = pd.DataFrame(data=bulk[:, 1],
                       dtype=np.float32,
                       index=np.array(bulk[:, 0], dtype=np.datetime64),
                       columns=['SP500IDX'])
    hi = pd.DataFrame(data=None, index=mkt.index, dtype=np.float32)
    lo = pd.DataFrame(data=None, index=mkt.index, dtype=np.float32)
    cl = pd.DataFrame(data=None, index=mkt.index, dtype=np.float32)
    vol = pd.DataFrame(data=None, index=mkt.index, dtype=np.float32)
    sma50 = pd.DataFrame(data=None, index=mkt.index, dtype=np.float32)
    volsma50 = pd.DataFrame(data=None, index=mkt.index, dtype=np.float32)
    sma200 = pd.DataFrame(data=None, index=mkt.index, dtype=np.float32)

    # join is slow so join in blocks
    hi_block = []
    lo_block = []
    cl_block = []
    vol_block = []
    sma50_block = []
    volsma50_block = []
    sma200_block = []
    ctr = 0
    for sym in sym_list:
        ctr += 1
        cur = price_hist_conn.cursor()
        query = "SELECT Date, High, Low, Close, Volume, sma50, volsma50, sma200 FROM '{}' ORDER BY Date ASC"
        res = cur.execute(query.format(sym))
        bulk = np.array(res.fetchall())
        cur.close()
        sym_dates = np.array(bulk[:, 0], dtype=np.datetime64)
        hi_block.append(pd.DataFrame(data=bulk[:, 1],
            dtype=np.float32,
            index=sym_dates,
            columns=[sym]))
        lo_block.append(pd.DataFrame(data=bulk[:, 2],
            dtype=np.float32,
            index=sym_dates,
            columns=[sym]))
        cl_block.append(pd.DataFrame(data=bulk[:, 3],
            dtype=np.float32,
            index=sym_dates,
            columns=[sym]))
        vol_block.append(pd.DataFrame(data=bulk[:, 4],
            dtype=np.float32,
            index=sym_dates,
            columns=[sym]))
        sma50_block.append(pd.DataFrame(data=bulk[:, 5],
            dtype=np.float32,
            index=sym_dates,
            columns=[sym]))
        volsma50_block.append(pd.DataFrame(data=bulk[:, 6],
            dtype=np.float32,
            index=sym_dates,
            columns=[sym]))
        sma200_block.append(pd.DataFrame(data=bulk[:, 7],
            dtype=np.float32,
            index=sym_dates,
            columns=[sym]))

        if ctr > 499:
            print(str(ctr) + ' ' + sym)
            hi = hi.join(hi_block, how='left')
            lo = lo.join(lo_block, how='left')
            cl = cl.join(cl_block, how='left')
            vol = vol.join(vol_block, how='left')
            sma50 = sma50.join(sma50_block, how='left')
            volsma50 = volsma50.join(volsma50_block, how='left')
            sma200 = sma200.join(sma200_block, how='left')

            ctr = 0
            hi_block = []
            lo_block = []
            cl_block = []
            vol_block = []
            sma50_block = []
            volsma50_block = []
            sma200_block = []

    else:
        # for when the loop exits
        if len(cl_block) > 0:
            hi = hi.join(hi_block, how='left')
            lo = lo.join(lo_block, how='left')
            cl = cl.join(cl_block, how='left')
            vol = vol.join(vol_block, how='left')
            sma50 = sma50.join(sma50_block, how='left')
            volsma50 = volsma50.join(volsma50_block, how='left')
            sma200 = sma200.join(sma200_block, how='left')
            del hi_block, lo_block, cl_block, vol_block, sma50_block, volsma50_block, sma200_block, ctr

    price_hist_conn.close()
    return hi, lo, cl, vol, sma50, volsma50, sma200, mkt

def tokenize(hi, lo, cl, vol, sma50, volsma50, sma200, mkt):

    # make comparisons of each close to prev close
    # must be in ascending date order
    chg = cl / cl.shift(1, axis=0) # must drop the NaN
    chg.drop(chg.index[0], inplace=True)
    hi.drop(hi.index[0], inplace=True)
    lo.drop(lo.index[0], inplace=True)
    cl.drop(cl.index[0], inplace=True)
    vol.drop(vol.index[0], inplace=True)
    sma50.drop(sma50.index[0], inplace=True)
    volsma50.drop(volsma50.index[0], inplace=True)
    sma200.drop(sma200.index[0], inplace=True)

    nan_mask = np.isnan(hi) | np.isnan(lo) | np.isnan(cl) | np.isnan(vol) | np.isnan(sma50) | np.isnan(volsma50)
    tokenized = pd.DataFrame(data=np.zeros_like(cl.values, dtype=np.int64),
                             index=cl.index,
                             columns=cl.columns)

    tokenized += np.digitize(chg.values, BINS_CHG)
    del chg

    hi_lo = hi - lo
    bar_size = hi_lo / cl
    del hi

    tokenized += MULT_BARSIZE * np.digitize(bar_size.values, BINS_BARSIZE)

    cl_in_bar = (cl - lo) / hi_lo
    del lo
    tokenized += MULT_CL * np.digitize(cl_in_bar.values, BINS_CL)

    mkt_chg = mkt / mkt.shift(1, axis=0)
    mkt_chg.drop(mkt_chg.index[0], inplace=True)
    del mkt
    # need numpy broadcasting of single mkt_chg column across cols of data in tokenized
    tokenized = pd.DataFrame(data=tokenized.values + MULT_MKT * np.digitize(mkt_chg.values, BINS_MKT),
                             index=tokenized.index,
                             columns=tokenized.columns)

    vol_to_sma50 = vol / volsma50
    del vol, volsma50
    tokenized += MULT_VOLSMA50 * np.digitize(vol_to_sma50.values, BINS_VOLSMA50)

    pr_to_sma50 = cl / sma50
    del sma50
    tokenized += MULT_PRSMA50 * np.digitize(pr_to_sma50.values, BINS_PRSMA50)

    pr_to_sma200 = cl / sma200
    del cl, sma200
    tokenized += MULT_PRSMA200 * np.digitize(pr_to_sma200.values, BINS_PRSMA200)

    tokenized[nan_mask] = -1

    return tokenized

def token_to_bar(token, prev_close):
    '''
    Reconstruct high-low-close price data from token
    '''

    def get_digit(number, n):
        return number // 10**n % 10

    def process_idx(i, bins):
        len_bin = bins.shape[0]
        if i == 0 and bins[0] > 0:
            return bins[0]
        elif i == 1 and bins[0] == -1:
            return bins[1]
        elif i == len_bin:
            return bins[-1]
        else:
            return np.mean((bins[i], bins[i-1]))

    chg = process_idx(get_digit(token, 0), BINS_CHG)
    bar_size = process_idx(get_digit(token, 1), BINS_BARSIZE)
    cl_in_bar = process_idx(get_digit(token, 2), BINS_CL)

    cl = prev_close * chg
    rescaled_bar = bar_size * cl
    lo = cl - cl_in_bar * rescaled_bar
    hi = lo + rescaled_bar

    return hi, lo, cl


def package_data_for_LSTM():
    '''
    crawl throuhgh the data and assemble an h5 file which contains a dataset for each symbol,
    where each dataset is 2 columns [embedding row index, class label] sorted by date ASC.
    '''

    kaggle_folder = "/home/carl/Trading/KaggleSecuritiesData/1Day/"

    with sqlite3.connect(kaggle_folder + 'BarTokens.db') as conn:
        cur = conn.cursor()
        res = cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        sym_list = [r[0] for r in res]

    with open(kaggle_folder + 'TokenToRow.txt') as f:
        tokens = list(map(int, f))
    token_to_row = {token: idx for idx, token in enumerate(tokens)}
    del tokens

    y_conn = sqlite3.connect(kaggle_folder + "StockData.db")
    x_conn = sqlite3.connect(kaggle_folder + 'BarTokens.db')
    ctr = 0
    for sym in sym_list:
        ctr += 1
        if ctr % 100 == 0:
            print(ctr)

        y_cur = y_conn.cursor()
        res = y_cur.execute("SELECT Date, Label FROM '{}' WHERE Date < '2017-12-01' AND Label IS NOT NULL ORDER BY Date ASC".format(sym))
        bulk = np.array(res.fetchall())
        labels = pd.DataFrame(data=bulk[:, 1].astype(np.float64),
                              index=bulk[:, 0].astype(np.datetime64),
                              columns=['Labels'])
        x_cur = x_conn.cursor()
        res = x_cur.execute("SELECT Date, Token FROM '{}' WHERE Date < '2017-12-01' ORDER BY Date ASC".format(sym))
        bulk = np.array(res.fetchall())
        embd_rows = pd.DataFrame(data=bulk[:, 1].astype(np.uint32),
                                 index=bulk[:, 0].astype(np.datetime64),
                                 columns=['EmbeddingMatrixRow'])

        embd_rows['EmbeddingMatrixRow'] = np.array([token_to_row[v] for v in embd_rows['EmbeddingMatrixRow'].values], dtype=np.uint32)

        embd_rows = embd_rows.join(labels, how='left')
        embd_rows.dropna(axis=0, how='any', inplace=True)

        with h5py.File('LSTM_data.h5', 'a') as hf:
            hf.create_dataset(sym,  data=embd_rows.values.astype(np.uint32))

    y_conn.close()
    x_conn.close()


def test_price_reconstruction():
    '''
    Transform price history to tokens, then transform back to price history and compare
    '''
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    sym_list = ['TWTR']
    hi, lo, cl, vol, sma50, volsma50, sma200, mkt = get_data(sym_list, '2017-07-07')
    tokenized = tokenize(hi, lo, cl, vol, sma50, volsma50, sma200, mkt)

    hi = hi.values.flatten()[-120:]
    lo = lo.values.flatten()[-120:]
    cl = cl.values.flatten()[-120:]
    tokenized = tokenized.values.flatten()[-120:]
    mask = tokenized > -1
    hi = hi[mask]
    lo = lo[mask]
    cl = cl[mask]
    tokenized = tokenized[mask]
    #bars = [token_to_bar(tokenized[i], cl[i-1]) for i in range(1, tokenized.shape[0])]
    #bars.insert(0, (hi[0], lo[0], cl[0]))
    bars = [(hi[0], lo[0], cl[0])]
    for i in range(1, tokenized.shape[0]):
        bars.append(token_to_bar(tokenized[i], bars[i-1][2]))


    shift = -3
    for i in range(len(bars)):
        plt.plot([i], [cl[i]], '.', color='k', markersize=8)
        plt.plot([i, i], [lo[i], hi[i]], color='k', linewidth=2)
        plt.plot([i], [shift + bars[i][2]], '.', color='b', markersize=8)
        plt.plot([i, i], [shift + bars[i][1], shift + bars[i][0]], color='b', linewidth=2)
    plt.savefig('test_reconstruction.png', dpi=300)

def class_labels_3days_ahead():
    '''
    crawl over price histories and determine class labels for training data, then add the class labels to the price history db. Labels are determined based on the magnitudes of the gain and maximum drawdown over the next 3 days. Total of 26 classes
    '''

    kaggle_conn = sqlite3.connect('/home/carl/Trading/KaggleSecuritiesData/1Day/StockData.db')
    kcur = kaggle_conn.cursor()
    query = "SELECT name FROM sqlite_master WHERE type='table';"
    res = kcur.execute(query).fetchall()
    sym_list = [r[0] for r in res]

    query = "SELECT Date, High, Low, Close FROM '{}' ORDER BY Date ASC"
    insert_label_stmnt = "UPDATE '{}' SET Label = {} WHERE Date = '{}'"

    ctr = 0
    for sym in sym_list:
        if sym == 'SP500IDX':
            continue
        ctr += 1
        if ctr % 200 == 0:
            print(ctr)

        res = np.array(kcur.execute(query.format(sym)).fetchall())
        dt = np.array(res[:,0], dtype=np.datetime64)
        hi = np.array(res[:,1], dtype=np.float64)
        lo = np.array(res[:,2], dtype=np.float64)
        cl = np.array(res[:,3], dtype=np.float64)

        for i in range(cl.shape[0] - 3):
            label = None
            changes = cl[i+1: i+4] - cl[i]
            highs = hi[i+1: i+4]
            lows = lo[i+1: i+4]
            largest_move_idx = np.argmax(np.abs(changes))
            largest_move_pct = abs(changes[largest_move_idx] / cl[i])
            if changes[largest_move_idx] >= 0:
                lowest_low = np.min(lows[:largest_move_idx+1])
                lowest_low_pct = (lowest_low / cl[i]) - 1
                if largest_move_pct < 0.01:
                    label = 7 # slightly up
                elif largest_move_pct >= 0.01 and largest_move_pct < 0.02:
                    if lowest_low_pct < -0.01:
                        label = 10 # choppy and risky, leaning weak bullish
                    elif lowest_low_pct < -0.005:
                        label = 8 # moderate risk, leaning weak bullish
                    else:
                        label = 2 # low risk, weak bullish
                elif largest_move_pct >= 0.02 and largest_move_pct < 0.03:
                    if lowest_low_pct < -0.02:
                        label = 9 # choppy and too risky, bullish
                    elif lowest_low_pct < -0.0125:
                        label = 5
                    else:
                        label = 1
                elif largest_move_pct >= 0.03 and largest_move_pct < 0.05:
                    if lowest_low_pct < -0.025:
                        label = 11
                    elif lowest_low_pct < -0.015:
                        label = 4
                    else:
                        label = 0 # Low risk, decisively bullish but not crazy strong
                elif largest_move_pct >= 0.05:
                    if lowest_low_pct < -0.03:
                        label = 12
                    elif lowest_low_pct < -0.02:
                        label = 6
                    else:
                        label = 3
            elif changes[largest_move_idx] < 0:
                highest_high = np.max(highs[:largest_move_idx+1])
                highest_high_pct = (highest_high / cl[i]) - 1
                if largest_move_pct < 0.01:
                    label = 20 # slightly up
                elif largest_move_pct >= 0.01 and largest_move_pct < 0.02:
                    if highest_high_pct > 0.01:
                        label = 23 # choppy and risky, leaning weak bearish
                    elif highest_high_pct > 0.005:
                        label = 21 # moderate risk, leaning weak bearish
                    else:
                        label = 15 # low risk, weak bearish
                elif largest_move_pct >= 0.02 and largest_move_pct < 0.03:
                    if highest_high_pct > 0.02:
                        label = 22 # choppy and too risky, bearish
                    elif highest_high_pct > 0.0125:
                        label = 18
                    else:
                        label = 14
                elif largest_move_pct >= 0.03 and largest_move_pct < 0.05:
                    if highest_high_pct > 0.025:
                        label = 24
                    elif highest_high_pct > 0.015:
                        label = 17
                    else:
                        label = 13
                elif largest_move_pct >= 0.05:
                    if highest_high_pct > 0.03:
                        label = 25
                    elif highest_high_pct > 0.02:
                        label = 19
                    else:
                        label = 16

            assert label is not None
            kcur.execute(insert_label_stmnt.format(sym, label, dt[i]))

        kaggle_conn.commit()
    kaggle_conn.close()


def class_labels_2xRisk_strategy():
    '''
    Simulate three months of trading a strategy with 6% allowed position risk (stop loss) seeking 2xRisk (12%) gain as exit criteria, moving stop loss order to breakeven at 1xRisk unrealized profit.
    '''
    kaggle_conn = sqlite3.connect('/home/carl/Trading/KaggleSecuritiesData/1Day/StockData.db')
    kcur = kaggle_conn.cursor()
    query = "SELECT name FROM sqlite_master WHERE type='table';"
    res = kcur.execute(query).fetchall()
    sym_list = [r[0] for r in res]

    query = "SELECT Date, High, Low, Close FROM '{}' ORDER BY Date ASC"
    insert_label_stmnt = "UPDATE '{}' SET Label = {} WHERE Date = '{}'"

    ctr = 0
    stop_loss = 0.06
    for sym in sym_list:
        if sym == 'SP500IDX':
            continue
        ctr += 1
        if ctr % 200 == 0:
            print(ctr)

        res = np.array(kcur.execute(query.format(sym)).fetchall())
        dt = np.array(res[:,0], dtype=np.datetime64)
        hi = np.array(res[:,1], dtype=np.float64)
        lo = np.array(res[:,2], dtype=np.float64)
        cl = np.array(res[:,3], dtype=np.float64)

        for i in range(cl.shape[0] - 63):
            label = None
            changes = 1. * lo[i+1: i+63] / cl[i]
            highs = hi[i+1: i+63]
            #lows = lo[i+1: i+63]
            stopped_out_idx = np.where(changes <= (1-stop_loss))[0]
            if stopped_out_idx.size > 0:
                stopped_out_idx = stopped_out_idx[0]
            else:
                stopped_out_idx = None
            if stopped_out_idx is not None:
                if stopped_out_idx == 0:
                    label = 0
                    continue
                max_gain = 1. * highs[:stopped_out_idx].max() / cl[i]
                if max_gain >= 2. * (1. + stop_loss):
                    label = 2 # made a gain of twice the size of the risk
                    continue
                elif max_gain >= 1. + stop_loss:
                    label = 1 # move stop up to breakeven, close at breakeven
                    continue
                else:
                    label = 0 # stopped out at a loss
                    continue
            else:
                max_gain = 1. * highs.max() / cl[i]
                if max_gain >= 2. * (1. + stop_loss):
                    label = 2 # made a gain of twice the size of the risk
                    continue
                else:
                    # its possible we won't be forced to close out, but as a conservative step assume brekeven closure
                    label = 1
                    continue

            assert label is not None
            kcur.execute(insert_label_stmnt.format(sym, label, dt[i]))

        for i in range(cl.shape[0] - 63, cl.shape[0]):
            kcur.execute(insert_label_stmnt.format(sym, 'null', dt[i]))

        kaggle_conn.commit()
    kaggle_conn.close()


def y_vals_regression():
    '''
    Experimenting with simple regression using next day's price change
    '''
    kaggle_conn = sqlite3.connect('/home/carl/Trading/KaggleSecuritiesData/1Day/StockData.db')
    kcur = kaggle_conn.cursor()
    query = "SELECT name FROM sqlite_master WHERE type='table';"
    res = kcur.execute(query).fetchall()
    sym_list = [r[0] for r in res]

    query = "SELECT Date, Close FROM '{}' ORDER BY Date ASC"
    insert_label_stmnt = "UPDATE '{}' SET Label = {} WHERE Date = '{}'"

    ctr = 0
    for sym in sym_list:
        if sym == 'SP500IDX':
            continue
        ctr += 1
        if ctr % 200 == 0:
            print(ctr)

        res = np.array(kcur.execute(query.format(sym)).fetchall())
        dt = np.array(res[:,0], dtype=np.datetime64)
        cl = np.array(res[:,1], dtype=np.float64)

        for i in range(cl.shape[0] - 1):
            chg = cl[i+1] / cl[i] - 1
            kcur.execute(insert_label_stmnt.format(sym, chg, dt[i]))

        kcur.execute(insert_label_stmnt.format(sym, 'Null', dt[-1]))

        kaggle_conn.commit()
    kaggle_conn.close()


if __name__ == '__main__':
    #test_price_reconstruction()
    class_labels_3days_ahead()
    package_data_for_LSTM()