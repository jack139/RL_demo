import os
import copy
import time
import warnings
from copy import deepcopy
from typing import List

import pandas as pd
from tqdm import tqdm

from meta.data_processors._base import _Base

warnings.filterwarnings("ignore")


class Qmt(_Base):
    """
    key-value in kwargs
    ----------
        token : str
            get from https://waditu.com/ after registration
        adj: str
            Whether to use adjusted closing price. Default is None.
            If you want to use forward adjusted closing price or 前复权. pleses use 'qfq'
            If you want to use backward adjusted closing price or 后复权. pleses use 'hfq'
    """

    def __init__(
        self,
        data_source: str,
        start_date: str,
        end_date: str,
        time_interval: str,
        **kwargs,
    ):
        super().__init__(data_source, start_date, end_date, time_interval, **kwargs)
        assert "csv_path" in kwargs.keys(), "Please input csv path!"
        self.csv_path = kwargs["csv_path"]

    def get_data(self, id) -> pd.DataFrame:
        df1 = pd.read_csv(os.path.join(self.csv_path, f'{id}.csv'))
        df1.columns=['date', 'open', 'high', 'low', 'close', 'volume' ]
        df1['tic']=[id]*len(df1)
        #df[ (df['date']>'20221001') & (df['date']<'20221015')]
        return df1[ (df1['date']>=self.start_date.replace('-','')) & (df1['date']<=self.end_date.replace('-',''))]

    # 从csv载入数据
    def download_data(self, ticker_list: List[str]):
        """
        `pd.DataFrame`
            7 columns: date, open, high, low, close, volume, tic
            for the specified stock ticker
        """
        #assert self.time_interval == "1d", "Not supported currently"

        self.ticker_list = ticker_list

        self.dataframe = pd.DataFrame()
        for i in tqdm(ticker_list, total=len(ticker_list)):
            # nonstandard_id = self.transfer_standard_ticker_to_nonstandard(i)
            # df_temp = self.get_data(nonstandard_id)
            df_temp = self.get_data(i)
            self.dataframe = self.dataframe.append(df_temp)
            # print("{} ok".format(i))
            #time.sleep(0.25)

        self.dataframe.sort_values(by=["date", "tic"], inplace=True)
        self.dataframe.reset_index(drop=True, inplace=True)

        self.dataframe = self.dataframe[
            ["tic", "date", "open", "high", "low", "close", "volume"]
        ]
        # self.dataframe.loc[:, 'tic'] = pd.DataFrame((self.dataframe['tic'].tolist()))
        self.dataframe["date"] = pd.to_datetime(self.dataframe["date"], format="%Y%m%d %H:%M:%S")
        self.dataframe["day"] = self.dataframe["date"].dt.dayofweek
        self.dataframe["date"] = self.dataframe.date.apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))

        self.dataframe.dropna(inplace=True)
        self.dataframe.sort_values(by=["date", "tic"], inplace=True)
        self.dataframe.reset_index(drop=True, inplace=True)

        print("Shape of DataFrame: ", self.dataframe.shape)


    # 直接从 df 设置数据
    def set_data(self, ticker_list: List[str], df):
        """
        `pd.DataFrame`
            7 columns: date, open, high, low, close, volume, tic
            for the specified stock ticker
        """
        #assert self.time_interval == "1d", "Not supported currently"

        self.ticker_list = ticker_list

        self.dataframe = df

        self.dataframe.sort_values(by=["date", "tic"], inplace=True)
        self.dataframe.reset_index(drop=True, inplace=True)

        self.dataframe = self.dataframe[
            ["tic", "date", "open", "high", "low", "close", "volume"]
        ]
        # self.dataframe.loc[:, 'tic'] = pd.DataFrame((self.dataframe['tic'].tolist()))
        self.dataframe["date"] = pd.to_datetime(self.dataframe["date"], format="%Y%m%d %H:%M:%S")
        self.dataframe["day"] = self.dataframe["date"].dt.dayofweek
        self.dataframe["date"] = self.dataframe.date.apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))

        self.dataframe.dropna(inplace=True)
        self.dataframe.sort_values(by=["date", "tic"], inplace=True)
        self.dataframe.reset_index(drop=True, inplace=True)

        print("Shape of DataFrame: ", self.dataframe.shape)


    def clean_data(self):
        dfc = copy.deepcopy(self.dataframe)

        dfcode = pd.DataFrame(columns=["tic"])
        dfcode.tic = dfc.tic.unique()

        if "time" in dfc.columns.values.tolist():
            dfc = dfc.rename(columns={"time": "date"})

        dfdate = pd.DataFrame(columns=["date"])
        dfdate.date = dfc.date.unique()
        dfdate.sort_values(by="date", ascending=False, ignore_index=True, inplace=True)

        # the old pandas may not support pd.merge(how="cross")
        try:
            df1 = pd.merge(dfcode, dfdate, how="cross")
        except:
            print("Please wait for a few seconds...")
            df1 = pd.DataFrame(columns=["tic", "date"])
            for i in range(dfcode.shape[0]):
                for j in range(dfdate.shape[0]):
                    df1 = df1.append(
                        pd.DataFrame(
                            data={
                                "tic": dfcode.iat[i, 0],
                                "date": dfdate.iat[j, 0],
                            },
                            index=[(i + 1) * (j + 1) - 1],
                        )
                    )

        df2 = pd.merge(df1, dfc, how="left", on=["tic", "date"])

        # back fill missing data then front fill
        df3 = pd.DataFrame(columns=df2.columns)
        for i in self.ticker_list:
            df4 = df2[df2.tic == i].fillna(method="bfill").fillna(method="ffill")
            df3 = pd.concat([df3, df4], ignore_index=True)

        df3 = df3.fillna(0)

        if "index" in df3.columns.values.tolist():
            df3 = df3.drop(columns=["index"])

        # reshape dataframe
        df3 = df3.sort_values(by=["date", "tic"]).reset_index(drop=True)

        print("Shape of DataFrame: ", df3.shape)

        self.dataframe = df3

    # def add_technical_indicator(self, tech_indicator_list: List[str], select_stockstats_talib: int=0):
    #     """
    #     calculate technical indicators
    #     use stockstats/talib package to add technical inidactors
    #     :param data: (df) pandas dataframe
    #     :return: (df) pandas dataframe
    #     """
    #     df = self.dataframe.copy()
    #     if "date" in df.columns.values.tolist():
    #         df = df.rename(columns={'date': 'time'})
    #
    #     if self.data_source == "ccxt":
    #         df = df.rename(columns={'index': 'time'})
    #
    #     # df = df.reset_index(drop=False)
    #     # df = df.drop(columns=["level_1"])
    #     # df = df.rename(columns={"level_0": "tic", "date": "time"})
    #     if select_stockstats_talib == 0:  # use stockstats
    #         stock = stockstats.StockDataFrame.retype(df.copy())
    #         unique_ticker = stock.tic.unique()
    #         #print(unique_ticker)
    #         for indicator in tech_indicator_list:
    #             indicator_df = pd.DataFrame()
    #             for i in range(len(unique_ticker)):
    #                 try:
    #                     temp_indicator = stock[stock.tic == unique_ticker[i]][indicator]
    #                     temp_indicator = pd.DataFrame(temp_indicator)
    #                     temp_indicator["tic"] = unique_ticker[i]
    #                     temp_indicator["time"] = df[df.tic == unique_ticker[i]][
    #                         "time"
    #                     ].to_list()
    #                     indicator_df = indicator_df.append(
    #                         temp_indicator, ignore_index=True
    #                     )
    #                 except Exception as e:
    #                     print(e)
    #             #print(indicator_df)
    #             df = df.merge(
    #                 indicator_df[["tic", "time", indicator]], on=["tic", "time"], how="left"
    #             )
    #     else:  # use talib
    #         final_df = pd.DataFrame()
    #         for i in df.tic.unique():
    #             tic_df = df[df.tic == i]
    #             tic_df['macd'], tic_df['macd_signal'], tic_df['macd_hist'] = MACD(tic_df['close'], fastperiod=12,
    #                                                                               slowperiod=26, signalperiod=9)
    #             tic_df['rsi'] = RSI(tic_df['close'], timeperiod=14)
    #             tic_df['cci'] = CCI(tic_df['high'], tic_df['low'], tic_df['close'], timeperiod=14)
    #             tic_df['dx'] = DX(tic_df['high'], tic_df['low'], tic_df['close'], timeperiod=14)
    #             final_df = final_df.append(tic_df)
    #         df = final_df
    #
    #     df = df.sort_values(by=["time", "tic"])
    #     df = df.rename(columns={'time': 'date'})    # 1/11 added by hx
    #     df = df.dropna()
    #     print("Succesfully add technical indicators")
    #     self.dataframe = df

    # def get_trading_days(self, start: str, end: str) -> List[str]:
    #     print('not supported currently!')
    #     return ['not supported currently!']

    # def add_turbulence(self, data: pd.DataFrame) \
    #         -> pd.DataFrame:
    #     print('not supported currently!')
    #     return pd.DataFrame(['not supported currently!'])

    # def calculate_turbulence(self, data: pd.DataFrame, time_period: int = 252) \
    #         -> pd.DataFrame:
    #     print('not supported currently!')
    #     return pd.DataFrame(['not supported currently!'])

    # def add_vix(self, data: pd.DataFrame) \
    #         -> pd.DataFrame:
    #     print('not supported currently!')
    #     return pd.DataFrame(['not supported currently!'])

    def add_vix(self, idx_code='000001.SH'):
        vix_df = self.get_data(idx_code)

        vix_df = vix_df[
            ["tic", "date", "open", "high", "low", "close", "volume"]
        ]
        vix_df["date"] = pd.to_datetime(vix_df["date"], format="%Y%m%d %H:%M:%S")
        vix_df["day"] = vix_df["date"].dt.dayofweek
        vix_df["date"] = vix_df.date.apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))

        vix_df.dropna(inplace=True)
        vix_df.sort_values(by=["date", "tic"], inplace=True)
        vix_df.reset_index(drop=True, inplace=True)

        vix = vix_df[["date", "close"]]
        vix = vix.rename(columns={"close": "vixy"})

        self.dataframe = self.dataframe.merge(vix, on="date")
        self.dataframe = self.dataframe.sort_values(["date", "tic"]).reset_index(drop=True)


    # def df_to_array(self, df: pd.DataFrame, tech_indicator_list: List[str], if_vix: bool) \
    #         -> List[np.array]:
    #     print('not supported currently!')
    #     return pd.DataFrame(['not supported currently!'])

    def data_split(self, df, start, end, target_date_col="date"):
        """
        split the dataset into training or testing using date
        :param data: (df) pandas dataframe, start, end
        :return: (df) pandas dataframe
        """
        data = df[(df[target_date_col] >= start) & (df[target_date_col] < end)]
        data = data.sort_values([target_date_col, "tic"], ignore_index=True)
        data.index = data[target_date_col].factorize()[0]
        return data

    # "600000.XSHG" -> "600000.SH"
    # "000612.XSHE" -> "000612.SZ"
    def transfer_standard_ticker_to_nonstandard(self, ticker: str) -> str:
        n, alpha = ticker.split(".")
        assert alpha in ["XSHG", "XSHE"], "Wrong alpha"
        if alpha == "XSHG":
            nonstandard_ticker = n + ".SH"
        elif alpha == "XSHE":
            nonstandard_ticker = n + ".SZ"
        return nonstandard_ticker


import tushare as ts
import pandas as pd
from matplotlib import pyplot as plt


class ReturnPlotter:
    """
    An easy-to-use plotting tool to plot cumulative returns over time.
    Baseline supports equal weighting(default) and any stocks you want to use for comparison.
    """

    def __init__(self, df_account_value, df_trade, start_date, end_date):
        self.start = start_date
        self.end = end_date
        self.trade = df_trade
        self.df_account_value = df_account_value

    def get_baseline(self, ticket):
        df = ts.get_hist_data(ticket, start=self.start, end=self.end)
        df.loc[:, "dt"] = df.index
        df.index = range(len(df))
        df.sort_values(axis=0, by="dt", ascending=True, inplace=True)
        df["date"] = pd.to_datetime(df["dt"], format="%Y-%m-%d")
        return df

    def get_baseline2(self, time_interval, csv_path, remove=0, ticket="000001.SH"):
        ts_processor_trade = Qmt(data_source="tushare", 
                                           start_date=self.start,
                                           end_date=self.end,
                                           time_interval=time_interval,
                                           csv_path=csv_path)
        baseline_df = ts_processor_trade.get_data(ticket)
        if remove>0:
            return baseline_df.head(len(baseline_df)-remove)
        else:
            return baseline_df

    def plot(self, baseline_ticket=None, csv_path='data', time_interval="15m", remove=0):
        """
        Plot cumulative returns over time.
        use baseline_ticket to specify stock you want to use for comparison
        (default: equal weighted returns)
        """
        baseline_label = "Equal-weight portfolio"
        tic2label = {"399300": "CSI 300 Index", "000016": "SSE 50 Index"}
        if baseline_ticket:
            # 使用指定ticket作为baseline
            baseline_df = self.get_baseline2(ticket=baseline_ticket, time_interval=time_interval,
                csv_path=os.path.join(csv_path, time_interval), remove=remove)
            #baseline_df = baseline_df[
            #    baseline_df.dt != "2020-06-26"
            #]  # ours don't have date=="2020-06-26"
            baseline = baseline_df.close.tolist()
            baseline_label = tic2label.get(baseline_ticket, baseline_ticket)
        else:
            # 均等权重
            all_date = self.trade.date.unique().tolist()
            if remove>0:
                all_date = all_date[:-remove]
            baseline = []
            for day in all_date:
                day_close = self.trade[self.trade["date"] == day].close.tolist()
                avg_close = sum(day_close) / len(day_close)
                baseline.append(avg_close)

        ours = self.df_account_value.account_value.tolist()
        ours = self.pct(ours)
        baseline = self.pct(baseline)

        days_per_tick = (
            60  # you should scale this variable accroding to the total trading days
        )
        time = list(range(len(ours)))
        datetimes = self.df_account_value.date.tolist()
        ticks = [tick for t, tick in zip(time, datetimes) if t % days_per_tick == 0]
        plt.title("Cumulative Returns")
        plt.plot(time, ours, label="Agent", color="green")
        plt.plot(time, baseline, label=baseline_label, color="grey")
        plt.xticks([i * days_per_tick for i in range(len(ticks))], ticks, fontsize=7)

        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")

        plt.legend()
        plt.show()

    def plot_all(self):
        baseline_label = "Equal-weight portfolio"
        tic2label = {"399300": "CSI 300 Index", "000016": "SSE 50 Index"}

        # 399300
        baseline_ticket = "399300"
        baseline_df = self.get_baseline(baseline_ticket)
        baseline_df = baseline_df[
            baseline_df.dt != "2020-06-26"
        ]  # ours don't have date=="2020-06-26"
        baseline_300 = baseline_df.close.tolist()
        baseline_label_300 = tic2label[baseline_ticket]

        # 000016
        baseline_ticket = "000016"
        baseline_df = self.get_baseline(baseline_ticket)
        baseline_df = baseline_df[
            baseline_df.dt != "2020-06-26"
        ]  # ours don't have date=="2020-06-26"
        baseline_50 = baseline_df.close.tolist()
        baseline_label_50 = tic2label[baseline_ticket]

        # 均等权重
        all_date = self.trade.date.unique().tolist()
        baseline_equal_weight = []
        for day in all_date:
            day_close = self.trade[self.trade["date"] == day].close.tolist()
            avg_close = sum(day_close) / len(day_close)
            baseline_equal_weight.append(avg_close)

        ours = self.df_account_value.account_value.tolist()

        ours = self.pct(ours)
        baseline_300 = self.pct(baseline_300)
        baseline_50 = self.pct(baseline_50)
        baseline_equal_weight = self.pct(baseline_equal_weight)

        days_per_tick = (
            60  # you should scale this variable accroding to the total trading days
        )
        time = list(range(len(ours)))
        datetimes = self.df_account_value.date.tolist()
        ticks = [tick for t, tick in zip(time, datetimes) if t % days_per_tick == 0]
        plt.title("Cumulative Returns")
        plt.plot(time, ours, label="DDPG Agent", color="darkorange")
        plt.plot(
            time,
            baseline_equal_weight,
            label=baseline_label,
            color="cornflowerblue",
        )  # equal weight
        plt.plot(
            time, baseline_300, label=baseline_label_300, color="lightgreen"
        )  # 399300
        plt.plot(time, baseline_50, label=baseline_label_50, color="silver")  # 000016
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")

        plt.xticks([i * days_per_tick for i in range(len(ticks))], ticks, fontsize=7)
        plt.legend()
        plt.show()

    def pct(self, l):
        """Get percentage"""
        base = l[0]
        return [x / base for x in l]

    def get_return(self, df, value_col_name="account_value"):
        df = deepcopy(df)
        df["daily_return"] = df[value_col_name].pct_change(1)
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
        df.set_index("date", inplace=True, drop=True)
        df.index = df.index.tz_localize("UTC")
        return pd.Series(df["daily_return"], index=df.index)
