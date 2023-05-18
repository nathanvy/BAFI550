import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression

def get_beta(equity, market):
    """Takes two arguments that should be Pandas series and computes
    the beta of the equity series with respect to the market series"""
    equity.fillna(0, inplace=True)
    market.fillna(0, inplace=True)
    covariance = market.cov(equity)
    variance = market.var()
    return (covariance / variance)


def get_rsquared(predictor, response):
    """Take two series as predictor and response variables, run
    a linear regression, and return the R^2 value"""
    predictor.fillna(0, inplace=True)
    response.fillna(0, inplace=True)

    p = predictor.values.reshape(-1, 1)
    r = response.values.reshape(-1, 1)
    model = LinearRegression()
    model.fit(p, r)
    return model.score(p, r)


sns.set_style("white")
sns.set_style("ticks")

startdate = "2017-01-01"
enddate = "2023-04-30"

spx = yf.Ticker("^GSPC").history(start=startdate, end=enddate, interval="1wk")
dis = yf.Ticker("DIS").history(start=startdate, end=enddate, interval="1wk")
# fox = yf.Ticker("FOX").history(start="2019-03-10", end=enddate, interval="1wk")
cmcsa = yf.Ticker("CMCSA").history(start=startdate, end=enddate, interval="1wk")
para = yf.Ticker("PARA").history(start=startdate, end=enddate, interval="1wk")
nflx = yf.Ticker("NFLX").history(start=startdate, end=enddate, interval="1wk")
tfcf = pd.read_csv("tfcf.csv", parse_dates=['Date'])
tfcf = tfcf.set_index(['Date'])   # convert to datetime64
tfcf = tfcf.sort_index(ascending=True)
fox = pd.read_csv("fox.csv", parse_dates=['Date']) # yfinance data was fucked, trust me
fox = fox.set_index(['Date'])
fox = fox.sort_index(ascending=True)

nflx['logret'] = np.log(nflx.Close) - np.log(nflx.Close.shift(1))
fox['logret'] = np.log(fox.Close) - np.log(fox.Close.shift(1))
spx['logret'] = np.log(spx.Close) - np.log(spx.Close.shift(1))
dis['logret'] = np.log(dis.Close) - np.log(dis.Close.shift(1))
cmcsa['logret'] = np.log(cmcsa.Close) - np.log(cmcsa.Close.shift(1))
para['logret'] = np.log(para.Close) - np.log(para.Close.shift(1))

# yes, Price not Close; different data source
tfcf['logret'] = np.log(tfcf.Price) - np.log(tfcf.Price.shift(1))

spx_upper = spx.logret.loc["2019-03-10":enddate]
spx_lower = spx.logret.loc[startdate:"2019-03-18"]

nflxbeta = get_beta(nflx.logret, spx.logret)
disbeta = get_beta(dis.logret, spx.logret)
foxbeta = get_beta(fox.logret, spx_upper)
tfcfbeta = get_beta(tfcf.logret, spx_lower)
cmcsabeta = get_beta(cmcsa.logret, spx.logret)
parabeta = get_beta(para.logret, spx.logret)

ax1 = plt.subplot()
sns.lineplot(data=dis, x="Date", y="Close", color='Orange', label="DIS, β=" + f'{disbeta:.2f}', ax=ax1)
sns.lineplot(data=tfcf, x="Date", y="Price", color="DarkBlue", label="TFCF, β=" + f'{tfcfbeta:.2f}', ax=ax1)
sns.lineplot(data=fox, x="Date", y="Close", color="LightBlue", label="FOX, β=" + f'{foxbeta:.2f}', ax=ax1)
sns.lineplot(data=para, x="Date", y="Close", color="Purple", label="PARA, β=" + f'{parabeta:.2f}', ax=ax1)
sns.lineplot(data=cmcsa, x="Date", y="Close", color="Green", label="CMCSA, β=" + f'{cmcsabeta:.2f}', ax=ax1)
sns.lineplot(data=nflx, x="Date", y="Close", color="Red", label="NFLX, β=" + f'{nflxbeta:.2f}', ax=ax1)

plt.title("Performance comparison of Disney and selected competitors")
ax1.set_ylabel("Weekly Closing Price")
ax1.tick_params(axis='x', rotation=30)
# plt.xticks(dis['Date'][::15])
# ax1.legend(['DIS'], loc="upper left")

plt.tight_layout()
plt.savefig('competitors.png', backend="cairo", format="png")
