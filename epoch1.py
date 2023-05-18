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
enddate = "2018-04-30"

dis = yf.Ticker("DIS").history(start=startdate, end=enddate, interval="1wk")
cmcsa = yf.Ticker("CMCSA").history(start=startdate, end=enddate, interval="1wk")
tfcf = pd.read_csv("tfcf.csv", parse_dates=['Date'])
tfcf = tfcf.set_index(['Date'])   # convert to datetime64
tfcf = tfcf.sort_index(ascending=True)
tfcf = tfcf[startdate:enddate]

ax1 = plt.subplot()
sns.lineplot(data=dis, x="Date", y="Close", color='Orange', label="DIS", ax=ax1)
sns.lineplot(data=cmcsa, x="Date", y="Close", color='Green', label="CMCSA", ax=ax1)
sns.lineplot(data=tfcf, x="Date", y="Price", color='DarkBlue', label="TFCF", ax=ax1)

ax1.annotate("1", xy=(pd.to_datetime('2017-11-6'), 27.5),  xycoords='data',
             bbox=dict(boxstyle="round", fc="none", ec='none'),
             xytext=(-20, 2), textcoords='offset points', ha='center',
             arrowprops=dict(arrowstyle="->", color='#333333'))

ax1.annotate("2", xy=(pd.to_datetime('2017-11-16'), 32.5),  xycoords='data',
             bbox=dict(boxstyle="round", fc="none", ec='none'),
             xytext=(-10, 20), textcoords='offset points', ha='center',
             arrowprops=dict(arrowstyle="->", color='#333333'))

ax1.annotate("3", xy=(pd.to_datetime('2017-11-28'), 31.5),  xycoords='data',
             bbox=dict(boxstyle="round", fc="none", ec='none'),
             xytext=(10, -20), textcoords='offset points', ha='center',
             arrowprops=dict(arrowstyle="->", color='#333333'))

ax1.annotate("4", xy=(pd.to_datetime('2017-12-5'), 33.5),  xycoords='data',
             bbox=dict(boxstyle="round", fc="none", ec='none'),
             xytext=(20, -10), textcoords='offset points', ha='center',
             arrowprops=dict(arrowstyle="->", color='#333333'))

ax1.annotate("5", xy=(pd.to_datetime('2017-12-11'), 34),  xycoords='data',
             bbox=dict(boxstyle="round", fc="none", ec='none'),
             xytext=(-10, 20), textcoords='offset points', ha='center',
             arrowprops=dict(arrowstyle="->", color='#333333'))

ax1.annotate("6", xy=(pd.to_datetime('2017-12-14'), 106),  xycoords='data',
             bbox=dict(boxstyle="round", fc="none", ec='none'),
             xytext=(10, -30), textcoords='offset points', ha='center',
             arrowprops=dict(arrowstyle="->", color='#333333'))

ax1.annotate("7", xy=(pd.to_datetime('2018-3-4'), 100),  xycoords='data',
             bbox=dict(boxstyle="round", fc="none", ec='none'),
             xytext=(-10, -30), textcoords='offset points', ha='center',
             arrowprops=dict(arrowstyle="->", color='#333333'))

ax1.annotate("8", xy=(pd.to_datetime('2018-4-12'), 98),  xycoords='data',
             bbox=dict(boxstyle="round", fc="none", ec='none'),
             xytext=(-10, 30), textcoords='offset points', ha='center',
             arrowprops=dict(arrowstyle="->", color='#333333'))

plt.title("Comcast and Disney Jockey to Acquire 21st Century Fox")
ax1.set_ylabel("Weekly Closing Price")
ax1.tick_params(axis='x', rotation=30)
# plt.xticks(dis['Date'][::15])
# ax1.legend(['DIS'], loc="upper left")

plt.tight_layout()
plt.savefig('early-developments.png', backend="cairo", format="png")
