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

startdate = pd.to_datetime("2019-03-21 00:00:00-04:00")
enddate = pd.to_datetime("2019-12-31 00:00:00-04:00")

dis = yf.Ticker("DIS").history(start=startdate, end=enddate, interval="1wk")
fox = pd.read_csv("fox.csv", parse_dates=['Date'])
fox = fox.set_index(['Date'])
fox = fox.sort_index(ascending=True)
fox = fox[startdate:enddate]

ax1 = plt.subplot()
sns.lineplot(data=dis, x="Date", y="Close", color='Orange', label="DIS", ax=ax1)
sns.lineplot(data=fox, x="Date", y="Close", color='LightBlue', label="FOX", ax=ax1)

ax1.annotate("1", xy=(pd.to_datetime('2019-3-21'), 109),  xycoords='data',
             bbox=dict(boxstyle="round", fc="none", ec='none'),
             xytext=(10, -20), textcoords='offset points', ha='center',
             arrowprops=dict(arrowstyle="->", color='#333333'))

ax1.annotate("2", xy=(pd.to_datetime('2019-4-10'), 129),  xycoords='data',
             bbox=dict(boxstyle="round", fc="none", ec='none'),
             xytext=(10, -20), textcoords='offset points', ha='center',
             arrowprops=dict(arrowstyle="->", color='#333333'))

ax1.annotate("3", xy=(pd.to_datetime('2019-4-24'), 139),  xycoords='data',
             bbox=dict(boxstyle="round", fc="none", ec='none'),
             xytext=(20, 20), textcoords='offset points', ha='center',
             arrowprops=dict(arrowstyle="->", color='#333333'))

ax1.annotate("4", xy=(pd.to_datetime('2019-7-3'), 140),  xycoords='data',
             bbox=dict(boxstyle="round", fc="none", ec='none'),
             xytext=(0, -40), textcoords='offset points', ha='center',
             arrowprops=dict(arrowstyle="->", color='#333333'))

ax1.annotate("5", xy=(pd.to_datetime('2019-8-1'), 138),  xycoords='data',
             bbox=dict(boxstyle="round", fc="none", ec='none'),
             xytext=(-10, -20), textcoords='offset points', ha='center',
             arrowprops=dict(arrowstyle="->", color='#333333'))

ax1.annotate("6", xy=(pd.to_datetime('2019-8-7'), 136),  xycoords='data',
             bbox=dict(boxstyle="round", fc="none", ec='none'),
             xytext=(-10, -30), textcoords='offset points', ha='center',
             arrowprops=dict(arrowstyle="->", color='#333333'))

ax1.annotate("7", xy=(pd.to_datetime('2019-8-9'), 138),  xycoords='data',
             bbox=dict(boxstyle="round", fc="none", ec='none'),
             xytext=(20, 10), textcoords='offset points', ha='center',
             arrowprops=dict(arrowstyle="->", color='#333333'))

ax1.annotate("8", xy=(pd.to_datetime('2019-9-10'), 136),  xycoords='data',
             bbox=dict(boxstyle="round", fc="none", ec='none'),
             xytext=(-10, -30), textcoords='offset points', ha='center',
             arrowprops=dict(arrowstyle="->", color='#333333'))

plt.title("Stock Performance after M&A Completion")
ax1.set_ylabel("Weekly Closing Price")
ax1.tick_params(axis='x', rotation=30)
# plt.xticks(dis['Date'][::15])
ax1.legend(loc="center right")

plt.tight_layout()
plt.savefig('aftermath.png', backend="cairo", format="png")
