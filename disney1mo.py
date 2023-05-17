import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression

def get_beta(equity, market):
    """Takes two arguments that should be Pandas series and computes
    the beta of the equity series with respect to the market series"""
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

# first let's pull down the data we want
dis1mo = yf.Ticker("DIS").history(start="2023-04-12", end="2023-05-12", interval="1d")
spx1mo = yf.Ticker("^GSPC").history(start="2023-04-12", end="2023-05-12", interval="1d")

# now let's compute logarithmic returns
dis1mo['logret'] = np.log(dis1mo.Close) - np.log(dis1mo.Close.shift(1))
spx1mo['logret'] = np.log(spx1mo.Close) - np.log(spx1mo.Close.shift(1))

beta = get_beta(dis1mo['logret'], spx1mo['logret'])
r_squared = get_rsquared(spx1mo['logret'], dis1mo['logret'])

# ok now let's plot

ax1 = plt.subplot()
ax2 = ax1.twinx()

sns.lineplot(data=dis1mo, x="Date", y="Close", color='Orange', label='DIS', ax=ax1)
sns.lineplot(data=spx1mo, x="Date", y="Close", color='DarkBlue', label='SP500', ax=ax2)

ax1.annotate("β (1-month daily) = " + f'{beta:.2f}', xy=(pd.to_datetime('2023-04-17 00:00:00-04:00', format='%Y-%m-%d %H:%M'), 102),  xycoords='data',
             xytext=(-10, 0), textcoords='offset points',
             arrowprops=dict(arrowstyle="->",
                             connectionstyle="arc3,rad=-0.2"))

ax1.annotate("R2  = " + f'{r_squared:.2f}', xy=(pd.to_datetime('2023-04-17 00:00:00-04:00', format='%Y-%m-%d %H:%M'), 101),  xycoords='data',
             xytext=(30, 0), textcoords='offset points',
             arrowprops=dict(arrowstyle="->",
                             connectionstyle="arc3,rad=-0.2"))

ax1.legend(loc="upper right")
ax2.legend(loc="lower left")

plt.title("One-month comparison of DIS vs SP 500 Index")
ax1.set_ylabel("DIS\nDaily Closing Price")
ax2.set_ylabel("SP500\nDaily Closing Price")
ax1.tick_params(axis='x', rotation=30)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))

plt.tight_layout()
plt.savefig('one-month.png', backend="cairo", format="png")
