import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression

sns.set_style("white")
sns.set_style("ticks")

# first let's pull down the data we want
dis1mo = yf.Ticker("DIS").history(period="1mo", interval="1d")
spx1mo = yf.Ticker("^GSPC").history(period="1mo", interval="1d")

# now let's compute logarithmic returns
dis1mo['logret'] = np.log(dis1mo.Close) - np.log(dis1mo.Close.shift(1))
spx1mo['logret'] = np.log(spx1mo.Close) - np.log(spx1mo.Close.shift(1))

# to calcluate Beta we'll need the covariance and variance of the returns,
# and for that we need to get them as Series objects per the Pandas API
dis_logret = pd.Series(dis1mo['logret'])
spx_logret = pd.Series(spx1mo['logret'])
covariance = dis_logret.cov(spx_logret)
variance = spx_logret.var()
beta = covariance / variance

dis_logret.fillna(0, inplace=True)
spx_logret.fillna(0, inplace=True)
# now for the R-squared value.  First initialize the linear regression model,
# then we'll set our predictor and response vars and run the model
model = LinearRegression()
p = spx_logret.values.reshape(-1, 1)
r = dis_logret.values.reshape(-1, 1)
model.fit(p, r)
r_squared = model.score(p, r)

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


plt.title("One-month comparison of DIS vs SP 500 Index")
ax1.set_ylabel("DIS\nWeekly Closing Price")
ax2.set_ylabel("SP500\nWeekly Closing Price")
ax1.tick_params(axis='x', rotation=30)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))

plt.tight_layout()
plt.savefig('one-month.png', backend="cairo", format="png")
