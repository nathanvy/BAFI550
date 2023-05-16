import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("white")
sns.set_style("ticks")

tfcf = pd.read_csv("tfcf.csv", parse_dates=['Date'])
dis = yf.Ticker("DIS").history(start="2015-01-04", end="2022-12-31", interval="1wk")
fox = yf.Ticker("FOX").history(period="5y", interval="1wk")
# cmcsa = yf.Ticker("CMCSA").history(start="2015-01-04", end="2022-12-31", interval="1wk")

ax1 = plt.subplot()
sns.lineplot(data=dis, x="Date", y="Close", color='Orange', label='DIS', ax=ax1)
sns.lineplot(data=tfcf, x="Date", y="Price", color="DarkBlue", label="TFCF", ax=ax1)
sns.lineplot(data=fox, x="Date", y="Close", color="LightBlue", label="FOX", ax=ax1)
# sns.lineplot(data=cmcsa, x="Date", y="Close", color="Green", label="CMCSA", ax=ax1)

plt.title("Longitudinal comparison during M&A")
ax1.set_ylabel("Weekly Closing Price")
ax1.tick_params(axis='x', rotation=90)
# plt.xticks(dis['Date'][::15])
# ax1.legend(['DIS'], loc="upper left")

plt.tight_layout()
plt.savefig('longitudinal-comparison.png', backend="cairo", format="png")
