import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import iexfinance
from iexfinance.stocks import Stock
from iexfinance.altdata import get_social_sentiment
from datetime import datetime
from iexfinance.stocks import get_historical_data

secret_key = 'sk_823a0d0e1269438498381d105780c56e'
public_key = 'pk_4d34c8d2197f40848419931605a6ca34'


def trial():
	aapl = Stock('AAPL', output_format='pandas', token=secret_key)
	print(aapl.get_balance_sheet())
	print('~~~~~~~~~~~~~~~~~~~~~')
	print(aapl.get_quote())
	print('~~~~~~~~~~~~~~~~~~~~~')
	# print(get_social_sentiment("AAPL", token=secret_key))
	tsla = Stock('TSLA', output_format='pandas', token=secret_key)
	print(tsla.get_price())
	batch = Stock(["TSLA", "AAPL"], token=secret_key)
	print(batch.get_price())
	print('~~~~~~~~~~~~~~~~~~~~~')
	start = datetime(2017, 1, 1)
	end = datetime(2018, 1, 1)
	df = get_historical_data("TSLA", start, end, output_format='pandas',token=secret_key)
	print(df.head())
	fig1, ax1 = plt.subplots()
	ax1.set_title("Open")
	ax1.plot(df['open'])
	fig2, ax2 = plt.subplots()
	ax2.set_title("Volume")
	ax2.plot(df['volume'])
	plt.show()
	pass


if __name__ == '__main__':
	trial()