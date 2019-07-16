import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import iexfinance
from iexfinance.stocks import Stock
from iexfinance.altdata import get_social_sentiment
from datetime import datetime
from iexfinance.stocks import get_historical_data
from iexfinance.stocks import get_historical_intraday
from iexfinance.refdata import get_symbols
from iexfinance.altdata import get_ceo_compensation
from iexfinance.refdata import get_iex_symbols
from iexfinance.account import get_metadata

secret_key = 'sk_823a0d0e1269438498381d105780c56e'
public_key = 'pk_4d34c8d2197f40848419931605a6ca34'


def trial1():
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
	df = get_historical_data("TSLA", start, end, output_format='pandas', token=secret_key)
	print(df.head())
	fig1, ax1 = plt.subplots()
	ax1.set_title("Open")
	ax1.plot(df['open'])
	fig2, ax2 = plt.subplots()
	ax2.set_title("Volume")
	ax2.plot(df['volume'])
	print('~~~~~~~~~~~~~~~~~~~~~')
	date = datetime(2018, 11, 27)
	df = get_historical_intraday("AAPL", output_format='pandas', token=secret_key)
	print(df.head())
	print('~~~~~~~~~~~~~~~~~~~~~')
	df = aapl.get_cash_flow()
	print(df.head())
	df = aapl.get_estimates()
	print(df.head())
	df = aapl.get_price_target()
	print(df.head())
	# print(get_social_sentiment("AAPL", token=secret_key))

	# plt.show()
	pass


def trial2():
	df = get_symbols(output_format='pandas', token=secret_key)
	for index, row in df.iterrows():
		print(row['symbol'])


def trial3():
	plan = Stock('PLAN', output_format='pandas', token=secret_key)
	print(plan.get_quote())
	print('~~~~~~~~~~~~~~~~~~~~~')
	print(get_ceo_compensation("PLAN", token=secret_key))


def trial4():
	df = get_iex_symbols(output_format='pandas', token=secret_key)
	for index, row in df.iterrows():
		print(row['symbol'])


if __name__ == '__main__':
	print(get_metadata(token=secret_key))
	# trial1()
	# trial2()
	# trial3()
	# trial4()