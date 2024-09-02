# import modin.pandas as pd
# import modin.config as cfg

# # Configure Modin ONCE
# os.environ["MODIN_ENGINE"] = "dask"  
# cfg.Engine.put("dask") 

# import modin.pandas as pd
# import modin.config as cfg

# # Configure Modin ONCE
# os.environ["MODIN_ENGINE"] = "ray"  
# cfg.Engine.put("ray") 
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Pick support for PolyCollection is missing")
### Tkinter ###
import tkinter as tk
from tkinter import ttk
from tkinter import ttk, messagebox, filedialog, Listbox, Label, Message
from tkcalendar import Calendar, DateEntry

#### Matplotlib mplfinance
import matplotlib as matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.lines import Line2D
import matplotlib.patches as patches
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import matplotlib.dates as mdates
import mplfinance as mpf
import mplcursors
from matplotlib.widgets import Cursor


### database, data processing  and historical data
import yfinance as yf
import psycopg2
import pandas as pd 

### Technical Analysis
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.trend import CCIIndicator
from ta.volatility import BollingerBands
from tradingview_ta import TA_Handler, Interval, Exchange

import talib
import talipp ## try to use this 
# from finta import TA

#### Standard Libraries
import numpy as np
from datetime import datetime, timedelta
import threading
import asyncio
import json
import redis
import pygame
import random
import html
import ctypes

import subprocess
import os

import sam_functions
# from detectCSpatterns import get_filtered_patterns


#################################################################


###################### py game audio player ############################################
# Initialize the mixer pygame mixer to play audio on my site.
pygame.mixer.init()

def play_audio():
    # Load the MP3 file
    pygame.mixer.music.load("audio/tradefearless.mp3")
    
    # Play the audio
    pygame.mixer.music.play()
    
################################################################

# Database connection
conn = psycopg2.connect(
    dbname="stockdata",
    user="postgres",
    password="Subhan$007",
    host="localhost",
    port="5432"
)
cur = conn.cursor()


# Create tables if they don't exist
cur.execute("""
CREATE TABLE IF NOT EXISTS tickers (
    symbol VARCHAR(20) PRIMARY KEY
);
""")
cur.execute("""
CREATE TABLE IF NOT EXISTS daily_prices (
    symbol VARCHAR(20),
    date DATE,
    open FLOAT,
    high FLOAT,
    low FLOAT,
    close FLOAT,
    volume BIGINT,
    PRIMARY KEY (symbol, date),
    FOREIGN KEY (symbol) REFERENCES tickers(symbol)
);
""")

# Add column to daily_prices table if it doesn't exist
cur.execute("""
ALTER TABLE daily_prices
ADD COLUMN  IF NOT EXISTS rsi3 FLOAT,
ADD COLUMN  IF NOT EXISTS rsi14 FLOAT,
ADD COLUMN  IF NOT EXISTS stochrsi14 FLOAT,
ADD COLUMN  IF NOT EXISTS stochrsi14_d FLOAT,
ADD COLUMN  IF NOT EXISTS stochrsi14_k FLOAT,
ADD COLUMN  IF NOT EXISTS bb_percent_b FLOAT,
ADD COLUMN  IF NOT EXISTS cci3 FLOAT,
ADD COLUMN  IF NOT EXISTS cci12 FLOAT,
ADD COLUMN  IF NOT EXISTS ma7 FLOAT,
ADD COLUMN  IF NOT EXISTS ma10 FLOAT,
ADD COLUMN  IF NOT EXISTS ma21 FLOAT,
ADD COLUMN  IF NOT EXISTS ma32 FLOAT,
ADD COLUMN  IF NOT EXISTS ma43 FLOAT,
ADD COLUMN  IF NOT EXISTS ma54 FLOAT,
ADD COLUMN  IF NOT EXISTS ma63 FLOAT,
ADD COLUMN  IF NOT EXISTS ma189 FLOAT,
ADD COLUMN  IF NOT EXISTS mamix14 FLOAT,
ADD COLUMN  IF NOT EXISTS mamix42 FLOAT,
ADD COLUMN  IF NOT EXISTS vwma25 FLOAT,
ADD COLUMN  IF NOT EXISTS volsma5 FLOAT,
ADD COLUMN  IF NOT EXISTS volsma20 FLOAT,
ADD COLUMN  IF NOT EXISTS traderule1 BOOLEAN,
ADD COLUMN  IF NOT EXISTS traderule2 BOOLEAN,
ADD COLUMN  IF NOT EXISTS traderule3 BOOLEAN,
ADD COLUMN  IF NOT EXISTS traderule4 BOOLEAN,
ADD COLUMN  IF NOT EXISTS traderule5 BOOLEAN,
ADD COLUMN  IF NOT EXISTS traderule6 BOOLEAN,
ADD COLUMN  IF NOT EXISTS traderule7 BOOLEAN,
ADD COLUMN  IF NOT EXISTS traderule8 BOOLEAN,
ADD COLUMN  IF NOT EXISTS traderule9 BOOLEAN,
ADD COLUMN  IF NOT EXISTS traderule10 BOOLEAN,
ADD COLUMN  IF NOT EXISTS notes TEXT; 

""")

conn.commit()


### CReating Custom Chart Style

# Define the style
    
samie_style = {
        "base_mpl_style": "nightclouds",
        "marketcolors": {
            "candle": {"up": "#8ad5ef", "down": "#f72e05"},  
            "edge": {"up": "#8ad5ef", "down": "#f72e05"},  
            "wick": {"up": "#247252", "down": "#ef4f60"},  
            "ohlc": {"up": "green", "down": "red"},
            "volume": {"up": "#067887", "down": "#ff6060"},  
            "vcedge": {"up": "#000000", "down": "#000000"},  
            "vcdopcod": False,
            "alpha": 1,
        },
        "mavcolors": ["#ad7739", "#a63ab2", "#62b8ba"],
        "facecolor": "#1b1f24",
        "gridcolor": "#2c2e31",
        "gridstyle": "--",
        "y_on_right": True,
        "rc": {
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.edgecolor": "#474d56",
            "axes.titlecolor": "red",
            "figure.facecolor": "#161a1e",
            "figure.titlesize": "x-large",
            "figure.titleweight": "semibold",
        },
        "base_mpf_style": "nightclouds",
    }


# samie_style_obj = mpf.make_mpf_style(
#         base_mpl_style="seaborn",
#         rc={
#             "axes.facecolor": "#202020",
#             "axes.grid": False,
#             "xtick.color": "w",
#             "ytick.color": "w",
#             "grid.color": "#31363F",
#             "text.color": "w",
#             "figure.facecolor": "#202020",
#         },
#         style_dict={
#             "candlestick": {
#                 "colorup": "#00ff00",
#                 "colordown": "#ff0000",
#                 "linewidth": 1,
#                 "edgecolor": "w",
#             },
#             "volume": {
#                 "colorup": "#00ff00",
#                 "colordown": "#ff0000",
#             },
#         },
#     )

    # samie_style = mpf.make_mpf_style(base_mpf_style='charles', 
    # marketcolors=mpf.make_marketcolors(up='green', down='black', inherit=True))

samie_style_obj = mpf.make_mpf_style(**samie_style)
    

# Create a Matplotlib figure and canvas
fig = Figure(figsize=(14.5, 8), dpi=100)
fig.set_edgecolor('blue')
fig.set_facecolor('#131722')
fig.text(0.99, 0.95, 'MashaAllah', fontsize=10, ha='right', color='green')
    

    
def fetch_data_from_db(symbol, selected_period):
        # Convert selected_period to days
        period_days = convert_period_to_days(selected_period)
        # Calculate the start date for the data
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=period_days)

        try:
        # Check if data exists for the entire period
            cur.execute("SELECT COUNT(*) FROM daily_prices WHERE symbol = %s AND date >= %s AND date <= %s", (symbol, start_date, end_date))
            data_count = cur.fetchone()[0]
            expected_data_points = period_days + 1  # Include the end date

            if data_count == expected_data_points:
                print(f"All data available for {symbol} for the past {selected_period}")
                return
        except Exception as e:
            print(f"An error occured: {e}")
            conn.rollback()
    # Fetch the data from the database

def convert_period_to_days(selected_period):
    if selected_period == '5d':
        return 5
    elif selected_period == '1mo':
        return 30
    elif selected_period == '3mo':
        return 90
    elif selected_period == '6mo':
        return 180
    elif selected_period == '9mo':
        return 270
    elif selected_period == '1y':
        return 365
    elif selected_period == '2y':
        return 730
    elif selected_period == '3y':
        return 1095
    elif selected_period == '4y':
        return 1460
    elif selected_period == '5y':
        return 1825
    elif selected_period == '6y':
        return 2190
    elif selected_period == '7y':
        return 2555
    elif selected_period == '8y':
        return 2920
    elif selected_period == '9y':
        return 3285
    elif selected_period == '10y':
        return 3650
    else:
        raise ValueError("Invalid period format")
    
def volume_formatter(x, pos):
    return f'{x:,.0f}'


def price_formatter(x, pos):
    return f'{x:,.2f}'

data_needed_days = 1460

async def calculate_and_store_rsi_cci(symbol):
    """Calculates RSI3 & RSI14 for the given symbol and stores it in the database."""

    # Fetch the last calculated RSI date
    cur.execute("SELECT MAX(date) FROM daily_prices WHERE symbol = %s AND rsi3 IS NOT NULL", (symbol,))
    # last_rsi_date = cur.fetchone()[0]
    end_date = datetime.now().date()
    last_rsi_date = end_date - timedelta(days=data_needed_days)
    

    # Fetch only new data
    cur.execute("SELECT date, close, high, low FROM daily_prices WHERE symbol = %s AND date > %s ORDER BY date", (symbol, last_rsi_date or '1900-01-01'))
    rows = cur.fetchall()

    if len(rows) < 3:
        print(f"Not enough new data to calculate RSI3, BBpB and CCIs for {symbol}")
        return

    df = pd.DataFrame(rows, columns=['date', 'close', 'high', 'low'])
    
    # Calculate RSI and CCI
    rsi_indicator = RSIIndicator(close=df['close'], window=3)
    rsi_indicator14 = RSIIndicator(close=df['close'], window=14)
    df['rsi3'] = rsi_indicator.rsi()
    df['rsi14'] = rsi_indicator14.rsi()
    
    ### CCI
    cci = CCIIndicator(high=df['high'], low=df['low'], close=df['close'], window=3, constant=0.015)
    df['cci3'] = cci.cci()
    
    cci12 = CCIIndicator(high=df['high'], low=df['low'], close=df['close'], window=12, constant=0.015)
    df['cci12'] = cci12.cci()
    
    #### stochrsi
    
    # Assuming df is your DataFrame
    stochrsi14 = StochRSIIndicator(close=df['close'], window=14, smooth1=3, smooth2=3)
    df['stochrsi14'] = stochrsi14.stochrsi()
    df['stochrsi14_k'] = stochrsi14.stochrsi_k()
    df['stochrsi14_d'] = stochrsi14.stochrsi_d()

    #### bollingerband %b
    
    # Calculate Bollinger Bands
    bb_indicator = BollingerBands(close=df['close'], window=20, window_dev=2)

    # Add Bollinger Band %B to the dataframe
    df['bb_percent_b'] = bb_indicator.bollinger_pband()
    # Prepare batch updates


    updates = [(row['rsi3'], row['rsi14'], row['cci3'], row['cci12'], row['stochrsi14'], row['stochrsi14_k'], row['stochrsi14_d'], row['bb_percent_b'], symbol, row['date']) for _, row in df.iterrows() if not pd.isna(row['rsi3']) and not pd.isna(row['rsi14']) and not pd.isna(row['cci3']) and not pd.isna(row['cci12']) and not pd.isna(row['stochrsi14']) and not pd.isna(row['stochrsi14_k']) and not pd.isna(row['stochrsi14_d']) ]

    # Perform bulk update
    cur.executemany(
        "UPDATE daily_prices SET rsi3 = %s, rsi14 = %s, cci3 = %s, cci12 = %s, stochrsi14 = %s, stochrsi14_k = %s, stochrsi14_d = %s, bb_percent_b = %s  WHERE symbol = %s AND date = %s",
        updates
    )
    conn.commit()

    print(f"RSI3,rsi14, CCI3 & 12, StochRSI14, BBpB calculated and stored for {symbol}")
    

async def calculate_and_store_ma(symbol):
    # Initialize Redis connection
    redis_client =  redis.Redis(host='localhost', port=6379, db=0)

    # Get last update timestamp
    last_update = await get_last_update(symbol)

    # Fetch only new data
    cur.execute("""
        SELECT date, close, volume, ma7, ma10, ma21, ma32, ma43, ma54, ma63, ma189, mamix14, mamix42, vwma25, volsma5, volsma20
        FROM daily_prices
        WHERE symbol = %s AND (date > %s OR %s IS NULL)
        ORDER BY date
    """, (symbol, last_update, last_update))
    rows = cur.fetchall()

    if not rows:
        print(f"No new data to process for {symbol}")
        return

    dates = [row[0] for row in rows]
    df = pd.DataFrame({
        'close': [row[1] for row in rows],
        'volume': [row[2] for row in rows]
    }, index=dates)
    df = df.fillna(0) 
    # Calculate moving averages
    for ma in [7, 10, 21, 32, 43, 54, 63, 189]:
        df[f'ma{ma}'] = df['close'].rolling(window=ma).mean()
    
    df['mamix14'] = ((df['ma7'] + df['ma21']) / 2).rolling(window=2).mean()
    df['mamix42'] = ((df['ma21'] + df['ma63']) / 2).rolling(window=2).mean()
    df['CloseVolume'] = df['close'] * df['volume']
    df['vwma25'] = df['CloseVolume'].rolling(window=25).sum() / df['volume'].rolling(window=25).sum()
    df['volsma5'] = (df['volume']).rolling(window=5).mean()
    df['volsma20'] = (df['volume']).rolling(window=20).mean()
    
    # Prepare batch updates
    updates = []
    for date, row in df.iterrows():
        updates.append((
            float(row['ma7']), 
            float(row['ma10']), 
            float(row['ma21']), 
            float(row['ma32']), 
            float(row['ma43']), 
            float(row['ma54']), 
            float(row['ma63']), 
            float(row['ma189']),
            float(row['mamix14']), 
            float(row['mamix42']), 
            float(row['vwma25']),
            float(row['volsma5']),
            float(row['volsma20']),
            symbol, date
        ))
        # Cache the data
        redis_client.set(f"{symbol}_{date}", json.dumps(row.to_dict()))


    # Perform bulk update
    await bulk_update(updates)

    print(f"Moving Averages are updated for {symbol}")

async def get_last_update(symbol):
    # cur.execute("SELECT MAX(date) FROM daily_prices WHERE symbol = %s", (symbol,))
    cur.execute("SELECT MIN(date) FROM daily_prices WHERE symbol = %s", (symbol,))
    last_date = cur.fetchone()[0]
    return last_date if last_date else datetime(1970, 1, 1).date()

async def bulk_update(updates):
    cur.executemany("""
        UPDATE daily_prices
        SET ma7 = COALESCE(%s, ma7), 
            ma10 = COALESCE(%s, ma10), 
            ma21 = COALESCE(%s, ma21), 
            ma32 = COALESCE(%s, ma32), 
            ma43 = COALESCE(%s, ma43), 
            ma54 = COALESCE(%s, ma54), 
            ma63 = COALESCE(%s, ma63), 
            ma189 = COALESCE(%s, ma189), 
            mamix14 = COALESCE(%s, mamix14), 
            mamix42 = COALESCE(%s, mamix42), 
            vwma25 = COALESCE(%s, vwma25),
            volsma5 = COALESCE(%s, volsma5),
            volsma20 = COALESCE(%s, volsma20)
        WHERE symbol = %s AND date = %s
    """, updates)
    conn.commit()

# Run the function asynchronously
# asyncio.run(calculate_and_store_ma(symbol))

#############################################???????????????????????????????#############################

def plot_resampled_chart(symbol, selected_period, resample_freq='ME'):
    # Convert period to a format suitable for SQL query
    period_map = {
        '1mo': '1 month', '3mo': '3 months', '6mo': '6 months', '9mo': '9 months',
        '1y': '1 year', '2y': '2 years', '3y': '3 years', '4y': '4 years',
        '5y': '5 years', '6y': '6 years', '7y': '7 years', '8y': '8 years',
        '9y': '9 years', '10y': '10 years', 'max': '100 years'
    }
    sql_period = period_map.get(selected_period)

    # Fetch data from database
    cur.execute("SELECT date, open, high, low, close, volume, rsi3, ma7, ma21, ma63, ma189, mamix14, mamix42, vwma25, traderule1, traderule2, traderule3, traderule4, traderule5, traderule6, traderule7, traderule8, volsma5, volsma20, cci3, cci12, bb_percent_b, rsi14, traderule9, traderule10  FROM daily_prices WHERE symbol = %s AND date >= NOW() - INTERVAL %s ORDER BY date", (symbol, sql_period))
    
    rows = cur.fetchall()
    if not rows:
        fetch_data_from_db(symbol, selected_period)
        cur.execute("SELECT date, open, high, low, close, volume, rsi3, ma7, ma21, ma63, ma189, mamix14, mamix42, vwma25, traderule1, traderule2, traderule3, traderule4, traderule5, traderule6, traderule7, traderule8, volsma5, volsma20, cci3, cci12, bb_percent_b, rsi14, traderule9, traderule10  FROM daily_prices WHERE symbol = %s AND date >= NOW() - INTERVAL %s ORDER BY date", (symbol, sql_period))
        rows = cur.fetchall()
    
    
 
    data = {
        'Date': [row[0] for row in rows],
        'Open': [round(row[1], 2) for row in rows],
        'High': [round(row[2], 2) for row in rows],
        'Low': [round(row[3], 2) for row in rows],
        'Close': [round(row[4], 2) for row in rows],
        'Volume': [row[5] for row in rows],
        'rsi3': [row[6] for row in rows],
        'ma7': [row[7] for row in rows],
        'ma21': [row[8] for row in rows],
        'ma63': [row[9] for row in rows],
        'ma189': [row[10] for row in rows],
        'mamix14':[row[11] for row in rows],
        'mamix42':[row[12] for row in rows],
        'vwma25':[row[13] for row in rows],
        'traderule1': [row[14] for row in rows],
        'traderule2':[row[15] for row in rows],
        'traderule3':[row[16] for row in rows],
        'traderule4':[row[17] for row in rows],
        'traderule5':[row[18] for row in rows],
        'traderule6':[row[19] for row in rows],
        'traderule7':[row[20] for row in rows],
        'traderule8':[row[21] for row in rows],
        'volsma5':[row[22] for row in rows],
        'volsma20':[row[23] for row in rows],
        'cci3':[row[24] for row in rows],
        'cci12':[row[25] for row in rows],
        'bb_percent_b':[row[26] for row in rows],
        'rsi14':[row[27] for row in rows],
        'traderule9':[row[28] for row in rows],
        'traderule10':[row[29] for row in rows]

        
        
    }
    
    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)
    df.index = df.index.astype(str)
    df.index = pd.to_datetime(df.index)  # Ensure the index is a DatetimeIndex
    
    
    
    # Define resampling rules
    resample_dict = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',
        'rsi3': 'last',
        'ma7': 'last',
        'ma21': 'last',
        'ma63': 'last',
        'ma189': 'last',
        'mamix14': 'last',
        'mamix42': 'last',
        'vwma25': 'last',
        'traderule1': 'last',
        'traderule2': 'last',
        'traderule3': 'last',
        'traderule4': 'last',
        'traderule5': 'last',
        'traderule6': 'last',
        'traderule7': 'last',
        'traderule8': 'last',
        'volsma5': 'last',
        'volsma20': 'last',
        'cci3': 'last',
        'cci12': 'last',
        'bb_percent_b': 'last',
        'rsi14': 'last',
        'traderule9': 'last',
        'traderule10': 'last'
    }

    
    ##### Tradingview Technical Analysis Recomendation.
    # clean_symbol = watchlist.get(watchlist.curselection()).replace(".NS", "")
    # print(clean_symbol)
    # ta_symbol = TA_Handler( symbol=clean_symbol, screener="india", exchange="NSE", interval=Interval.INTERVAL_1_DAY) 
    # print(ta_symbol.get_analysis().summary)['RECOMMENDATION']
    # recommend = ta_symbol.get_analysis().summary
    # ta_symbol.get_analysis().summary# Resample data only if necessary
    if resample_freq != 'D':
        resampled_df = df.resample(resample_freq).agg(resample_dict)
    else:
        resampled_df = df  # Use the original daily data

    if not resampled_df.empty:
        latest_data = resampled_df.iloc[-1]
        prev_close = resampled_df.iloc[-2]['Close'] if len(resampled_df) > 1 else 0
        percent_change = ((latest_data['Close'] - prev_close) / prev_close) * 100 if prev_close != 0 else 0
        latest_info = (
            f"TF:  {resample_freq}, "
            f"Date: {latest_data.name.strftime('%b %d %Y')}, "
            f"Open: {latest_data['Open']:.2f}, "
            f"High: {latest_data['High']:.2f}, "
            f"Low: {latest_data['Low']:.2f}, "
            f"Close: {latest_data['Close']:.2f}, "
            f"Chg: {percent_change:.2f}% , "
            # f"Recommendation: {recommend['RECOMMENDATION']}"
        )
    else:
        latest_info = "No data available"

    #these lines are highlighting the Low values in your DataFrame where each respective trading rule is met, 
    # and setting the value to NaN where the rule is not met.
    ### to plot on low of the candle .. we are collecting lows
    resampled_df['traderule1_highlight'] = np.where(resampled_df['traderule1'], resampled_df['Low'] * 0.999, np.nan)
    resampled_df['traderule2_highlight'] = np.where(resampled_df['traderule2'], resampled_df['Low'] * 0.999, np.nan)
    resampled_df['traderule3_highlight'] = np.where(resampled_df['traderule3'], resampled_df['Low'] * 0.999, np.nan)
    resampled_df['traderule4_highlight'] = np.where(resampled_df['traderule4'], resampled_df['Low'] * 0.999, np.nan)
    resampled_df['traderule5_highlight'] = np.where(resampled_df['traderule5'], resampled_df['Low'] * 0.995, np.nan)
    resampled_df['traderule6_highlight'] = np.where(resampled_df['traderule6'], resampled_df['Low'] * 0.990, np.nan)
    resampled_df['traderule7_highlight'] = np.where(resampled_df['traderule7'], resampled_df['Low'] * 0.985, np.nan)
    resampled_df['traderule8_highlight'] = np.where(resampled_df['traderule8'], resampled_df['Low'] * 0.980, np.nan)
    resampled_df['traderule9_highlight'] = np.where(resampled_df['traderule9'], resampled_df['Low'] * 0.975, np.nan)
    resampled_df['traderule10_highlight'] = np.where(resampled_df['traderule10'], resampled_df['Low'] * 0.970, np.nan)
    
    # resampled_df['traderule3_highlight'] = np.where(resampled_df['traderule3'], resampled_df['Low'] * 1.05, np.nan) # to add 5%
    # resampled_df['traderule3_highlight'] = np.where(resampled_df['traderule3'], resampled_df['Low'] * 0.95, np.nan) # to minus 5%
    
    resampled_df['traderule3_high'] = np.where(resampled_df['traderule3'], resampled_df['High'], np.nan) # to plot a line taking high
    resampled_df['traderule3_low'] = np.where(resampled_df['traderule3'], resampled_df['Low'], np.nan) # to plot a line taking high
    resampled_df['traderule4_high'] = np.where(resampled_df['traderule4'], resampled_df['High'], np.nan) # to plot a line taking high
    resampled_df['traderule4_low'] = np.where(resampled_df['traderule4'], resampled_df['Low'], np.nan) # to plot a line taking high
    resampled_df['traderule5_high'] = np.where(resampled_df['traderule5'], resampled_df['High'], np.nan) # to plot a line taking high
    resampled_df['traderule5_low'] = np.where(resampled_df['traderule5'], resampled_df['Low'], np.nan) # to plot a line taking high
    resampled_df['traderule7_high'] = np.where(resampled_df['traderule7'], resampled_df['High'], np.nan) # to plot a line taking high
    resampled_df['traderule7_low'] = np.where(resampled_df['traderule7'], resampled_df['Low'], np.nan) # to plot a line taking high
    resampled_df['traderule8_high'] = np.where(resampled_df['traderule8'], resampled_df['Low'] + (resampled_df['High'] - resampled_df['Low']) * 0.3, np.nan) # to plot a line taking a value below hl2 and above 30% of candle.
    resampled_df['traderule8_low'] = np.where(resampled_df['traderule8'], resampled_df['Low'], np.nan) # to plot a line taking high
    resampled_df['traderule9_high'] = np.where(resampled_df['traderule9'], resampled_df['Low'] + (resampled_df['High'] - resampled_df['Low']) * 0.3, np.nan) # to plot a line taking a value below hl2 and above 30% of candle.
    resampled_df['traderule9_low'] = np.where(resampled_df['traderule9'], resampled_df['Low'], np.nan) # to plot a line taking high
    resampled_df['traderule10_high'] = np.where(resampled_df['traderule10'], resampled_df['High'], np.nan) # to plot a line taking high
    resampled_df['traderule10_low'] = np.where(resampled_df['traderule10'], resampled_df['Low'], np.nan) # to plot a line taking high
    
    
    traderule3_high = resampled_df['traderule3_high'].dropna()
    traderule3_low = resampled_df['traderule3_low'].dropna()
    traderule4_high = resampled_df['traderule4_high'].dropna()
    traderule4_low = resampled_df['traderule4_low'].dropna()
    traderule5_high = resampled_df['traderule5_high'].dropna()
    traderule5_low = resampled_df['traderule5_low'].dropna()
    traderule7_high = resampled_df['traderule7_high'].dropna()
    traderule7_low = resampled_df['traderule7_low'].dropna()
    
    ## Traderule 8 is high probable to taking these values and ploting on chart.
    traderule8_high_values = resampled_df['traderule8_high'].dropna()
    traderule8_low_values = resampled_df['traderule8_low'].dropna()
    
    # I am collecting the high and low values of traderules only if more than one rule is met. To plot lines below.
    
    # To complare all traderules taking them in variable. Add the traderules as you add new ones
    traderule_columns = resampled_df[['traderule1', 'traderule2', 'traderule3', 'traderule4', 'traderule5', 'traderule6', 'traderule7', 'traderule8', 'traderule9', 'traderule10']]

    # Use the variable in your np.where condition
    resampled_df['highlight2_high'] = np.where((traderule_columns.sum(axis=1) > 2) & (traderule_columns.sum(axis=1) < 4), resampled_df['High'], np.nan)
    resampled_df['highlight2_low'] = np.where((traderule_columns.sum(axis=1) > 2) & (traderule_columns.sum(axis=1) < 4), resampled_df['Low'], np.nan)
    
    highlight2_high_values = resampled_df['highlight2_high'].dropna()
    highlight2_low_values = resampled_df['highlight2_low'].dropna()
    
    resampled_df['highlight3_high'] = np.where((traderule_columns.sum(axis=1) >= 4) & (traderule_columns.sum(axis=1) < 6), resampled_df['High'], np.nan)
    resampled_df['highlight3_low'] = np.where((traderule_columns.sum(axis=1) >= 4) & (traderule_columns.sum(axis=1) < 6), resampled_df['Low'], np.nan)
    
    
    highlight3_high_values = resampled_df['highlight3_high'].dropna()
    highlight3_low_values = resampled_df['highlight3_low'].dropna()
    
    resampled_df['highlight5_high'] = np.where((traderule_columns.sum(axis=1) >= 6) & (traderule_columns.sum(axis=1) < 10), resampled_df['High'], np.nan)
    resampled_df['highlight5_low'] = np.where((traderule_columns.sum(axis=1) >= 6) & (traderule_columns.sum(axis=1) < 10), resampled_df['Low'], np.nan)
    
    highlight5_high_values = resampled_df['highlight5_high'].dropna()
    highlight5_low_values = resampled_df['highlight5_low'].dropna()
    
    # if not traderule3_high.empty:
    #     addplot = []
    #     for value in traderule3_high:
    #         addplot.append(mpf.make_addplot([value]*len(resampled_df), ax=ax1, color='#7572f1', linestyle='--'))
    
    fig.clear()
    
    # Adjust the layout to stretch the chart
    fig.tight_layout()
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    # Create the subplots
    ax1 = fig.add_subplot(4, 1, 1)
    ax2 = fig.add_subplot(4, 1, 2, sharex=ax1)
    ax3 = fig.add_subplot(4, 1, 3, sharex=ax1)
    ax4 = fig.add_subplot(4, 1, 4, sharex=ax1)
    
    AXS_BG_Color = '#000000ff'
    
    ax1.set_facecolor(AXS_BG_Color) 
    ax2.set_facecolor(AXS_BG_Color)
    ax3.set_facecolor(AXS_BG_Color)
    ax4.set_facecolor(AXS_BG_Color)
    
    
    addplot = [
        # mpf.make_addplot(resampled_df['ma7'], ax=ax1, color='red'),
        # mpf.make_addplot(resampled_df['ma21'], ax=ax1, color='black'),
        # mpf.make_addplot(resampled_df['ma63'], ax=ax1, color='green'),
        mpf.make_addplot(resampled_df['traderule1_highlight'], ax=ax1, type='scatter', markersize=100, marker='s', color='#bcf5bc'), # working code
        mpf.make_addplot(resampled_df['traderule2_highlight'], ax=ax1,  type='scatter', markersize=100, marker='^', color='#1e90ff'), # working code
        mpf.make_addplot(resampled_df['traderule3_highlight'], ax=ax1,  type='scatter', markersize=100, marker='s', color='#7572f1'), # working code
        mpf.make_addplot(resampled_df['traderule4_highlight'], ax=ax1, type='scatter', markersize=100, marker='d', color='#ff00ff'),  # New color for traderule4
        mpf.make_addplot(resampled_df['traderule5_highlight'], ax=ax1, type='scatter', markersize=100, marker='^', color='#067887'),  # New color for traderule5
        mpf.make_addplot(resampled_df['traderule6_highlight'], ax=ax1, type='scatter', markersize=100, marker='^', color='#007117'),  # New color for traderule6
        mpf.make_addplot(resampled_df['traderule7_highlight'], ax=ax1, type='scatter', markersize=100, marker='*', color='#fff000'),  # New color for traderule7
        mpf.make_addplot(resampled_df['traderule8_highlight'], ax=ax1, type='scatter', markersize=100, marker='*', color='#0fff00'),  # New color for traderule8
        mpf.make_addplot(resampled_df['traderule9_highlight'], ax=ax1, type='scatter', markersize=50, marker='x', color='#000fff'),  # New color for traderule9
        mpf.make_addplot(resampled_df['traderule10_highlight'], ax=ax1, type='scatter', markersize=100, marker='1', color='#00fff0'),  # New color for traderule10
        
        # mpf.make_addplot([horizontal_line]*len(resampled_df), ax=ax1, color='#7572f1', linestyle='--') working code for only one time occurance

    ]
    
    
    ################################################################################################################
    ########################################################## Working Code below #################################################

    # Define your market colors
    mcgreen = mpf.make_marketcolors(base_mpf_style='yahoo', up='limegreen', down='yellow', ohlc='limegreen', edge='inherit', wick='inherit')
    style = mpf.make_mpf_style(marketcolors=mcgreen)

    # Define volume conditions
    VolumeUpCondition = (resampled_df['Volume'] > resampled_df['Volume'].rolling(window=20).mean() * 1.5) & (resampled_df['Close'] > resampled_df['Open'])
    VolumeDownCondition = (resampled_df['Volume'] > resampled_df['Volume'].rolling(window=20).mean() * 1.5) & (resampled_df['Close'] <= resampled_df['Open'])

    # Initialize an empty DataFrame to store the candles to be plotted
    new_resampled_df = resampled_df.copy()
    new_resampled_df.loc[~(VolumeUpCondition | VolumeDownCondition), ['Open', 'High', 'Low', 'Close']] = new_resampled_df.loc[VolumeUpCondition | VolumeDownCondition, ['Open', 'High', 'Low', 'Close']].ffill()

    
    # Define the length of the trend triangle
    triangle_len = triangle_length_var.get() + 1

    def create_trend_line_data(resampled_df, condition_index, is_up_condition):
        today_high = resampled_df.loc[condition_index, 'High']
        today_low = resampled_df.loc[condition_index, 'Low']
        initial_diff = today_high - today_low
        
        end_date = min(condition_index + pd.Timedelta(days=triangle_len), resampled_df.index[-1])
        date_range = (end_date - condition_index).days + 1
        step_size = 1 / date_range
        
        trend_high = []
        trend_low = []
        
        for i in range(date_range):
            current_date = condition_index + pd.Timedelta(days=i)
            if current_date <= resampled_df.index[-1]:
                if is_up_condition:
                    new_high = today_high - (initial_diff * step_size * i)
                    new_low = today_low  # Keeping low constant
                else:
                    new_high = today_high  # Keeping high constant
                    new_low = today_low + (initial_diff * step_size * i)
                trend_high.append(new_high)
                trend_low.append(new_low)
            else:
                trend_high.append(np.nan)
                trend_low.append(np.nan)

        return trend_high, trend_low

    # Find the indices where the volume conditions are True
    up_condition_indices = resampled_df.index[VolumeUpCondition].tolist()
    down_condition_indices = resampled_df.index[VolumeDownCondition].tolist()

    # Create trend line data for up conditions
    trend_highs_up = pd.DataFrame(index=resampled_df.index, columns=[f'TrendHighUp_{i}' for i in range(len(up_condition_indices))])
    trend_lows_up = pd.DataFrame(index=resampled_df.index, columns=[f'TrendLowUp_{i}' for i in range(len(up_condition_indices))])

    for i, idx in enumerate(up_condition_indices):
        high, low = create_trend_line_data(resampled_df, idx, is_up_condition=True)
        start_idx = resampled_df.index.get_loc(idx)
        end_idx = min(start_idx + triangle_len, len(resampled_df))
        trend_highs_up.iloc[start_idx:end_idx, i] = high[:end_idx-start_idx]
        trend_lows_up.iloc[start_idx:end_idx, i] = low[:end_idx-start_idx]

        addplot.append(mpf.make_addplot(trend_highs_up[f'TrendHighUp_{i}'], color='#ad0afd', linestyle='--', ax=ax1, alpha=1))
        addplot.append(mpf.make_addplot(trend_lows_up[f'TrendLowUp_{i}'], color='#0bf77d', linestyle='--', ax=ax1, alpha=1))

    # Create trend line data for down conditions
    trend_highs_down = pd.DataFrame(index=resampled_df.index, columns=[f'TrendHighDown_{i}' for i in range(len(down_condition_indices))])
    trend_lows_down = pd.DataFrame(index=resampled_df.index, columns=[f'TrendLowDown_{i}' for i in range(len(down_condition_indices))])

    for i, idx in enumerate(down_condition_indices):
        high, low = create_trend_line_data(resampled_df, idx, is_up_condition=False)
        start_idx = resampled_df.index.get_loc(idx)
        end_idx = min(start_idx + triangle_len, len(resampled_df))
        trend_highs_down.iloc[start_idx:end_idx, i] = high[:end_idx-start_idx]
        trend_lows_down.iloc[start_idx:end_idx, i] = low[:end_idx-start_idx]

        addplot.append(mpf.make_addplot(trend_highs_down[f'TrendHighDown_{i}'], color='#de0c62', linestyle='--', ax=ax1, alpha=1))
        addplot.append(mpf.make_addplot(trend_lows_down[f'TrendLowDown_{i}'], color='#08ff08', linestyle='--', ax=ax1, alpha=1))
    
    #################################################################################################################
    
    
    show_ma189 = False
    show_mamix14 = True
    show_mamix42 = True
    show_vwma25 = True
    show_volsma5 = True
    show_rsi3 = True
    show_rsi14 = True
    show_cci3 = True
    show_cci12 = True
    show_volume_candle = True
    show_trendlines = True
    ### RSI SEttings
    
    rsi_overbought_mask = resampled_df['rsi3'] > 60
    rsi_oversold_mask = resampled_df['rsi3'] < 40
    cci_overbought_mask = resampled_df['cci3'] > 50
    cci_oversold_mask = resampled_df['cci3'] < -50
    
    
    if show_ma189:
        addplot.append(mpf.make_addplot(resampled_df['ma189'], ax=ax1, color='purple', width=0.5))
    if show_mamix14:        
        addplot.append(mpf.make_addplot(resampled_df['mamix14'], ax=ax1, color='blue')),
    if show_mamix42:    
        addplot.append(mpf.make_addplot(resampled_df['mamix42'], ax=ax1, color='red', width=0.5)),
    if show_vwma25:
        addplot.append(mpf.make_addplot(resampled_df['vwma25'], ax=ax1, color='#37B7C3')),
    if show_volsma5:
        addplot.append(mpf.make_addplot(resampled_df['volsma5'], ax=ax2, color='#ff00ff')),
    if show_rsi3:
        addplot.append(mpf.make_addplot(resampled_df['rsi3'], ax=ax3, color='#1e90ff'))
        addplot.append(mpf.make_addplot(resampled_df['rsi3'].where(rsi_overbought_mask), ax=ax3, color='green', fill_between={'y1': 60, 'y2': 100, 'alpha': 0.3}))
        addplot.append(mpf.make_addplot(resampled_df['rsi3'].where(rsi_oversold_mask), ax=ax3, color='red', fill_between={'y1': 0, 'y2': 40, 'alpha': 0.3}))
    if show_rsi14:
        addplot.append(mpf.make_addplot(resampled_df['rsi14'], ax=ax3, color='#2f01ee'))
    if show_cci3:
        addplot.append(mpf.make_addplot(resampled_df['cci3'], ax=ax4, color='#1e90ff'))
        addplot.append(mpf.make_addplot(resampled_df['cci3'].where(cci_overbought_mask), ax=ax4, color='green', fill_between={'y1': 50, 'y2': 100, 'alpha': 0.3}))
        addplot.append(mpf.make_addplot(resampled_df['cci3'].where(cci_oversold_mask), ax=ax4, color='red', fill_between={'y1': -100, 'y2': -50, 'alpha': 0.3}))
    if show_cci12:
        addplot.append(mpf.make_addplot(resampled_df['cci12'], ax=ax4, color='#6600ee'))
    if show_volume_candle:# Check if there are any rows that meet the condition
        if not new_resampled_df.empty:
            addplot.append(mpf.make_addplot(new_resampled_df[['Open', 'High', 'Low', 'Close', 'Volume']], ax=ax1, type='candle', marketcolors=mcgreen))

    
            
    
    
    
    
    ################################## main plot dont delete #########################################################################
    # mpf.plot(resampled_df, type='candle', style=samie_style_obj, ax=ax1, volume=ax2, datetime_format='%b %d', addplot=addplot) # working code
    mpf.plot(resampled_df, type='candle', style=samie_style_obj, ax=ax1, volume=ax2, datetime_format='%m-%d', addplot=addplot, returnfig=True)
    ##################################################################################################################################


    # Create a custom cursor
    cursor = mplcursors.cursor(ax1, hover=True)

    # Function to find the nearest candle
    def find_nearest_candle(sel):
        x, y = sel.target
        # Convert x from float to nearest index
        index = int(round(x))
        return index, resampled_df.iloc[index]

    # Custom picking function
    def custom_pick(artist, mouseevent):
        if isinstance(artist, plt.Rectangle):
            mouse_x = mouseevent.xdata
            candle_index = int(round(mouse_x))
            return True, {'index': candle_index}
        return False, {}

    # Modified tooltip function
    def custom_tooltip(sel):
        index, row = find_nearest_candle(sel)
        
        open_price = row['Open']
        high_price = row['High']
        low_price = row['Low']
        close_price = row['Close']
        date = row.name
        
        prev_close = resampled_df['Close'].iloc[index-1] if index > 0 else open_price
        pct_chg = ((close_price - prev_close) / prev_close) * 100 if prev_close != 0 else 0
        
        sel.annotation.set_text(
            f"Date: {date.strftime('%Y-%m-%d')}\n"
            f"Open: {open_price:.2f}\n"
            f"High: {high_price:.2f}\n"
            f"Low: {low_price:.2f}\n"
            f"Close: {close_price:.2f}\n"
            f"Chg: {pct_chg:.2f}%"
        )
        sel.annotation.get_bbox_patch().set_facecolor("lightgray")
        sel.annotation.get_bbox_patch().set_alpha(0.7)
        sel.annotation.set_fontsize(10)
        sel.annotation.get_bbox_patch().set_edgecolor("darkgray")
        sel.annotation.get_bbox_patch().set_linewidth(1)
        sel.annotation.set_zorder(1000)

    # Set the custom picking function
    ax1.figure.canvas.mpl_connect('pick_event', lambda event: custom_pick(event.artist, event.mouseevent))

    # Connect the modified tooltip function
    cursor.connect("add", custom_tooltip)



    
#####################################################################################################################
    ax1.set_title(f' {selected_period} :  {latest_info}', loc='left',  fontsize=10, color='dodgerblue', y=1)
    # ax1.set_ylabel('Price')
    ax1.grid(axis='both', which='both', linestyle='-.', linewidth=0.3, color='#31363F')
    ax2.grid(axis='both', which='both', linestyle='-.', linewidth=0.3, color='#31363F')
    ax3.grid(axis='both', which='both', linestyle='-.', linewidth=0.3, color='#31363F')
    ax4.grid(axis='both', which='both', linestyle='-.', linewidth=0.3, color='#31363F')
    
    ax1.tick_params(axis='y', colors='dodgerblue')
    ax2.tick_params(axis='y', colors='dodgerblue')
    ax3.tick_params(axis='y', colors='dodgerblue')
    ax4.tick_params(axis='y', colors='dodgerblue')
    ax4.tick_params(axis='x', colors='dodgerblue')
    
    # Enhanced Gridlines for ax1 (Price Chart)
    # ax1.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')  # Major and minor grids for ax1
    
    data_range = resampled_df['Close'].max() - resampled_df['Close'].min()
    # tick_interval = data_range * 0.05  # 5% of the data range
    
    def round_to_nearest_five_paisa(value):
        rounded_value = round(value * 20) / 20.0
        return rounded_value

    # data_range = 1.23  # Replace with your actual data range
    # tick_interval = data_range * 0.10
    # rounded_tick_interval = round_to_nearest_five_cents(tick_interval)
    # print(rounded_tick_interval)

    tick_interval = data_range * 0.033  # 3.3% of the data range
    rounded_tick_interval = round_to_nearest_five_paisa(tick_interval)
    
    # Set y-axis major ticks at every 5% of the data range
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(rounded_tick_interval))
    # ax1.set_ylim([resampled_df['Low'].min(), resampled_df['High'].max()])  # working code.. sets you graph to max and min values
    
    # Set the title on the primary axes to include the symbol name
    # ax1.set_title(f'{symbol} Stock Price ({selected_period})', loc='center', fontsize=16, fontweight='bold')
    ############ working code for line below for drawing lines.. dont delete.
    
    # if not traderule3_high.empty:
    #     for value in traderule3_high:
    #         ax1.axhline(y=value, color='#7572f1', linestyle=':')	
    
    # if not traderule3_low.empty:
    #     # for value in traderule3_low:                                              ### working code
    #     #     ax1.axhline(y=value * 0.95, color='#ff000f', linestyle='--')          ### working code
    #     for value  in traderule3_low:
    #         ax1.axhline(y=value * 0.98, color='#ff000f', linestyle=':')
    #         ax1.text(resampled_df.index[-1], value * 0.98, f'{value:.2f}', va='center', ha='left', 
    #             bbox=dict(facecolor='#0042a1', edgecolor='black', alpha=0.7, boxstyle='round,pad=0.5'))
                
    # if not traderule4_high.empty:
    #     for value in traderule4_high:
    #         ax1.axhline(y=value, color='#abcddcba', linestyle='--')
    
    # if not traderule4_low.empty:
    #     # for value in traderule4_low:                                              ### working code
    #     #     ax1.axhline(y=value * 0.95, color='#fabfab', linestyle='--')          ### working code
    #     for value  in traderule4_low:
    #         ax1.axhline(y=value * 0.98, color='#fabcfabc', linestyle='--')
    #         ax1.text(resampled_df.index[-1], value * 0.98, f'{value:.2f}', va='center', ha='left', 
    #             bbox=dict(facecolor='#7572f1', edgecolor='black', alpha=0.7))
                

    # ax1.set_yscale('log')   
    # ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:.2f}'))
    # ax1.yaxis.set_major_formatter(ticker.FuncFormatter(price_formatter))
    ############################################################################################
    
    if not highlight5_high_values.empty:
        for value in highlight5_high_values:
            ax1.axhline(y=value, color='#ff000f', linestyle='--', linewidth=0.8)	
    
    if not highlight5_low_values.empty:
        for value in highlight5_low_values:
            ax1.axhline(y=value * 0.95, color='#0042a1', linestyle='--', linewidth=0.8)          ### working code
    
    if not highlight3_high_values.empty:
        for value in highlight3_high_values:
            ax1.axhline(y=value, color='#ff3b46', linestyle='--', linewidth=0.8)	
    
    if not highlight3_low_values.empty:
        for value in highlight3_low_values:
            ax1.axhline(y=value * 0.95, color='#004cbb', linestyle='--', linewidth=0.8)          ### working code
    
    if not highlight2_high_values.empty:
        for value in highlight2_high_values:
            ax1.axhline(y=value, color='#ff626b', linestyle='--', linewidth=0.8)	
    
    if not highlight2_low_values.empty:
        for value in highlight2_low_values:
            ax1.axhline(y=value * 0.95, color='#0061ee', linestyle='--', linewidth=0.8)          ### working code
    
    
    
    #### Plotting for TRaderule 8 
    if not traderule8_high_values.empty:
        for index, value in resampled_df['traderule8_high'].items():
            if pd.notnull(value):
                ax1.axhline(y=value, color='#0fffa1', linestyle='--', linewidth=0.8)
    if not traderule8_low_values.empty:
        for index, value in resampled_df['traderule8_low'].items():
            if pd.notnull(value):
                ax1.axhline(y=value, color='#0042a1', linestyle='--', linewidth=0.8)

    #################################################################################################
    fig.suptitle(f'{symbol}', fontsize=14, y=0.95, color='#000ff0ff')  # Adjust the y position as needed

    # # Volume Chart
    # # Plot the volume chart on ax2
    # ax2.set_title('  Volume', loc='left',  fontsize=10, pad=5)
    # ax2.set_ylabel('Volume')
    # ax2.yaxis.set_label_position('right')
    ax2.yaxis.tick_left()
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(volume_formatter))
    ax3.yaxis.tick_right()
    ax4.yaxis.tick_left()
        
    # Enhanced Gridlines for ax2 (Volume Chart)
    ax2.grid(axis='both', which='both', linestyle='-.', linewidth=0.3, color='#31363F')  # Major and minor grids for ax2

    #################################################################################################
    ##### Configured the below after so many trails.. dont change. best for 4 chart layout.
    ax1.set_position([0.09, 0.41, 0.84, 0.55])  # [left, bottom, width, height]
    ax2.set_position([0.09, 0.29, 0.84, 0.12])
    ax3.set_position([0.09, 0.17, 0.84, 0.12])
    ax4.set_position([0.09, 0.05, 0.84, 0.12])
    
    # Disable x-axis tickers for all subplots except ax4
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=False)

    # Assuming you have a figure and two axes, ax1 and ax2, set up as part of your plotting code
    ################################################################################################
    # Calculate the interval for 5% of the price range
    data_range = resampled_df['Close'].max() - resampled_df['Close'].min()
    # tick_interval = data_range * 0.05  # 5% of the data range
    tick_interval = data_range * 0.10  # 10% of the data range

    # Set y-axis major ticks at every 5% of the data range
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(tick_interval)) 
    
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(10)) # working
    ax2.xaxis.tick_bottom()
    plt.setp(ax2.get_xticklabels(), rotation=0, ha='center' )
    
    
    
    # for RSI
    ax3.axhline(y=40, color='red', linestyle='--', linewidth=.4)
    ax3.axhline(y=60, color='green', linestyle='--', linewidth=.4)
    # for CCI
    ax4.axhline(y=-50, color='red', linestyle='--', linewidth=.4)
    ax4.axhline(y=50, color='green', linestyle='--', linewidth=.4)
################################### crosshair ################################################    

    # Initialize the crosshair lines
    crosshair_ax1_v = ax1.axvline(x=0, color='#074173', linestyle=':')
    crosshair_ax1_h = ax1.axhline(y=0, color='#074173', linestyle=':')
    crosshair_ax2_v = ax2.axvline(x=0, color='#075173', linestyle=':')
    crosshair_ax2_h = ax2.axhline(y=0, color='#075173', linestyle=':')
    crosshair_ax3_v = ax3.axvline(x=0, color='#076173', linestyle=':')
    crosshair_ax3_h = ax3.axhline(y=0, color='#076173', linestyle=':')
    crosshair_ax4_v = ax4.axvline(x=0, color='#077173', linestyle=':')
    crosshair_ax4_h = ax4.axhline(y=0, color='#077173', linestyle=':')
    

    # Function to update the crosshair position this work.. 
    # def update_crosshair(event):
    #     if event.inaxes == ax1 or ax2 or ax3 or ax4:
    #         crosshair_ax1_v.set_xdata([event.xdata, event.xdata])
    #         crosshair_ax2_v.set_xdata([event.xdata, event.xdata])
    #         crosshair_ax3_v.set_xdata([event.xdata, event.xdata])
    #         crosshair_ax4_v.set_xdata([event.xdata, event.xdata])
    #         ax1ymin, ax1ymax = ax1.get_ylim() 
    #         ax2ymin, ax2ymax = ax2.get_ylim()
    #         ax3ymin, ax3ymax = ax3.get_ylim()
    #         ax4ymin, ax4ymax = ax4.get_ylim()
            
    #         if event.ydata is not None and ((ax1ymin <= event.ydata <= ax1ymax) or (ax2ymin <= event.ydata <= ax2ymax) or (ax3ymin <= event.ydata <= ax3ymax) or (ax4ymin <= event.ydata <= ax4ymax)):

    #             crosshair_ax1_h.set_ydata([event.ydata, event.ydata])
    #             crosshair_ax2_h.set_ydata([event.ydata, event.ydata])
    #             crosshair_ax3_h.set_ydata([event.ydata, event.ydata])
    #             crosshair_ax3_h.set_ydata([event.ydata, event.ydata])
                
    #         else:
    #             crosshair_ax1_h.set_ydata([ax1ymin, ax1ymin])
    #             crosshair_ax2_h.set_ydata([ax2ymin, ax2ymin])
    #             crosshair_ax3_h.set_ydata([ax3ymin, ax3ymin])
    #             crosshair_ax4_h.set_ydata([ax4ymin, ax4ymin])
    #         fig.canvas.draw_idle()

    def update_crosshair(event):
        if event.inaxes in (ax1, ax2, ax3, ax4):
            # Update vertical crosshairs
            for crosshair in (crosshair_ax1_v, crosshair_ax2_v, crosshair_ax3_v, crosshair_ax4_v):
                crosshair.set_xdata([event.xdata, event.xdata])

            # Get y-limits for all axes
            y_limits = [ax.get_ylim() for ax in (ax1, ax2, ax3, ax4)]

            if event.ydata is not None and any(ymin <= event.ydata <= ymax for ymin, ymax in y_limits):
                # Update horizontal crosshairs if within any axis limits
                for crosshair in (crosshair_ax1_h, crosshair_ax2_h, crosshair_ax3_h, crosshair_ax4_h):
                    crosshair.set_ydata([event.ydata, event.ydata])
            else:
                # Set horizontal crosshairs to bottom of each axis if outside limits
                for crosshair, (ymin, _) in zip((crosshair_ax1_h, crosshair_ax2_h, crosshair_ax3_h, crosshair_ax4_h), y_limits):
                    crosshair.set_ydata([ymin, ymin])

            # Redraw the figure
            fig.canvas.draw_idle()

    # Connect the function to the figure's motion_notify_event
    fig.canvas.mpl_connect('motion_notify_event', update_crosshair)

    # Optionally, set the y-axis limits to match your data's range
    ymin, ymax = resampled_df['Low'].min() * 0.98, resampled_df['High'].max() * 1.02 # Adjust for your DataFrame's columns
    ax1.set_ylim([ymin, ymax])
    
    ### drawing a line in center of the chart.
    chart_center = ( ymin + ymax ) / 2 
    ax1.axhline(y=chart_center, color='orange', linestyle='--', linewidth=.4)
    
		
		
    custom_lines = [
							Line2D([0], [0], color=addplot[0]['color'], label='[ma189]', lw=2), 
							Line2D([0], [0], color=addplot[1]['color'], label='[mamix14]', lw=2),
							Line2D([0], [0], color=addplot[2]['color'], label='[mamix42]', lw=2), 
							Line2D([0], [0], color=addplot[3]['color'], label='[vwma25]', lw=2),
							Line2D([0], [0], color=addplot[4]['color'], label='[volsma5]', lw=2) 
							# Line2D([0], [0], color=addplot[5]['color'], label='[traderule1_highlight]', lw=2),
							# Line2D([0], [0], color=addplot[6]['color'], label='[traderule2_highlight]', lw=2), 
							# Line2D([0], [0], color=addplot[7]['color'], label='[traderule3_highlight]', lw=2),
							# Line2D([0], [0], color=addplot[8]['color'], label='[traderule4_highlight]', lw=2), 
							# Line2D([0], [0], color=addplot[9]['color'], label='[traderule5_highlight]', lw=2),
							# Line2D([0], [0], color=addplot[10]['color'], label='[traderule6_highlight]', lw=2) 
							]
    ax1.legend(custom_lines, ['ma189','mamix14', 'mamix42', 'vwma25', 'volsma5'], loc='upper left', fontsize=8, facecolor='#202020',labelcolor='dodgerblue')
    
    canvas.draw()
 

    
# Usage
def plot_monthly_chart(symbol, selected_period):
    plot_resampled_chart(symbol, selected_period, resample_freq='ME')

def plot_weekly_chart(symbol, selected_period):
    plot_resampled_chart(symbol, selected_period, resample_freq='W')

def plot_daily_chart(symbol, selected_period):
    plot_resampled_chart(symbol, selected_period, resample_freq='D')

##############################################????????????????????????????????#############################


def add_stock():
    symbol = stock_entry.get().upper()
    if symbol:
        if symbol not in watchlist.get(0, tk.END):
            watchlist.insert(tk.END, symbol)
            # watchlist.set_values(sorted(watchlist))
            cur.execute("INSERT INTO tickers (symbol) VALUES (%s) ON CONFLICT (symbol) DO NOTHING", (symbol,))
            conn.commit()
            asyncio.run(calculate_and_store_rsi_cci(symbol))
            asyncio.run(calculate_and_store_ma(symbol))
            asyncio.run(update_traderules(symbol))
            

        else:
            messagebox.showwarning("Duplicate Entry", "Stock symbol already in watchlist.")
    else:
        messagebox.showwarning("Input Error", "Please enter a valid stock symbol.")
        
# Adding stocks which are listed in chartink screeners from local text file

def add_chartink_stock():
    file_path = filedialog.askopenfilename(title="Select Chartink Stock File", filetypes=(("CSV Files", "*.csv"),))
    if file_path:
        try:
            with open(file_path, 'r') as file:
                stocks = file.readlines()
                for stock in stocks:
                    symbol = stock.strip()
                    # Insert symbol into database
                    cur.execute("INSERT INTO tickers (symbol) VALUES (%s) ON CONFLICT (symbol) DO NOTHING", (symbol,))
                    conn.commit()
                    
                    # Check if symbol is already in watchlist before adding
                    if symbol not in watchlist.get(0, tk.END):
                        watchlist.insert(tk.END, symbol)
            
            messagebox.showinfo("Success", "Stocks added to watchlist!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read file: {e}")
                                    

def delete_stock():
    selected = watchlist.curselection()
    if selected:
        symbol = watchlist.get(selected)
        watchlist.delete(selected)
        cur.execute("DELETE FROM daily_prices WHERE symbol = %s", (symbol,))
        cur.execute("DELETE FROM tickers WHERE symbol = %s", (symbol,))
        conn.commit()
    else:
        messagebox.showwarning("Selection Error", "Please select a stock to delete.")
        

def on_select(event):
    if watchlist.curselection():
        selected_symbol = watchlist.get(watchlist.curselection())
        selected_period = period_var.get()
        # fetch_data_from_db(selected_symbol, selected_period)
        selected_timeframe = timeframe_var.get()
        if selected_timeframe == "Daily":
            plot_daily_chart(selected_symbol, selected_period)
        elif selected_timeframe == "Weekly":
            plot_weekly_chart(selected_symbol, selected_period)
        elif selected_timeframe == "Monthly":
            plot_monthly_chart(selected_symbol, selected_period)        


async def update_traderules(symbol):
    cur.execute("""
        SELECT date, open, high, low, close, rsi3, rsi14, mamix14, mamix42, vwma25, volsma5, ma10, ma21, ma32, ma43, ma54, stochrsi14, volume, bb_percent_b,
               traderule1, traderule2, traderule3, traderule4, traderule5, traderule6, traderule7, traderule8, traderule9, traderule10 
        FROM daily_prices 
        WHERE symbol = %s 
        ORDER BY date
    """, (symbol,))
    rows = cur.fetchall()
    
    df = pd.DataFrame(rows, columns=['date', 'open', 'high', 'low', 'close', 'rsi3', 'rsi14', 'mamix14', 'mamix42', 'vwma25', 'volsma5', 'ma10','ma21', 'ma32', 'ma43', 'ma54', 'stochrsi14', 'volume', 'bb_percent_b', 
                                     'traderule1', 'traderule2', 'traderule3', 'traderule4', 'traderule5', 'traderule6', 'traderule7', 'traderule8', 'traderule9', 'traderule10'])
    df.set_index('date', inplace=True)
    df = df.fillna(value=np.nan)


    # Calclulate slope of mamix 14 for entries
    mamix14slope = df['mamix14'].shift(8) / df['mamix14']
    # Calculate traderule1
    df['new_traderule1'] = (df['close'].shift(3) < df['open'].shift(3)) & \
                           (df['close'].shift(2) < df['open'].shift(2)) & \
                           (df['close'].shift(1) < df['open'].shift(1)) & \
                           (df['close'] > df['open']) & \
                           (mamix14slope < 1.01)

    # Calculate traderule2
    df['new_traderule2'] = (df['close'].shift(2) < df['open'].shift(2)) & \
                           (df['close'].shift(1) > df['open'].shift(1)) & \
                           (df['close'].shift(1) < df['open'].shift(2)) & \
                           (df['open'].shift(1) > df['close'].shift(2)) & \
                           (df['open'] < df['open'].shift(1)) & \
                           (df['close'] > df['close'].shift(1)) & \
                           (df['close'] > df['open'].shift(2)) 

    # Calculate traderule3
    df['new_traderule3'] = (df['high'] > df['high'].shift(2)) & \
                           (df['high'].shift(1) < df['high'].shift(2)) & \
                           (df['open'].shift(2) > df['close'].shift(2)) & \
                           (df['open'] < df['close']) & \
                           (df['rsi3'] < 60)

    # Calculate traderule4
    df['new_traderule4'] = (df['close'].shift(2) < df['open'].shift(2)) & \
                           (df['close'].shift(1) > df['open'].shift(1)) & \
                           (df['close'].shift(1) > ((df['close'].shift(1).abs() + df['open'].shift(1).abs()) / 2)) & \
                           (df['close'].shift(1) < ((df['close'].shift(2).abs() + df['open'].shift(2).abs()) / 2)) & \
                           (df['open'] > ((df['high'].shift(1) + df['low'].shift(1)) / 2)) & \
                           (df['close'] > df['open'])

    # Calculate traderule5
    df['new_traderule5'] = (df['close'].gt(df['mamix14'])) & \
                       (df['close'].shift(1).le(df['mamix14'].shift(1))) & \
                       (df['close'].gt(df['mamix42'])) & \
                       (df['close'].shift(1).le(df['mamix42'].shift(1)))
    
    # Calculate traderule6
    df['new_traderule6'] = (df['close'].shift(1) < df['open'].shift(1)) & \
                        (df['close'] > df['open']) & \
                        (df['open'] > df['close'].shift(1)) & \
                        (df['close'] < df['open'].shift(1)) & \
                        (df['close'] / df['open'] < 1.002) & \
                        (df['close'] < df['vwma25'] * 1.01)
    # Calculate traderule7
    df['samies_morningstar'] = (df['close'].shift(2) < df['open'].shift(2)) & \
                        (df['close'].shift(1) > df['low'].shift(2)) & \
                        (df['close'].shift(1) > ((df['high'].shift(1) + df['low'].shift(1)) / 2)) & \
                        (df['close'].shift(1) < df['high'].shift(2)) & \
                        (df['close'] > df['high'].shift(1)) & \
                        (df['close'] > ((df['high'] + df['low']) / 2)) & \
                        (mamix14slope < 1.01)
    # Calculate traderule8
    df['new_traderule8'] = (df['close'] > df['ma10']) & \
                        (df['close'] > df['ma21']) & \
                        (df['close'] > df['ma32']) & \
                        (df['close'] > df['ma43']) & \
                        (df['close'] > df['ma54']) & \
                        (df['volume'] > df['volsma5']) & \
                        (df['close'] > df['high'].shift(1)) & \
                        (df['close'].shift(1) <= df['ma54'].shift(1)) & \
                        (df['close'] > ((df['high'] + df['low']) / 2)) & \
                        (mamix14slope < 1.01)
    # Bollinger band percent b breakout                        
    df['bbpct_b'] = (df['bb_percent_b'] > 0.5) & (df['bb_percent_b'].shift(1) < 0.5) & (mamix14slope < 1.01)
    
    ## bbpct_b and rsi14 oversold reversal candle.
    df['perfect_bottom'] =  (df['bb_percent_b'].shift(1) < 0.4) & \
                            (df['rsi3'].shift(1) < 20) & \
                            (df['rsi14'].shift(1) < 40) & \
                            (df['close'] > ((df['high'] + df['low']) / 2)) & \
                            (mamix14slope < 1.01)
                                
    # Find rows where traderules have changed
    changed_rows = df[(df['traderule1'] != df['new_traderule1']) | 
                      (df['traderule2'] != df['new_traderule2']) |
                      (df['traderule3'] != df['new_traderule3']) |
                      (df['traderule4'] != df['new_traderule4']) |
                      (df['traderule5'] != df['new_traderule5']) |
                      (df['traderule6'] != df['new_traderule6']) |
                      (df['traderule7'] != df['samies_morningstar']) | 
                      (df['traderule8'] != df['new_traderule8']) |
                      (df['traderule9'] != df['bbpct_b'])  |
                      (df['traderule10'] != df['perfect_bottom']) ]

    # Prepare batch updates
    updates = [(bool(row['new_traderule1']), 
                bool(row['new_traderule2']),
                bool(row['new_traderule3']),
                bool(row['new_traderule4']),
                bool(row['new_traderule5']),
                bool(row['new_traderule6']),
                bool(row['samies_morningstar']),
                bool(row['new_traderule8']),
                bool(row['bbpct_b']),
                bool(row['perfect_bottom']),
                
                symbol, row.name) for _, row in changed_rows.iterrows()]

    # Perform bulk update
    cur.executemany("""
        UPDATE daily_prices 
        SET traderule1 = %s, traderule2 = %s, traderule3 = %s, traderule4 = %s, traderule5 = %s, traderule6 = %s, traderule7 = %s, traderule8 = %s, traderule9 = %s, traderule10 = %s
        WHERE symbol = %s AND date = %s
    """, updates)
    
    conn.commit()
    print(f"Updated traderules columns in the database for symbol {symbol}")

# Call the function
# update_traderules(symbol)


def format_tooltip(x, y, data):
    # Access the data for the hovered candle
    candle_data = data[x]
    open_price, high_price, low_price, close_price = candle_data[1:5]
    return f"Open: {open_price:.2f}\nHigh: {high_price:.2f}\nLow: {low_price:.2f}\nClose: {close_price:.2f}"


def download_histdata(start_date, end_date):
    selected_symbol = watchlist.get(watchlist.curselection())

    # Fetch missing data
    stock_hist_data = yf.download(selected_symbol, start=start_date, end=end_date)

    if stock_hist_data.empty:
        print(f"No data available for {selected_symbol} between {start_date} and {end_date}")
        return

# Prepare data for bulk insert
    data_to_insert = []
    for index, row in stock_hist_data.iterrows():
        data_to_insert.append((
            selected_symbol,
            index.date(),
            float(row['Open']),
            float(row['High']),
            float(row['Low']),
            float(row['Close']),
            int(row['Volume'])
        ))
    # Bulk insert/update
    try:
        cur.executemany("""
            INSERT INTO daily_prices (symbol, date, open, high, low, close, volume)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (symbol, date) DO UPDATE SET
                open = excluded.open,
                high = excluded.high,
                low = excluded.low,
                close = excluded.close,
                volume = excluded.volume
        """, data_to_insert)
        conn.commit()
        print(f"Successfully updated data for {selected_symbol}")
    except Exception as e:
        print(f"Error inserting data: {e}")
        conn.rollback()

    
def download_last_3_days_histdata():
    
    # Calculate the start date for the last 3 days
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=3)
    
    # Download the historical data for each symbol and update the database

    # Fetch symbols from the database
    cur.execute("SELECT symbol FROM tickers")
    symbols = [row[0] for row in cur.fetchall()]
    symbols.sort()

    # Check data availability for the first 3 symbols
    available_symbols = []
    for symbol in symbols[:3]:
        stock_hist_data = yf.download(symbol, start=start_date, end=end_date)
        if not stock_hist_data.empty:
            available_symbols.append(symbol)

    # If data is available for at least 3 symbols, fetch data for all symbols
    if len(available_symbols) >= 3:
        for symbol in symbols:
            stock_hist_data = yf.download(symbol, start=start_date, end=end_date)
            
            if stock_hist_data.empty:
                print(f"No data available for symbol {symbol}")
                continue
            
            # Insert the data into the database
            for index, row in stock_hist_data.iterrows():
                data = {
                    'symbol': symbol,
                    'date': index.date(),
                    'open': row['Open'],
                    'high': row['High'],
                    'low': row['Low'],
                    'close': row['Close'],
                    'volume': row['Volume']
                }
                try:
                    cur.execute("""
                        INSERT INTO daily_prices (symbol, date, open, high, low, close, volume)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (symbol, date) DO NOTHING
                    """, (
                        symbol,
                        row.name.date(),
                        float(row['Open']),
                        float(row['High']),
                        float(row['Low']),
                        float(row['Close']),
                        int(row['Volume'])
                    ))
                    conn.commit()
                except Exception as e:
                    print(f"Error downloading data for symbol {symbol}: {e}")
#################################### All Rules update section ####################################                
def update_Screensers_thread():
    threading.Thread(target=update_Screeners, daemon=True).start()
        
    root.after(0, lambda: messagebox.showinfo("Success", "All traderules updated for all symbols!"))
    root.after(0, lambda: progress_bar.pack_forget())  # Hide the progress bar when done


def update_Screeners(): # updates every symbol
    cur.execute("SELECT symbol FROM tickers")
    symbols = [row[0] for row in cur.fetchall()]
    
    total_symbols = len(symbols)
    for index, symbol in enumerate(symbols, 1):
        asyncio.run(update_traderules(symbol))
        
        # Update progress
        progress = (index / total_symbols) * 100
        root.after(0, lambda p=progress: update_progress(p))
    
    # root.after(0, lambda: messagebox.showinfo("Success", "All traderules updated for all symbols!"))
    # root.after(0, lambda: progress_bar.pack_forget())  # Hide the progress bar when done
def update_progress(progress):
    # Update a progress bar or label in your UI
    # For example, if you have a progress bar widget named progress_bar:
    progress_bar['value'] = progress
    root.update_idletasks()
##################################################################
# def detect_cspatterns():
#     try:
#         filtered_patterns = get_filtered_patterns()
#         filtered_patterns = filtered_patterns.sort_values(by='Date', ascending=False)


#         # Create a popup window with a table to display the results
#         popup =  tk.Toplevel()
#         popup.title("CSpatterns Detection Results")

#         # Create a table to display the results
#         table = ttk.Treeview(popup)
#         table['columns'] = tuple(filtered_patterns.columns)

#         # Format the table columns
#         table.column("#0", width=0, stretch=tk.NO)
#         for col in filtered_patterns.columns:
#             table.column(col, anchor=tk.W, width=100)
#             table.heading(col, text=col, anchor=tk.W)

#         # Insert data into the table
#         for index, row in filtered_patterns.iterrows():
#             values = [('✅' if v == True else ' ' if v == False else v) for v in row]
#             table.insert('', 'end', values=values)

#         # Pack the table
#         table.pack(fill=tk.BOTH, expand=True)

#         # Make the popup window visible
#         popup.mainloop()

#     except Exception as e:
#         print(f"An error occurred: {e}")



###########################################################################################
def up2date_chart():
    try:
        selected_symbol = watchlist.get(watchlist.curselection())
    except tk.TclError:
        messagebox.showerror("Selection Error", "No symbol selected.")
        return

    async def update_tasks(symbol):
        await asyncio.gather(
            calculate_and_store_rsi_cci(symbol),
            calculate_and_store_ma(symbol)
        )
        await update_traderules(symbol)
    def run_async_tasks():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(update_tasks(selected_symbol))
        loop.close()

    threading.Thread(target=run_async_tasks, daemon=True).start()

async def update_everything():
    cur.execute("SELECT symbol FROM tickers")
    symbols = [symbol[0] for symbol in cur.fetchall()]
    # Sort the symbols in ascending order
    symbols.sort()
    
    total_symbols = len(symbols)
    for index, symbol in enumerate(symbols):
            await calculate_and_store_rsi_cci(symbol),
            await calculate_and_store_ma(symbol),
            await update_traderules(symbol)   ### enable later.. 
            
            # Update progress
            progress = (index + 1 / total_symbols) * 100
            root.after(0, lambda p=progress: update_progress(p))
    
    root.after(0, lambda: messagebox.showinfo("Success", "All traderules updated for all symbols!"))
    root.after(0, lambda: progress_bar.pack_forget())  # Hide the progress bar when done

        
        
def show_recent_signals():
    popup =  tk.Toplevel()
    popup.title("Recent Signals")
    
    # Create a style
    style = ttk.Style()
    style.configure("Treeview", font=('Nirmala UI', 10), foreground="black", rowheight=20)
    style.configure("BoldText", font=('Nirmala UI', 10, 'bold'), foreground="black")

    style.configure(
        "Treeview",
        background="#D3DEDC",
        fieldbackground="#aeae",
        foreground="#272727",
        font=('Nirmala UI', 10, "bold"),
        rowheight=20,
        height=1,
        borderwidth=0,
        relief="flat",
    )
    style.configure(
        "Treeview.Heading",
        background="#aeae",
        foreground="#275427",
        borderwidth=2,
        font=('Tahoma', 11, "bold"),
        relief="flat",
    )
    style.map(
        "Treeview.Heading",
        background=[("active", "#aeae"), ("selected", "#aeae")],
        foreground=[("active", "#275427"), ("selected", "#275427")],
    )

    # Create a treeview to display the data
    tree = ttk.Treeview(popup, columns=("Date", "Symbol", "Rule2", "Rule3", "Rule4", "Rule5", "Rule6", "Rule7", "Rule8", "Rule9", "Rule10"), show="headings", height=10)
    
    # Set column widths and alignment
    tree.column("Date", width=100, anchor='center')
    tree.column("Symbol", width=100, anchor='center')
    tree.column("Rule2", width=100, anchor='center')
    tree.column("Rule3", width=100, anchor='center')
    tree.column("Rule4", width=100, anchor='center')
    tree.column("Rule5", width=100, anchor='center')
    tree.column("Rule6", width=100, anchor='center')
    tree.column("Rule7", width=100, anchor='center')
    tree.column("Rule8", width=100, anchor='center')
    tree.column("Rule9", width=100, anchor='center')
    tree.column("Rule10", width=100, anchor='center')

    tree.heading("Date", text="Date")
    tree.heading("Symbol", text="Symbol")
    tree.heading("Rule2", text="Rule 2")
    tree.heading("Rule3", text="Rule 3")
    tree.heading("Rule4", text="Rule 4")
    tree.heading("Rule5", text="Rule 5")
    tree.heading("Rule6", text="Rule 6")
    tree.heading("Rule7", text="SaMornStar")
    tree.heading("Rule8", text="Ma54xOver")
    tree.heading("Rule9", text="BBpctBxOver")
    tree.heading("Rule10", text="PerfectBottom")
    
    # Fetch data from the database
    days_ago = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
    
    query = """
    SELECT date, symbol, 
            BOOL_OR(traderule2) AS traderule2, 
            BOOL_OR(traderule3) AS traderule3, 
            BOOL_OR(traderule4) AS traderule4,
            BOOL_OR(traderule5) AS traderule5,
            BOOL_OR(traderule6) AS traderule6,
            BOOL_OR(traderule7) AS traderule7,
            BOOL_OR(traderule8) AS traderule8,
            BOOL_OR(traderule9) AS traderule9,
            BOOL_OR(traderule10) AS traderule10

    FROM daily_prices
    WHERE date >= %s AND (traderule2 OR traderule3 OR traderule4 OR traderule5 OR traderule6 OR traderule7 OR traderule8 OR traderule9 OR traderule10)
    GROUP BY date, symbol
    """
    cur.execute(query, (days_ago,))
    # results = cur.fetchall()
    results = sorted(cur.fetchall(), reverse=True)
    
    # Define a tag for highlighting
    tree.tag_configure('highlight', background="#fDf3fD")

    # Populate the treeview
    for row in results:
        date, symbol, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10 = row
            # Convert symbol to sentence case
        # symbol_sentence_case = symbol.capitalize() #works
        values = (date, symbol, "✅" if rule2 else " ", "✅" if rule3 else " ", "✅" if rule4 else " ", "✅" if rule5 else " ", "✅" if rule6 else " ", "✅" if rule7 else " ", "✅" if rule8 else " ", "✅" if rule9 else " ", "✅" if rule10 else " ")
        item = tree.insert("", "end", values=values)
        # # Check if any rule is True and apply the highlight tag to the entire row if so
        if any(values[2:]):  # Assuming the first two columns are date and symbol, and the rest are boolean rules
            tree.item(item, tags=('highlight',))
                    
            # Apply the 'BoldText' tag to cells that are "TRUE"
        if rule2:
            tree.item(item, tags=('BoldText', item))  # Apply to the third column (Rule2)
        if rule3:
            tree.item(item, tags=('BoldText', item))  # Apply to the fourth column (Rule3)
        if rule4:
            tree.item(item, tags=('BoldText', item))  # Apply to the fifth column (Rule4)
        if rule5:
            tree.item(item, tags=('BoldText', item))  # Apply to the sixth column (Rule5)
        if rule6:
            tree.item(item, tags=('BoldText', item))  # Apply to the sixth column (Rule6)
        if rule7:
            tree.item(item, tags=('BoldText', item))  # Apply to the sixth column (Rule7)
        if rule8:
            tree.item(item, tags=('BoldText', item))  # Apply to the sixth column (Rule8)
        if rule9:
            tree.item(item, tags=('BoldText', item))  # Apply to the sixth column (Rule8)
        if rule10:
            tree.item(item, tags=('BoldText', item))  # Apply to the sixth column (Rule8)

        tree.pack(expand=True, fill="both", padx=10, pady=10)
        # Bind double-click event
        tree.bind("<Double-1>", lambda event: on_tree_double_click(tree))
    popup.mainloop()
    
def on_tree_double_click(tree):
        selected_item = tree.selection()
        if selected_item:
            item_values = tree.item(selected_item[0], "values")
            symbol = item_values[1]  # Get the symbol from the second column
            selected_period = period_var.get()  # Assuming you have a period_var
            selected_timeframe = timeframe_var.get()  # Assuming you have a timeframe_var
            
            # Clear the current selection in the watchlist
            watchlist.selection_clear(0, tk.END)

            # Find the index of the symbol in the watchlist
            index = 0
            while index < watchlist.size():
                if watchlist.get(index) == symbol:
                    # Select the symbol in the watchlist
                    watchlist.selection_set(index)
                    break
                index += 1
            

            if selected_timeframe == "Daily":
                plot_daily_chart(symbol, selected_period)
            elif selected_timeframe == "Weekly":
                plot_weekly_chart(symbol, selected_period)
            elif selected_timeframe == "Monthly":
                plot_monthly_chart(symbol, selected_period)

############################## For Traders Dairy ############################################
################## Traders Dairy function ###############


# ... (Your existing imports and database connection) ...

def create_traders_diary_notes_popup():
    """Opens a popup for managing trader's notes for a selected stock."""

    try:
        selected_symbol = watchlist.get(watchlist.curselection())
    except tk.TclError:
        messagebox.showwarning("No Symbol Selected", 
                               "Please select a symbol from the watchlist first.")
        return

    # Fetch existing notes for selected symbol from the database
    cur.execute("SELECT date, notes, symbol FROM daily_prices WHERE notes IS NOT NULL AND symbol = %s", (selected_symbol,))
    notes_data = cur.fetchall()
    
    # Create the popup window
    traders_diary_notes_create_popup = tk.Toplevel(root)
    traders_diary_notes_create_popup.title("Traders Diary")
    traders_diary_notes_create_popup.geometry("+%d+%d" % (root.winfo_screenwidth() - 400, 0))

    # Symbol Display
    selected_symbol = watchlist.get(watchlist.curselection())
    symbol_label =  tk.Label(traders_diary_notes_create_popup, text="Symbol:")
    symbol_label.pack()
    symbol_entry =  tk.Entry(traders_diary_notes_create_popup)
    symbol_entry.insert(0, selected_symbol)
    symbol_entry.config(state='readonly')
    symbol_entry.pack()

    # Fetch the latest date from daily_prices
    cur.execute("SELECT MAX(date) FROM daily_prices WHERE symbol = %s", (selected_symbol,))
    last_chart_date_result = cur.fetchone()
    last_chart_date = last_chart_date_result[0].strftime('%Y-%m-%d') if last_chart_date_result[0] else datetime.now().strftime('%Y-%m-%d')

    # Date Display (Last chart date)
    date_label =  tk.Label(traders_diary_notes_create_popup, text="Date:")
    date_label.pack()
    date_entry =  tk.Entry(traders_diary_notes_create_popup)
    date_entry.insert(0, last_chart_date)
    date_entry.config(state='readonly')
    date_entry.pack()

    # Notes Input Area
    notes_label =  tk.Label(traders_diary_notes_create_popup, text="Notes:")
    notes_label.pack()
    notes_text =  tk.Text(traders_diary_notes_create_popup, height=5, width=30)
    notes_text.pack()

    def save_notes():
        """Saves the current notes to the database."""
        notes = notes_text.get("1.0", tk.END).strip()
        if notes:
            try:
                cur.execute("""
                    UPDATE daily_prices 
                    SET notes = %s 
                    WHERE symbol = %s AND date = %s
                """, (notes, selected_symbol, last_chart_date))
                conn.commit()
                messagebox.showinfo("Success", "Notes saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", str(e))
        else:
            messagebox.showwarning("Empty Notes", "Please enter some notes before saving.")

    # --- Buttons ---
    save_button =  tk.Button(traders_diary_notes_create_popup, text="Save", command=save_notes)
    save_button.pack()
    traders_diary_notes_create_popup.mainloop()

    
def view_traders_diary_notes_popup():
        # Fetch existing notes for selected symbol from the database
        cur.execute("SELECT date, symbol, notes  FROM daily_prices WHERE notes IS NOT NULL ORDER BY date DESC")
        diary_notes_data = cur.fetchall()
        # print(diary_notes_data)

        """Opens a new window to display all previous notes."""
        view_notes =  tk.Toplevel(root)
        view_notes.title("All Previous Notes for all Symbols")
        view_notes.geometry("+%d+%d" % (root.winfo_screenwidth() - 800, 0))


        # Create a Treeview to display the notes
        tree = ttk.Treeview(view_notes, columns=("Date", "Symbol", "Notes"), 
                           show="headings", height=10)

        # Configure Treeview columns
        tree.column("Date", width=100, anchor='center')
        tree.column("Symbol", width=100, anchor='center')
        tree.column("Notes", width=300, anchor='w')  # Adjusted width for notes

        tree.heading("Date", text="Date")
        tree.heading("Symbol", text="Symbol")
        tree.heading("Notes", text="Notes")

        tree.pack(expand=True, fill="both", padx=10, pady=10)

        # Populate the Treeview with notes data
        for date, symbol, notes in diary_notes_data:
            tree.insert("", "end", values=(date, symbol, notes))

        # Add a delete button to the Treeview
        delete_button =  tk.Button(view_notes, text="Delete Note", 
                                 command=lambda: delete_note(tree.item(tree.selection())['values'][0],
                                                             tree.item(tree.selection())['values'][1]) 
                                 if tree.selection() else None)
        delete_button.pack()


        def delete_note(date, symbol):
            """Deletes a note from the database."""
            try:
                cur.execute("""
                    UPDATE daily_prices 
                    SET notes = NULL 
                    WHERE date = %s AND symbol = %s
                """, (date, symbol))
                conn.commit()
                messagebox.showinfo("Success", "Note deleted successfully!")
                view_traders_diary_notes_popup()  # Refresh the view
            except Exception as e:
                messagebox.showerror("Error", str(e))

    
        view_button =  tk.Button(view_notes, text="View Previous Notes", 
                            command=view_traders_diary_notes_popup)
        view_button.pack()
        
        tree.bind("<Double-1>", lambda event: on_tree_double_click(tree))
        
        view_notes.mainloop()
        
        
#########################################################################################################################################
def resize_figure(event):
    # Get the dimensions of the main_frame
    width = main_frame.winfo_width()
    height = main_frame.winfo_height()
    
    # Convert dimensions to inches (assuming 100 dpi)
    width_in = width / 100
    height_in = height / 100
    
    # Update the figure size
    fig.set_size_inches(width_in, height_in)

########################################################################################################################################################################    
BG_COLOR = '#162636'
# Create the main window
root = tk.Tk()
root.option_add("*Font", "Ebrima 8")
root.wm_state('zoomed')

style = ttk.Style()
style.theme_use('clam')

# root =  tk.Window(themename="darkly")
root.title("Subhantech Stock Watchlist")
root.configure(background=BG_COLOR)
root.geometry("1900x1000")


def set_title_bar_color(root):
    root.update()
    DWMWA_CAPTION_COLOR = 35
    hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
    color = 0x002B2B2B  # Replace with your desired color in hex format
    ctypes.windll.dwmapi.DwmSetWindowAttribute(
        hwnd, 
        DWMWA_CAPTION_COLOR,
        ctypes.byref(ctypes.c_int(color)),
        ctypes.sizeof(ctypes.c_int)
    )

# Call this function after creating your root window
set_title_bar_color(root)


# text_label = Label(root, text="You hold the power to make your life what you want...!", font=("Arial", 10))
# text_label.pack()  # or use grid() or place() for more precise positioning


def load_quotes():
    with open('quotes.txt', 'r', encoding='utf-8') as file:
        return [html.escape(line.strip()) for line in file if line.strip()]

def update_quote():
    quotes = load_quotes()
    random_quote = random.choice(quotes)
    text_label.config(text=random_quote)
    root.after(30000, update_quote)  # 30000 milliseconds = 30 seconds

# Create the Label
text_label = Label(root, text="", font=("Nirmala UI", 10), wraplength=1000)
text_label.pack()

# Start updating quotes
update_quote()

# # Create a sidebar frame    
# sidebar = tk.Frame(root, width=300)
# sidebar.pack(expand=False, fill='y', side='left', anchor='nw', ipadx=5, padx=5)
# sidebar.pack_propagate(False)


# Create the sidebar
sidebar = tk.Frame(root, width=250, background=BG_COLOR)
sidebar.pack(expand=False, side='left', fill='y', anchor='nw', ipadx=5, padx=5)
sidebar.pack_propagate(False)

# Create the watchlist frame with a fixed height
watchlist_frame = tk.Frame(sidebar, width=180, height=400, background=BG_COLOR)  # Adjust height as needed
watchlist_frame.pack(side='top', fill='y', anchor='nw',expand=False)
watchlist_frame.pack_propagate(False)

# Create a listbox for the watchlist
watchlist = tk.Listbox(watchlist_frame, fg="black", font=("Nirmala UI", 10), bd=0, borderwidth=0, relief=tk.FLAT, selectborderwidth=2, selectforeground="white", selectbackground=BG_COLOR)
watchlist.pack(side='left', expand=True, fill='y', anchor='nw', padx=5, pady=1)


# Add a scrollbar to the watchlist frame
scrollbar = tk.Scrollbar(watchlist_frame, command=watchlist.yview)
scrollbar.pack(side='right', fill='y')

# Configure the listbox to use the scrollbar
watchlist.config(yscrollcommand=scrollbar.set)

# # Apply a custom style to the Listbox
# style = ttk.Style()
# style.theme_use("clam")  # or any other theme you prefer
# style.configure("TListbox", background="dark red", foreground="white", font=("Arial", 10))


# Create the sort frame
sort_frame = tk.Frame(sidebar, width=180, background=BG_COLOR)
sort_frame.pack(side='top', fill='x', anchor='nw')

def sort_watchlist(reverse=False):
    # Get the current watchlist items
    watchlist_items = list(watchlist.get(0, tk.END))

    # Sort the watchlist items
    watchlist_items.sort(reverse=reverse)

    # Clear the current watchlist
    watchlist.delete(0, tk.END)

    # Insert the sorted watchlist items
    for item in watchlist_items:
        watchlist.insert(tk.END, item)

# Create the sort buttons
sort_button = tk.Button(sort_frame, text="▲", width=4, command=lambda: sort_watchlist())
sort_button.pack(side=tk.LEFT, pady=10, padx=5)

reverse_sort_button = tk.Button(sort_frame, text="▼", width=4, command=lambda: sort_watchlist(reverse=True))
reverse_sort_button.pack(side=tk.LEFT, pady=10, padx=5)

#######################################################################
##########################################################################
# frame2 = tk.Frame(top_container, width=120, height=500, bg='blue')
# frame2.pack(side='left')

# Create the third frame below
sidebar_bottom_frame =  tk.Frame(sidebar, width=180, height=800, background=BG_COLOR)
sidebar_bottom_frame.pack(side='top', anchor='nw')

def search_stocks(event=None):
    search_query = search_entry.get().lower()
    watchlist.selection_clear(0, tk.END)  # Clear previous selection

    for i in range(watchlist.size()):
        symbol = watchlist.get(i).lower()
        if search_query in symbol:
            watchlist.selection_set(i)  # Highlight matching items
        else:
            watchlist.selection_clear(i)  # Unhighlight non-matching items


stock_search_section =  tk.Frame(sidebar_bottom_frame, background=BG_COLOR)
stock_search_section.pack(fill=tk.Y, anchor='nw')

# Create a search entry and button
search_entry =  tk.Entry(stock_search_section, width=12)
search_entry.pack(side=tk.LEFT, pady=10, padx=5)
search_button =  tk.Button(stock_search_section, text="Search", command=search_stocks)
search_button.pack(pady=10, padx=5, side=tk.LEFT)
# Bind the Enter key to the search_watchlist method
search_entry.bind("<Return>", search_stocks)

###########################################################

# Add items to the watchlist from the database
cur.execute("SELECT symbol FROM tickers")
symbols = cur.fetchall()
# Sort the symbols in ascending order
symbols.sort()
for symbol in symbols:
    watchlist.insert(tk.END, symbol[0])

# Bind the listbox selection event to the plot function
watchlist.bind('<<ListboxSelect>>', on_select)



############################################################
stock_entry_section =  tk.Frame(sidebar_bottom_frame, background=BG_COLOR)
stock_entry_section.pack(fill=tk.Y, side=tk.TOP, anchor='nw')

stock_entry =  tk.Entry(stock_entry_section, width=12)
stock_entry.pack(side=tk.LEFT, padx=5, pady=5)

add_button =  tk.Button(stock_entry_section, text="Add Stock", command=add_stock)
add_button.pack(padx=5, pady=5, side=tk.RIGHT)

delete_button =  tk.Button(stock_entry_section, text="Delete Stock", command=delete_stock)
delete_button.pack(padx=5,pady=5, side=tk.RIGHT)

###################################################
stock_other_section =  tk.Frame(sidebar_bottom_frame, background=BG_COLOR)
stock_other_section.pack(fill=tk.Y, side=tk.TOP)

update_all_rules_button =  tk.Button(stock_other_section, text="Update Screeners", command=update_Screeners)
update_all_rules_button.pack(side=tk.RIGHT, padx=10, pady=5, anchor='nw')

add_fromfile_button =  tk.Button(stock_other_section, text="Add Chartink Stocks", command=add_chartink_stock)
add_fromfile_button.pack(padx=10,pady=5, side=tk.LEFT)

##########################################################
download_data_section =  tk.Frame(sidebar_bottom_frame, background=BG_COLOR)
download_data_section.pack(fill=tk.Y, side=tk.TOP)

# Add start date and end date date pickers
date_pickers_frame =  tk.Frame(download_data_section, width=24, background=BG_COLOR)
date_pickers_frame.pack(side=tk.TOP, padx=10, pady=5, fill=tk.X)

start_date_label =  tk.Label(date_pickers_frame, text="SD:")
start_date_label.pack(side=tk.LEFT, padx=1, pady=5)

start_date_picker =    DateEntry(date_pickers_frame, width=12)
start_date_picker.pack(side=tk.LEFT, padx=1, pady=5)

end_date_label =  tk.Label(date_pickers_frame, text="ED:")
end_date_label.pack(side=tk.LEFT, padx=1, pady=5)

end_date_picker =    DateEntry(date_pickers_frame, width=12)
end_date_picker.pack(side=tk.LEFT, padx=1, pady=5)

# Update the Download HistData button to use the start and end dates
download_histdata_button =  tk.Button(download_data_section, text="Download HistData", command=lambda: download_histdata(start_date_picker.get_date(), end_date_picker.get_date()))
download_histdata_button.pack(side=tk.RIGHT, padx=10, pady=5)


data_update_section =  tk.Frame(sidebar_bottom_frame, background=BG_COLOR)
data_update_section.pack(side=tk.TOP,fill=tk.Y)

up2date_chart_button =  tk.Button(data_update_section, text=" Update RSI, MA, and Traderules ", command=up2date_chart)
up2date_chart_button.pack(side=tk.LEFT, padx=10, pady=5)

progress_bar_section =  tk.Frame(sidebar_bottom_frame, background=BG_COLOR)
progress_bar_section.pack(side=tk.TOP,fill=tk.Y)

progress_bar =  ttk.Progressbar(progress_bar_section, orient="horizontal", length=200, mode="determinate")
progress_bar.pack(side=tk.LEFT, padx=10, pady=5)

# # Prevent the frames from shrinking
# top_container.pack_propagate(False)
# watchlist_frame.pack_propagate(False)
# # frame2.pack_propagate(False)
# sidebar_bottom_frame.pack_propagate(False)





main_frame =  tk.Frame(root, background=BG_COLOR)
main_frame.pack(side=tk.LEFT, fill='both', expand=True)

# Create a frame to simulate the embossed shadow
embossed_frame =  tk.Frame(main_frame, background=BG_COLOR)  # Slightly darker background
embossed_frame.pack(padx=3, pady=3, expand=True, fill='both')  # Add padding for the effect


timeframe_section =  tk.Frame(main_frame, background=BG_COLOR)
timeframe_section.pack(side=tk.TOP,fill=tk.X)

period_var =  tk.StringVar(value='1y')
period_label =  tk.Label(timeframe_section, text="Select Period:")
period_label.pack(side=tk.LEFT, padx=5)

period_options = ['6mo', '1d', '5d', '1mo', '3mo', '6mo', '9mo', '1y', '2y', '3y', '4y', '5y', '6y','7y', '8y', '9y', '10y', 'max']
period_menu = ttk.OptionMenu(timeframe_section, period_var, *period_options)
period_menu.pack(side=tk.LEFT, padx=1)
period_menu.config(width=5)
# Bind to period_var changes
# With this:
period_var.trace_add("write", lambda *args: on_select(None))

# Add TimeFrame selection
timeframe_var =  tk.StringVar(value='Daily')
timeframe_label =  tk.Label(timeframe_section, text="TF:")
timeframe_label.pack(side=tk.LEFT, padx=5)

# timeframe_options = ['Daily', 'Weekly', 'Monthly']
timeframe_options = {
    '𝐃': 'Daily',
    '𝐖': 'Weekly',
    '𝐌': 'Monthly',
}

for option, value in timeframe_options.items():
    button =  tk.Button(timeframe_section, text=option, command=lambda value=value: [timeframe_var.set(value), on_select(None)])
    button.pack(side=tk.LEFT, padx=5)
    
##################################################################################
# Create an IntVar to hold the value
triangle_length_var = tk.IntVar(value=12)

# Create the Entry widget and link it to the IntVar
triangle_length = tk.Entry(timeframe_section, width=3, textvariable=triangle_length_var)
triangle_length.pack(side=tk.LEFT, padx=5, pady=5)

# Function to retrieve and use the value
def use_triangle_length():
    value = triangle_length_var.get()
    # print(f"Triangle Length: {value}")
    # Use the value in another place
    # For example, you can pass it to another function or use it in calculations

# Button to trigger the function
use_button = tk.Button(timeframe_section, text="Triangle Size", command=use_triangle_length)
use_button.pack(side=tk.LEFT)

  
update_3days_data_button =  tk.Button(timeframe_section, text="Price Up2date", command=lambda: download_last_3_days_histdata())
update_3days_data_button.pack(side=tk.RIGHT, padx=10, pady=5)

detect_cspatterns_button =  tk.Button(timeframe_section, text="Candle patterns", command=lambda: detect_cspatterns())
detect_cspatterns_button.pack(side=tk.RIGHT, padx=10, pady=5)

show_signals_button =  tk.Button(timeframe_section, text="Show Signals", command=show_recent_signals)
show_signals_button.pack(side=tk.RIGHT, padx=10, pady=5)

update_everything_button =  tk.Button(timeframe_section, text="Update Everything", command=lambda: asyncio.run(update_everything()))
update_everything_button.pack(side=tk.RIGHT, padx=10, pady=5)


# Place the canvas inside the embossed frame
canvas = FigureCanvasTkAgg(fig, master=embossed_frame)

canvas.get_tk_widget().config(highlightthickness=5, highlightbackground="#1f3b4d" )
canvas.get_tk_widget().pack(expand=True, fill='both')


def on_closing():
    root.quit()
    root.destroy()

button = tk.Button(master=root, text="Quit", command=on_closing)
button.pack(side=tk.BOTTOM)

# horizonScrollbar =  tk.Scrollbar(main_frame, orient='horizontal')

# Create a right sidebar frame
right_sidebar =  tk.Frame(root, width=100, background=BG_COLOR)
right_sidebar.pack(side=tk.RIGHT, fill='y')

#################################### Traders Section ########################################################

# traders_diary_section =  tk.Frame(right_sidebar)
# traders_diary_section.pack(fill=tk.Y)

# Create a button to play the audio
play_audio_button =  tk.Button(right_sidebar, text="Anxiety", command=play_audio)
play_audio_button.pack(pady=10, padx=5)


diary_create_notes_button =  tk.Button(right_sidebar, text="Create Notes",   command=create_traders_diary_notes_popup)
diary_create_notes_button.pack(pady=10, padx=5,side=tk.TOP)

diary_view_notes_button =  tk.Button(right_sidebar, text="View Notes", command=view_traders_diary_notes_popup)
diary_view_notes_button.pack(pady=10, padx=5,side=tk.TOP)

################################### Progress Bar Section ########################################################


# Add navigation toolbar for scrolling and zooming
# toolbar = NavigationToolbar2Tk(canvas, main_frame)
# toolbar.update()
# canvas.get_tk_widget().pack(expand=True, fill='both')

# Start the Tkinter main loop
root.mainloop()

# Close the database connection when the application is closed
conn.close()

