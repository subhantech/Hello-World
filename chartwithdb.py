import pandas as pd 
import os
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

import tkinter as tk
from tkinter import ttk
from tkinter import ttk, messagebox, filedialog, Listbox, Label, Message
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import matplotlib.cbook as cbook
import matplotlib.dates as mdates
import yfinance as yf
import psycopg2
import mplfinance as mpf
from ta.momentum import RSIIndicator
import mplcursors
import numpy as np
from datetime import datetime, timedelta
from tkcalendar import Calendar, DateEntry
import threading
from joblib import Parallel, delayed ## need to work on this for speed processing.
import asyncio
import json
import redis
import pygame
import random
import html
import subprocess
from detectCSpatterns import get_filtered_patterns
# import ttkbootstrap as  tk
# from ttkbootstrap.constants import *
#################################################################


##################################################################
# Initialize the mixer pygame mixer to play audio on my site.
pygame.mixer.init()

def play_audio():
    # Load the MP3 file
    pygame.mixer.music.load("audio/tradefearless.mp3")
    
    # Play the audio
    pygame.mixer.music.play()
    
### Dont download data with yahoo finance always.. got blocked by that.. use the button to download only missing data. download historical data should have more options
# like startdate and enddate to download missing data.


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
ADD COLUMN IF NOT EXISTS rsi3 FLOAT;
ALTER TABLE daily_prices
ADD COLUMN  IF NOT EXISTS ma7 FLOAT,
ADD COLUMN  IF NOT EXISTS ma21 FLOAT,
ADD COLUMN  IF NOT EXISTS ma63 FLOAT,
ADD COLUMN  IF NOT EXISTS ma189 FLOAT,
ADD COLUMN  IF NOT EXISTS mamix14 FLOAT,
ADD COLUMN  IF NOT EXISTS mamix42 FLOAT,
ADD COLUMN  IF NOT EXISTS vwma25 FLOAT,
ADD COLUMN  IF NOT EXISTS volsma5 FLOAT,
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


async def calculate_and_store_rsi(symbol):
    """Calculates RSI(3) for the given symbol and stores it in the database."""

    # Fetch the last calculated RSI date
    cur.execute("SELECT MAX(date) FROM daily_prices WHERE symbol = %s AND rsi3 IS NOT NULL", (symbol,))
    # last_rsi_date = cur.fetchone()[0]
    end_date = datetime.now().date()
    last_rsi_date = end_date - timedelta(days=140)
    

    # Fetch only new data
    cur.execute("SELECT date, close FROM daily_prices WHERE symbol = %s AND date > %s ORDER BY date", (symbol, last_rsi_date or '1900-01-01'))
    rows = cur.fetchall()

    if len(rows) < 3:
        print(f"Not enough new data to calculate RSI(3) for {symbol}")
        return

    df = pd.DataFrame(rows, columns=['date', 'close'])
    
    # Calculate RSI
    rsi_indicator = RSIIndicator(close=df['close'], window=3)
    df['rsi3'] = rsi_indicator.rsi()

    # Prepare batch updates
    updates = [(row['rsi3'], symbol, row['date']) for _, row in df.iterrows() if not pd.isna(row['rsi3'])]

    # Perform bulk update
    cur.executemany(
        "UPDATE daily_prices SET rsi3 = %s WHERE symbol = %s AND date = %s",
        updates
    )
    conn.commit()

    print(f"RSI(3) calculated and stored for {symbol}")
    

async def calculate_and_store_ma(symbol):
    # Initialize Redis connection
    redis_client =  redis.Redis(host='localhost', port=6379, db=0)

    # Get last update timestamp
    last_update = await get_last_update(symbol)

    # Fetch only new data
    cur.execute("""
        SELECT date, close, volume, ma7, ma21, ma63, ma189, mamix14, mamix42, vwma25, volsma5
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
    for ma in [7, 21, 63, 189]:
        df[f'ma{ma}'] = df['close'].rolling(window=ma).mean()
    
    df['mamix14'] = ((df['ma7'] + df['ma21']) / 2).rolling(window=2).mean()
    df['mamix42'] = ((df['ma21'] + df['ma63']) / 2).rolling(window=2).mean()
    df['CloseVolume'] = df['close'] * df['volume']
    df['vwma25'] = df['CloseVolume'].rolling(window=25).sum() / df['volume'].rolling(window=25).sum()
    df['volsma5'] = (df['volume']).rolling(window=5).mean()
    
    # Prepare batch updates
    updates = []
    for date, row in df.iterrows():
        updates.append((
            float(row['ma7']), 
            float(row['ma21']), 
            float(row['ma63']), 
            float(row['ma189']),
            float(row['mamix14']), 
            float(row['mamix42']), 
            float(row['vwma25']),
            float(row['volsma5']),
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
            ma21 = COALESCE(%s, ma21), 
            ma63 = COALESCE(%s, ma63), 
            ma189 = COALESCE(%s, ma189), 
            mamix14 = COALESCE(%s, mamix14), 
            mamix42 = COALESCE(%s, mamix42), 
            vwma25 = COALESCE(%s, vwma25),
            volsma5 = COALESCE(%s, volsma5)
        WHERE symbol = %s AND date = %s
    """, updates)
    conn.commit()

# Run the function asynchronously
# asyncio.run(calculate_and_store_ma(symbol))


def plot_daily_chart(symbol, selected_period):
    
    # Convert period to a format suitable for SQL query
    period_map = {
        '1d': '1 day',
        '5d': '5 days',
        '1mo': '1 month',
        '3mo': '3 months',
        '6mo': '6 months',
        '1y': '1 year',
        '2y': '2 years',
        '3y': '3 years',
        '4y': '4 years',
        '5y': '5 years',
        '6y': '6 years',
        '7y': '7 years',
        '8y': '8 years',
        '9y': '9 years',
        '10y': '10 years',
        'max': '100 years'  # Assuming max means as much data as possible
    }
    sql_period = period_map.get(selected_period)
    
    

    cur.execute("SELECT date, open, high, low, close, volume, rsi3,  ma7, ma21, ma63, ma189, mamix14, mamix42, vwma25, traderule1, traderule2, traderule3, traderule4, traderule5, traderule6, volsma5  FROM daily_prices WHERE symbol = %s AND date >= NOW() - INTERVAL %s ORDER BY date", (symbol, sql_period))
    
    rows = cur.fetchall()
    if not rows:
        fetch_data_from_db(symbol, selected_period)
        cur.execute("SELECT date, open, high, low, close, volume, rsi3,  ma7, ma21, ma63, ma189, mamix14, mamix42, vwma25, traderule1, traderule2, traderule3, traderule4, traderule5, traderule6, volsma5  FROM daily_prices WHERE symbol = %s AND date >= NOW() - INTERVAL %s ORDER BY date", (symbol, sql_period))
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
        'volsma5':[row[20] for row in rows]
        
        
        
    }
    
    
    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)
    df.index = df.index.astype(str)
    df.index = pd.to_datetime(df.index)  # Ensure the index is a DatetimeIndex
        
    if not df.empty:
        latest_data = df.iloc[-1]
        prev_close = df.iloc[-2]['Close'] if len(df) > 1 else 0  # Handle the case where there's no previous data
        percent_change = ((latest_data['Close'] - prev_close) / prev_close) * 100 if prev_close != 0 else 0  # Avoid division by zero
        latest_info = (
            f"Date: {latest_data.name.strftime('%b %d %Y')}, "
            f"Open: {latest_data['Open']:.2f}, "
            f"High: {latest_data['High']:.2f}, "
            f"Low: {latest_data['Low']:.2f}, "
            f"Close: {latest_data['Close']:.2f}, "
            f"RSI3: {latest_data['rsi3']:.2f}, "
            f"Chg: {percent_change:.2f}%"
        )
    else:
        latest_info = "No data available"
    
    #these lines are highlighting the Low values in your DataFrame where each respective trading rule is met, 
    # and setting the value to NaN where the rule is not met.
    ### to plot on low of the candle .. we are collecting lows
    df['traderule1_highlight'] = np.where(df['traderule1'], df['Low'] * 0.999, np.nan)
    df['traderule2_highlight'] = np.where(df['traderule2'], df['Low'] * 0.999, np.nan)
    df['traderule3_highlight'] = np.where(df['traderule3'], df['Low'] * 0.999, np.nan)
    df['traderule4_highlight'] = np.where(df['traderule4'], df['Low'] * 0.999, np.nan)
    df['traderule5_highlight'] = np.where(df['traderule5'], df['Low'] * 0.999, np.nan)
    df['traderule6_highlight'] = np.where(df['traderule6'], df['Low'] * 0.999, np.nan)
    
    # df['traderule3_highlight'] = np.where(df['traderule3'], df['Low'] * 1.05, np.nan) # to add 5%
    # df['traderule3_highlight'] = np.where(df['traderule3'], df['Low'] * 0.95, np.nan) # to minus 5%
    
    df['traderule3_high'] = np.where(df['traderule3'], df['High'], np.nan) # to plot a line taking high
    df['traderule3_low'] = np.where(df['traderule3'], df['Low'], np.nan) # to plot a line taking high
    df['traderule4_high'] = np.where(df['traderule4'], df['High'], np.nan) # to plot a line taking high
    df['traderule4_low'] = np.where(df['traderule4'], df['Low'], np.nan) # to plot a line taking high
    df['traderule5_high'] = np.where(df['traderule5'], df['High'], np.nan) # to plot a line taking high
    df['traderule5_low'] = np.where(df['traderule5'], df['Low'], np.nan) # to plot a line taking high
    
    
    traderule3_high = df['traderule3_high'].dropna()
    traderule3_low = df['traderule3_low'].dropna()
    traderule4_high = df['traderule4_high'].dropna()
    traderule4_low = df['traderule4_low'].dropna()
    traderule5_high = df['traderule5_high'].dropna()
    traderule5_low = df['traderule5_low'].dropna()
    
    # I am collecting the high and low values of traderules only if more than one rule is met. To plot lines below.
    
    df['highlight_high'] = np.where((df[['traderule1', 'traderule2', 'traderule3', 'traderule4', 'traderule5', 'traderule6']].sum(axis=1) > 1), df['High'], np.nan)
    df['highlight_low'] = np.where((df[['traderule1', 'traderule2', 'traderule3', 'traderule4', 'traderule5', 'traderule6']].sum(axis=1) > 1), df['Low'], np.nan)
    
    highlight_high_values = df['highlight_high'].dropna()
    highlight_low_values = df['highlight_low'].dropna()
    
    # if not traderule3_high.empty:
    #     addplot = []
    #     for value in traderule3_high:
    #         addplot.append(mpf.make_addplot([value]*len(df), ax=ax1, color='#7572f1', linestyle='--'))
    
    fig.clear()
    
    # Adjust the layout to stretch the chart
    fig.tight_layout()
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    # Create the subplots
    ax1 = fig.add_subplot(4, 1, 1)
    ax2 = fig.add_subplot(4, 1, 2, sharex=ax1)
    ax3 = fig.add_subplot(4, 1, 3, sharex=ax1)
    ax4 = fig.add_subplot(4, 1, 4, sharex=ax1)
    
    
    
    ax1.set_facecolor('#000000ff') 
    ax2.set_facecolor('#000000ff')
    ax3.set_facecolor('#000000ff')
    ax4.set_facecolor('#000000ff')
    
    
    addplot = [
        # mpf.make_addplot(df['ma7'], ax=ax1, color='red'),
        # mpf.make_addplot(df['ma21'], ax=ax1, color='black'),
        # mpf.make_addplot(df['ma63'], ax=ax1, color='green'),
        mpf.make_addplot(df['traderule1_highlight'], ax=ax1, type='scatter', markersize=100, marker='s', color='#bcf5bc'), # working code
        mpf.make_addplot(df['traderule2_highlight'], ax=ax1,  type='scatter', markersize=100, marker='^', color='#1e90ff'), # working code
        mpf.make_addplot(df['traderule3_highlight'], ax=ax1,  type='scatter', markersize=100, marker='s', color='#7572f1'), # working code
        mpf.make_addplot(df['traderule4_highlight'], ax=ax1, type='scatter', markersize=100, marker='d', color='#ff00ff'),  # New color for traderule4
        mpf.make_addplot(df['traderule5_highlight'], ax=ax1, type='scatter', markersize=100, marker='^', color='#067887'),  # New color for traderule4
        mpf.make_addplot(df['traderule6_highlight'], ax=ax1, type='scatter', markersize=100, marker='^', color='#007117'),  # New color for traderule4
        
        # mpf.make_addplot([horizontal_line]*len(df), ax=ax1, color='#7572f1', linestyle='--') working code for only one time occurance

    ]
    
    show_ma189 = True
    show_mamix14 = True
    show_mamix42 = True
    show_vwma25 = True
    show_volsma5 = True
    show_rsi3 = True
    
    ### RSI SEttings
    
    overbought_mask = df['rsi3'] > 60
    oversold_mask = df['rsi3'] < 40
    
    if show_ma189:
        addplot.append(mpf.make_addplot(df['ma189'], ax=ax1, color='purple', width=0.5))
    if show_mamix14:        
        addplot.append(mpf.make_addplot(df['mamix14'], ax=ax1, color='blue')),
    if show_mamix42:    
        addplot.append(mpf.make_addplot(df['mamix42'], ax=ax1, color='red', width=0.5)),
    if show_vwma25:
        addplot.append(mpf.make_addplot(df['vwma25'], ax=ax1, color='#37B7C3')),
    if show_volsma5:
        addplot.append(mpf.make_addplot(df['volsma5'], ax=ax2, color='#ff00ff')),
    if show_rsi3:
        addplot.append(mpf.make_addplot(df['rsi3'], ax=ax3, color='#1e90ff'))
        addplot.append(mpf.make_addplot(df['rsi3'].where(overbought_mask), ax=ax3, color='green', fill_between={'y1': 60, 'y2': 100, 'alpha': 0.3}))
        addplot.append(mpf.make_addplot(df['rsi3'].where(oversold_mask), ax=ax3, color='red', fill_between={'y1': 0, 'y2': 40, 'alpha': 0.3}))


        

    
    # def on_click(event):
    #     if event.inaxes == ax2:
    #         date = mdates.num2date(event.xdata).strftime('%Y-%m-%d')
    #         price = event.ydata
    #         print(f"Selected: Date = {date}, Price = {price:.2f}")

    # selected_line = ax2.axvline(color='r', linestyle='--', linewidth=1)

    # def on_click(event):
    #     if event.inaxes == ax2:
    #         date = mdates.num2date(event.xdata).strftime('%Y-%m-%d')
    #         price = event.ydata
    #         print(f"Selected: Date = {date}, Price = {price:.2f}")
    #         selected_line.set_xdata(event.xdata)
    #         fig.canvas.draw()
    # fig.canvas.mpl_connect('button_press_event', on_click)
    
    # mpf.plot(df, type='candle', style=samie_style_obj, ax=ax1, volume=ax2, datetime_format='%b %d', addplot=addplot) # working code
    mpf.plot(df, type='candle', style=samie_style_obj, ax=ax1, volume=ax2, datetime_format='%m-%d', addplot=addplot, returnfig=True)
    
    ###########/////////////////////// Fix this code ///////////////#######################
    # Define a custom tooltip function
    # def custom_tooltip(sel):
    #     # Get the x-coordinate of the selected point, which should correspond to the date
    #     x_data = sel.target.index
    #     # Convert x_data to a datetime object if necessary, depending on your DataFrame's index type
    #     selected_date = df.index[x_data]  # Adjust if your x_data is in a different format
    #     close_price = df.loc[selected_date, 'Close']
    #     # Customize the tooltip text
    #     sel.annotation.set_text(f"Date: {selected_date.strftime('%Y-%m-%d')}\nClose: {close_price:.2f}")
    #     sel.annotation.get_bbox_patch().set_facecolor("lightgray")
    #     sel.annotation.get_bbox_patch().set_alpha(0.7)
    #     sel.annotation.set_color("black")
    #     sel.annotation.set_fontsize(12)
    #     sel.annotation.get_bbox_patch().set_edgecolor("darkgray")
    #     sel.annotation.get_bbox_patch().set_linewidth(1)
    #     sel.annotation.set_zorder(1000)  # Ensure tooltip is on top
    #     sel.annotation.set_visible(True)  # Ensure the annotation is visible

    #     # Your existing code to finalize the plot

    # # Add this line after plotting to enable mplcursors
    # mplcursors.cursor(ax1, hover=True).connect("add", custom_tooltip)

    


    ax1.set_title(f' {selected_period} :  {latest_info}', loc='left',  fontsize=10, color='dodgerblue', y=1)
    # ax1.set_ylabel('Price')
    ax1.grid(axis='both', which='both', linestyle='-.', linewidth=0.3, color='#31363F')
    ax2.grid(axis='both', which='both', linestyle='-.', linewidth=0.3, color='#31363F')
    ax3.grid(axis='both', which='both', linestyle='-.', linewidth=0.3, color='#31363F')
    ax4.grid(axis='both', which='both', linestyle='-.', linewidth=0.3, color='#31363F')
    
    ax1.tick_params(axis='y', colors='dodgerblue')
    ax4.tick_params(axis='y', colors='dodgerblue')
    ax4.tick_params(axis='x', colors='dodgerblue')
    
    # Enhanced Gridlines for ax1 (Price Chart)
    # ax1.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')  # Major and minor grids for ax1
    
    data_range = df['Close'].max() - df['Close'].min()
    # tick_interval = data_range * 0.05  # 5% of the data range
    
    def round_to_nearest_five_paisa(value):
        rounded_value = round(value * 20) / 20.0
        return rounded_value

    # data_range = 1.23  # Replace with your actual data range
    # tick_interval = data_range * 0.10
    # rounded_tick_interval = round_to_nearest_five_cents(tick_interval)
    # print(rounded_tick_interval)

    tick_interval = data_range * 0.10  # 10% of the data range
    rounded_tick_interval = round_to_nearest_five_paisa(tick_interval)
    
    # Set y-axis major ticks at every 5% of the data range
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(rounded_tick_interval))
    # ax1.set_ylim([df['Low'].min(), df['High'].max()])  # working code.. sets you graph to max and min values
    
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
    #         ax1.text(df.index[-1], value * 0.98, f'{value:.2f}', va='center', ha='left', 
    #             bbox=dict(facecolor='#0042a1', edgecolor='black', alpha=0.7, boxstyle='round,pad=0.5'))
                
    # if not traderule4_high.empty:
    #     for value in traderule4_high:
    #         ax1.axhline(y=value, color='#abcddcba', linestyle='--')
    
    # if not traderule4_low.empty:
    #     # for value in traderule4_low:                                              ### working code
    #     #     ax1.axhline(y=value * 0.95, color='#fabfab', linestyle='--')          ### working code
    #     for value  in traderule4_low:
    #         ax1.axhline(y=value * 0.98, color='#fabcfabc', linestyle='--')
    #         ax1.text(df.index[-1], value * 0.98, f'{value:.2f}', va='center', ha='left', 
    #             bbox=dict(facecolor='#7572f1', edgecolor='black', alpha=0.7))
                

    # ax1.set_yscale('log')   
    # ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:.2f}'))
    # ax1.yaxis.set_major_formatter(ticker.FuncFormatter(price_formatter))
    ############################################################################################
    
    if not highlight_high_values.empty:
        for value in highlight_high_values:
            ax1.axhline(y=value, color='#ff000f', linestyle='--', linewidth=1)	
    
    if not highlight_low_values.empty:
        for value in highlight_low_values:
            ax1.axhline(y=value * 0.95, color='#0042a1', linestyle='--', linewidth=1)          ### working code
    
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
    data_range = df['Close'].max() - df['Close'].min()
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
    ymin, ymax = df['Low'].min() * 0.98, df['High'].max() * 1.02 # Adjust for your DataFrame's columns
    ax1.set_ylim([ymin, ymax])
    
		
		
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
    
    
    # Define a custom tooltip function
    # def custom_tooltip(selection):
    #     """Show a tooltip with OHLC data for the selected date."""
    #     x = selection.target[0]
    #     if isinstance(df.index, pd.DatetimeIndex):
    #         date = df.index.get_loc(x, method='nearest')
    #     else:
    #         date = df.index[df.index.get_loc(x, method='nearest')]
    #     candle_data = df.loc[date]
    #     text = (
    #         f"Date: {date}\n"
    #         f"Open: {candle_data['open']:.2f}\n"
    #         f"High: {candle_data['high']:.2f}\n"
    #         f"Low: {candle_data['low']:.2f}\n"
    #         f"Close: {candle_data['close']:.2f}")
    #     selection.annotation.set_text(text)
    # # Add cursor after plotting
    # cursor = mplcursors.cursor(ax1, hover=True)
    # cursor.connect("add", custom_tooltip)
    #     # Add this line after plotting to enable mplcursors
    # mplcursors.cursor(ax1, hover=True).connect("add", custom_tooltip)

    canvas.draw()
 

def plot_monthly_chart(symbol, selected_period):
    
    # Convert period to a format suitable for SQL query
    period_map = {
        '1mo': '1 month',
        '3mo': '3 months',
        '6mo': '6 months',
        '1y': '1 year',
        '2y': '2 years',
        '3y': '3 years',
        '4y': '4 years',
        '5y': '5 years',
        '6y': '6 years',
        '7y': '7 years',
        '8y': '8 years',
        '9y': '9 years',
        '10y': '10 years',
        'max': '100 years'  # Assuming max means as much data as possible
    }
    sql_period = period_map.get(selected_period)
    
    cur.execute("SELECT date, open, high, low, close, volume FROM daily_prices WHERE symbol = %s AND date >= NOW() - INTERVAL %s ORDER BY date", (symbol, sql_period))
    
    rows = cur.fetchall()
    
    data = {
        'Date': [row[0] for row in rows],
        'Open': [round(row[1], 2) for row in rows],
        'High': [round(row[2], 2) for row in rows],
        'Low': [round(row[3], 2) for row in rows],
        'Close': [round(row[4], 2) for row in rows],
        'Volume': [row[5] for row in rows]
        
    }
    
    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)
    df.index = pd.to_datetime(df.index)  # Ensure the index is a DatetimeIndex
    
    # Resample to monthly data
    monthly_df = df.resample('M').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })

    if not monthly_df.empty:
            latest_data = monthly_df.iloc[-1]
            prev_close = monthly_df.iloc[-2]['Close'] if len(monthly_df) > 1 else 0  # Handle the case where there's no previous data
            percent_change = ((latest_data['Close'] - prev_close) / prev_close) * 100 if prev_close != 0 else 0  # Avoid division by zero
            latest_info = (
                f"Date: {latest_data.name.strftime('%b %d %Y')}, "
                f"Open: {latest_data['Open']:.2f}, "
                f"High: {latest_data['High']:.2f}, "
                f"Low: {latest_data['Low']:.2f}, "
                f"Close: {latest_data['Close']:.2f}, "
                f"Chg: {percent_change:.2f}%"
            )
    else:
            latest_info = "No data available"
        

    fig.clear()
    
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
    
    # Add the highlight plot to the original plot
    addplot = [
        # mpf.make_addplot(df['ma7'], ax=ax1, color='red'),

    ]
    # # Create a figure and axis
    # ax1 = fig.add_subplot(111)

 

    # Plot the candlestick chart
    mpf.plot(monthly_df, type='candle', style=samie_style_obj, ax=ax1, volume=ax2, datetime_format='%Y-%m')
    ax1.set_title(f' ({selected_period}) :  {latest_info}', loc='left',  fontsize=10, color='dodgerblue', y=1)
    ax1.set_ylabel('Price')
    ax1.grid(axis='both', which='both', linestyle='--', linewidth=0.5, color='#31363F')
    data_range = monthly_df['Close'].max() - monthly_df['Close'].min()
    # tick_interval = data_range * 0.05  # 5% of the data range
    tick_interval = data_range * 0.10  # 10% of the data range
    
    # Set y-axis major ticks at every 5% of the data range
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(tick_interval))
    # ax1.set_ylim([df['Low'].min(), df['High'].max()])  # working code.. sets you graph to max and min values
    
    # Set the title on the primary axes to include the symbol name
    # ax1.set_title(f'{symbol} Stock Price ({selected_period})', loc='center', fontsize=16, fontweight='bold')
    fig.suptitle(f'{symbol}', fontsize=14, y=0.95, color='#ff000f')  # Adjust the y position as needed
    fig.text(0.99, 0.95, 'MashaAllah', fontsize=10, ha='right', color='green')
    # Enhanced Gridlines for ax1 (Price Chart)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')  # Major and minor grids for ax1
    
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')
# # Plot the volume chart on ax2
    ax2.set_title('  Volume', loc='left',  fontsize=10, pad=5)
    ax2.set_ylabel('Volume')
    ax2.yaxis.set_label_position('right')
    ax2.yaxis.tick_right()
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(volume_formatter))
        
    # Enhanced Gridlines for ax2 (Volume Chart)
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5, color='#31363F')  # Major and minor grids for ax2

    ax1.set_position([0.1, 0.3, 0.8, 0.6])  # Adjust position to make main chart larger
    ax2.set_position([0.1, 0.1, 0.8, 0.2])  # Adjust position to make volume chart smaller

    
    # Initialize the crosshair lines
    crosshair_v = ax1.axvline(x=0, color='#074173', linestyle='--')
    crosshair_h = ax1.axhline(y=0, color='#074173', linestyle='--')

    # Function to update the crosshair position
    def update_crosshair(event):
        if event.inaxes == ax1:
            crosshair_v.set_xdata([event.xdata, event.xdata])
            ymin, ymax = ax1.get_ylim()
            if ymin <= event.ydata <= ymax:
                crosshair_h.set_ydata([event.ydata, event.ydata])
            else:
                crosshair_h.set_ydata([ymin, ymin])
            fig.canvas.draw_idle()

    # Connect the function to the figure's motion_notify_event
    fig.canvas.mpl_connect('motion_notify_event', update_crosshair)

    # Optionally, set the y-axis limits to match your data's range
    ymin, ymax = monthly_df['Low'].min(), monthly_df['High'].max()  # Adjust for your DataFrame's columns
    ax1.set_ylim([ymin, ymax])
    
    # Show the plot
    canvas.draw()

    # Assuming you want to draw the plot on a canvas in a Tkinter GUI, you should replace plt.show() with canvas.draw() if you're integrating this into a Tkinter application
    # canvas.draw()

def plot_weekly_chart(symbol, selected_period):
    
    # Convert period to a format suitable for SQL query
    period_map = {
        '1mo': '1 month',
        '3mo': '3 months',
        '6mo': '6 months',
        '1y': '1 year',
        '2y': '2 years',
        '3y': '3 years',
        '4y': '4 years',
        '5y': '5 years',
        '6y': '6 years',
        '7y': '7 years',
        '8y': '8 years',
        '9y': '9 years',
        '10y': '10 years',
        'max': '100 years'  # Assuming max means as much data as possible
    }
    sql_period = period_map.get(selected_period)
    
    cur.execute("SELECT date, open, high, low, close, volume FROM daily_prices WHERE symbol = %s AND date >= NOW() - INTERVAL %s ORDER BY date", (symbol, sql_period))
    
    rows = cur.fetchall()
    
    data = {
        'Date': [row[0] for row in rows],
        'Open': [round(row[1], 2) for row in rows],
        'High': [round(row[2], 2) for row in rows],
        'Low': [round(row[3], 2) for row in rows],
        'Close': [round(row[4], 2) for row in rows],
        'Volume': [row[5] for row in rows]
        
    }
    
    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)
    df.index = pd.to_datetime(df.index)  # Ensure the index is a DatetimeIndex
    
    # Resample to monthly data
    # Resample to weekly data
    weekly_df = df.resample('W').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })

    if not weekly_df.empty:
            latest_data = weekly_df.iloc[-1]
            prev_close = weekly_df.iloc[-2]['Close'] if len(weekly_df) > 1 else 0  # Handle the case where there's no previous data
            percent_change = ((latest_data['Close'] - prev_close) / prev_close) * 100 if prev_close != 0 else 0  # Avoid division by zero
            latest_info = (
                f"Date: {latest_data.name.strftime('%b %d %Y')}, "
                f"Open: {latest_data['Open']:.2f}, "
                f"High: {latest_data['High']:.2f}, "
                f"Low: {latest_data['Low']:.2f}, "
                f"Close: {latest_data['Close']:.2f}, "
                f"Chg: {percent_change:.2f}%"
            )
    else:
            latest_info = "No data available"
        

    fig.clear()
    
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
    
    # Add the highlight plot to the original plot
    addplot = [
        # mpf.make_addplot(df['ma7'], ax=ax1, color='red'),

    ]
    # # Create a figure and axis
    # ax1 = fig.add_subplot(111)

    # Plot the candlestick chart
    mpf.plot(weekly_df, type='candle', style=samie_style_obj, ax=ax1, volume=ax2, datetime_format='%Y-%m')

    ax1.set_title(f' ({selected_period}) :  {latest_info}', loc='left',  fontsize=10, color='dodgerblue')
    ax1.set_ylabel('Price')
    ax1.grid(axis='both', which='both', linestyle='--', linewidth=0.5, color='#31363F')
    data_range = weekly_df['Close'].max() - weekly_df['Close'].min()
    # tick_interval = data_range * 0.05  # 5% of the data range
    tick_interval = data_range * 0.10  # 10% of the data range
    
    # Set y-axis major ticks at every 5% of the data range
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(tick_interval))
    # ax1.set_ylim([df['Low'].min(), df['High'].max()])  # working code.. sets you graph to max and min values
    
    # Set the title on the primary axes to include the symbol name
    # ax1.set_title(f'{symbol} Stock Price ({selected_period})', loc='center', fontsize=16, fontweight='bold')
    fig.suptitle(f'{symbol}', fontsize=14, y=0.98, color='#ff000f')  # Adjust the y position as needed
    fig.text(0.99, 0.95, 'MashaAllah', fontsize=10, ha='right', color='green')
    # Enhanced Gridlines for ax1 (Price Chart)
    # ax1.grid(True, which='both', linestyle='--', linewidth=0.5, color='#31363F')  # Major and minor grids for ax1
    
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')
# # Plot the volume chart on ax2
    ax2.set_title('  Volume', loc='left',  fontsize=10, pad=5)
    ax2.set_ylabel('Volume')
    ax2.yaxis.set_label_position('right')
    ax2.yaxis.tick_right()
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(volume_formatter))
        
    # Enhanced Gridlines for ax2 (Volume Chart)
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5, color='#31363F')  # Major and minor grids for ax2

    ax1.set_position([0.1, 0.3, 0.8, 0.6])  # Adjust position to make main chart larger
    ax2.set_position([0.1, 0.1, 0.8, 0.2])  # Adjust position to make volume chart smaller

    
    # Initialize the crosshair lines
    crosshair_v = ax1.axvline(x=0, color='#074173', linestyle='--')
    crosshair_h = ax1.axhline(y=0, color='#074173', linestyle='--')

    # Function to update the crosshair position
    def update_crosshair(event):
        if event.inaxes == ax1:
            crosshair_v.set_xdata([event.xdata, event.xdata])
            ymin, ymax = ax1.get_ylim()
            if ymin <= event.ydata <= ymax:
                crosshair_h.set_ydata([event.ydata, event.ydata])
            else:
                crosshair_h.set_ydata([ymin, ymin])
            fig.canvas.draw_idle()

    # Connect the function to the figure's motion_notify_event
    fig.canvas.mpl_connect('motion_notify_event', update_crosshair)

    # Optionally, set the y-axis limits to match your data's range
    ymin, ymax = weekly_df['Low'].min(), weekly_df['High'].max()  # Adjust for your DataFrame's columns
    ax1.set_ylim([ymin, ymax])
    
    # Show the plot
    canvas.draw()


def add_stock():
    symbol = stock_entry.get().upper()
    if symbol:
        if symbol not in watchlist.get(0, tk.END):
            watchlist.insert(tk.END, symbol)
            # watchlist.set_values(sorted(watchlist))
            cur.execute("INSERT INTO tickers (symbol) VALUES (%s) ON CONFLICT (symbol) DO NOTHING", (symbol,))
            conn.commit()
            asyncio.run(calculate_and_store_rsi(symbol))
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
        SELECT date, open, high, low, close, rsi3, mamix14, mamix42, vwma25, volsma5,
               traderule1, traderule2, traderule3, traderule4, traderule5, traderule6 
        FROM daily_prices 
        WHERE symbol = %s 
        ORDER BY date
    """, (symbol,))
    rows = cur.fetchall()
    
    df = pd.DataFrame(rows, columns=['date', 'open', 'high', 'low', 'close', 'rsi3', 'mamix14', 'mamix42', 'vwma25', 'volsma5',
                                     'traderule1', 'traderule2', 'traderule3', 'traderule4', 'traderule5', 'traderule6'])
    df.set_index('date', inplace=True)
    df = df.fillna(value=np.nan)

    # Calculate traderule1
    df['new_traderule1'] = (df['close'].shift(3) < df['open'].shift(3)) & \
                           (df['close'].shift(2) < df['open'].shift(2)) & \
                           (df['close'].shift(1) < df['open'].shift(1)) & \
                           (df['close'] > df['open'])

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
                       
    # Find rows where traderules have changed
    changed_rows = df[(df['traderule1'] != df['new_traderule1']) | 
                      (df['traderule2'] != df['new_traderule2']) |
                      (df['traderule3'] != df['new_traderule3']) |
                      (df['traderule4'] != df['new_traderule4']) |
                      (df['traderule5'] != df['new_traderule5']) |
                      (df['traderule6'] != df['new_traderule6'])]

    # Prepare batch updates
    updates = [(bool(row['new_traderule1']), 
                bool(row['new_traderule2']),
                bool(row['new_traderule3']),
                bool(row['new_traderule4']),
                bool(row['new_traderule5']),
                bool(row['new_traderule6']),
                symbol, row.name) for _, row in changed_rows.iterrows()]

    # Perform bulk update
    cur.executemany("""
        UPDATE daily_prices 
        SET traderule1 = %s, traderule2 = %s, traderule3 = %s, traderule4 = %s, traderule5 = %s, traderule6 = %s
        WHERE symbol = %s AND date = %s
    """, updates)
    
    conn.commit()
    print(f"Updated traderule columns in the database for symbol {symbol}")

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

def update_Screeners(): # updates every symbol
    cur.execute("SELECT symbol FROM tickers")
    symbols = [row[0] for row in cur.fetchall()]
    
    total_symbols = len(symbols)
    for index, symbol in enumerate(symbols, 1):
        asyncio.run(update_traderules(symbol))
        
        # Update progress
        progress = (index / total_symbols) * 100
        root.after(0, lambda p=progress: update_progress(p))
    
    root.after(0, lambda: messagebox.showinfo("Success", "All traderules updated for all symbols!"))
    root.after(0, lambda: progress_bar.pack_forget())  # Hide the progress bar when done
def update_progress(progress):
    # Update a progress bar or label in your UI
    # For example, if you have a progress bar widget named progress_bar:
    progress_bar['value'] = progress
    root.update_idletasks()
##################################################################
def detect_cspatterns():
    try:
        filtered_patterns = get_filtered_patterns()
        filtered_patterns = filtered_patterns.sort_values(by='Date', ascending=False)


        # Create a popup window with a table to display the results
        popup =  tk.Toplevel()
        popup.title("CSpatterns Detection Results")

        # Create a table to display the results
        table = ttk.Treeview(popup)
        table['columns'] = tuple(filtered_patterns.columns)

        # Format the table columns
        table.column("#0", width=0, stretch=tk.NO)
        for col in filtered_patterns.columns:
            table.column(col, anchor=tk.W, width=100)
            table.heading(col, text=col, anchor=tk.W)

        # Insert data into the table
        for index, row in filtered_patterns.iterrows():
            values = [('✅' if v == True else ' ' if v == False else v) for v in row]
            table.insert('', 'end', values=values)

        # Pack the table
        table.pack(fill=tk.BOTH, expand=True)

        # Make the popup window visible
        popup.mainloop()

    except Exception as e:
        print(f"An error occurred: {e}")



###########################################################################################
def up2date_chart():
    try:
        selected_symbol = watchlist.get(watchlist.curselection())
    except tk.TclError:
        messagebox.showerror("Selection Error", "No symbol selected.")
        return

    async def update_tasks(symbol):
        await asyncio.gather(
            calculate_and_store_rsi(symbol),
            calculate_and_store_ma(symbol),
            update_traderules(symbol)
        )

    def run_async_tasks():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(update_tasks(selected_symbol))
        loop.close()

    threading.Thread(target=run_async_tasks, daemon=True).start()

async def update_everything():
    cur.execute("SELECT symbol FROM tickers")
    symbols = cur.fetchall()
    # Sort the symbols in ascending order
    symbols.sort()
    for symbol in symbols:
            await calculate_and_store_rsi(symbol),
            await calculate_and_store_ma(symbol),
            await update_traderules(symbol)
        
        
def show_recent_signals():
    popup =  tk.Toplevel()
    popup.title("Recent Signals")
    
    # Create a style
    style = ttk.Style()
    style.configure("Treeview", font=('Arial', 10), foreground="black", rowheight=20)
    style.configure("BoldText", font=('Arial', 10, 'bold'), foreground="black")

    style.configure(
        "Treeview",
        background="#D3DEDC",
        fieldbackground="#aeae",
        foreground="#272727",
        font=('Arial', 10, "bold"),
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
    tree = ttk.Treeview(popup, columns=("Date", "Symbol", "Rule2", "Rule3", "Rule4", "Rule5", "Rule6"), show="headings", height=10)
    
    # Set column widths and alignment
    tree.column("Date", width=100, anchor='center')
    tree.column("Symbol", width=100, anchor='center')
    tree.column("Rule2", width=100, anchor='center')
    tree.column("Rule3", width=100, anchor='center')
    tree.column("Rule4", width=100, anchor='center')
    tree.column("Rule5", width=100, anchor='center')
    tree.column("Rule6", width=100, anchor='center')

    tree.heading("Date", text="Date")
    tree.heading("Symbol", text="Symbol")
    tree.heading("Rule2", text="Rule 2")
    tree.heading("Rule3", text="Rule 3")
    tree.heading("Rule4", text="Rule 4")
    tree.heading("Rule5", text="Rule 5")
    tree.heading("Rule6", text="Rule 6")
    
    # Fetch data from the database
    days_ago = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
    
    query = """
    SELECT date, symbol, 
            BOOL_OR(traderule2) AS traderule2, 
            BOOL_OR(traderule3) AS traderule3, 
            BOOL_OR(traderule4) AS traderule4,
            BOOL_OR(traderule5) AS traderule5,
            BOOL_OR(traderule6) AS traderule6

    FROM daily_prices
    WHERE date >= %s AND (traderule2 OR traderule3 OR traderule4 OR traderule5 OR traderule6)
    GROUP BY date, symbol
    """
    cur.execute(query, (days_ago,))
    # results = cur.fetchall()
    results = sorted(cur.fetchall(), reverse=True)
    
    # Define a tag for highlighting
    tree.tag_configure('highlight', background="#fDf3fD")

    # Populate the treeview
    for row in results:
        date, symbol, rule2, rule3, rule4, rule5, rule6 = row
            # Convert symbol to sentence case
        # symbol_sentence_case = symbol.capitalize() #works
        values = (date, symbol, "✅" if rule2 else " ", "✅" if rule3 else " ", "✅" if rule4 else " ", "✅" if rule5 else " ", "✅" if rule6 else " ")
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
    traders_diary_notes_create_popup =  tk.Toplevel(root, position="right")
    traders_diary_notes_create_popup.title("Traders Diary")

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

# root =  tk.Window(themename="darkly")
root.title("Subhantech Stock Watchlist")
root.configure(background=BG_COLOR)
# style.configure("TFrame", background="#162636")
# style.theme_use("darkly")
# style.configure("darkly", background="#162636")
root.geometry("1900x1000")

import ctypes

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
    threading.Timer(120, update_quote).start()  # 300 seconds = 5 minutes

# Create the Label
text_label = Label(root, text="", font=("Arial", 10), wraplength=1000)
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
watchlist_frame.pack(side='top', fill='y', expand=False)
watchlist_frame.pack_propagate(False)

# Create a listbox for the watchlist
watchlist = tk.Listbox(watchlist_frame, fg="black", font=("Arial", 10), bd=0, borderwidth=0, relief=tk.FLAT)
watchlist.pack(side='left', expand=True, fill='y', padx=5, pady=5)

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
update_all_rules_button.pack(side=tk.RIGHT, padx=10, pady=5)

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

period_options = ['6mo', '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '3y', '4y', '5y', '6y','7y', '8y', '9y', '10y', 'max']
period_menu = ttk.OptionMenu(timeframe_section, period_var, *period_options)
period_menu.pack(side=tk.LEFT, padx=5)

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

