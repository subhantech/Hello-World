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
from tradingview_ta import TA_Handler, Interval  # instead of storing values in db try to execute this and get it.
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


# Database connection
conn = psycopg2.connect(
    dbname="stockdata",
    user="postgres",
    password="Subhan$007",
    host="localhost",
    port="5432"
)
cur = conn.cursor()


def volume_formatter(x, pos):
    return f'{x:,.0f}'


def price_formatter(x, pos):
    return f'{x:,.2f}'


def plot_monthly_chart(symbol, selected_period):
    
    # Convert period to a format suitable for SQL query
    period_map = {
        '1mo': '1 month',
        '3mo': '3 months',
        '6mo': '6 months',
        '9mo': '9 months',
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
    monthly_df = df.resample('ME').agg({
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
        

    # fig.clear()
    
    # Add the highlight plot to the original plot
    addplot = [
        # mpf.make_addplot(df['ma7'], ax=ax1, color='red'),

    ]
    # # Create a figure and axis
    # ax1 = fig.add_subplot(111)

 

    # Plot the candlestick chart
    mpf.plot(monthly_df, type='candle', style='yahoo', ax=ax1, volume=ax2, datetime_format='%Y-%m')
    
    # fig.show()
    # Show the plot
    
    # Assuming you want to draw the plot on a canvas in a Tkinter GUI, you should replace plt.show() with canvas.draw() if you're integrating this into a Tkinter application
    