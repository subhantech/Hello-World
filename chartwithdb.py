import tkinter as tk
from tkinter import ttk
from tkinter import ttk, messagebox, filedialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import yfinance as yf
import psycopg2
import pandas as pd
import mplfinance as mpf
from mplfinance import make_marketcolors
import mplcursors
from ta.momentum import RSIIndicator
import numpy as np
import matplotlib.cbook as cbook
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from tkcalendar import Calendar, DateEntry
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from numba import jit
import threading
from joblib import Parallel, delayed ## need to work on this for speed processing.

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
ADD COLUMN IF NOT EXISTS notes TEXT; 

""")

conn.commit()


try:
    
    def fetch_data_from_db(symbol, selected_period):
        # Convert selected_period to days
        if selected_period == '5d':
            period_days = 5
        elif selected_period == '1mo':
            months = 1
            period_days = months * 30
        elif selected_period == '3mo':
            months = 3
            period_days = months * 30
        elif selected_period == '6mo':
            months = 6
            period_days = months * 30
        elif selected_period == '1y':
            years = 1
            period_days = years * 365
        elif selected_period == '2y':
            years = 2
            period_days = years * 365
        elif selected_period == '3y':
            years = 3
            period_days = years * 365
        elif selected_period == '4y':
            years = 4
            period_days = years * 365
        elif selected_period == '5y':
            years = 5
            period_days = years * 365
        elif selected_period == '6y':
            years = 6
            period_days = years * 365
        elif selected_period == '7y':
            years = 7
            period_days = years * 365
        elif selected_period == '8y':
            years = 8
            period_days = years * 365
        elif selected_period == '9y':
            years = 9
            period_days = years * 365
        elif selected_period == '10y':
            years = 10
            period_days = years * 365
        else:
            raise ValueError("Invalid period format")

        # Calculate the start date for the data
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=period_days)

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

def volume_formatter(x, pos):
    return f'{x:,.0f}'


def price_formatter(x, pos):
    return f'{x:,.2f}'


def calculate_and_store_rsi(symbol):
    """Calculates RSI(3) for the given symbol and stores it in the database."""

    cur.execute("SELECT date, close FROM daily_prices WHERE symbol = %s ORDER BY date", (symbol,))
    rows = cur.fetchall()
    dates = [row[0] for row in rows]
    close_prices = [row[1] for row in rows]

    if len(close_prices) >= 3:  # Need at least 3 data points for RSI(3)
        rsi_indicator = RSIIndicator(close=pd.Series(close_prices), window=3)
        rsi_values = rsi_indicator.rsi().tolist()

        # Update the database with calculated RSI values
        for i in range(len(dates)):
            cur.execute(
                "UPDATE daily_prices SET rsi3 = %s WHERE symbol = %s AND date = %s",
                (rsi_values[i], symbol, dates[i])
            )
        conn.commit()
        print(f"RSI(3) calculated and stored for {symbol}")
    else:
        print(f"Not enough data to calculate RSI(3) for {symbol}")
    

def calculate_and_store_ma(symbol):
    cur.execute("SELECT date, close, ma7, ma21, ma63, ma189, mamix14, mamix42 FROM daily_prices WHERE symbol = %s ORDER BY date", (symbol,))
    rows = cur.fetchall()
    dates = [row[0] for row in rows]
    close_prices = [row[1] for row in rows]
    
    df = pd.DataFrame({'close': close_prices}, index=dates)
    df['ma7'] = df['close'].rolling(window=7).mean()
    df['ma21'] = df['close'].rolling(window=21).mean()
    df['ma63'] = df['close'].rolling(window=63).mean()
    df['ma189'] = df['close'].rolling(window=189).mean()
    df['mamix14'] = ((df['ma7'] + df['ma21'] ) / 2).rolling(window=2).mean()
    df['mamix42'] = ((df['ma21'] + df['ma63'] ) / 2).rolling(window=2).mean()
    
    for date, row in df.iterrows():
        cur.execute("""
            UPDATE daily_prices 
            SET ma7 = %s, ma21 = %s, ma63 = %s, ma189 = %s, mamix14 = %s, mamix42 = %s 
            WHERE symbol = %s AND date = %s
        """, (row['ma7'], row['ma21'], row['ma63'], row['ma189'],row['mamix14'],row['mamix42'], symbol, date))
    conn.commit()
    print(f"Moving Averages are updated for {symbol}")



def plot_chart(symbol, selected_period):
    
    # Define the style
    
    samie_style = {
        "base_mpl_style": "nightclouds",
        "marketcolors": {
            "candle": {"up": "#ffffff", "down": "#ef4f60"},  
            "edge": {"up": "#000000", "down": "#ef4f60"},  
            "wick": {"up": "#247252", "down": "#ef4f60"},  
            "ohlc": {"up": "green", "down": "red"},
            "volume": {"up": "#067887", "down": "#ff6060"},  
            "vcedge": {"up": "#ffffff", "down": "#ffffff"},  
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

    # samie_style = mpf.make_mpf_style(base_mpf_style='charles', 
    # marketcolors=mpf.make_marketcolors(up='green', down='black', inherit=True))

    samie_style_obj = mpf.make_mpf_style(**samie_style)
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
    
    

    cur.execute("SELECT date, open, high, low, close, volume, rsi3,  ma7, ma21, ma63, ma189, mamix14, mamix42, traderule1, traderule2, traderule3, traderule4, traderule5  FROM daily_prices WHERE symbol = %s AND date >= NOW() - INTERVAL %s ORDER BY date", (symbol, sql_period))
    
    rows = cur.fetchall()
    if not rows:
        fetch_data_from_db(symbol, selected_period)
        cur.execute("SELECT date, open, high, low, close, volume, rsi3,  ma7, ma21, ma63, ma189, mamix14, mamix42, traderule1, traderule2, traderule3, traderule4, traderule5  FROM daily_prices WHERE symbol = %s AND date >= NOW() - INTERVAL %s ORDER BY date", (symbol, sql_period))
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
        'traderule1': [row[13] for row in rows],
        'traderule2':[row[14] for row in rows],
        'traderule3':[row[15] for row in rows],
        'traderule4':[row[16] for row in rows],
        'traderule5':[row[17] for row in rows]
        
        
        
    }
    
    
    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)
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
    

    ### to plot on low of the candle .. we are collecting lows
    df['traderule1_highlight'] = np.where(df['traderule1'], df['Low'], np.nan)
    df['traderule2_highlight'] = np.where(df['traderule2'], df['Low'], np.nan)
    df['traderule3_highlight'] = np.where(df['traderule3'], df['Low'], np.nan)
    df['traderule4_highlight'] = np.where(df['traderule4'], df['Low'], np.nan)
    df['traderule5_highlight'] = np.where(df['traderule5'], df['Low'], np.nan)
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
    
    
    # if not traderule3_high.empty:
    #     addplot = []
    #     for value in traderule3_high:
    #         addplot.append(mpf.make_addplot([value]*len(df), ax=ax1, color='#7572f1', linestyle='--'))
    
    fig.clear()
    
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
    
    #Example apd = [mpf.make_addplot(buy, scatter=True, markersize=100, marker=r'$\Uparrow$', color='green')]
    #mpf.add_line(price=100, color='red', style='dashed', width=1)
    # Add the highlight plot to the original plot
    addplot = [
        # mpf.make_addplot(df['ma7'], ax=ax1, color='red'),
        # mpf.make_addplot(df['ma21'], ax=ax1, color='black'),
        # mpf.make_addplot(df['ma63'], ax=ax1, color='green'),
        mpf.make_addplot(df['ma189'], ax=ax1, color='purple'),
        mpf.make_addplot(df['mamix14'], ax=ax1, color='blue'),
        mpf.make_addplot(df['mamix42'], ax=ax1, color='red'),
        mpf.make_addplot(df['traderule1_highlight'], ax=ax1, type='scatter', markersize=100, marker='s', color='#bcf5bc'), # working code
        mpf.make_addplot(df['traderule2_highlight'], ax=ax1,  type='scatter', markersize=100, marker='^', color='#1e90ff'), # working code
        mpf.make_addplot(df['traderule3_highlight'], ax=ax1,  type='scatter', markersize=100, marker='s', color='#7572f1'), # working code
        mpf.make_addplot(df['traderule4_highlight'], ax=ax1, type='scatter', markersize=100, marker='d', color='#ff00ff'),  # New color for traderule4
        mpf.make_addplot(df['traderule5_highlight'], ax=ax1, type='scatter', markersize=100, marker='^', color='#067887'),  # New color for traderule4
        # mpf.make_addplot([horizontal_line]*len(df), ax=ax1, color='#7572f1', linestyle='--') working code for only one time occurance

    ]
    
    
    
    #mpf.plot(df, hlines=['traderule3_high'], type='candle')
    # mpf.plot(df, type='candle', style=samie_style_obj, ax=ax1, volume=ax2, datetime_format='%b %d', addplot=addplot) # working code
    mpf.plot(df, type='candle', style=samie_style_obj, ax=ax1, volume=ax2, datetime_format='%Y-%m-%d', addplot=addplot, returnfig=True)
    
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


    ax1.set_title(f' ({selected_period}) :  {latest_info}', loc='left',  fontsize=10)
    ax1.set_ylabel('Price')
    ax1.grid(axis='both', which='both')
    # ax1.set_ylim([df['Low'].min(), df['High'].max()])  # working code.. sets you graph to max and min values
    
    # Set the title on the primary axes to include the symbol name
    # ax1.set_title(f'{symbol} Stock Price ({selected_period})', loc='center', fontsize=16, fontweight='bold')
    fig.suptitle(f'{symbol}', fontsize=14, y=0.98)  # Adjust the y position as needed
    fig.text(0.99, 0.95, 'MashaAllah', fontsize=10, ha='right')
    # Enhanced Gridlines for ax1 (Price Chart)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')  # Major and minor grids for ax1
    ############ working code for line below.
    if not traderule3_high.empty:
        for value in traderule3_high:
            ax1.axhline(y=value, color='#7572f1', linestyle=':')	
    
    if not traderule3_low.empty:
        # for value in traderule3_low:                                              ### working code
        #     ax1.axhline(y=value * 0.95, color='#ff000f', linestyle='--')          ### working code
        for value  in traderule3_low:
            ax1.axhline(y=value * 0.98, color='#ff000f', linestyle=':')
            ax1.text(df.index[-1], value * 0.98, f'{value:.2f}', va='center', ha='left', 
                bbox=dict(facecolor='#0042a1', edgecolor='black', alpha=0.7, boxstyle='round,pad=0.5'))
                
    if not traderule4_high.empty:
        for value in traderule4_high:
            ax1.axhline(y=value, color='#abcddcba', linestyle='--')
    
    if not traderule4_low.empty:
        # for value in traderule3_low:                                              ### working code
        #     ax1.axhline(y=value * 0.95, color='#fabfab', linestyle='--')          ### working code
        for value  in traderule4_low:
            ax1.axhline(y=value * 0.98, color='#fabcfabc', linestyle='--')
            ax1.text(df.index[-1], value * 0.98, f'{value:.2f}', va='center', ha='left', 
                bbox=dict(facecolor='#0042a1', edgecolor='black', alpha=0.7))
                

    # ax1.set_yscale('log')   
    # ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:.2f}'))
    # ax1.yaxis.set_major_formatter(ticker.FuncFormatter(price_formatter))
    # # Volume Chart
    # # Plot the volume chart on ax2
    ax2.set_title('  Volume', loc='left',  fontsize=10, pad=5)
    ax2.set_ylabel('Volume')
    ax2.yaxis.set_label_position('right')
    ax2.yaxis.tick_right()
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(volume_formatter))
    
    
    # Enhanced Gridlines for ax2 (Volume Chart)
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')  # Major and minor grids for ax2

     

    ax1.set_position([0.1, 0.3, 0.8, 0.6])  # Adjust position to make main chart larger
    ax2.set_position([0.1, 0.1, 0.8, 0.2])  # Adjust position to make volume chart smaller


    canvas.draw()


def add_stock():
    symbol = stock_entry.get().upper()
    if symbol:
        if symbol not in watchlist.get(0, tk.END):
            watchlist.insert(tk.END, symbol)
            # watchlist.set_values(sorted(watchlist))
            cur.execute("INSERT INTO tickers (symbol) VALUES (%s) ON CONFLICT (symbol) DO NOTHING", (symbol,))
            conn.commit()
            calculate_and_store_rsi(symbol)
            calculate_and_store_ma(symbol)
            update_traderule1(symbol)
            update_traderule2(symbol)
            update_traderule3(symbol)
            update_traderule4(symbol)
            update_traderule5(symbol)
            

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
        fetch_data_from_db(selected_symbol, selected_period)
        plot_chart(selected_symbol, selected_period)



def update_traderule1(symbol):
    cur.execute("SELECT date, close, open, traderule1 FROM daily_prices WHERE symbol = %s ORDER BY date", (symbol,))
    rows = cur.fetchall()
    dates, close_prices, open_prices, traderule1 = zip(*rows)
    
    df = pd.DataFrame({'close': close_prices, 'open': open_prices, 'traderule1': traderule1}, index=dates)
    
    # Calculate traderule1 - https://chartink.com/screener/samie-daily-trend-reverse
    df['traderule1'] = (df['close'].shift(3) < df['open'].shift(3)) & \
                       (df['close'].shift(2) < df['open'].shift(2)) & \
                       (df['close'].shift(1) < df['open'].shift(1)) & \
                       (df['close'] > df['open'])

    # Update the traderule1 column in the database
    for i in range(len(df)):
        cur.execute("UPDATE daily_prices SET traderule1 = %s WHERE symbol = %s AND date = %s", 
                    (bool(df['traderule1'].iloc[i]), symbol, df.index[i]))
        conn.commit()
    print(f"Updated the traderule1 column in the database for symbol {symbol}")

def update_traderule2(symbol):
    cur.execute("SELECT date, close, open, traderule2 FROM daily_prices WHERE symbol = %s ORDER BY date", (symbol,))
    rows = cur.fetchall()
    dates, close_prices, open_prices, traderule1 = zip(*rows)
    
    df = pd.DataFrame({'close': close_prices, 'open': open_prices, 'traderule1': traderule1}, index=dates)
    
    # Calculate traderule2 - https://chartink.com/screener/samie-candle-stick-based-long
    df['traderule2'] = (df['close'].shift(2) < df['open'].shift(2)) & \
                 (df['close'].shift(1) > df['open'].shift(1)) & \
                 (df['close'].shift(1) < df['open'].shift(2)) & \
                 (df['open'].shift(1) > df['close'].shift(2)) & \
                 (df['open'] < df['open'].shift(1)) & \
                 (df['close'] > df['close'].shift(1)) & \
                 (df['close'] > df['open'].shift(2))
    # Update the traderule1 column in the database
    for i in range(len(df)):
        cur.execute("UPDATE daily_prices SET traderule2 = %s WHERE symbol = %s AND date = %s", 
                    (bool(df['traderule2'].iloc[i]), symbol, df.index[i]))
        conn.commit()
    print(f"Updated the traderule2 column in the database for symbol {symbol}")

def update_traderule3(symbol):
    cur.execute("SELECT date, open, high, low, close, rsi3 FROM daily_prices WHERE symbol = %s ORDER BY date", (symbol,))
    rows = cur.fetchall()
    dates, open_prices, high_prices, low_prices, close_prices, rsi3_values = zip(*rows)
    
    df = pd.DataFrame({
        'open': open_prices, 
        'high': high_prices, 
        'low': low_prices, 
        'close': close_prices, 
        'rsi3': rsi3_values
    }, index=dates)
    
    df['traderule3'] = (
        (df['high'] > df['high'].shift(2)) & 
        (df['high'].shift(1) < df['high'].shift(2)) & 
        (df['open'].shift(2) > df['close'].shift(2)) & 
        (df['open'] < df['close']) & 
        (df['rsi3'] < 60)
    )

    for i in range(len(df)):
        cur.execute("UPDATE daily_prices SET traderule3 = %s WHERE symbol = %s AND date = %s", 
                    (bool(df['traderule3'].iloc[i]), symbol, df.index[i]))
    conn.commit()
    print(f"Updated the traderule3 column in the database for symbol {symbol}")
        
def update_traderule4(symbol):
    cur.execute("SELECT date, open, high, low, close FROM daily_prices WHERE symbol = %s ORDER BY date", (symbol,))
    rows = cur.fetchall()
    dates, open_prices, high_prices, low_prices, close_prices = zip(*rows)
    
    df = pd.DataFrame({
        'open': open_prices, 
        'high': high_prices, 
        'low': low_prices, 
        'close': close_prices
    }, index=dates)
    
    df['traderule4'] = (
        (df['close'].shift(2) < df['open'].shift(2)) &  # 2 days ago close < 2 days ago open
        (df['close'].shift(1) > df['open'].shift(1)) &  # 1 day ago close > 1 day ago open
        (df['close'].shift(1) > ((df['close'].shift(1).abs() + df['open'].shift(1).abs()) / 2)) &  # 1 day ago close > (abs(1 day ago close + 1 day ago open) / 2)
        (df['close'].shift(1) < ((df['close'].shift(2).abs() + df['open'].shift(2).abs()) / 2)) &  # 1 day ago close < (abs(2 days ago close + 2 days ago open) / 2)
        (df['open'] > ((df['high'].shift(1) + df['low'].shift(1)) / 2)) &  # latest open > (abs(1 day ago high + 1 day ago low) / 2)
        (df['close'] > df['open'])  # latest close > latest open
    )

    for i in range(len(df)):
        cur.execute("UPDATE daily_prices SET traderule4 = %s WHERE symbol = %s AND date = %s", 
                    (bool(df['traderule4'].iloc[i]), symbol, df.index[i]))
    conn.commit()
    print(f"Updated the traderule4 column in the database for symbol {symbol}")
    
def update_traderule5(symbol):
    cur.execute("SELECT date, close, mamix14, mamix42 FROM daily_prices WHERE symbol = %s ORDER BY date", (symbol,))
    rows = cur.fetchall()
    dates, close_prices, mamix14_prices, mamix42_prices  = zip(*rows)
    
    df = pd.DataFrame({
        'close': close_prices,
        'mamix14': mamix14_prices, 
        'mamix42': mamix42_prices 
    }, index=dates)
    
    df['traderule5'] = (
            (df['close'] > df['mamix14']) &  # latest close > latest mamix14
            (df['close'].shift(1) <= df['mamix14'].shift(1)) &  # 1 day ago close <= 1 day ago mamix14
            (df['close'] > df['mamix42']) &  # latest close > latest mamix42
            (df['close'].shift(1) <= df['mamix42'].shift(1))  # 1 day ago close <= 1 day ago mamix42
        )

    for i in range(len(df)):
        cur.execute("UPDATE daily_prices SET traderule5 = %s WHERE symbol = %s AND date = %s", 
                    (bool(df['traderule5'].iloc[i]), symbol, df.index[i]))
    conn.commit()
    print(f"Updated the traderule5 column in the database for symbol {symbol}")

def format_tooltip(x, y, data):
    # Access the data for the hovered candle
    candle_data = data[x]
    open_price, high_price, low_price, close_price = candle_data[1:5]
    return f"Open: {open_price:.2f}\nHigh: {high_price:.2f}\nLow: {low_price:.2f}\nClose: {close_price:.2f}"


def download_histdata(start_date, end_date):
    # cur.execute("SELECT symbol FROM tickers")
    # symbols = cur.fetchall()

    # for symbol, in symbols:
    #     # Check if maximum data already exists
    #     cur.execute("SELECT MAX(date) FROM daily_prices WHERE symbol = %s", (symbol,))
    #     max_date = cur.fetchone()[0]

    #     if max_date:
    #         # Check if the maximum date is yesterday
    #         yesterday = (datetime.now() - timedelta(days=1)).date()
    #         if max_date >= yesterday:
    #             continue  # Skip if data is already up-to-date

    selected_symbol = watchlist.get(watchlist.curselection())
    # Fetch missing data
        # Fetch missing data
    stock_hist_data = yf.download(selected_symbol, start=start_date, end=end_date)
    # print(stock_hist_data)
    
        # Define the column mapping
    column_mapping = {
            'symbol': 'symbol',
            'date': 'Date',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }

        # Insert data into database
    for index, row in stock_hist_data.iterrows():
            
                data = {
                    'symbol': selected_symbol,
                    'date': index.date(),
                    'open': row['Open'],
                    'high': row['High'],
                    'low': row['Low'],
                    'close': row['Close'],
                    'volume': row['Volume']
                }
                try: 
                    cur.execute( 
                            "INSERT INTO daily_prices (symbol, date, open, high, low, close, volume) VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT (symbol, date) DO UPDATE SET open = excluded.open, high = excluded.high, low = excluded.low, close = excluded.close, volume = excluded.volume RETURNING *",
                    (data['symbol'], data['date'], data['open'], data['high'], data['low'], data['close'], data['volume']))
                    conn.commit()
                except Exception as e:
                    print(f"Error inserting data: {e}")
                    conn.rollback() 
                    # messagebox.showinfo("Success", "Data downloaded and inserted into the database!")
    # Calculate and store RSI(3) if not already calculated

def download_last_3_days_histdata():
    # Get the list of symbols from the database
    cur.execute("SELECT symbol FROM tickers")
    symbols = [row[0] for row in cur.fetchall()]
    
    # Calculate the start date for the last 3 days
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=3)
    
    # Download the historical data for each symbol and update the database
    for symbol in symbols:
        stock_hist_data = yf.download(symbol, start=start_date, end=end_date)
        # print(stock_hist_data)
        
        # Define the column mapping
        column_mapping = {
            'symbol': 'symbol',
            'date': 'Date',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }

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
                cur.execute( 
                        "INSERT INTO daily_prices (symbol, date, open, high, low, close, volume) VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT (symbol, date) DO UPDATE SET open = excluded.open, high = excluded.high, low = excluded.low, close = excluded.close, volume = excluded.volume RETURNING *",
                        (data['symbol'], data['date'], data['open'], data['high'], data['low'], data['close'], data['volume']))
                conn.commit()
            except Exception as e:
                print(f"Error downloading data for symbol {symbol}: {e}")
                
def update_all_traderules_thread():
    threading.Thread(target=update_all_traderules, daemon=True).start()

def update_all_traderules():
    cur.execute("SELECT symbol FROM tickers")
    symbols = [row[0] for row in cur.fetchall()]
    
    total_symbols = len(symbols)
    for index, symbol in enumerate(symbols, 1):
        update_traderule1(symbol)
        update_traderule2(symbol)
        update_traderule3(symbol)
        update_traderule4(symbol)  
        update_traderule5(symbol)  

        calculate_and_store_rsi(symbol)
        calculate_and_store_ma(symbol)
        
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

def update_data(selected_symbol):
    selected_symbol = watchlist.get(watchlist.curselection())
    calculate_and_store_rsi(selected_symbol)
    calculate_and_store_ma(selected_symbol)
    update_traderule1(selected_symbol)
    update_traderule2(selected_symbol)
    update_traderule3(selected_symbol)
    update_traderule4(selected_symbol)
    update_traderule5(selected_symbol)

def show_recent_signals():
    popup = tk.Toplevel()
    popup.title("Recent Signals")
    
    # Create a style
    style = ttk.Style()
    # style.configure("Treeview", font=('Tahoma', 10))
    # style.configure("Treeview.Heading", font=('Arial', 12, 'bold'))
    # style.map('Treeview', background=[('alternate', '#272727'), ('selected', '#aeae')])
    style.configure(
        "Treeview",
        background="#ECECEC",
        fieldbackground="#aeae",
        foreground="#272727",
        font=('Tahoma', 10),
        rowheight=20,
        height=1,
        borderwidth=0,
        relief="flat",
    )
    style.configure(
        "Treeview.Heading",
        background="#ECECEC",
        foreground="#272727",
        borderwidth=2,
        font=('Tahoma', 10, "bold"),
        relief="flat",
    )
    style.map(
        "Treeview.Heading",
        background=[("active", "#383838"), ("selected", "#383838")],
        foreground=[("active", "#ECECEC"), ("selected", "#ECECEC")],
    )
    
    # Create a treeview to display the data
    tree = ttk.Treeview(popup, columns=("Date", "Symbol", "Rule2", "Rule3", "Rule4", "Rule5"), show="headings", height=10)
    
        # Set column widths and alignment
    tree.column("Date", width=100, anchor='center')
    tree.column("Symbol", width=100, anchor='center')
    tree.column("Rule2", width=100, anchor='center')
    tree.column("Rule3", width=100, anchor='center')
    tree.column("Rule4", width=100, anchor='center')
    tree.column("Rule5", width=100, anchor='center')


    tree.heading("Date", text="Date")
    tree.heading("Symbol", text="Symbol")
    tree.heading("Rule2", text="Rule 2")
    tree.heading("Rule3", text="Rule 3")
    tree.heading("Rule4", text="Rule 4")
    tree.heading("Rule5", text="Rule 5")
    
    # Configure tag for highlighting
    tree.tag_configure("highlight", background="yellow")
    # Fetch data from the database
    # Fetch data from the database
    three_days_ago = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
    
    query = """
    SELECT date, symbol, 
            BOOL_OR(traderule2) AS traderule2, 
            BOOL_OR(traderule3) AS traderule3, 
            BOOL_OR(traderule4) AS traderule4,
            BOOL_OR(traderule5) AS traderule5
    FROM daily_prices
    WHERE date >= %s AND (traderule2 OR traderule3 OR traderule4 OR traderule5)
    GROUP BY date, symbol
    """
    cur.execute(query, (three_days_ago,))
    results = cur.fetchall()
    
    # Populate the treeview
    for row in results:
        date, symbol, rule2, rule3, rule4, rule5 = row
        item = tree.insert("", "end", values=(date, symbol, rule2, rule3, rule4, rule5))
    
        if any(rule == "true" for rule in ( rule2, rule3, rule4, rule5)):
            tree.item(item, tags=("highlight",))
    
    tree.pack(expand=True, fill="both",padx=10, pady=10)
    popup.mainloop()
############################## For Traders Dairy ############################################
################## Traders Dairy function ###############
def open_diary_popup():
    try:
        selected_symbol = watchlist.get(watchlist.curselection())
    except tk.TclError:
        messagebox.showwarning("No Symbol Selected", "Please select a symbol from the watchlist first.")
        return
    cur.execute("SELECT date, notes, symbol FROM daily_prices WHERE symbol = %s", (selected_symbol,))
    rows = cur.fetchall()
    # symbols = [row[0] for row in cur.fetchall()]
    
    data = {
        'date': [row[0] for row in rows],
        'symbol': [row[1] for row in rows],
        'notes':[row[2] for row in rows]
    }
    
    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)
    df = df.sort_index(ascending=False) 
    df.index = pd.to_datetime(df.index)  # Ensure the index is a DatetimeIndex
    
    traders_diary_popup = tk.Toplevel(root)
    traders_diary_popup.title("Traders Diary")

    symbol_label = tk.Label(traders_diary_popup, text="Symbol:")
    symbol_label.pack()
    symbol_entry = tk.Entry(traders_diary_popup)
    symbol_entry.insert(0, selected_symbol)
    symbol_entry.config(state='readonly')
    symbol_entry.pack()
    
    last_chart_date = df.index[0].strftime('%Y-%m-%d')
    date_label = tk.Label(traders_diary_popup, text="Date:")
    date_label.pack()
    date_entry = tk.Entry(traders_diary_popup)
    date_entry.insert(0, last_chart_date)
    date_entry.config(state='readonly')
    date_entry.pack()

    
    notes_label = tk.Label(traders_diary_popup, text="Notes:")
    notes_label.pack()
    notes_text = tk.Text(traders_diary_popup, height=5, width=30)
    notes_text.pack()
    
    def view_previous_notes():
        view_popup = tk.Toplevel(root)
        view_popup.title("Previous Notes")
        
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT date, symbol, notes FROM daily_prices WHERE notes IS NOT NULL ORDER BY date DESC")
                notes = cur.fetchall()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return
        notes_text = tk.Text(view_popup, height=20, width=60)
        notes_text.pack()

        for note in notes:
            notes_text.insert(tk.END, f"Date: {note[0]}\n")
            notes_text.insert(tk.END, f"Symbol: {note[1]}\n")
            notes_text.insert(tk.END, f"Notes: {note[2]}\n\n")
        notes_text.config(state='disabled')

    def save_notes():
        notes = notes_text.get("1.0", tk.END)
        try:
            with conn.cursor() as cur:
                cur.execute("UPDATE daily_prices SET notes = %s WHERE symbol = %s AND date = %s", (notes, selected_symbol, last_chart_date))
                conn.commit()
                messagebox.showinfo("Success", "Notes saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            
            
    save_button = tk.Button(traders_diary_popup, text="Save", command=lambda: save_notes())
    save_button.pack()
    
    view_button = tk.Button(traders_diary_popup, text="View Previous Notes", command=view_previous_notes)
    view_button.pack()
    
    
    def delete_note(date, symbol):
        # with conn.cursor() as cur:
            cur.execute("UPDATE daily_prices SET notes = NULL WHERE date = %s AND symbol = %s", (date, symbol))
            conn.commit()
            # conn.close()
            messagebox.showinfo("Success", "Note deleted successfully!")
            view_previous_notes()  # Refresh the view

#########################################################################################################################################

########################################################################################################################################################################    
# Create the main window
root = tk.Tk()
root.title("Subhantech Stock Watchlist")
root.geometry("1800x900")

# Create a sidebar frame
sidebar = tk.Frame(root, width=250, bg='dodgerblue')
sidebar.pack(expand=False, fill='y', side='left', anchor='nw', ipadx=5, padx=5)
sidebar.pack_propagate(False)

# Create a listbox for the watchlist
watchlist = tk.Listbox(sidebar, borderwidth=0, relief=tk.FLAT)
watchlist.pack(expand=True, fill='both', padx=5, ipadx=5)
# watchlist = tk.Listbox(sidebar, borderwidth=0, relief=tk.FLAT)
# watchlist.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))

# Add a scrollbar to the frame
scrollbar = ttk.Scrollbar(watchlist)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# Configure the scrollbar to control the listbox
watchlist.config(yscrollcommand=scrollbar.set)
scrollbar.config(command=watchlist.yview)


def search_stocks():
    search_query = search_entry.get().lower()
    
    for i in range(watchlist.size()):
        symbol = watchlist.get(i).lower()
        if search_query in symbol:
            watchlist.itemconfig(i, bg="yellow")
        else:
            watchlist.itemconfig(i, bg="SystemButtonFace")
            
###########################################################################
stock_search_section = tk.Frame(sidebar, bg='dodgerblue')
stock_search_section.pack(fill=tk.Y)

# Create a search entry and button
search_entry = tk.Entry(stock_search_section, width=12)
search_entry.pack(side=tk.LEFT, pady=10, padx=5)
search_button = tk.Button(stock_search_section, text="Search", command=search_stocks)
search_button.pack(pady=10, padx=5, side=tk.LEFT)
# Bind the Enter key to the search_watchlist method
search_entry.bind("<Return>", search_stocks)

# Add items to the watchlist from the database
cur.execute("SELECT symbol FROM tickers")
symbols = cur.fetchall()
for symbol in symbols:
    watchlist.insert(tk.END, symbol[0])

# Bind the listbox selection event to the plot function
watchlist.bind('<<ListboxSelect>>', on_select)



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


# sort_watchlist_frame = tk.Frame(sidebar, bg='dodgerblue')
# sort_watchlist_frame.pack()

# Create buttons to trigger the sort functionality
sort_button = tk.Button(stock_search_section, text="Sort Asc", command=lambda: sort_watchlist())
sort_button.pack(side="right")

reverse_sort_button = tk.Button(stock_search_section, text="Sort Dsc", command=lambda: sort_watchlist(reverse=True))
reverse_sort_button.pack(side="right")


period_section = tk.Frame(sidebar, bg='dodgerblue')
period_section.pack()
# Add period selection
period_var = tk.StringVar(value='1y')
period_label = tk.Label(period_section, text="Select Period:")
period_label.pack(side=tk.LEFT, padx=10, pady=10)

period_options = ['6mo', '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '3y', '4y', '5y', '6y','7y', '8y', '9y', '10y', 'max']
period_menu = ttk.OptionMenu(period_section, period_var, *period_options)
period_menu.pack(side=tk.LEFT, padx=5)

# Bind to period_var changes
# With this:
period_var.trace_add("write", lambda *args: on_select(None))
############################################################
stock_entry_section = tk.Frame(sidebar, bg='dodgerblue')
stock_entry_section.pack(fill=tk.Y)

stock_entry = tk.Entry(stock_entry_section)
stock_entry.pack(side=tk.LEFT, padx=10, pady=5)

add_button = tk.Button(stock_entry_section, text="Add Stock", command=add_stock)
add_button.pack(padx=10, pady=5)

delete_button = tk.Button(stock_entry_section, text="Delete Stock", command=delete_stock)
delete_button.pack(padx=10,pady=5, side=tk.RIGHT)

###################################################
stock_other_section = tk.Frame(sidebar, bg='dodgerblue')
stock_other_section.pack(fill=tk.Y)

update_all_rules_button = tk.Button(stock_other_section, text="Update All Rules", command=update_all_traderules_thread, bg='orange')
update_all_rules_button.pack(side=tk.RIGHT, padx=10, pady=5)

add_fromfile_button = tk.Button(stock_other_section, text="Add Chartink Stocks", command=add_chartink_stock)
add_fromfile_button.pack(padx=10,pady=5, side=tk.LEFT)

##########################################################
download_data_section = tk.Frame(sidebar, bg='dodgerblue')
download_data_section.pack(fill=tk.Y)

# Add start date and end date date pickers
start_date_frame = tk.Frame(download_data_section)
start_date_frame.pack(side=tk.TOP, padx=10, pady=5)

start_date_label = tk.Label(start_date_frame, text="Start Date:")
start_date_label.pack(side=tk.LEFT, padx=10, pady=5)
start_date_picker = DateEntry(start_date_frame, width=20, bg="darkblue", fg="white", borderwidth=2, date_pattern='yyyy/mm/dd')
# start_date_picker.place(relx=0.1, rely=0.5, anchor=tk.CENTER)
start_date_picker.pack(side=tk.LEFT, padx=5, pady=5)

end_date_frame = tk.Frame(download_data_section)
end_date_frame.pack(side=tk.TOP, padx=10, pady=5)

end_date_label = tk.Label(end_date_frame, text="End Date:")
end_date_label.pack(side=tk.LEFT, padx=10, pady=5)
end_date_picker = DateEntry(end_date_frame, width=20, bg="darkblue", fg="white", borderwidth=2, date_pattern='yyyy/mm/dd')
# end_date_picker.place(relx=0.9, rely=0.5, anchor=tk.CENTER)
end_date_picker.pack(side=tk.LEFT, padx=5, pady=5)

# Update the Download HistData button to use the start and end dates
download_histdata_button = tk.Button(download_data_section, text="Download HistData", command=lambda: download_histdata(start_date_picker.get_date(), end_date_picker.get_date()))
download_histdata_button.pack(side=tk.RIGHT, padx=10, pady=5)


data_update_section = tk.Frame(sidebar, bg='dodgerblue')
data_update_section.pack(fill=tk.Y)

# Update the Download HistData button to use the start and end dates
update_3days_data_button = tk.Button(data_update_section, text="3 Days Data", command=lambda: download_last_3_days_histdata())
update_3days_data_button.pack(side=tk.LEFT, padx=10, pady=5)
update_data_button = tk.Button(data_update_section, text=" Update Data", command=lambda: update_data(symbol))
update_data_button.pack(side=tk.RIGHT, padx=10, pady=5)

progress_bar_section = tk.Frame(sidebar, bg='dodgerblue')
progress_bar_section.pack(fill=tk.Y)

progress_bar = ttk.Progressbar(progress_bar_section, orient="horizontal", length=200, mode="determinate")
progress_bar.pack(side=tk.LEFT, padx=10, pady=5)

show_signals_button = ttk.Button(sidebar, text="Show Recent Signals", command=show_recent_signals)
show_signals_button.pack(pady=10)


traders_dairy_section = tk.Frame(sidebar, bg='dodgerblue')
traders_dairy_section.pack(fill=tk.Y)

diary_button = tk.Button(traders_dairy_section, text="Diary", command=open_diary_popup)
diary_button.pack()

# Create a main frame
main_frame = tk.Frame(root, bg='turquoise')
main_frame.pack(expand=True, fill='both', side='right')

# Create a Matplotlib figure and canvas
fig = Figure(figsize=(8, 6), dpi=100)
canvas = FigureCanvasTkAgg(fig, master=main_frame)
canvas.get_tk_widget().pack(expand=True, fill='both')

horizonScrollbar = tk.Scrollbar(main_frame, orient='horizontal')



# Add navigation toolbar for scrolling and zooming
toolbar = NavigationToolbar2Tk(canvas, main_frame)
toolbar.update()
canvas.get_tk_widget().pack(expand=True, fill='both')

# Start the Tkinter main loop
root.mainloop()

# Close the database connection when the application is closed
conn.close()
