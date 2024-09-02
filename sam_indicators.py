import psycopg2
import pandas as pd 

### Technical Analysis
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.trend import CCIIndicator
from ta.volatility import BollingerBands
from tradingview_ta import TA_Handler, Interval  # instead of storing values in db try to execute this and get it.
import talib

import numpy as np
from datetime import datetime, timedelta
import threading
import asyncio

import redis
import json
# Database connection
conn = psycopg2.connect(
    dbname="stockdata",
    user="postgres",
    password="Subhan$007",
    host="localhost",
    port="5432"
)
cur = conn.cursor()


async def calculate_and_store_rsi_cci(symbol):
    """Calculates RSI3 & RSI14 for the given symbol and stores it in the database."""

    # Fetch the last calculated RSI date
    cur.execute("SELECT MAX(date) FROM sam_indicators WHERE symbol = %s AND rsi3 IS NOT NULL", (symbol,))
    # last_rsi_date = cur.fetchone()[0]
    end_date = datetime.now().date()
    last_rsi_date = end_date - timedelta(days=500)
    

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
        "UPDATE sam_indicators SET rsi3 = %s, rsi14 = %s, cci3 = %s, cci12 = %s, stochrsi14 = %s, stochrsi14_k = %s, stochrsi14_d = %s, bb_percent_b = %s  WHERE symbol = %s AND date = %s",
        updates
    )
    conn.commit()

    print(f"RSI3,rsi14, CCI3 & 12, StochRSI14, BBpB calculated and stored for {symbol}")
    

async def calculate_and_store_ma(symbol):
    # Initialize Redis connection
    # redis_client =  redis.Redis(host='localhost', port=6379, db=0)

    # Get last update timestamp
    # last_update = await get_last_update(symbol)

    # # Fetch only new data
    # cur.execute("""
    #     SELECT date, ma7, ma10, ma21, ma32, ma43, ma54, ma63, ma189, mamix14, mamix42, vwma15, vwma25, volsma5, volsma20
    #     FROM sam_indicators
    #     WHERE symbol = %s AND (date > %s OR %s IS NULL)
    #     ORDER BY date
    # """, (symbol, last_update, last_update))
    # rows = cur.fetchall()

    # if not rows:
    #     print(f"No new data to process for {symbol}")
    #     return
    
    end_date = datetime.now().date()
    last_ma_date = end_date - timedelta(days=500)
    
    
    daily_prices_columns = ["date", "open", "high", "low", "close", "volume"]
    sam_indicators_columns = ["rsi3", "rsi14", "ma7", "ma21", "ma63", "ma189", "mamix14", "mamix42", "vwma15", "vwma25", "volsma5", "volsma20", "cci3", "cci12", "bb_percent_b"]

    daily_prices_query = f"SELECT {', '.join(daily_prices_columns)} FROM daily_prices WHERE symbol = %s AND date >= %s ORDER BY date"
    cur.execute(daily_prices_query, (symbol, last_ma_date))
    daily_prices_rows = cur.fetchall()

    sam_indicators_query = f"SELECT {', '.join(sam_indicators_columns)} FROM sam_indicators WHERE symbol = %s AND date >= %s ORDER BY date"
    cur.execute(sam_indicators_query, (symbol, last_ma_date))
    sam_indicators_rows = cur.fetchall()

    
    
    daily_prices_df = pd.DataFrame(daily_prices_rows,  columns=daily_prices_columns)
    sam_indicators_df = pd.DataFrame(sam_indicators_rows, columns=sam_indicators_columns )
    
    # merged_data = pd.merge(daily_prices_data, sam_indicators_data, on='date')
    
    dates = daily_prices_df.loc[:, 'date'],
    # close = daily_prices_df.loc[:, 'close'],
    # volume = daily_prices_df.loc[:, 'volume'],
    

    df = pd.DataFrame({
    'close': [row[4] for row in daily_prices_rows],
    'volume': [row[5] for row in daily_prices_rows]
    }, index=dates)# close

    df = df.fillna(0)
    # Calculate moving averages
    
    for ma in [7, 10, 21, 32, 43, 54, 63, 189]:
        daily_prices_df[f'ma{ma}'] = daily_prices_df['close'].rolling(window=ma).mean()
    
    df['mamix14'] = ((df['ma7'] + df['ma21']) / 2).rolling(window=2).mean()
    df['mamix42'] = ((df['ma21'] + df['ma63']) / 2).rolling(window=2).mean()
    df['CloseVolume'] = df['close'] * df['volume']
    df['vwma15'] = df['CloseVolume'].rolling(window=15).sum() / df['volume'].rolling(window=15).sum()
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
            float(row['vwma15']),
            float(row['vwma25']),
            float(row['volsma5']),
            float(row['volsma20']),
            symbol, date
        ))
        # Cache the data
        # redis_client.set(f"{symbol}_{date}", json.dumps(row.to_dict()))


    # Perform bulk update
    await bulk_update(updates)

    print(f"Moving Averages are updated for {symbol}")

async def get_last_update(symbol):
    # cur.execute("SELECT MAX(date) FROM daily_prices WHERE symbol = %s", (symbol,))
    cur.execute("SELECT MIN(date) FROM sam_indicators WHERE symbol = %s", (symbol,))
    last_date = cur.fetchone()[0]
    return last_date if last_date else datetime(1970, 1, 1).date()

async def bulk_update(updates):
    cur.executemany("""
        UPDATE sam_indicators
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
            vwma15 = COALESCE(%s, vwma15),
            vwma25 = COALESCE(%s, vwma25),
            volsma5 = COALESCE(%s, volsma5),
            volsma20 = COALESCE(%s, volsma20)
        WHERE symbol = %s AND date = %s
    """, updates)
    conn.commit()

# asyncio.run(calculate_and_store_rsi_cci('ITC.NS'))
asyncio.run(calculate_and_store_ma('ITC.NS'))

conn.close()