import psycopg2
import pandas as pd 

conn = psycopg2.connect(
    dbname="stockdata",
    user="postgres",
    password="Subhan$007",
    host="localhost",
    port="5432"
)
cur = conn.cursor()


async def update_traderules(symbol):
    cur.execute("""
        SELECT date, open, high, low, close, rsi3, rsi14, mamix14, mamix42, vwma15, vwma25, volsma5, ma10, ma21, ma32, ma43, ma54, stochrsi14, volume, bb_percent_b,
               traderule1, traderule2, traderule3, traderule4, traderule5, traderule6, traderule7, traderule8, traderule9, traderule10 
        FROM daily_prices 
        WHERE symbol = %s 
        ORDER BY date
    """, (symbol,))
    rows = cur.fetchall()
    
    df = pd.DataFrame(rows, columns=['date', 'open', 'high', 'low', 'close', 'rsi3', 'rsi14', 'mamix14', 'mamix42', 'vwma15', 'vwma25','volsma5', 'ma10','ma21', 'ma32', 'ma43', 'ma54', 'stochrsi14', 'volume', 'bb_percent_b', 
                                     'traderule1', 'traderule2', 'traderule3', 'traderule4', 'traderule5', 'traderule6', 'traderule7', 'traderule8', 'traderule9', 'traderule10'])
    df.set_index('date', inplace=True)
    df = df.fillna(value=np.nan)


    hl2 = (df['high'] / df['low']) / 2
    # Calculate traderule1
    df['3down1up'] = (df['close'].shift(3) < df['open'].shift(3)) & \
                           (df['close'].shift(2) < df['open'].shift(2)) & \
                           (df['close'].shift(1) < df['open'].shift(1)) & \
                           (df['close'] > df['open']) #& \
                        #    (df['mamix42'].shift(5) / df['mamix42'] < 1 )
    # Calculate traderule2
    df['new_traderule2'] = (df['close'].shift(2) < df['open'].shift(2)) & \
                           (df['close'].shift(1) > df['open'].shift(1)) & \
                           (df['close'].shift(1) < df['open'].shift(2)) & \
                           (df['open'].shift(1) > df['close'].shift(2)) & \
                           (df['open'] < df['open'].shift(1)) & \
                           (df['close'] > df['close'].shift(1)) & \
                           (df['close'] > df['open'].shift(2)) #& \
                        #    (df['mamix42'].shift(5) / df['mamix42'] < 1 )

    # Calculate traderule3
    df['new_traderule3'] = (df['high'] > df['high'].shift(2)) & \
                           (df['high'].shift(1) < df['high'].shift(2)) & \
                           (df['open'].shift(2) > df['close'].shift(2)) & \
                           (df['open'] < df['close']) & \
                           (df['rsi3'] < 60) # & \
                        #    (df['mamix42'].shift(5) / df['mamix42'] < 1 )

    # Calculate traderule4
    df['new_traderule4'] = (df['close'].shift(2) < df['open'].shift(2)) & \
                           (df['close'].shift(1) > df['open'].shift(1)) & \
                           (df['close'].shift(1) > ((df['close'].shift(1).abs() + df['open'].shift(1).abs()) / 2)) & \
                           (df['close'].shift(1) < ((df['close'].shift(2).abs() + df['open'].shift(2).abs()) / 2)) & \
                           (df['open'] > ((df['high'].shift(1) + df['low'].shift(1)) / 2)) & \
                           (df['close'] > df['open']) #& \
                        #    (df['mamix42'].shift(5) / df['mamix42'] < 1 )
                           

    # Calculate traderule5
    df['new_traderule5'] = (df['close'].gt(df['mamix14'])) & \
                       (df['close'].shift(1).le(df['mamix14'].shift(1))) & \
                       (df['close'].gt(df['mamix42'])) & \
                       (df['close'].shift(1).le(df['mamix42'].shift(1))) # & \
                    #    (df['mamix42'].shift(5) / df['mamix42'] < 1 )
    
    # Calculate traderule6
    df['new_traderule6'] = (df['close'].shift(1) < df['open'].shift(1)) & \
                        (df['close'] > df['open']) & \
                        (df['open'] > df['close'].shift(1)) & \
                        (df['close'] < df['open'].shift(1)) & \
                        (df['close'] / df['open'] < 1.002) & \
                        (df['close'] < df['vwma25'] * 1.01) #& \
                        # (df['mamix42'].shift(5) / df['mamix42'] < 1 )
                            
    # Calculate traderule7
    df['samies_morningstar'] = (df['close'].shift(2) < df['open'].shift(2)) & \
                        (df['close'].shift(1) > df['low'].shift(2)) & \
                        (df['close'].shift(1) > ((df['high'].shift(1) + df['low'].shift(1)) / 2)) & \
                        (df['close'].shift(1) < df['high'].shift(2)) & \
                        (df['close'] > df['high'].shift(1)) & \
                        (df['close'] > ((df['high'] + df['low']) / 2)) #& \
                        # (df['mamix42'].shift(5) / df['mamix42'] < 1 )
                            
    # Calculate traderule8 - already update database with the close > mamix
    df['new_traderule8'] = (df['close'] > df['ma10']) & \
                        (df['close'] > df['ma21']) & \
                        (df['close'] > df['ma32']) & \
                        (df['close'] > df['ma43']) & \
                        (df['close'] > df['ma54']) & \
                        (df['volume'] > df['volsma5']) & \
                        (df['close'] > df['high'].shift(1)) & \
                        (df['close'].shift(1) <= df['ma54'].shift(1)) & \
                        (df['close'] > ((df['high'] + df['low']) / 2)) #& \
                        # (df['mamix42'].shift(5) / df['mamix42'] < 1 )
                        
    # Bollinger band percent b breakout                        
    df['bbpct_b'] = (df['bb_percent_b'] > 0.5) & (df['bb_percent_b'].shift(1) < 0.5) & \
                                (df['mamix42'].shift(5) / df['mamix42'] < 1 )
                                # (df['close'] > df['mamix42'])

    
    ## bbpct_b and rsi14 oversold reversal candle.
    df['perfect_bottom'] =  (df['bb_percent_b'].shift(1) < 0.4) & \
                            (df['rsi3'].shift(1) < 20) & \
                            (df['rsi14'].shift(1) < 40) & \
                            (df['close'] > ((df['high'] + df['low']) / 2)) #& \
                            # (df['mamix42'].shift(5) / df['mamix42'] < 1 )
                            
    # Find rows where traderules have changed
    changed_rows = df[(df['traderule1'] != df['3down1up']) | 
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
    updates = [(bool(row['3down1up']), 
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
