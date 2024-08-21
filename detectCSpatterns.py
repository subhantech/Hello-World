import tulipy as ti
import numpy as np
import pandas as pd
import psycopg2
import datetime

def get_filtered_patterns():
    # Database connection
    try:
        conn = psycopg2.connect(
            dbname="stockdata",
            user="postgres",
            password="Subhan$007",
            host="localhost",
            port="5432"
        )
        cur = conn.cursor()

        cur.execute("SELECT date, symbol, open, high, low, close FROM daily_prices WHERE date >= NOW() - INTERVAL '10 days' ORDER BY date")
        rows = cur.fetchall()
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        return None
    # finally:
    #     if conn:
    #         conn.close()

    # Convert fetched data to DataFrame
    data = {
        'Date': [row[0] for row in rows],
        'Symbol': [row[1] for row in rows],
        'Open': [round(row[2], 2) for row in rows],
        'High': [round(row[3], 2) for row in rows],
        'Low': [round(row[4], 2) for row in rows],
        'Close': [round(row[5], 2) for row in rows],
    }

    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)
    df.index = pd.to_datetime(df.index)  # Ensure the index is a DatetimeIndex

    open_prices = df['Open'].values
    high_prices = df['High'].values
    low_prices = df['Low'].values
    close_prices = df['Close'].values

    # Define a function to detect patterns
    def detect_patterns(open_prices, high_prices, low_prices, close_prices):
        # Initialize arrays to hold pattern detections
        morning_star = np.zeros(len(open_prices), dtype=bool)
        three_white_soldiers = np.zeros(len(open_prices), dtype=bool)
        bullish_harami = np.zeros(len(open_prices), dtype=bool)
        rising_three_methods = np.zeros(len(open_prices), dtype=bool)
        three_inside_up = np.zeros(len(open_prices), dtype=bool)
        inverted_hammer = np.zeros(len(open_prices), dtype=bool)
        tasuki_gap = np.zeros(len(open_prices), dtype=bool)
        mat_hold = np.zeros(len(open_prices), dtype=bool)
        upside_tasuki_gap = np.zeros(len(open_prices), dtype=bool)
        three_line_strike = np.zeros(len(open_prices), dtype=bool)

        # Define helper functions
        def is_small_body(open_price, high_price, low_price, close_price):
            body_size = abs(close_price - open_price)
            total_range = high_price - low_price
            return body_size < (total_range * 0.2)

        def is_long(open_price, high_price, low_price, close_price, threshold=0.05):
            return (high_price - low_price) > (high_price * threshold)

        def is_bullish(open_price, close_price):
            return close_price > open_price

        def is_bearish(open_price, close_price):
            return open_price > close_price

        def has_gap(open_price, prev_close_price, threshold=0.05):
            return abs(open_price - prev_close_price) > (open_price * threshold)

        def has_long_upper_shadow(open_price, high_price, close_price):
            upper_shadow = high_price - max(open_price, close_price)
            body_size = abs(close_price - open_price)
            return upper_shadow > (body_size * 5)

        def has_small_lower_shadow(open_price, low_price, close_price):
            lower_shadow = min(open_price, close_price) - low_price
            body_size = abs(close_price - open_price)
            return lower_shadow < (body_size * 0.1)

        # Iterate over the data to detect patterns
        for i in range(2, len(open_prices) - 2):
            # Morning Star
            if (is_bearish(open_prices[i-2], close_prices[i-2]) and
                is_small_body(open_prices[i-1], high_prices[i-1], low_prices[i-1], close_prices[i-1]) and
                is_bullish(open_prices[i], close_prices[i]) and
                close_prices[i] > (open_prices[i-2] + (close_prices[i-2] - open_prices[i-2]) / 2)):
                morning_star[i] = True


            # Three White Soldiers
            if (is_bullish(open_prices[i-2], close_prices[i-2]) and
                is_bullish(open_prices[i-1], close_prices[i-1]) and
                is_bullish(open_prices[i], close_prices[i]) and
                close_prices[i-2] < close_prices[i-1] and
                close_prices[i-1] < close_prices[i]):
                three_white_soldiers[i] = True

            # Bullish Harami
            if (is_bearish(open_prices[i-2], close_prices[i-2]) and
                is_small_body(open_prices[i-1], high_prices[i-1], low_prices[i-1], close_prices[i-1]) and
                is_bullish(open_prices[i], close_prices[i]) and
                close_prices[i] > (open_prices[i-2] + (close_prices[i-2] - open_prices[i-2]) / 2) and
                close_prices[i-1] > open_prices[i-2] and close_prices[i-1] < close_prices[i-2]):
                bullish_harami[i] = True

            # Rising Three Methods
            if (is_bearish(open_prices[i-2], close_prices[i-2]) and
                is_small_body(open_prices[i-1], high_prices[i-1], low_prices[i-1], close_prices[i-1]) and
                is_bearish(open_prices[i], close_prices[i]) and
                open_prices[i] > close_prices[i-1] and
                close_prices[i] > open_prices[i-2]):
                rising_three_methods[i] = True

            # Three Inside Up
            if (is_bullish(open_prices[i-2], close_prices[i-2]) and
                is_bearish(open_prices[i-1], close_prices[i-1]) and
                is_bullish(open_prices[i], close_prices[i]) and
                open_prices[i-1] > close_prices[i-2] and
                close_prices[i] > open_prices[i-2]):
                three_inside_up[i] = True

            # Inverted Hammer
            if (is_bearish(open_prices[i-2], close_prices[i-2]) and
                is_small_body(open_prices[i], high_prices[i], low_prices[i], close_prices[i]) and
                has_long_upper_shadow(open_prices[i], high_prices[i], close_prices[i]) and
                has_small_lower_shadow(open_prices[i], low_prices[i], close_prices[i]) and
                is_bullish(open_prices[i], close_prices[i]) and
                close_prices[i] > (open_prices[i-2] + (close_prices[i-2] - open_prices[i-2]) / 2)):
                inverted_hammer[i] = True

            # Tasuki Gap
            if (has_gap(open_prices[i-2], close_prices[i-3]) and
                has_gap(open_prices[i-1], close_prices[i-2]) and
                has_gap(open_prices[i], close_prices[i-1])):
                tasuki_gap[i] = True

            # Mat Hold
            if (is_long(open_prices[i-2], high_prices[i-2], low_prices[i-2], close_prices[i-2]) and
                is_small_body(open_prices[i-1], high_prices[i-1], low_prices[i-1], close_prices[i-1]) and
                has_gap(open_prices[i-1], close_prices[i-2]) and
                is_long(open_prices[i], high_prices[i], low_prices[i], close_prices[i]) and
                open_prices[i] < close_prices[i-1] and close_prices[i] > open_prices[i-2]):
                mat_hold[i] = True

            # Upside Tasuki Gap
            if (is_bearish(open_prices[i-2], close_prices[i-2]) and
                has_gap(open_prices[i-1], close_prices[i-2]) and
                is_bullish(open_prices[i], close_prices[i]) and
                open_prices[i] > close_prices[i-1] and close_prices[i] > open_prices[i-2]):
                upside_tasuki_gap[i] = True

            # Three-Line Strike
            if (abs(open_prices[i-2] - close_prices[i-2]) < 0.01 * (high_prices[i-2] - low_prices[i-2]) and
                abs(open_prices[i-1] - close_prices[i-1]) < 0.01 * (high_prices[i-1] - low_prices[i-1]) and
                abs(open_prices[i] - close_prices[i]) < 0.01 * (high_prices[i] - low_prices[i])):
                three_line_strike[i] = True

        # Return a DataFrame with pattern detections
        patterns = pd.DataFrame({
            'Date': df.index.strftime('%Y-%m-%d'),
            'Symbol': df['Symbol'],
            'MorningStar': morning_star,
            'ThreeWhiteSoldiers': three_white_soldiers,
            'BullishHarami': bullish_harami,
            'RisingThreeMethods': rising_three_methods,
            'ThreeInsideUp': three_inside_up,
            'InvertedHammer': inverted_hammer,
            'TasukiGap': tasuki_gap,
            'MatHold': mat_hold,
            'UpsideTasukiGap': upside_tasuki_gap,
            'ThreeLineStrike': three_line_strike
        })

        return patterns

    patterns = detect_patterns(open_prices, high_prices, low_prices, close_prices)
    filtered_patterns = patterns[patterns.iloc[:, 2:].any(axis=1)]
    return filtered_patterns

if __name__ == "__main__":
    # This block will only run if the script is executed directly
    result = get_filtered_patterns()
    print(result)
    
    
else:
    # This allows other scripts to import and use get_filtered_patterns
    pass

