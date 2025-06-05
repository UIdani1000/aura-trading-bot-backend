import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from pybit.unified_trading import HTTP
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
import random
import json # Import json for structured responses
import numpy as np # Import numpy to handle NaN values
import time # Import time for delays
import traceback # Import traceback for detailed error logging

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__) # Initialize your Flask app first
CORS(app) # THEN enable CORS for the app instance

# --- Configuration ---
BYBIT_API_KEY = os.environ.get('BYBIT_API_KEY')
BYBIT_API_SECRET = os.environ.get('BYBIT_API_SECRET')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY') # Ensure this matches your .env

# Initialize Bybit client
if BYBIT_API_KEY and BYBIT_API_SECRET:
    bybit_client = HTTP(
        api_key=BYBIT_API_KEY,
        api_secret=BYBIT_API_SECRET,
        testnet=False, # Set to True for testnet
        timeout=30 # Increased timeout to 30 seconds
    )
    print("--- Bybit client initialized ---")
else:
    bybit_client = None
    print("--- Bybit API keys not found. Bybit client not initialized. ---")

# Initialize Google Gemini
try:
    import google.generativeai as genai
    genai.configure(api_key=GOOGLE_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.0-flash') # Using gemini-2.0-flash
    print("--- Successfully initialized model to gemini-2.0-flash at startup ---")
except Exception as e:
    gemini_model = None
    print(f"--- Failed to initialize Gemini model: {e} ---")

# --- Helper Functions ---

def get_indicator_value(df, indicator_name, default_val="N/A"):
    """
    Helper to safely get the most recent non-NaN indicator value from the DataFrame.
    Assumes DataFrame is ordered oldest to newest (iloc[-1] is most recent).
    """
    if indicator_name in df.columns:
        # Drop NaN values and get the last (most recent) valid value
        valid_values = df[indicator_name].dropna()
        if not valid_values.empty:
            return round(float(valid_values.iloc[-1]), 2) # Use iloc[-1] as DataFrame is oldest to newest
    return default_val

# Function to fetch live market data (for dashboard)
@app.route('/all_market_prices', methods=['GET'])
def get_all_market_prices():
    if not bybit_client:
        print("Error: Bybit client not initialized in get_all_market_prices.")
        return jsonify({"error": "Bybit client not initialized"}), 500

    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT"]
    market_data = {}

    for symbol in symbols:
        print(f"\n--- Fetching daily kline data for {symbol} ---")
        try:
            # Fetch kline data for 1-day interval to get OHLCV for indicator calculation
            # Bybit kline data is ordered from newest to oldest.
            kline_response = bybit_client.get_kline(
                category="spot",
                symbol=symbol,
                interval="D", # Daily interval for general market overview
                limit=500 # Increased limit to ensure enough data for indicators
            )
            kline_data = kline_response.get('result', {}).get('list', [])
            # print(f"Raw kline_data for {symbol} (first 5 and last 5): {kline_data[:5]} ... {kline_data[-5:]}")
            # print(f"Number of kline entries for {symbol}: {len(kline_data)}")

            if kline_data and len(kline_data) >= 2: # Need at least 2 candles for percent change
                # Convert to DataFrame
                df = pd.DataFrame(kline_data, columns=['start', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
                df['close'] = pd.to_numeric(df['close'])
                df['volume'] = pd.to_numeric(df['volume'])
                df['start'] = pd.to_datetime(df['start'], unit='ms') # Convert start to datetime
                df.set_index('start', inplace=True) # Set index to datetime
                df = df.iloc[::-1] # CRITICAL FIX: Reverse DataFrame to be OLDEST TO NEWEST

                # Now, last_close is df['close'].iloc[-1] (most recent)
                last_close = df['close'].iloc[-1]
                # Previous day's close is df['close'].iloc[-2]
                prev_close = df['close'].iloc[-2] if len(df) > 1 else last_close
                percent_change = ((last_close - prev_close) / prev_close) * 100 if prev_close != 0 else 0

                # Calculate indicators using pandas_ta, which adds specific column names
                df.ta.rsi(append=True) # Adds 'RSI_14'
                df.ta.macd(append=True) # Adds 'MACD_12_26_9', 'MACDH_12_26_9', 'MACDS_12_26_9'
                df.ta.stoch(append=True) # Adds 'STOCHk_14_3_3', 'STOCHd_14_3_3'

                # print(f"DataFrame Tail for {symbol} (after indicator calculation, should be newest data):\n{df.tail()}")

                orscr_signal = "NEUTRAL"
                # Access indicators using their pandas_ta generated names from the most recent candle (iloc[-1])
                rsi_val = get_indicator_value(df, 'RSI_14', default_val="N/A")
                macd_val = get_indicator_value(df, 'MACD_12_26_9', default_val="N/A")
                macds_val = get_indicator_value(df, 'MACDS_12_26_9', default_val="N/A")
                stoch_k_val = get_indicator_value(df, 'STOCHk_14_3_3', default_val="N/A")

                # Check if indicators are numeric before applying ORSCR logic
                if isinstance(rsi_val, (int, float)) and isinstance(macd_val, (int, float)) and isinstance(macds_val, (int, float)):
                    if rsi_val > 60 and macd_val > macds_val:
                        orscr_signal = "BUY"
                    elif rsi_val < 40 and macd_val < macds_val:
                        orscr_signal = "SELL"
                else:
                    orscr_signal = "N/A (Indicators not available for ORSCR logic)"

                market_data[symbol] = {
                    "price": round(float(last_close), 2),
                    "percent_change": round(float(percent_change), 2),
                    "rsi": rsi_val, # Use the safely retrieved value
                    "macd": macd_val, # Use the safely retrieved value
                    "stoch_k": stoch_k_val, # Use the safely retrieved value
                    "volume": round(float(df['volume'].iloc[-1]), 2), # Use iloc[-1] for newest volume
                    "orscr_signal": orscr_signal
                }
                print(f"Final market_data[{symbol}]: {market_data[symbol]}")
            else:
                print(f"Insufficient kline data for {symbol} to calculate full market details.")
                market_data[symbol] = {
                    "price": "N/A", "percent_change": "N/A", "rsi": "N/A",
                    "macd": "N/A", "stoch_k": "N/A", "volume": "N/A", "orscr_signal": "N/A"
                }

        except Exception as e:
            print(f"Error fetching or processing data for {symbol}: {e}")
            traceback.print_exc() # Print full traceback
            market_data[symbol] = {
                "price": "N/A", "percent_change": "N/A", "rsi": "N/A",
                "macd": "N/A", "stoch_k": "N/A", "volume": "N/A", "orscr_signal": "N/A"
            }
        
        time.sleep(0.5) # Add a small delay between API calls

    return jsonify(market_data)

# --- Chat Endpoint (Existing) ---
@app.route('/chat', methods=['POST'])
def chat():
    if not gemini_model:
        return jsonify({"error": "Gemini model not initialized"}), 500

    user_message = request.json.get('message')
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    # Fetch live market data for context
    market_data_response = get_all_market_prices()
    market_data = market_data_response.json # Get the JSON content directly

    # Fetch mock trade history for context
    trade_history = get_trades().json # Get the JSON content directly

    # Construct context for Gemini - MODIFIED FOR FRIENDLIER TONE
    context = f"""
    You are Aura, an AI trading assistant. Your goal is to provide insightful, helpful, and **friendly** responses to the user's trading-related questions.
    Adopt a **conversational, approachable, and encouraging tone**. Avoid overly formal or robotic language.
    You can use emojis if appropriate to convey friendliness (e.g., 😊📈).

    Here is the current live market data:
    {json.dumps(market_data, indent=2)}

    Here is the user's recent trade history:
    {json.dumps(trade_history, indent=2)}

    User's question: {user_message}
    """

    try:
        response = gemini_model.generate_content(context)
        ai_response = response.candidates[0].content.parts[0].text
        return jsonify({"response": ai_response})
    except Exception as e:
        print(f"Error generating content with Gemini: {e}")
        traceback.print_exc() # Print full traceback
        return jsonify({"error": f"Failed to get AI response: {e}"}), 500

# --- Trade Log Endpoints (Existing - using local JSON file) ---
TRADE_LOG_FILE = 'trades.json'

def load_trades():
    if os.path.exists(TRADE_LOG_FILE):
        with open(TRADE_LOG_FILE, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []

def save_trades(trades):
    with open(TRADE_LOG_FILE, 'w') as f:
        json.dump(trades, f, indent=4)

@app.route('/log_trade', methods=['POST'])
def log_trade():
    trade_data = request.json
    if not trade_data:
        return jsonify({"error": "No trade data provided"}), 400

    trades = load_trades()
    trade_data['id'] = len(trades) + 1 # Simple ID generation
    trade_data['timestamp'] = datetime.now().isoformat()
    trades.append(trade_data)
    save_trades(trades)
    return jsonify({"message": "Trade logged successfully!", "trade": trade_data}), 201

@app.route('/get_trades', methods=['GET'])
def get_trades():
    trades = load_trades()
    return jsonify(trades)

# --- ORMCR Analysis Endpoint ---

# Mapping for Bybit interval strings and data limits
BYBIT_INTERVAL_MAP = {
    "M1": {"interval": "1", "limit": 500}, # Increased limit
    "M5": {"interval": "5", "limit": 500}, # Increased limit
    "M15": {"interval": "15", "limit": 500}, # Increased limit
    "M30": {"interval": "30", "limit": 500}, # Increased limit
    "H1": {"interval": "60", "limit": 500}, # Increased limit
    "H4": {"interval": "240", "limit": 500}, # Increased limit
    "D1": {"interval": "D", "limit": 500}, # Increased limit
}

def fetch_real_ohlcv(symbol, interval_key):
    """Fetches real OHLCV data from Bybit for a given symbol and interval."""
    if not bybit_client:
        print("Bybit client not initialized. Cannot fetch real data.")
        return pd.DataFrame()

    bybit_interval = BYBIT_INTERVAL_MAP[interval_key]["interval"]
    limit = BYBIT_INTERVAL_MAP[interval_key]["limit"]

    try:
        # Bybit kline data is ordered from newest to oldest.
        kline_response = bybit_client.get_kline(
            category="spot", # Assuming spot market for now
            symbol=symbol,
            interval=bybit_interval,
            limit=limit
        )
        kline_data = kline_response.get('result', {}).get('list', [])

        if not kline_data:
            print(f"No kline data found for {symbol} on {interval_key} interval.")
            return pd.DataFrame()

        # Convert to DataFrame and reverse to oldest to newest for pandas_ta
        df = pd.DataFrame(kline_data, columns=['start', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
        df['start'] = pd.to_datetime(df['start'], unit='ms')
        df[['open', 'high', 'low', 'close', 'volume', 'turnover']] = df[['open', 'high', 'low', 'close', 'volume', 'turnover']].apply(pd.to_numeric)
        df.set_index('start', inplace=True)
        df = df.iloc[::-1] # Reverse the DataFrame to be oldest to newest

        return df

    except Exception as e:
        print(f"Error fetching real OHLCV data for {symbol} ({interval_key}): {e}")
        traceback.print_exc()
        return pd.DataFrame()


def calculate_indicators_for_df(df, indicators):
    """Calculates selected indicators for a DataFrame."""
    df_copy = df.copy()
    
    # Use append=True to add indicators directly to the DataFrame
    if "RSI" in indicators:
        df_copy.ta.rsi(append=True) # Adds 'RSI_14'
    if "MACD" in indicators:
        df_copy.ta.macd(append=True) # Adds 'MACD_12_26_9', 'MACDH_12_26_9', 'MACDS_12_26_9'
    if "Moving Averages" in indicators:
        df_copy.ta.ema(length=9, append=True) # Adds 'EMA_9' for ORMCR EMA alignment
        df_copy.ta.sma(length=20, append=True) # Adds 'SMA_20'
        df_copy.ta.ema(length=50, append=True) # Adds 'EMA_50'
    if "Bollinger Bands" in indicators:
        df_copy.ta.bbands(append=True) # Adds 'BBL_5_2.0', 'BBM_5_2.0', 'BBU_5_2.0' (default length 5, std 2)
    if "Stochastic Oscillator" in indicators:
        df_copy.ta.stoch(append=True) # Adds 'STOCHk_14_3_3', 'STOCHd_14_3_3'
    if "Volume" in indicators:
        # Volume is already in the data, just ensure it's numeric
        df_copy['volume'] = pd.to_numeric(df_copy['volume'])
    if "ATR" in indicators:
        df_copy.ta.atr(append=True) # Adds 'ATR_14'

    return df_copy

def apply_ormcr_logic(analysis_data):
    """Applies simplified ORMCR logic to the analysis data and prepares for Gemini."""
    overall_bias = "NEUTRAL"
    confirmation_status = "PENDING"
    confirmation_reason = "Initial analysis."
    entry_suggestion = "MONITOR"
    sl_price = "N/A"
    tp1_price = "N/A"
    tp2_price = "N/A"
    risk_in_points = "N/A"
    position_size_suggestion = "User to calculate"
    
    calculated_confidence_score = 50 # Base confidence
    calculated_signal_strength = "NEUTRAL"

    # Sort timeframes from highest to lowest for top-down analysis
    sorted_timeframes = sorted(analysis_data.keys(), key=lambda x: int(BYBIT_INTERVAL_MAP.get(x, {"interval": "0"}).get("interval")) if BYBIT_INTERVAL_MAP[x]["interval"].isdigit() else 999999, reverse=True)

    print(f"\n--- Starting ORMCR Logic ---")
    print(f"Sorted Timeframes: {sorted_timeframes}")

    trend_analysis = {}
    for tf in sorted_timeframes:
        df = analysis_data[tf]['df']
        if df.empty:
            trend_analysis[tf] = "No data for analysis."
            print(f"  {tf}: No data for analysis.")
            continue

        last_close = get_indicator_value(df, 'close')
        ema9 = get_indicator_value(df, 'EMA_9')
        rsi = get_indicator_value(df, 'RSI_14')
        macd_hist = get_indicator_value(df, 'MACDH_12_26_9')

        tf_trend = "Neutral"
        if len(df) >= 2 and isinstance(ema9, (int, float)) and isinstance(last_close, (int, float)) and 'EMA_9' in df.columns:
            prev_ema9 = df['EMA_9'].iloc[-2] if not pd.isna(df['EMA_9'].iloc[-2]) else np.nan
            prev_close = df['close'].iloc[-2] if not pd.isna(df['close'].iloc[-2]) else np.nan

            if not pd.isna(prev_ema9) and not pd.isna(prev_close):
                if last_close > ema9 and prev_close > prev_ema9:
                    tf_trend = "Uptrend"
                elif last_close < ema9 and prev_close < prev_ema9:
                    tf_trend = "Downtrend"
        
        trend_analysis[tf] = {
            "trend": tf_trend,
            "last_close": last_close,
            "ema9": ema9,
            "rsi": rsi,
            "macd_hist": macd_hist
        }
        print(f"  {tf} Trend Analysis: {trend_analysis[tf]}")
    
    # Determine overall bias from higher timeframes (prioritizing D1, then H4, H1)
    for tf in ["D1", "H4", "H1", "M30", "M15", "M5", "M1"]:
        if tf in trend_analysis and trend_analysis[tf]["trend"] != "Neutral":
            overall_bias = trend_analysis[tf]["trend"].upper()
            print(f"  Overall Bias determined from {tf}: {overall_bias}")
            break
    print(f"Final Overall Bias: {overall_bias}")

    # Confirmation (focus on lowest timeframe for entry)
    lowest_tf = sorted_timeframes[-1] if sorted_timeframes else None
    print(f"Lowest Timeframe for Confirmation: {lowest_tf}")

    if lowest_tf and lowest_tf in analysis_data and not analysis_data[lowest_tf]['df'].empty:
        df_lowest = analysis_data[lowest_tf]['df']
        last_close_lowest = get_indicator_value(df_lowest, 'close')
        prev_close_lowest = df_lowest['close'].iloc[-2] if len(df_lowest) > 1 and not pd.isna(df_lowest['close'].iloc[-2]) else np.nan

        ema9_lowest = get_indicator_value(df_lowest, 'EMA_9')
        rsi_lowest = get_indicator_value(df_lowest, 'RSI_14')
        macd_lowest = get_indicator_value(df_lowest, 'MACD_12_26_9')
        macds_lowest = get_indicator_value(df_lowest, 'MACDS_12_26_9')
        stoch_k_lowest = get_indicator_value(df_lowest, 'STOCHk_14_3_3')
        stoch_d_lowest = get_indicator_value(df_lowest, 'STOCHd_14_3_3') # Added Stochastic D

        print(f"  Lowest TF ({lowest_tf}) Indicator Values:")
        print(f"    Last Close: {last_close_lowest}, Prev Close: {prev_close_lowest}")
        print(f"    EMA9: {ema9_lowest}, RSI: {rsi_lowest}")
        print(f"    MACD: {macd_lowest}, MACDS: {macds_lowest}, STOCH_K: {stoch_k_lowest}, STOCH_D: {stoch_d_lowest}")

        required_indicators = {
            'Last Close': last_close_lowest,
            'Prev Close': prev_close_lowest,
            'EMA9': ema9_lowest,
            'RSI': rsi_lowest,
            'MACD': macd_lowest,
            'MACDS': macds_lowest,
            'STOCHk': stoch_k_lowest,
            'STOCHd': stoch_d_lowest # Include STOCHd in check
        }
        
        missing_indicators = [name for name, val in required_indicators.items() if val == "N/A" or pd.isna(val)]

        if missing_indicators:
            confirmation_reason = f"Insufficient numeric data for lowest timeframe confirmation. Missing or NaN indicators: {', '.join(missing_indicators)}."
            calculated_confidence_score = 30 # Low confidence if key indicators are missing
        else:
            # Initialize conditions met counter
            conditions_met = 0
            total_conditions = 5 # Price action, EMA, RSI, MACD, Stochastic

            # Condition 1: Price Action (Strong Candle)
            if overall_bias == "BULLISH" and last_close_lowest > prev_close_lowest * 1.001:
                conditions_met += 1
                confirmation_reason = "Strong bullish candle detected."
            elif overall_bias == "BEARISH" and last_close_lowest < prev_close_lowest * 0.999:
                conditions_met += 1
                confirmation_reason = "Strong bearish candle detected."
            else:
                confirmation_reason = "No strong directional candle for lowest timeframe."

            # Condition 2: Price vs. EMA9
            if overall_bias == "BULLISH" and last_close_lowest > ema9_lowest:
                conditions_met += 1
                confirmation_reason += " Price above EMA9."
            elif overall_bias == "BEARISH" and last_close_lowest < ema9_lowest:
                conditions_met += 1
                confirmation_reason += " Price below EMA9."
            else:
                confirmation_reason += " Price not aligned with EMA9."

            # Condition 3: RSI
            if overall_bias == "BULLISH" and rsi_lowest > 50:
                conditions_met += 1
                confirmation_reason += " RSI is bullish (>50)."
            elif overall_bias == "BEARISH" and rsi_lowest < 50:
                conditions_met += 1
                confirmation_reason += " RSI is bearish (<50)."
            else:
                confirmation_reason += " RSI is neutral (40-60)."

            # Condition 4: MACD Crossover
            if overall_bias == "BULLISH" and macd_lowest > macds_lowest:
                conditions_met += 1
                confirmation_reason += " MACD shows bullish crossover."
            elif overall_bias == "BEARISH" and macd_lowest < macds_lowest:
                conditions_met += 1
                confirmation_reason += " MACD shows bearish crossover."
            else:
                confirmation_reason += " MACD is not confirming direction."

            # Condition 5: Stochastic Oscillator (not overbought/oversold)
            if stoch_k_lowest > 20 and stoch_k_lowest < 80: # General mid-range for confirmation
                conditions_met += 1
                confirmation_reason += " Stochastic is in mid-range (20-80)."
            else:
                confirmation_reason += " Stochastic is in overbought/oversold zone."

            # Calculate confidence based on met conditions
            calculated_confidence_score = int((conditions_met / total_conditions) * 100)
            if calculated_confidence_score < 40: # Prevent very low scores from being 'BUY' or 'SELL'
                calculated_signal_strength = "NEUTRAL"
            elif overall_bias == "BULLISH":
                if calculated_confidence_score >= 80:
                    calculated_signal_strength = "STRONG BUY"
                    entry_suggestion = "ENTER NOW"
                elif calculated_confidence_score >= 60:
                    calculated_signal_strength = "BUY"
                    entry_suggestion = "ENTER NOW"
                else:
                    calculated_signal_strength = "MONITOR"
            elif overall_bias == "BEARISH":
                if calculated_confidence_score >= 80:
                    calculated_signal_strength = "STRONG SELL"
                    entry_suggestion = "ENTER NOW"
                elif calculated_confidence_score >= 60:
                    calculated_signal_strength = "SELL"
                    entry_suggestion = "ENTER NOW"
                else:
                    calculated_signal_strength = "MONITOR"
            else: # Overall bias is neutral
                calculated_signal_strength = "NEUTRAL"
                entry_suggestion = "MONITOR"

            # Set confirmation status based on entry suggestion
            if entry_suggestion == "ENTER NOW":
                confirmation_status = "STRONG CONFIRMATION"
                # Mock SL/TP based on a simple percentage of current price if confirmed
                if overall_bias == "BULLISH":
                    sl_price = round(last_close_lowest * 0.995, 2) # 0.5% SL
                    tp1_price = round(last_close_lowest * 1.01, 2) # 1% TP1
                    tp2_price = round(last_close_lowest * 1.02, 2) # 2% TP2
                    risk_in_points = round(last_close_lowest - sl_price, 2)
                elif overall_bias == "BEARISH":
                    sl_price = round(last_close_lowest * 1.005, 2) # 0.5% SL
                    tp1_price = round(last_close_lowest * 0.99, 2) # 1% TP1
                    tp2_price = round(last_close_lowest * 0.98, 2) # 2% TP2
                    risk_in_points = round(sl_price - last_close_lowest, 2)
                position_size_suggestion = "2.5% of balance (example)" # Placeholder
            else:
                confirmation_status = "PENDING"
                sl_price = "N/A"
                tp1_price = "N/A"
                tp2_price = "N/A"
                risk_in_points = "N/A"
                position_size_suggestion = "User to calculate"
            
            print(f"  Conditions Met: {conditions_met}/{total_conditions}")
            print(f"  Calculated Confidence Score: {calculated_confidence_score}%")
            print(f"  Calculated Signal Strength: {calculated_signal_strength}")
            print(f"  Confirmation Status: {confirmation_status}, Reason: {confirmation_reason}")
    else:
        confirmation_reason = "Not enough data for lowest timeframe confirmation or lowest timeframe not selected."
        calculated_confidence_score = 30 # Low confidence if no lowest TF data
        print(f"  Confirmation Reason: {confirmation_reason}")
        
    print(f"--- ORMCR Logic Finished ---")
    print(f"Final ORMCR Results: Overall Bias={overall_bias}, Confirmation Status={confirmation_status}, Reason={confirmation_reason}, Confidence={calculated_confidence_score}, Signal={calculated_signal_strength}")

    return {
        "overall_bias": overall_bias,
        "confirmation_status": confirmation_status,
        "confirmation_reason": confirmation_reason,
        "entry_suggestion": entry_suggestion,
        "sl_price": sl_price,
        "tp1_price": tp1_price,
        "tp2_price": tp2_price,
        "risk_in_points": risk_in_points,
        "position_size_suggestion": position_size_suggestion,
        "calculated_confidence_score": calculated_confidence_score, # New
        "calculated_signal_strength": calculated_signal_strength,   # New
        "trend_analysis_by_tf": {tf: {k: v for k, v in data.items() if k != 'df'} for tf, data in analysis_data.items()} # Exclude DataFrame
    }

@app.route('/run_ormcr_analysis', methods=['POST'])
def run_ormcr_analysis():
    if not gemini_model:
        return jsonify({"error": "Gemini model not initialized"}), 500
    if not bybit_client:
        return jsonify({"error": "Bybit client not initialized"}), 500

    data = request.json
    currency_pair = data.get('currencyPair')
    timeframes = data.get('timeframes', []) # Now expects a list
    trade_type = data.get('tradeType')
    indicators = data.get('indicators', [])
    available_balance = data.get('availableBalance')
    leverage = data.get('leverage')

    if not currency_pair or not timeframes:
        return jsonify({"error": "Currency pair and at least one timeframe are required"}), 400

    # Ensure timeframes are valid and sorted for top-down analysis
    valid_timeframes = [tf for tf in timeframes if tf in BYBIT_INTERVAL_MAP]
    if not valid_timeframes:
        return jsonify({"error": "No valid timeframes provided"}), 400
    
    # Sort timeframes from highest to lowest for top-down analysis based on interval value
    valid_timeframes.sort(key=lambda x: int(BYBIT_INTERVAL_MAP[x]["interval"]) if BYBIT_INTERVAL_MAP[x]["interval"].isdigit() else 999999, reverse=True)


    analysis_data_by_tf = {}
    bybit_symbol = currency_pair.replace('/', '') + 'T' # Convert BTC/USD to BTCUSDT

    # Step 1: Fetch real OHLCV data for each selected timeframe
    for tf in valid_timeframes:
        df = fetch_real_ohlcv(bybit_symbol, tf)
        
        if df.empty:
            print(f"Skipping {tf} due to empty DataFrame after fetching real data.")
            continue # Skip this timeframe if no data was fetched

        df_with_indicators = calculate_indicators_for_df(df, indicators)
        
        # Debugging prints for analysis dataframes
        # print(f"\n--- DataFrame for {tf} after indicator calculation (Head) ---")
        # print(df_with_indicators.head()) # Head will be oldest data
        # print(f"\n--- DataFrame for {tf} after indicator calculation (Tail) ---")
        # print(df_with_indicators.tail()) # Tail will be newest data

        # Ensure last_price and volume are floats and handle potential NaNs
        # Use iloc[-1] because DataFrame is now oldest to newest
        last_price_val = get_indicator_value(df_with_indicators, 'close') # Most recent close
        volume_val = get_indicator_value(df_with_indicators, 'volume') # Most recent volume


        analysis_data_by_tf[tf] = {
            "df": df_with_indicators,
            "last_price": last_price_val,
            "volume": volume_val
        }
        # Add a small delay between fetching data for different timeframes
        time.sleep(0.2) # 200 ms delay

    if not analysis_data_by_tf:
        return jsonify({"error": "No valid market data could be fetched for analysis. Please check currency pair and timeframes."}), 500


    # Step 2: Apply ORMCR logic
    ormcr_results = apply_ormcr_logic(analysis_data_by_tf)

    # Step 3: Construct prompt for Gemini
    # Ensure all numerical values in detailed_timeframe_data are properly converted
    detailed_tf_data_for_prompt = {}
    for tf, data in analysis_data_by_tf.items():
        indicators_snapshot_dict = {}
        # List all possible pandas_ta indicator column names
        indicator_cols = [
            'RSI_14', 'MACD_12_26_9', 'MACDH_12_26_9', 'MACDS_12_26_9',
            'EMA_9', 'SMA_20', 'EMA_50',
            'BBL_5_2.0', 'BBM_5_2.0', 'BBU_5_2.0', # Bollinger Bands
            'STOCHk_14_3_3', 'STOCHd_14_3_3', # Stochastic
            'ATR_14', # ATR
            'volume' # Volume is also a numeric column
        ]
        for ind_col in indicator_cols:
            # Use the improved get_indicator_value here as well
            indicators_snapshot_dict[ind_col] = get_indicator_value(data["df"], ind_col)
        
        detailed_tf_data_for_prompt[tf] = {
            "last_price": data["last_price"],
            "volume": data["volume"],
            "indicators_snapshot": indicators_snapshot_dict
        }
    
    # Debugging print for indicators_snapshot_dict before sending to Gemini
    print(f"\n--- Indicators Snapshot before sending to Gemini ---")
    print(json.dumps(detailed_tf_data_for_prompt, indent=2))
    print("-" * 40)


    prompt_parts = [
        "You are Aura, an advanced AI trading assistant specializing in the ORMCR strategy.",
        "The user has requested an analysis for:",
        f"- Currency Pair: {currency_pair}",
        f"- Selected Timeframes (highest to lowest): {', '.join(valid_timeframes)}",
        f"- Intended Trade Type: {trade_type}",
        f"- Selected Indicators: {', '.join(indicators) if indicators else 'None'}",
        f"- Available Balance: ${available_balance}",
        f"- Leverage: {leverage}",
        "\nHere is the detailed market data and ORMCR analysis results from our internal system:",
        json.dumps({
            "ormcr_analysis": ormcr_results,
            "detailed_timeframe_data": detailed_tf_data_for_prompt
        }, indent=2),
        "\nBased on this information, provide a comprehensive AI ANALYSIS RESULTS. Follow this exact structure:",
        """
        {
            "confidence_score": "X%",
            "signal_strength": "STRONG BUY/BUY/NEUTRAL/SELL/STRONG SELL",
            "market_summary": "Detailed summary based on multi-timeframe trend, price action, and key indicators.",
            "ai_suggestion": {
                "entry_type": "BUY ORDER/SELL ORDER/WAIT",
                "recommended_action": "ENTER NOW/MONITOR/AVOID",
                "position_size": "X% of balance"
            },
            "stop_loss": {
                "price": "$X.XX",
                "percentage_change": "X.XX%"
            },
            "take_profit_1": {
                "price": "$X.XX",
                "percentage_change": "X.XX%"
            },
            "take_profit_2": {
                "price": "$X.XX",
                "percentage_change": "X.XX%"
            },
            "technical_indicators_analysis": "Based on the 'indicators_snapshot' provided in 'detailed_timeframe_data', interpret the values for all selected indicators (RSI, MACD, Stochastic, Moving Averages, Bollinger Bands, Volume, ATR, Fibonacci Retracements if applicable) across the relevant timeframes, focusing on the lowest timeframe for potential entry signals. Explain what each indicator suggests about market conditions (e.g., overbought/oversold for RSI, momentum for MACD, volatility for Bollinger Bands, etc.).",
            "next_step_for_user": "What the user should do next (e.g., 'Monitor for confirmation', 'Proceed with the trade', 'Review other timeframes')."
        }
        """,
        "\n**IMPORTANT:**",
        f"- The 'confidence_score' should be '{ormcr_results['calculated_confidence_score']}%' based on the backend's calculation.",
        f"- The 'signal_strength' should be '{ormcr_results['calculated_signal_strength']}' based on the backend's calculation.",
        f"- If 'ormcr_confirmation_status' is 'PENDING', ensure 'entry_type' is 'WAIT' and 'recommended_action' is 'MONITOR', and explain why in 'market_summary' and 'next_step_for_user'. Also, if 'ormcr_confirmation_status' is 'PENDING', provide 'N/A' for SL/TP prices and percentages.",
        "\n**IMPORTANT: Maintain a friendly, conversational, and encouraging tone throughout your response. Use simple, clear language and feel free to include relevant emojis to enhance friendliness (e.g., 😊📈).**"
    ]

    try:
        # Use response_schema for structured output
        response = gemini_model.generate_content(
            prompt_parts,
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": {
                    "type": "OBJECT",
                    "properties": {
                        "confidence_score": {"type": "STRING"},
                        "signal_strength": {"type": "STRING"},
                        "market_summary": {"type": "STRING"},
                        "ai_suggestion": {
                            "type": "OBJECT",
                            "properties": {
                                "entry_type": {"type": "STRING"},
                                "recommended_action": {"type": "STRING"},
                                "position_size": {"type": "STRING"}
                            }
                        },
                        "stop_loss": {
                            "type": "OBJECT",
                            "properties": {
                                "price": {"type": "STRING"},
                                "percentage_change": {"type": "STRING"}
                            }
                        },
                        "take_profit_1": {
                            "type": "OBJECT",
                            "properties": {
                                "price": {"type": "STRING"},
                                "percentage_change": {"type": "STRING"}
                            }
                        },
                        "take_profit_2": {
                            "type": "OBJECT",
                            "properties": {
                                "price": {"type": "STRING"},
                                "percentage_change": {"type": "STRING"}
                            }
                        },
                        "technical_indicators_analysis": {"type": "STRING"}, # Changed to STRING
                        "next_step_for_user": {"type": "STRING"}
                    },
                    "required": ["confidence_score", "signal_strength", "market_summary", "ai_suggestion", "stop_loss", "take_profit_1", "take_profit_2", "technical_indicators_analysis", "next_step_for_user"]
                }
            }
        )
        
        # Parse the JSON string response
        ai_analysis_results = json.loads(response.candidates[0].content.parts[0].text)

        # Add ORMCR specific flags for frontend to interpret
        ai_analysis_results['ormcr_confirmation_status'] = ormcr_results['confirmation_status']
        ai_analysis_results['ormcr_overall_bias'] = ormcr_results['overall_bias']
        ai_analysis_results['ormcr_reason'] = ormcr_results['confirmation_reason']

        # If confirmation is pending, ensure SL/TP are N/A in the final response to frontend
        if ai_analysis_results['ormcr_confirmation_status'] == "PENDING":
            ai_analysis_results['stop_loss']['price'] = "N/A"
            ai_analysis_results['stop_loss']['percentage_change'] = "N/A"
            ai_analysis_results['take_profit_1']['price'] = "N/A"
            ai_analysis_results['take_profit_1']['percentage_change'] = "N/A"
            ai_analysis_results['take_profit_2']['price'] = "N/A"
            ai_analysis_results['take_profit_2']['percentage_change'] = "N/A"


        return jsonify(ai_analysis_results)

    except Exception as e:
        print(f"Error generating ORMCR analysis with Gemini: {e}")
        traceback.print_exc() # Print full traceback
        return jsonify({"error": f"Failed to get AI analysis: {e}"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 10000), debug=False)
