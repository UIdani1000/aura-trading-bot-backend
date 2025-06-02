import os
import datetime
import json
import time
import requests
import google.generativeai as genai
import logging

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# --- NEW IMPORTS FOR BINANCE DATA & INDICATORS ---
from binance.client import Client # type: ignore
import pandas as pd # type: ignore
import pandas_ta as ta # type: ignore
# --- END NEW IMPORTS ---

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)

TRADES_FILE = 'trades.json'
ANALYSIS_FILE = 'analysis_results.json'

app.logger.info("--- APP.PY STARTED SUCCESSFULLY ---")

# --- Configure Gemini API Key ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    app.logger.error("GOOGLE_API_KEY not found in environment variables.")
else:
    genai.configure(api_key=GOOGLE_API_KEY)


# --- Configure Binance API Keys ---
BINANCE_API_KEY = os.environ.get('BINANCE_API_KEY')
BINANCE_API_SECRET = os.environ.get('BINANCE_API_SECRET')

# Initialize Binance client (for fetching candlestick data for indicators)
binance_client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
app.logger.info("--- Binance client initialized ---")


# --- IMMEDIATE DEBUG: List available Gemini models at startup ---
app.logger.info("--- Attempting to list available Gemini models at application startup ---")
try:
    startup_models = []
    for m in genai.list_models():
        startup_models.append(m.name)
    app.logger.info(f"--- STARTUP MODELS: {', '.join(startup_models) if startup_models else 'None found'} ---")
except Exception as e:
    app.logger.error(f"--- ERROR LISTING MODELS AT STARTUP: {e} ---")
app.logger.info("--- Finished attempting to list models at startup ---")

model = None
try:
<<<<<<< HEAD
    model = genai.GenerativeModel("gemini-1.5-flash")
    app.logger.info("--- Successfully initialized model to gemini-1.5-flash at startup ---")
=======
    # --- IMPORTANT CHANGE: Using gemini-1.5-flash now ---
    model = genai.GenerativeModel("gemini-1.5-flash")
    print("--- Successfully initialized model to gemini-1.5-flash at startup ---")
>>>>>>> 83585c9c02c56cf767234a8b28cd9add0124928d
except Exception as e:
    app.logger.error(f"Initial attempt to load gemini-1.5-flash failed at startup: {e}")


# --- Caching for Market Prices (Global for app.py) ---
<<<<<<< HEAD
# This cache will now store data from Binance including indicators
market_prices_cache = {}
last_updated_time = 0
# Dashboard cache duration: 50 seconds (slightly less than frontend's 60s fetch to ensure fresh data)
CACHE_DURATION = 50 
=======
last_market_data = None
last_fetch_time = 0
# Increased cache duration to 30 seconds to reduce CoinGecko 429 errors
CACHE_DURATION = 30
>>>>>>> 83585c9c02c56cf767234a8b28cd9add0124928d

# Ensure trades.json and analysis_results.json exist
def init_db():
    if not os.path.exists(TRADES_FILE):
        with open(TRADES_FILE, 'w') as f:
            json.dump([], f)
    if not os.path.exists(ANALYSIS_FILE):
        with open(ANALYSIS_FILE, 'w') as f:
            json.dump([], f)

init_db()

# Function to read trades from JSON file
def read_trades():
    if not os.path.exists(TRADES_FILE):
        return []
    with open(TRADES_FILE, 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []

# Function to write trades to JSON file
def write_trades(trades):
    with open(TRADES_FILE, 'w') as f:
        json.dump(trades, f, indent=4)

# Function to read analysis results
def read_analysis_results():
    if not os.path.exists(ANALYSIS_FILE):
        return []
    with open(ANALYSIS_FILE, 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []

# Function to write analysis results
def write_analysis_results(results):
    with open(ANALYSIS_FILE, 'w') as f:
        json.dump(results, f, indent=4)

@app.route('/')
def home():
    return "Aura Trading Bot Backend is running!"

@app.route('/log_trade', methods=['POST'])
def log_trade():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    required_fields = ['pair', 'trade_type', 'entry_price', 'exit_price', 'profit_loss']
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400

    trades = read_trades()
    new_id = len(trades) + 1
    
    trade_entry = {
        "id": new_id,
        "pair": data['pair'],
        "trade_type": data['trade_type'],
        "entry_price": data['entry_price'],
        "exit_price": data['exit_price'],
        "profit_loss": data['profit_loss'],
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    trades.append(trade_entry)
    write_trades(trades)
    
    return jsonify({"message": "Trade logged successfully!", "trade": trade_entry}), 201

@app.route('/get_trades', methods=['GET'])
def get_trades():
    trades = read_trades()
    return jsonify(trades), 200

@app.route('/get_trade_summary', methods=['GET'])
def get_trade_summary():
    trades = read_trades()

    total_profit_loss = sum(trade['profit_loss'] for trade in trades)
    total_trades = len(trades)
    
    profitable_trades = sum(1 for trade in trades if trade['profit_loss'] > 0)
    win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0.0

    average_profit_per_trade = (total_profit_loss / total_trades) if total_trades > 0 else 0.0

    summary = {
        "total_profit_loss": total_profit_loss,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "average_profit_per_trade": average_profit_per_trade
    }
    
    return jsonify(summary), 200

# --- NEW FUNCTION: Fetch market data with indicators from Binance ---
def get_market_data_with_indicators(symbol: str, interval: str = '1h', limit: int = 100):
    """
    Fetches candlestick data from Binance and calculates RSI, MACD, Stochastic, and Volume.
    """
    try:
<<<<<<< HEAD
        klines = binance_client.get_klines(symbol=symbol, interval=interval, limit=limit)

        # Convert to Pandas DataFrame
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        df['close'] = pd.to_numeric(df['close'])
        df['open'] = pd.to_numeric(df['open'])
        df['high'] = pd.to_numeric(df['high'])
        df['low'] = pd.to_numeric(df['low'])
        df['volume'] = pd.to_numeric(df['volume'])
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')

        # Calculate indicators using pandas_ta
        # Ensure 'close', 'high', 'low', 'open', 'volume' columns are available for indicators
        df.ta.macd(append=True) # MACD
        df.ta.rsi(append=True)  # RSI
        df.ta.stoch(append=True)# Stochastic Oscillator (K & D)

        # Get the latest data point
        latest_data = df.iloc[-1]
        previous_data = df.iloc[-2] if len(df) > 1 else None

        current_price = latest_data['close']
        previous_close_price = previous_data['close'] if previous_data is not None else current_price

        # Calculate percentage change for the last period
        percent_change = ((current_price - previous_close_price) / previous_close_price) * 100 if previous_close_price else 0

        # Implement your actual ORSCR Strategy logic here
        # This is a highly simplified example. You'll need to define your actual ORSCR logic.
        orscr_signal = "NEUTRAL"
        if latest_data['RSI_14'] and latest_data['RSI_14'] > 70:
            orscr_signal = "SELL"
        elif latest_data['RSI_14'] and latest_data['RSI_14'] < 30:
            orscr_signal = "BUY"
        elif latest_data['MACDh_12_26_9'] and latest_data['MACDh_12_26_9'] > 0 and previous_data is not None and previous_data['MACDh_12_26_9'] <= 0:
            orscr_signal = "BUY" # MACD Histogram crossing above zero
        elif latest_data['MACDh_12_26_9'] and latest_data['MACDh_12_26_9'] < 0 and previous_data is not None and previous_data['MACDh_12_26_9'] >= 0:
            orscr_signal = "SELL" # MACD Histogram crossing below zero

        return {
            "price": current_price,
            "percent_change": percent_change,
            "rsi": round(latest_data['RSI_14'], 2) if 'RSI_14' in latest_data and pd.notna(latest_data['RSI_14']) else "N/A",
            "macd": round(latest_data['MACD_12_26_9'], 2) if 'MACD_12_26_9' in latest_data and pd.notna(latest_data['MACD_12_26_9']) else "N/A",
            "macd_histogram": round(latest_data['MACDh_12_26_9'], 2) if 'MACDh_12_26_9' in latest_data and pd.notna(latest_data['MACDh_12_26_9']) else "N/A",
            "stoch_k": round(latest_data['STOCHk_14_3_3'], 2) if 'STOCHk_14_3_3' in latest_data and pd.notna(latest_data['STOCHk_14_3_3']) else "N/A",
            "stoch_d": round(latest_data['STOCHd_14_3_3'], 2) if 'STOCHd_14_3_3' in latest_data and pd.notna(latest_data['STOCHd_14_3_3']) else "N/A",
            "volume": round(latest_data['volume'], 2) if 'volume' in latest_data and pd.notna(latest_data['volume']) else "N/A",
            "orscr_signal": orscr_signal # Your calculated signal
        }
    except Exception as e:
        app.logger.error(f"Error fetching/calculating data for {symbol}: {e}")
        return None
=======
        response = requests.get(
            f"https://api.coingecko.com/api/v3/simple/price?ids={ids_string}&vs_currencies=usd&include_24hr_change=true"
        )
        response.raise_for_status()
        market_data = response.json()
        
        formatted_data = {}
        for pair, info_id in coin_ids.items():
            if info_id in market_data and 'usd' in market_data[info_id]:
                price = market_data[info_id]['usd']
                change_24h = market_data[info_id].get('usd_24h_change', 0)
                formatted_data[pair] = {
                    "price": price,
                    "change": change_24h / 100 * price,
                    "percent_change": change_24h
                }
            else:
                formatted_data[pair] = {
                    "price": 0.0,
                    "change": 0.0,
                    "percent_change": 0.0
                }
        
        last_market_data = formatted_data
        last_fetch_time = time.time()
        return formatted_data

    except requests.exceptions.RequestException as e:
        app.logger.error(f"Error fetching market prices from CoinGecko for internal use: {e}")
        # IMPORTANT CHANGE: Return last known good data if new fetch fails
        return last_market_data if last_market_data is not None else {}
    except Exception as e:
        app.logger.error(f"An unexpected error occurred in _get_cached_market_data: {e}")
        # IMPORTANT CHANGE: Return last known good data if new fetch fails
        return last_market_data if last_market_data is not None else {}

>>>>>>> 83585c9c02c56cf767234a8b28cd9add0124928d

# --- MODIFIED: get_all_market_prices endpoint to use Binance and indicators ---
@app.route('/all_market_prices', methods=['GET'])
def get_all_market_prices():
    """Endpoint for dashboard display, now using Binance data with indicators."""
    global last_updated_time, market_prices_cache
    current_time = time.time()

    if current_time - last_updated_time < CACHE_DURATION and market_prices_cache:
        app.logger.info("Serving market data from cache for dashboard.")
        return jsonify(market_prices_cache)

    app.logger.info("Fetching fresh market prices and indicators for dashboard...")
    prices = {}

    # Fetch data for BTC/USD (Binance symbol: BTCUSDT)
    btc_data = get_market_data_with_indicators("BTCUSDT")
    if btc_data:
        prices["BTC/USD"] = btc_data
    else:
        # Fallback to mock data if API fails
        prices["BTC/USD"] = {
            "price": round(42000 + random.uniform(-500, 500), 2),
            "percent_change": round(random.uniform(-3, 3), 2),
            "rsi": round(random.uniform(30, 70), 2),
            "macd": round(random.uniform(-500, 500), 2),
            "macd_histogram": round(random.uniform(-100, 100), 2),
            "stoch_k": round(random.uniform(20, 80), 2),
            "stoch_d": round(random.uniform(20, 80), 2),
            "volume": round(random.uniform(100000, 500000), 2),
            "orscr_signal": random.choice(["BUY", "SELL", "NEUTRAL"])
        }

    # Fetch data for ETH/USD (Binance symbol: ETHUSDT)
    eth_data = get_market_data_with_indicators("ETHUSDT")
    if eth_data:
        prices["ETH/USD"] = eth_data
    else:
        # Fallback to mock data if API fails
        prices["ETH/USD"] = {
            "price": round(2500 + random.uniform(-50, 50), 2),
            "percent_change": round(random.uniform(-3, 3), 2),
            "rsi": round(random.uniform(30, 70), 2),
            "macd": round(random.uniform(-50, 50), 2),
            "macd_histogram": round(random.uniform(-10, 10), 2),
            "stoch_k": round(random.uniform(20, 80), 2),
            "stoch_d": round(random.uniform(20, 80), 2),
            "volume": round(random.uniform(50000, 200000), 2),
            "orscr_signal": random.choice(["BUY", "SELL", "NEUTRAL"])
        }

    market_prices_cache = prices
    last_updated_time = current_time
    return jsonify(prices)

# --- Kept _get_live_price_for_pair using CoinGecko for analysis requests ---
def _get_live_price_for_pair(coin_gecko_id):
    """Fetches the live price for a single coin directly from CoinGecko, bypassing cache, for analysis."""
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_gecko_id}&vs_currencies=usd&include_24hr_change=true"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        if coin_gecko_id in data and 'usd' in data[coin_gecko_id]:
            app.logger.info(f"Successfully fetched live price for {coin_gecko_id} from CoinGecko for analysis.")
            return data[coin_gecko_id]['usd']
        app.logger.warning(f"Live price data for {coin_gecko_id} not found in CoinGecko response for analysis.")
        return 0.0
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Error fetching LIVE price for {coin_gecko_id} from CoinGecko for analysis: {e}")
        return 0.0
    except Exception as e:
        app.logger.error(f"An unexpected error occurred fetching LIVE price for {coin_gecko_id} for analysis: {e}")
        return 0.0


@app.route('/generate_analysis', methods=['POST'])
def generate_analysis():
    """Generates analysis based on live price for the requested pair (using CoinGecko for price)."""
    data = request.get_json()
    pair = data.get('pair')
    timeframes = data.get('timeframes')
    indicators = data.get('indicators')
    trade_type = data.get('trade_type')
    balance_range = data.get('balance_range')
    leverage = data.get('leverage')
    
    # IMPORTANT CHANGE: Only map for Bitcoin and Ethereum for live fetches for analysis
    coin_gecko_id_map = {
        "BTC/USD": "bitcoin",
        "ETH/USD": "ethereum"
    }
    coingecko_id = coin_gecko_id_map.get(pair)

    live_price = 0.0
    if coingecko_id:
        live_price = _get_live_price_for_pair(coingecko_id)
        if live_price == 0.0:
            app.logger.warning(f"Could not fetch live price for {pair} for analysis. Falling back to 0.0.")
    else:
        app.logger.warning(f"CoinGecko ID not found for pair: {pair}. Cannot fetch live price for analysis.")

    current_price_for_pair = live_price 

    if not all([pair, timeframes, indicators, trade_type, balance_range, leverage]):
        return jsonify({"error": "Missing analysis parameters"}), 400
    
    if current_price_for_pair <= 0:
        return jsonify({"error": f"Cannot generate analysis for {pair}: Live price could not be fetched, or is invalid."}), 400

    # These are placeholder signals based on simple conditions
    # In a real bot, this would involve complex analysis
    if trade_type == "BUY":
        signal = "BUY"
        confidence = "High"
        entry = current_price_for_pair
        tp1 = round(current_price_for_pair * 1.005, 2)
        tp2 = round(current_price_for_pair * 1.01, 2)
        tp3 = round(current_price_for_pair * 1.02, 2)
        sl = round(current_price_for_pair * 0.99, 2)
        rr_ratio = "1:2.0"
    elif trade_type == "SELL":
        signal = "SELL"
        confidence = "Medium"
        entry = current_price_for_pair
        tp1 = round(current_price_for_pair * 0.995, 2)
        tp2 = round(current_price_for_pair * 0.99, 2)
        tp3 = round(current_price_for_pair * 0.98, 2)
        sl = round(current_price_for_pair * 1.01, 2)
        rr_ratio = "1:1.5"
    else: # Default to HOLD if trade_type is not BUY/SELL or invalid
        signal = "HOLD"
        confidence = "Moderate"
        entry = current_price_for_pair
        tp1 = round(current_price_for_pair * 1.002, 2)
        tp2 = round(current_price_for_pair * 1.005, 2)
        tp3 = round(current_price_for_pair * 1.01, 2)
        sl = round(current_price_for_pair * 0.998, 2)
        rr_ratio = "1:1.0"


    ai_analysis_text = (
        f"Based on your request for {pair} across {', '.join(timeframes)} timeframes, "
        f"utilizing {', '.join(indicators)} indicators, and considering a '{trade_type}' trade style "
        f"with a balance range of '{balance_range}' and '{leverage}' leverage, "
        f"Aura suggests a **{signal}** opportunity. "
        f"The current market conditions indicate {signal} with a {confidence} confidence level. "
        f"Monitor price action around key support/resistance levels. "
        f"Always conduct your own research before making trading decisions."
    )

    analysis_data = {
        "signal": signal,
        "confidence": f"{confidence} Confidence",
        "entry": entry,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "sl": sl,
        "rr_ratio": rr_ratio,
        "leverage": leverage
    }

    analysis_results = read_analysis_results()
    new_analysis_id = len(analysis_results) + 1
    analysis_results.append({
        "id": new_analysis_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "pair": pair,
        "timeframes": timeframes,
        "indicators": indicators,
        "trade_type": trade_type,
        "balance_range": balance_range,
        "leverage": leverage,
        "ai_analysis_text": ai_analysis_text,
        "analysis_data": analysis_data
    })
    write_analysis_results(analysis_results)

    return jsonify({
        "ai_analysis_text": ai_analysis_text,
        "analysis_data": analysis_data
    }), 200

@app.route('/chat', methods=['POST'])
def chat_with_gemini():
    data = request.get_json()
    user_message = data.get('message')
    user_name = data.get('userName', 'Trader')
    ai_name = data.get('aiName', 'Aura')

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    # --- Fetch Live Market Data for AI Context (now from dashboard cache with indicators) ---
    market_prices = get_all_market_prices().json # This will use the new Binance data + indicators
    market_context = ""
    if market_prices:
        market_context = "Current Market Prices and Indicators:\n"
        for pair, info in market_prices.items():
HEAD
            market_context += (
                f"- {pair}: {info['price']:.2f} USD ({info['percent_change']:.2f}% in 24h)\n"
                f"  RSI: {info.get('rsi', 'N/A')}, MACD: {info.get('macd', 'N/A')}, Stoch K: {info.get('stoch_k', 'N/A')}, "
                f"Stoch D: {info.get('stoch_d', 'N/A')}, Volume: {info.get('volume', 'N/A')}, "
                f"ORSCR Signal: {info.get('orscr_signal', 'N/A')}\n"
            )
            

            market_context += f"- {pair}: {info['price']:.2f} USD ({info['percent_change']:.2f}% in 24h)\n"
    
 83585c9c02c56cf767234a8b28cd9add0124928d
    # --- DEBUG LOGGING ---
    app.logger.info(f"Market context sent to Gemini: '{market_context.strip()}'")
    # --- END DEBUG LOGGING ---

    # --- Fetch Trade Logs for AI Context ---
    trades = read_trades()
    trade_context = ""
    
    if trades:
        total_profit_loss = sum(trade['profit_loss'] for trade in trades)
        total_trades = len(trades)
        profitable_trades = sum(1 for trade in trades if trade['profit_loss'] > 0)
        win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0.0
        avg_profit_per_trade = (total_profit_loss / total_trades) if total_trades > 0 else 0.0

        trade_context = "\nYour Trading History Summary:\n"
        trade_context += f"- Total Trades: {total_trades}\n"
        trade_context += f"- Total P/L: {total_profit_loss:.2f} USD\n"
        trade_context += f"- Win Rate: {win_rate:.2f}%\n"
        trade_context += f"- Avg. P/L per Trade: {avg_profit_per_trade:.2f} USD\n"
        
    # Construct the full prompt for Gemini, including context
    full_prompt = (
        f"You are a helpful and knowledgeable AI trading assistant named {ai_name}. "
        f"Your purpose is to assist {user_name} with trading-related questions, market analysis, "
        f"and general inquiries. You now have access to live market data including technical indicators and {user_name}'s trading history. "
        f"Use this context to provide more informed and personalized answers. "
        f"Be concise, informative, and always encourage users to do their own research. "
        f"Do not provide financial advice or recommendations to buy/sell. "
        f"Do not act as a trading bot or execute trades. "
        f"Do not make up prices or trade data if not explicitly provided.\n\n"
        f"--- Context ---\n"
        f"{market_context}\n"
        f"{trade_context}\n"
        f"--- End Context ---\n\n"
        f"User: {user_message}"
    )

    try:
        global model
        if model is None:
            app.logger.warning("Gemini model not initialized globally. Attempting to initialize now within /chat.")
            model = genai.GenerativeModel('gemini-1.5-flash')
            app.logger.info("Gemini model initialized successfully within /chat.")


        app.logger.info("Attempting to generate content with gemini-1.5-flash...")
        response = model.generate_content(full_prompt)
        gemini_response_text = response.text
        app.logger.info("Gemini content generation successful.")

        return jsonify({"response": gemini_response_text}), 200

    except Exception as e:
        app.logger.error(f"Error communicating with Gemini API for generateContent. Exception details: {e}")
        return jsonify({"error": f"Failed to get response from AI. Please check your Gemini API key and backend logs. Details: {str(e)}"}), 500

if __name__ == '__main__':
    # For local development, you can set dummy API keys
    if not BINANCE_API_KEY:
        os.environ['BINANCE_API_KEY'] = 'YOUR_BINANCE_API_KEY' # Replace with actual key for local testing
    if not BINANCE_API_SECRET:
        os.environ['BINANCE_API_SECRET'] = 'YOUR_BINANCE_API_SECRET' # Replace with actual secret for local testing

    app.run(host='0.0.0.0', port=os.environ.get('PORT', 10000), debug=False)