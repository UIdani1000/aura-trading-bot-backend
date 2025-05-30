from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
import datetime
import requests
import time
import google.generativeai as genai # Import the Gemini library

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

TRADES_FILE = 'trades.json'
ANALYSIS_FILE = 'analysis_results.json'

# --- Configure Gemini API Key ---
# IMPORTANT: Set this as an environment variable in Render!
# In Render, go to your backend service -> Environment -> Add Environment Variable
# Key: GOOGLE_API_KEY
# Value: YOUR_ACTUAL_GEMINI_API_KEY
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

# --- DEBUG: List available Gemini models ---
try:
    app.logger.info("Attempting to list available Gemini models...")
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            app.logger.info(f"Available model: {m.name}")
except Exception as e:
    app.logger.error(f"Error listing Gemini models: {e}")
# --- END DEBUG ---

# If you want to use the model immediately after confirming it works, uncomment this:
model = genai.GenerativeModel("gemini-1.0-pro")

# --- Caching for Market Prices (Global for app.py) ---
last_market_data = None
last_fetch_time = 0
CACHE_DURATION = 5 # Changed to 5 seconds

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
            return [] # Return empty list if file is empty or corrupted

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
    new_id = len(trades) + 1 # Simple ID generation
    
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

# Endpoint to get trade summary statistics
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

# Helper function to get cached market data
def _get_cached_market_data():
    global last_market_data, last_fetch_time

    if last_market_data and (time.time() - last_fetch_time < CACHE_DURATION):
        app.logger.info("Serving market data from cache for internal use.")
        return last_market_data

    app.logger.info("Fetching new market data from CoinGecko for internal use.")
    coin_ids = {
        "BTC/USD": "bitcoin", "ETH/USD": "ethereum", "SOL/USD": "solana",
        "XRP/USD": "ripple", "ADA/USD": "cardano", "DOGE/USD": "dogecoin",
        "RVN/USD": "ravencoin"
    }
    ids_string = ",".join(coin_ids.values())

    try:
        response = requests.get(
            f"https://api.coingecko.com/api/v3/simple/price?ids={ids_string}&vs_currencies=usd&include_24hr_change=true"
        )
        response.raise_for_status()
        market_data = response.json()
        
        formatted_data = {}
        for pair, coin_id in coin_ids.items():
            if coin_id in market_data and 'usd' in market_data[coin_id]:
                price = market_data[coin_id]['usd']
                change_24h = market_data[coin_id].get('usd_24h_change', 0)
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
        return {}
    except Exception as e:
        app.logger.error(f"An unexpected error occurred in _get_cached_market_data: {e}")
        return {}


# Endpoint to fetch real-time market prices from CoinGecko
@app.route('/all_market_prices', methods=['GET'])
def get_all_market_prices():
    market_data = _get_cached_market_data()
    return jsonify(market_data), 200


@app.route('/generate_analysis', methods=['POST'])
def generate_analysis():
    data = request.get_json()
    pair = data.get('pair')
    timeframes = data.get('timeframes')
    indicators = data.get('indicators')
    trade_type = data.get('trade_type')
    balance_range = data.get('balance_range')
    leverage = data.get('leverage')
    current_price_for_pair = data.get('current_price_for_pair', 0)

    if not all([pair, timeframes, indicators, trade_type, balance_range, leverage]):
        return jsonify({"error": "Missing analysis parameters"}), 400

    if current_price_for_pair > 0:
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
        else:
            signal = "HOLD"
            confidence = "Moderate"
            entry = current_price_for_pair
            tp1 = round(current_price_for_pair * 1.002, 2)
            tp2 = round(current_price_for_pair * 1.005, 2)
            tp3 = round(current_price_for_pair * 1.01, 2)
            sl = round(current_price_for_pair * 0.998, 2)
            rr_ratio = "1:1.0"
    else:
        signal = "NEUTRAL"
        confidence = "Low (No live price)"
        entry = 0.0
        tp1 = 0.0
        tp2 = 0.0
        tp3 = 0.0
        sl = 0.0
        rr_ratio = "N/A"


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

    # --- Fetch Live Market Data for AI Context ---
    market_prices = _get_cached_market_data()
    market_context = ""
    if market_prices:
        market_context = "Current Market Prices:\n"
        for pair, info in market_prices.items():
            market_context += f"- {pair}: {info['price']:.2f} USD ({info['percent_change']:.2f}% in 24h)\n"

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
        f"and general inquiries. You now have access to live market data and {user_name}'s trading history. "
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
        # Ensure the GOOGLE_API_KEY environment variable is set on Render!
        # If the API key is not set or invalid, this will raise an exception.
        # Attempting to initialize Gemini model 'gemini-1.0-pro'
        model = genai.GenerativeModel('gemini-1.0-pro')
        app.logger.info("Attempting to generate content with gemini-1.0-pro...") # Added this line for better logging
        response = model.generate_content(full_prompt)
        gemini_response_text = response.text
        app.logger.info("Gemini content generation successful.")

        return jsonify({"response": gemini_response_text}), 200

    except Exception as e:
        app.logger.error(f"Error communicating with Gemini API. Exception details: {e}")
        # Provide a more helpful error message in the frontend if API fails
        return jsonify({"error": f"Failed to get response from AI. Please check your Gemini API key and backend logs. Details: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 5000), debug=True)