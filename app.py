from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
import datetime
import requests
import time # Import the time module for caching

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

TRADES_FILE = 'trades.json'
ANALYSIS_FILE = 'analysis_results.json' # To store analysis for persistence (optional, but good practice)

# --- Caching for Market Prices ---
last_market_data = None
last_fetch_time = 0
CACHE_DURATION = 30 # Cache data for 30 seconds to avoid hitting CoinGecko rate limits

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

# Endpoint to fetch real-time market prices from CoinGecko
@app.route('/all_market_prices', methods=['GET'])
def get_all_market_prices():
    global last_market_data, last_fetch_time

    # Check if cached data is still valid
    if last_market_data and (time.time() - last_fetch_time < CACHE_DURATION):
        app.logger.info("Serving market data from cache.")
        return jsonify(last_market_data), 200

    # If cache is expired or empty, fetch new data
    app.logger.info("Fetching new market data from CoinGecko.")
    coin_ids = {
        "BTC/USD": "bitcoin",
        "ETH/USD": "ethereum",
        "SOL/USD": "solana",
        "XRP/USD": "ripple",
        "ADA/USD": "cardano",
        "DOGE/USD": "dogecoin",
        "RVN/USD": "ravencoin" # This might not be directly available as USD pair
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
        
        # Update cache
        last_market_data = formatted_data
        last_fetch_time = time.time()
        
        return jsonify(formatted_data), 200

    except requests.exceptions.RequestException as e:
        app.logger.error(f"Error fetching market prices from CoinGecko: {e}")
        return jsonify({"error": "Failed to fetch market prices", "details": str(e)}), 500
    except Exception as e:
        app.logger.error(f"An unexpected error occurred in get_all_market_prices: {e}")
        return jsonify({"error": "An internal server error occurred", "details": str(e)}), 500

# Endpoint for AI analysis (using a mock response or integrating with an actual LLM later)
@app.route('/generate_analysis', methods=['POST'])
def generate_analysis():
    data = request.get_json()
    pair = data.get('pair')
    timeframes = data.get('timeframes')
    indicators = data.get('indicators')
    trade_type = data.get('trade_type')
    balance_range = data.get('balance_range')
    leverage = data.get('leverage')
    current_price_for_pair = data.get('current_price_for_pair', 0) # Get the current price from frontend

    if not all([pair, timeframes, indicators, trade_type, balance_range, leverage]):
        return jsonify({"error": "Missing analysis parameters"}), 400

    # Basic mock analysis response for demonstration
    # In a real application, you'd send this data to an LLM like Gemini
    # or a dedicated analysis engine.
    
    # Generate dynamic entry/TP/SL based on current_price_for_pair
    # These are just illustrative calculations
    if current_price_for_pair > 0:
        if trade_type == "BUY":
            signal = "BUY"
            confidence = "High"
            entry = current_price_for_pair
            tp1 = round(current_price_for_pair * 1.005, 2) # 0.5% up
            tp2 = round(current_price_for_pair * 1.01, 2)  # 1% up
            tp3 = round(current_price_for_pair * 1.02, 2)  # 2% up
            sl = round(current_price_for_pair * 0.99, 2)   # 1% down
            rr_ratio = "1:2.0" # Example ratio
        elif trade_type == "SELL":
            signal = "SELL"
            confidence = "Medium"
            entry = current_price_for_pair
            tp1 = round(current_price_for_pair * 0.995, 2) # 0.5% down
            tp2 = round(current_price_for_pair * 0.99, 2)  # 1% down
            tp3 = round(current_price_for_pair * 0.98, 2)  # 2% down
            sl = round(current_price_for_pair * 1.01, 2)   # 1% up
            rr_ratio = "1:1.5" # Example ratio
        else: # Scalp or Long Hold, defaulting to a neutral or general signal
            signal = "HOLD"
            confidence = "Moderate"
            entry = current_price_for_pair
            tp1 = round(current_price_for_pair * 1.002, 2)
            tp2 = round(current_price_for_pair * 1.005, 2)
            tp3 = round(current_price_for_pair * 1.01, 2)
            sl = round(current_price_for_pair * 0.998, 2)
            rr_ratio = "1:1.0"
    else: # Fallback if no current price is provided or it's zero
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

    # Store analysis result (optional, but good for history/debugging)
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
    user_name = data.get('userName', 'Trader') # Get user's name
    ai_name = data.get('aiName', 'Aura')     # Get AI's name

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    # Construct the full prompt for Gemini
    # This example uses a very basic prompt. For a real application, you'd
    # include conversation history and more sophisticated context.
    full_prompt = (
        f"You are a helpful and knowledgeable AI trading assistant named {ai_name}. "
        f"Your purpose is to assist {user_name} with trading-related questions, market analysis, "
        f"and general inquiries. Be concise, informative, and always encourage users to do their own research. "
        f"If the user asks about a price, try to give a brief, relevant market overview without guaranteeing future prices. "
        f"Do not provide financial advice or recommendations to buy/sell. "
        f"Do not act as a trading bot or execute trades. "
        f"User: {user_message}"
    )

    try:
        # Placeholder for Gemini API call
        # Replace with your actual Gemini API integration
        # Example using a hypothetical client (you would integrate your actual API key and library)
        # from google.generativeai import GenerativeModel
        # model = GenerativeModel('gemini-pro')
        # response = model.generate_content(full_prompt)
        # gemini_response_text = response.text

        # For now, a mock response for demonstration:
        gemini_response_text = f"Hello {user_name}! You asked: '{user_message}'. As {ai_name}, I can help you with trading concepts, market data, and analysis. What specifically would you like to know?"
        if "hello" in user_message.lower():
            gemini_response_text = f"Hello {user_name}! How can {ai_name} assist you today with your trading inquiries?"
        elif "price" in user_message.lower() or "market" in user_message.lower():
            gemini_response_text = f"The crypto market is highly volatile. For real-time prices, please check reputable exchanges. Always do your own research before making decisions."
        elif "buy" in user_message.lower() or "sell" in user_message.lower() or "trade" in user_message.lower():
            gemini_response_text = f"I cannot provide financial advice or execute trades. My purpose is to provide information and analysis for {user_name} to make informed decisions."


        return jsonify({"response": gemini_response_text}), 200

    except Exception as e:
        app.logger.error(f"Error communicating with Gemini API: {e}")
        return jsonify({"error": f"Failed to get response from AI: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 5000), debug=True)