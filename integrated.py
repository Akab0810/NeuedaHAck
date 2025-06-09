# !pip install yfinance pandas numpy scikit-learn matplotlib seaborn
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# Part 1: Portfolio Allocation Prediction
def predict_portfolio(age, amount, duration, risk, goal):
    # Generate dataset
    def generate_dataset(n=10000):
        data = []
        for _ in range(n):
            age = random.randint(20, 65)
            amount = random.randint(50000, 1000000)
            duration = random.randint(1, 20)
            risk = random.choice(['low', 'medium', 'high'])
            goal = random.choice(['retirement', 'short-term', 'wealth growth'])

            if risk == 'low':
                stocks = 0.1
                bonds = 0.4
                mutual_funds = 0.3
                gold = 0.1
                etfs = 0.1
            elif risk == 'medium':
                stocks = 0.3
                bonds = 0.2
                mutual_funds = 0.3
                gold = 0.1
                etfs = 0.1
            else:
                stocks = 0.6
                bonds = 0.05
                mutual_funds = 0.2
                gold = 0.05
                etfs = 0.1

            data.append([age, amount, duration, risk, goal, stocks, mutual_funds, bonds, etfs, gold])
        
        df = pd.DataFrame(data, columns=[
            'age', 'amount', 'duration', 'risk', 'goal',
            'stocks', 'mutual_funds', 'bonds', 'etfs', 'gold'
        ])
        return df

    # Create and save model
    df = generate_dataset()
    df.to_csv('investment_data.csv', index=False)

    df = pd.read_csv("investment_data.csv")

    # Encode categorical
    le_risk = LabelEncoder()
    le_goal = LabelEncoder()

    df['risk_enc'] = le_risk.fit_transform(df['risk'])
    df['goal_enc'] = le_goal.fit_transform(df['goal'])

    X = df[['age', 'amount', 'duration', 'risk_enc', 'goal_enc']]
    y = df[['stocks', 'mutual_funds', 'bonds', 'etfs', 'gold']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Save model and encoders
    joblib.dump(model, 'portfolio_model.pkl')
    joblib.dump(le_risk, 'risk_encoder.pkl')
    joblib.dump(le_goal, 'goal_encoder.pkl')

    # Load model and encoders
    model = joblib.load('portfolio_model.pkl')
    le_risk = joblib.load('risk_encoder.pkl')
    le_goal = joblib.load('goal_encoder.pkl')
    risk_enc = le_risk.transform([risk])[0]
    goal_enc = le_goal.transform([goal])[0]
    
    input_features = np.array([[age, amount, duration, risk_enc, goal_enc]])
    prediction = model.predict(input_features)[0]

    return {
        "Stocks": round(prediction[0] * 100, 2),
        "Mutual_Funds": round(prediction[1] * 100, 2),
        "Bonds": round(prediction[2] * 100, 2),
        "ETFs": round(prediction[3] * 100, 2),
        "Gold": round(prediction[4] * 100, 2)
    }

# Part 2: Stock Recommendation System
def get_stock_recommendations(investment_amount, top_n=5):
    def fetch_stock_data(tickers, start_date, end_date):
        data = {}
        for ticker in tickers:
            try:
                df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if not df.empty:
                    data[ticker] = df
            except:
                continue
        return data

    def prepare_features(stock_data, ticker):
        df = stock_data[ticker].copy()
        required_columns = ['Close', 'Open', 'High', 'Low', 'Volume']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Missing columns in {ticker}")
        if 'Adj Close' not in df.columns:
            df['Adj Close'] = df['Close']
        df['Daily_Return'] = df['Adj Close'].pct_change()
        df['SMA_50'] = df['Adj Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Adj Close'].rolling(window=200).mean()
        df['EMA_20'] = df['Adj Close'].ewm(span=20, adjust=False).mean()
        for i in range(1, 6):
            df[f'Return_Lag_{i}'] = df['Daily_Return'].shift(i)
        df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
        df.dropna(inplace=True)
        feature_cols = ['SMA_50', 'SMA_200', 'EMA_20', 'Volatility'] + [f'Return_Lag_{i}' for i in range(1, 6)]
        X = df[feature_cols]
        y = df['Daily_Return'].shift(-1)
        return X[:-1], y[:-1]

    def train_models(tickers, stock_data):
        models = {}
        for ticker in tickers:
            try:
                if ticker not in stock_data:
                    continue
                X, y = prepare_features(stock_data, ticker)
                if len(X) < 100:
                    continue
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                models[ticker] = model
            except:
                continue
        return models

    def recommend_stocks(investment_amount, models, stock_data, top_n):
        predicted_returns = {}
        current_prices = {}
        for ticker, model in models.items():
            try:
                X, _ = prepare_features(stock_data, ticker)
                if len(X) < 1:
                    continue
                latest_features = X.iloc[-1:].values
                predicted_return = float(model.predict(latest_features)[0])
                current_price = stock_data[ticker]['Close'].iloc[-1].item()
                predicted_returns[ticker] = predicted_return
                current_prices[ticker] = current_price
            except:
                continue
        if not predicted_returns:
            raise ValueError("No valid predictions could be made.")
        sorted_stocks = sorted(predicted_returns.items(), key=lambda x: x[1], reverse=True)[:top_n]
        total_weight = sum(abs(ret) for _, ret in sorted_stocks)
        recommendations = []
        for ticker, ret in sorted_stocks:
            weight = abs(ret) / total_weight
            allocation = investment_amount * weight
            shares = allocation / current_prices[ticker]
            recommendations.append({
                'ticker': ticker,
                'shares': round(shares, 2),
                'allocation': round(allocation, 2),
                'predicted_return': round(ret * 100, 2),
                'current_price': round(current_prices[ticker], 2)
            })
        return recommendations

    def get_diversification_advice(recommendations):
        sectors = {
            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
            'AMZN': 'Consumer Discretionary', 'META': 'Communication Services',
            'TSLA': 'Consumer Discretionary', 'JNJ': 'Healthcare',
            'PFE': 'Healthcare', 'XOM': 'Energy', 'WMT': 'Consumer Staples'
        }
        sector_allocation = {}
        total = sum(rec['allocation'] for rec in recommendations)
        for rec in recommendations:
            sector = sectors.get(rec['ticker'], 'Other')
            sector_allocation[sector] = sector_allocation.get(sector, 0) + rec['allocation']
        for sector in sector_allocation:
            sector_allocation[sector] = round(sector_allocation[sector] / total * 100, 2)
        advice = []
        if len(sector_allocation) < 3:
            advice.append("Consider diversifying across more sectors for better risk management.")
        if sector_allocation.get('Technology', 0) > 50:
            advice.append("Your portfolio is heavily weighted in Technology. Consider adding Healthcare or Consumer Staples.")
        if not any(s in sector_allocation for s in ['Healthcare', 'Consumer Staples']):
            advice.append("Adding Healthcare or Consumer Staples could provide stability during downturns.")
        if not advice:
            advice.append("Your portfolio is well diversified across sectors.")
        return {'sector_allocation': sector_allocation, 'advice': advice}

    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'JNJ', 'PFE', 'XOM', 'WMT']
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)

    stock_data = fetch_stock_data(tickers, start_date, end_date)
    models = train_models(tickers, stock_data)
    recommendations = recommend_stocks(investment_amount, models, stock_data, top_n)
    diversification = get_diversification_advice(recommendations)

    return {
        'investment_amount': investment_amount,
        'top_recommendations': recommendations,
        'diversification': diversification
    }

# Main Function to Connect Both Parts
def investment_advisor():
    print("Welcome to the Smart Investment Advisor!")
    print("This system will help you allocate your investment and select optimal stocks.")
    
    # Get user inputs
    while True:
        try:
            age = int(input("\nEnter your age: "))
            if age < 18 or age > 100:
                print("Please enter a valid age between 18 and 100.")
                continue
            break
        except ValueError:
            print("Please enter a valid number for age.")

    while True:
        try:
            amount = float(input("Enter your investment amount in USD: $"))
            if amount <= 0:
                print("Please enter a positive amount.")
                continue
            break
        except ValueError:
            print("Please enter a valid number for amount.")

    while True:
        try:
            duration = int(input("Enter investment duration in years (1-30): "))
            if duration < 1 or duration > 30:
                print("Please enter a duration between 1 and 30 years.")
                continue
            break
        except ValueError:
            print("Please enter a valid number for duration.")

    while True:
        risk = input("Enter your risk tolerance (low/medium/high): ").lower()
        if risk in ['low', 'medium', 'high']:
            break
        print("Please enter either 'low', 'medium', or 'high'.")

    while True:
        goal = input("Enter your investment goal (retirement/short-term/wealth growth): ").lower()
        if goal in ['retirement', 'short-term', 'wealth growth']:
            break
        print("Please enter either 'retirement', 'short-term', or 'wealth growth'.")

    # Get portfolio allocation
    print("\nCalculating optimal portfolio allocation...")
    portfolio = predict_portfolio(age, amount, duration, risk, goal)
    
    print("\n=== RECOMMENDED PORTFOLIO ALLOCATION ===")
    print(f"Total Investment Amount: ${amount:,.2f}")
    for asset, percent in portfolio.items():
        allocation = amount * percent / 100
        print(f"{asset}: {percent}% (${allocation:,.2f})")

    # Get stock recommendations for the stock portion
    stock_amount = amount * portfolio['Stocks'] / 100
    print(f"\nGetting stock recommendations for ${stock_amount:,.2f}...")
    
    while True:
        try:
            num_stocks = int(input("How many stocks would you like to consider? (1-10): "))
            if 1 <= num_stocks <= 10:
                break
            print("Please enter a number between 1 and 10.")
        except ValueError:
            print("Please enter a valid integer.")

    stock_recommendations = get_stock_recommendations(stock_amount, num_stocks)
    
    print("\n=== TOP STOCK RECOMMENDATIONS ===")
    print(f"Amount allocated to stocks: ${stock_amount:,.2f}")
    for i, stock in enumerate(stock_recommendations['top_recommendations'], 1):
        print(f"\n{i}. {stock['ticker']}:")
        print(f"   - Shares to buy: {stock['shares']}")
        print(f"   - Allocation: ${stock['allocation']:,.2f}")
        print(f"   - Current Price: ${stock['current_price']:,.2f}")
        print(f"   - Predicted Daily Return: {stock['predicted_return']}%")

    print("\n=== PORTFOLIO DIVERSIFICATION ===")
    print("Sector Allocation:")
    for sector, percent in stock_recommendations['diversification']['sector_allocation'].items():
        print(f" - {sector}: {percent}%")

    print("\nDiversification Advice:")
    for advice in stock_recommendations['diversification']['advice']:
        print(f" - {advice}")

    print("\nNote: Past performance is not indicative of future results. Consider consulting a financial advisor.")

# Run the program
if __name__ == "__main__":
    investment_advisor()