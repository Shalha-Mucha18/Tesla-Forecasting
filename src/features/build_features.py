def add_features(df):
    df = df.copy()

    # Ensure the Date index is unique (if it's not, you can drop duplicates or aggregate them)
    df = df.loc[~df.index.duplicated(keep='first')]

    # Monthly return (use 'ME' to avoid FutureWarning)
    monthly_returns = df['Close'].resample('ME').ffill().pct_change()
    df['Monthly_Return'] = monthly_returns.reindex(df.index, method='ffill')

    # Moving Averages
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()

    # Volatility (rolling std dev)
    df['Volatility_5'] = df['Close'].rolling(window=5).std()
    df['Volatility_10'] = df['Close'].rolling(window=10).std()

    # Drop rows with any NaN values (if any remain after rolling operations)
    df.dropna(inplace=True)

    return df