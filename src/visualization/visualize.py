import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib.lines import Line2D
from sklearn.ensemble import RandomForestRegressor
import pandas as pd


import plotly.graph_objects as go
import plotly.io as pio

# Ensure inline rendering in VS Code notebook
pio.renderers.default = "notebook"

def plot_moving_averages(df):
    plot_data = df.copy()

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['Close'],
                             mode='lines', name='Close Price',
                             line=dict(color='black', width=2)))
    fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['MA5'],
                             mode='lines', name='5-Day MA',
                             line=dict(color='blue', width=1.5)))
    fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['MA10'],
                             mode='lines', name='10-Day MA',
                             line=dict(color='orange', dash='dash', width=1.5)))
    fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['MA20'],
                             mode='lines', name='20-Day MA',
                             line=dict(color='green', dash='dot', width=1.5)))

    fig.update_layout(
        title='Close Price with 5, 10 & 20-Day Moving Averages',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        legend_title='Legend',
        template='plotly_white',
        height=500,
        width=1100
    )

    fig.show()





# def decompose_and_plot(df, column='Close', period=30):

#     # Styling
#     sns.set_style("whitegrid")
#     plt.rcParams['figure.figsize'] = [10, 7]
#     palette = sns.color_palette("husl", 4)

#     # Decompose
#     result = seasonal_decompose(df[column], model='additive', period=period)

#     # Plot
#     fig, axes = plt.subplots(4, 1, sharex=True, gridspec_kw={'hspace': 0.3})

#     # Observed
#     axes[0].plot(result.observed, color=palette[0], linewidth=1.5)
#     axes[0].set_title('Original Price Data', fontsize=12)
#     axes[0].set_ylabel('Price')

#     # Trend
#     axes[1].plot(result.trend, color=palette[1], linewidth=2)
#     axes[1].set_title('Long-Term Trend', fontsize=12)
#     axes[1].set_ylabel('Trend')

#     # Seasonality
#     axes[2].plot(result.seasonal, color=palette[2], linewidth=1)
#     axes[2].set_title(f'Seasonal Patterns ({period}-Day Cycle)', fontsize=12)
#     axes[2].set_ylabel('Seasonality')

#     # Highlight peaks/troughs
#     peaks = np.where(result.seasonal == result.seasonal.max())[0]
#     troughs = np.where(result.seasonal == result.seasonal.min())[0]
#     for p in peaks:
#         axes[2].axvline(x=result.seasonal.index[p], color='red', alpha=0.3, linestyle='--')
#     for t in troughs:
#         axes[2].axvline(x=result.seasonal.index[t], color='green', alpha=0.3, linestyle='--')
#     axes[2].legend(['Seasonal Pattern', 'Peaks', 'Troughs'], loc='upper left')

#     # Residuals
#     axes[3].scatter(result.resid.index, result.resid, color=palette[3], s=10, alpha=0.6)
#     axes[3].axhline(y=0, color='gray', linestyle='--')
#     axes[3].set_title('Random Noise / Residuals', fontsize=12)
#     axes[3].set_ylabel('Residuals')
#     axes[3].set_xlabel('Date')

#     # Title and layout
#     fig.suptitle('Seasonality Breakdown of Stock Prices', y=0.98, fontsize=14, fontweight='bold')
#     plt.tight_layout()
#     plt.show()  

import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
from plotly.subplots import make_subplots

def decompose_and_plot(df, column='Close', period=30):
    result = seasonal_decompose(df[column], model='additive', period=period)

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Original Price Data', 'Long-Term Trend', 'Random Noise / Residuals')
    )

    # Observed
    fig.add_trace(
        go.Scatter(x=result.observed.index, y=result.observed, name='Observed', line=dict(color='royalblue')),
        row=1, col=1
    )

    # Trend
    fig.add_trace(
        go.Scatter(x=result.trend.index, y=result.trend, name='Trend', line=dict(color='darkorange')),
        row=2, col=1
    )

    # Residuals
    fig.add_trace(
        go.Scatter(x=result.resid.index, y=result.resid, name='Residuals',
                   mode='markers', marker=dict(color='green', size=4, opacity=0.6)),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=result.resid.index, y=[0]*len(result.resid), 
                   name='Zero Line', line=dict(color='gray', dash='dash')),
        row=3, col=1
    )

    fig.update_layout(
        height=500,
        width=1300,
        title_text='Seasonality Breakdown of Stock Prices (No Seasonality)',
        template='plotly_white',
        showlegend=False
    )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Trend", row=2, col=1)
    fig.update_yaxes(title_text="Residuals", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)

    fig.show()








# def plot_volatility_analysis(df):
 
#     # Create subplots
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

#     # 1. Price and Volatility Bands
#     ax1.plot(df.index, df['Close'], label='Closing Price', color='#2c3e50', linewidth=2)
#     ax1.fill_between(df.index,
#                      df['Close'] + df['Volatility_5'],
#                      df['Close'] - df['Volatility_5'],
#                      alpha=0.3, color='#3498db', label='±5-Day Volatility Band')
#     ax1.set_title('Price with Volatility Bands', pad=10, fontsize=12)

#     # 2. Rolling Volatility Comparison
#     ax2.plot(df.index, df['Volatility_5'], label='5-Day Volatility', linewidth=1.5, color='#3498db')
#     ax2.plot(df.index, df['Volatility_10'], label='10-Day Volatility', linewidth=1.5, color='#e74c3c')
#     ax2.axhline(y=df['Volatility_5'].mean(), color='#3498db', linestyle='--', alpha=0.7)
#     ax2.axhline(y=df['Volatility_10'].mean(), color='#e74c3c', linestyle='--', alpha=0.7)
#     ax2.set_title('Rolling Volatility Comparison', pad=10, fontsize=12)
#     ax2.set_xlabel('Date', labelpad=10)

#     # Shared formatting
#     for ax in (ax1, ax2):
#         ax.grid(True, linestyle=':', alpha=0.6)
#         ax.legend(loc='upper left')

#     fig.suptitle('Volatility Analysis', y=0.98, fontsize=14, fontweight='bold')
#     plt.tight_layout()
#     plt.show()  




import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_volatility_analysis(df):
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(
            "Price with Volatility Bands",
            "Rolling Volatility Comparison"
        )
    )

    # 1. Price with Volatility Bands
    fig.add_trace(
        go.Scatter(x=df.index, y=df['Close'], name='Closing Price',
                   line=dict(color='black', width=2)),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df.index, 
            y=df['Close'] + df['Volatility_5'], 
            name='+5-Day Vol Band',
            line=dict(width=0),
            showlegend=False
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Close'] - df['Volatility_5'],
            name='±5-Day Vol Band',
            fill='tonexty',
            fillcolor='rgba(255, 255, 0, 0.3)',  # yellow with transparency
            line=dict(width=0),
            showlegend=True
        ),
        row=1, col=1
    )

    # 2. Rolling Volatility
    fig.add_trace(
        go.Scatter(x=df.index, y=df['Volatility_5'], name='5-Day Volatility',
                   line=dict(color='red', width=1.5)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['Volatility_10'], name='10-Day Volatility',
                   line=dict(color='green', width=1.5)),
        row=2, col=1
    )

    # Mean lines
    fig.add_trace(
        go.Scatter(x=df.index, y=[df['Volatility_5'].mean()]*len(df),
                   name='5-Day Mean', line=dict(color='red', dash='dash'),
                   showlegend=False),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=[df['Volatility_10'].mean()]*len(df),
                   name='10-Day Mean', line=dict(color='green', dash='dash'),
                   showlegend=False),
        row=2, col=1
    )

    # Layout
    fig.update_layout(
        title='Volatility Analysis',
        height=500,
        width=1140,
        template='plotly_white'
    )

    fig.update_xaxes(title_text='Date', row=2, col=1)
    fig.update_yaxes(title_text='Price (USD)', row=1, col=1)
    fig.update_yaxes(title_text='Volatility', row=2, col=1)

    fig.show()


def plot_correlation_matrix(df):
    
    # Calculate correlation matrix
    corr = df.corr()

    # Mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up figure
    plt.figure(figsize=(10, 8))

    # Create heatmap
    heatmap = sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )

    # Formatting
    plt.title("Feature Correlation Matrix", pad=20, fontsize=14, fontweight='bold')
    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.show()    
    
def plot_price_volume_heatmap(df):
 
    plt.figure(figsize=(10, 6))

    # Price trend lines
    for i in range(len(df) - 1):
        if df['Close'].iloc[i + 1] > df['Close'].iloc[i]:
            plt.plot(df.index[i:i + 2], df['Close'].iloc[i:i + 2],
                     color='#4CAF50', linewidth=3, alpha=0.8)  # Green
        else:
            plt.plot(df.index[i:i + 2], df['Close'].iloc[i:i + 2],
                     color='#F44336', linewidth=3, alpha=0.8)  # Red

    # Volume bars with gradient
    bar_width = 0.8
    for idx, date in enumerate(df.index):
        change_pct = (df['Close'].iloc[idx] - df['Open'].iloc[idx]) / df['Open'].iloc[idx]
        color_intensity = min(abs(change_pct) * 10, 1)  # Cap at 1
        if change_pct >= 0:
            color = (0, 0.5 + color_intensity / 2, 0, 0.6)
        else:
            color = (0.5 + color_intensity / 2, 0, 0, 0.6)

        plt.bar(date,
                df['Volume'].iloc[idx] / 1e6,
                width=bar_width,
                color=color,
                bottom=df['Close'].min() * 0.9,
                edgecolor='white',
                linewidth=0.5)

    # Titles and labels
    plt.title('Price Movement & Volume Patterns', pad=20, fontsize=16, fontweight='bold')
    plt.xlabel('Date', labelpad=10)
    plt.ylabel('Price ($)', labelpad=10)
    plt.xticks(rotation=45)

    # Custom legend
    legend_elements = [
        Line2D([0], [0], color='#4CAF50', lw=4, label='Upward Price Movement'),
        Line2D([0], [0], color='#F44336', lw=4, label='Downward Price Movement'),
        plt.Rectangle((0, 0), 1, 1, fc=(0, 0.7, 0, 0.6), ec='white', label='High Volume (Up)'),
        plt.Rectangle((0, 0), 1, 1, fc=(0.7, 0, 0, 0.6), ec='white', label='High Volume (Down)')
    ]
    plt.legend(handles=legend_elements, loc='upper left', framealpha=1)

    # Volume secondary y-axis
    ax2 = plt.gca().twinx()
    ax2.set_ylim(0, df['Volume'].max() / 50000)
    ax2.set_ylabel('Volume (Millions)', labelpad=10, color='#555555')
    ax2.tick_params(axis='y', colors='#555555')

    # Grid and layout
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.tight_layout()
    plt.show()  
# def plot_outlier_boxplots(df):
#     # Select numeric columns only
#     numeric_cols = df.select_dtypes(include=['float64', 'int64'])

#     # Check required columns
#     required_cols = ['Close', 'Volume', 'Volatility_5']
#     for col in required_cols:
#         if col not in numeric_cols.columns:
#             raise ValueError(f"Column '{col}' not found in DataFrame.")

#     # Create boxplot
#     plt.figure(figsize=(10, 6))
#     sns.boxplot(data=numeric_cols[required_cols],
#                 palette="viridis",
#                 whis=1.5)

#     # Formatting
#     plt.title('Outlier Detection in Key Metrics', fontsize=14, pad=20)
#     plt.xticks(rotation=45)
#     plt.ylabel('Value Range', labelpad=10)
#     plt.grid(axis='y', linestyle=':', alpha=0.4)

#     plt.tight_layout()
#     plt.show()

import plotly.graph_objects as go

def plot_outlier_boxplots(df):
    # Select numeric columns only
    numeric_cols = df.select_dtypes(include=['float64', 'int64'])

    # Check required columns
    required_cols = ['Close', 'Volume', 'Volatility_5']
    for col in required_cols:
        if col not in numeric_cols.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    # Create Plotly boxplot
    fig = go.Figure()

    colors = ['black', 'green', 'orange']  # Customize colors for each metric

    for i, col in enumerate(required_cols):
        fig.add_trace(go.Box(
            y=numeric_cols[col],
            name=col,
            marker_color=colors[i],
            boxpoints='outliers',  # show only outliers
            line=dict(width=1),
            fillcolor='rgba(0,0,0,0)',  # transparent fill
        ))

    # Layout settings
    fig.update_layout(
        title='Outlier Detection ',
        yaxis_title='Value Range',
        template='plotly_white',
        width=1100,
        height=500
    )

    fig.show() 
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

def plot_feature_importance(df):
    # Split data into features and target
    X = df.drop(columns=['Target'])
    y = df['Target']

    # Train RandomForest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Get feature importance
    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=True)  # ascending for horizontal bar plot

    # Plot with Plotly Express (horizontal bar)
    fig = px.bar(
        importance,
        x='Importance',
        y='Feature',
        orientation='h',
        color='Importance',
        color_continuous_scale='Viridis',
        title='Feature Importance for Price Prediction',
        labels={'Importance': 'Relative Importance', 'Feature': 'Feature'},
        width=1100,
        height=500
    )

    fig.update_layout(
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray', zeroline=True),
        yaxis=dict(tickfont=dict(size=12)),
        coloraxis_showscale=False,  # hides the color scale legend
        margin=dict(l=150, r=40, t=80, b=50)
    )

    fig.show()
 

# def plot_evaluation_metrics_1(evaluation_df):
#     import matplotlib.pyplot as plt
#     import seaborn as sns

#     metrics = ['MAE', 'MSE', 'RMSE', 'R²']
#     sns.set_style("whitegrid")
#     palette = sns.color_palette("viridis", len(metrics))

#     # Create 2 rows × 2 columns subplot grid
#     fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(300, 30))
#     axes = axes.flatten()  # Flatten 2D array to 1D for easier iteration

#     for i, metric in enumerate(metrics):
#         sns.barplot(x=evaluation_df.index, y=evaluation_df[metric], ax=axes[i], palette=palette)
#         axes[i].set_title(metric, fontsize=14)
#         axes[i].set_xlabel("Model")
#         axes[i].set_ylabel(metric)
#         axes[i].tick_params(axis='x', rotation=30)
#         axes[i].grid(True, linestyle='--', alpha=0.5)

#     fig.suptitle('Model Evaluation Metrics Comparison', fontsize=18, fontweight='bold')
#     plt.tight_layout(rect=[1, 1, 1, 0.95])
#     plt.show()

import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_metrics_from_dict(metrics_dict):
    metrics = ['MAE', 'MSE', 'RMSE', 'R²']
    colors = ['red', 'green', 'yellow', 'black']
    
    models = list(metrics_dict.keys())
    data_by_metric = {metric: [metrics_dict[model][metric] for model in models] for metric in metrics}

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=metrics,
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )

    for i, metric in enumerate(metrics):
        row = i // 2 + 1
        col = i % 2 + 1

        fig.add_trace(
            go.Bar(
                x=models,
                y=data_by_metric[metric],
                name=metric,
                marker_color=colors[i % len(colors)],
                text=[round(val, 4) for val in data_by_metric[metric]],
                textposition='auto'
            ),
            row=row, col=col
        )

        fig.update_xaxes(title_text="Model", tickangle=30, row=row, col=col)
        fig.update_yaxes(title_text=metric, row=row, col=col)

    fig.update_layout(
        title_text='Model Evaluation Metrics Comparison,Liner',
        height=700,
        width=1300,
        template='plotly_white',
        showlegend=False
    )

    fig.show()




def plot_test_train_prediction_scaled(model_name, y_train_scaled, y_test_scaled, y_pred_scaled, df_train, df_test):

    plt.figure(figsize=(13,5))

    # Plot scaled actual target values
    plt.plot(df_train.index, y_train_scaled, label='Train Target (Scaled)', color='teal')
    plt.plot(df_test.index, y_test_scaled, label='Test Target (Scaled)', color='magenta')

    # Plot scaled predicted values
    plt.plot(df_test.index, y_pred_scaled, label=f'{model_name} Prediction (Scaled)', 
             color='blue', linestyle='--')

    # Formatting
    plt.title(f'{model_name} Predictions vs Actual Target', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Target (Scaled)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_test_train_prediction_dl_model(model_name, df_train, df_test, y_train_scaled, y_test_scaled, y_pred_scaled):

    plt.figure(figsize=(13, 5))

    # Ensure shapes match by using correct trailing indices
    plt.plot(df_train.index[-len(y_train_scaled):], y_train_scaled.ravel(), label='Train Target (Scaled)', color='teal')
    plt.plot(df_test.index[-len(y_test_scaled):], y_test_scaled.ravel(), label='Test Target (Scaled)', color='magenta')
    plt.plot(df_test.index[-len(y_pred_scaled):], y_pred_scaled.ravel(), label=f'{model_name} Prediction (Scaled)', 
             color='blue', linestyle='--')

    # Formatting
    plt.title(f'{model_name} Predictions vs Actual Target (Scaled)', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Target (Scaled)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()






