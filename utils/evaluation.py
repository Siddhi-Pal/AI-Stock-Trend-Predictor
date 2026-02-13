"""
Evaluation Module
Model evaluation and visualization functions
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os


# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def calculate_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics
    
    Parameters:
    -----------
    y_true : np.array
        True values
    y_pred : np.array
        Predicted values
    
    Returns:
    --------
    dict
        Dictionary containing RMSE and MAE
    """
    # Root Mean Squared Error
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Mean Absolute Error
    mae = mean_absolute_error(y_true, y_pred)
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }
    
    return metrics


def compare_with_baseline(model_metrics, baseline_metrics):
    """
    Compare model performance with baseline
    
    Parameters:
    -----------
    model_metrics : dict
        Metrics from the model
    baseline_metrics : dict
        Metrics from baseline
    
    Returns:
    --------
    dict
        Comparison report with improvement percentages
    """
    comparison = {}
    
    for metric_name in model_metrics.keys():
        model_value = model_metrics[metric_name]
        baseline_value = baseline_metrics[metric_name]
        
        # Calculate improvement (lower is better for error metrics)
        improvement = ((baseline_value - model_value) / baseline_value) * 100
        
        comparison[metric_name] = {
            'model': model_value,
            'baseline': baseline_value,
            'improvement_%': improvement
        }
    
    return comparison


def print_metrics(metrics, title="Model Metrics"):
    """
    Print metrics in a formatted way
    
    Parameters:
    -----------
    metrics : dict
        Dictionary of metrics
    title : str
        Title for the metrics display
    """
    print(f"\n{'='*50}")
    print(f"{title:^50}")
    print(f"{'='*50}")
    for metric, value in metrics.items():
        print(f"{metric:.<20} {value:.4f}")
    print(f"{'='*50}\n")


def print_comparison(comparison):
    """
    Print comparison between model and baseline
    
    Parameters:
    -----------
    comparison : dict
        Comparison dictionary from compare_with_baseline
    """
    print(f"\n{'='*70}")
    print(f"{'Model vs Baseline Comparison':^70}")
    print(f"{'='*70}")
    print(f"{'Metric':<15} {'Model':<15} {'Baseline':<15} {'Improvement':<15}")
    print(f"{'-'*70}")
    
    for metric, values in comparison.items():
        improvement = values['improvement_%']
        improvement_str = f"{improvement:+.2f}%"
        print(f"{metric:<15} {values['model']:<15.4f} {values['baseline']:<15.4f} {improvement_str:<15}")
    
    print(f"{'='*70}\n")


def plot_predictions(y_true, y_pred, title="Actual vs Predicted", 
                     save_path=None, dates=None):
    """
    Plot actual vs predicted values
    
    Parameters:
    -----------
    y_true : np.array
        True values
    y_pred : np.array
        Predicted values
    title : str
        Plot title
    save_path : str
        Path to save the plot (if None, display only)
    dates : array-like
        Date values for x-axis
    
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Use dates if provided, otherwise use index
    x_values = dates if dates is not None else range(len(y_true))
    
    # Plot actual and predicted
    ax.plot(x_values, y_true, label='Actual', color='#2E86AB', linewidth=2, alpha=0.8)
    ax.plot(x_values, y_pred, label='Predicted', color='#A23B72', linewidth=2, alpha=0.8)
    
    # Styling
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Stock Price', fontsize=12)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    return fig


def plot_training_history(history, save_path=None):
    """
    Plot training and validation loss curves
    
    Parameters:
    -----------
    history : keras.callbacks.History
        Training history from model.fit()
    save_path : str
        Path to save the plot
    
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot loss
    ax.plot(history.history['loss'], label='Training Loss', color='#2E86AB', linewidth=2)
    ax.plot(history.history['val_loss'], label='Validation Loss', color='#F18F01', linewidth=2)
    
    # Styling
    ax.set_title('Model Training History', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    return fig


def plot_future_forecast(historical_prices, future_predictions, 
                         n_historical=100, title="Future Forecast",
                         save_path=None):
    """
    Plot historical prices and future predictions
    
    Parameters:
    -----------
    historical_prices : np.array
        Historical price data
    future_predictions : np.array
        Future predicted prices
    n_historical : int
        Number of recent historical points to show
    title : str
        Plot title
    save_path : str
        Path to save the plot
    
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Get recent historical data
    recent_history = historical_prices[-n_historical:]
    
    # Create x-axis indices
    hist_x = range(len(recent_history))
    future_x = range(len(recent_history), len(recent_history) + len(future_predictions))
    
    # Plot historical and forecast
    ax.plot(hist_x, recent_history, label='Historical', 
            color='#2E86AB', linewidth=2, alpha=0.8)
    ax.plot(future_x, future_predictions, label='Forecast', 
            color='#C73E1D', linewidth=2, alpha=0.8, linestyle='--')
    
    # Mark the connection point
    ax.axvline(x=len(recent_history)-1, color='gray', linestyle=':', alpha=0.5)
    ax.text(len(recent_history)-1, ax.get_ylim()[1]*0.95, 'Forecast Start', 
            ha='center', fontsize=10, color='gray')
    
    # Styling
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Time Steps', fontsize=12)
    ax.set_ylabel('Stock Price', fontsize=12)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Forecast plot saved to {save_path}")
    
    return fig


if __name__ == "__main__":
    # Example usage
    print("=== Testing Evaluation Module ===\n")
    
    # Generate sample data
    np.random.seed(42)
    y_true = np.random.randn(100) * 10 + 100
    y_pred = y_true + np.random.randn(100) * 2
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)
    print_metrics(metrics, "Sample Model Metrics")
    
    # Baseline metrics (worse performance)
    y_baseline = y_true + np.random.randn(100) * 5
    baseline_metrics = calculate_metrics(y_true, y_baseline)
    print_metrics(baseline_metrics, "Baseline Metrics")
    
    # Compare
    comparison = compare_with_baseline(metrics, baseline_metrics)
    print_comparison(comparison)
    
    # Plot
    fig = plot_predictions(y_true, y_pred, title="Test Plot")
    plt.show()
