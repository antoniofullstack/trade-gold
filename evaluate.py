import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from env.gold_env import GoldTradingEnv
from agent.dqn_agent import DQNAgent
from utils.data_processor import load_and_prepare_data, create_sample_data


def evaluate_agent(env, agent, episodes=1):
    """
    Evaluate the trained agent without exploration
    
    Args:
        env: Trading environment
        agent: Trained DQN agent
        episodes: Number of evaluation episodes
    
    Returns:
        dict: Evaluation metrics
    """
    # Set epsilon to 0 for greedy evaluation
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    all_results = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            state, reward, done = env.step(action)
            total_reward += reward
        
        # Calculate metrics
        final_equity = env.net_worth
        total_return = (final_equity - env.initial_balance) / env.initial_balance
        max_drawdown = max(env.drawdown_history) if env.drawdown_history else 0
        
        # Calculate Sharpe ratio
        if len(env.equity_history) > 1:
            returns = np.diff(env.equity_history) / env.initial_balance
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8)
        else:
            sharpe_ratio = 0
        
        results = {
            'episode': episode + 1,
            'total_reward': total_reward,
            'final_equity': final_equity,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'num_trades': len(env.trade_history),
            'equity_history': env.equity_history.copy(),
            'drawdown_history': env.drawdown_history.copy(),
            'trade_history': env.trade_history.copy()
        }
        
        all_results.append(results)
        
        print(f"Episode {episode + 1}: "
              f"Return: {total_return:.2%} | "
              f"Sharpe: {sharpe_ratio:.2f} | "
              f"Max DD: {max_drawdown:.2%} | "
              f"Trades: {len(env.trade_history)}")
    
    # Restore original epsilon
    agent.epsilon = original_epsilon
    
    return all_results


def plot_results(results, df, save_dir='plots'):
    """
    Plot evaluation results
    
    Args:
        results: List of evaluation results
        df: Original dataframe with price data
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for i, result in enumerate(results):
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # Get price data for the episode length
        price_data = df["Close"].iloc[:len(result['equity_history'])]
        
        # Plot 1: Price with trades
        axes[0].plot(price_data.values, label="Gold Price", color='black', alpha=0.7)
        
        # Plot trades
        labels_used = set()
        for step, price, trade_type in result['trade_history']:
            if step < len(price_data):
                if trade_type == 'BUY':
                    label = 'Buy' if 'Buy' not in labels_used else ''
                    labels_used.add('Buy')
                    axes[0].scatter(step, price, marker='^', color='green', s=50, label=label)
                elif trade_type == 'SELL':
                    label = 'Sell' if 'Sell' not in labels_used else ''
                    labels_used.add('Sell')
                    axes[0].scatter(step, price, marker='v', color='red', s=50, label=label)
                elif trade_type in ('CLOSE', 'CLOSE_LONG', 'CLOSE_SHORT'):
                    label = 'Close' if 'Close' not in labels_used else ''
                    labels_used.add('Close')
                    axes[0].scatter(step, price, marker='o', color='blue', s=30, alpha=0.6, label=label)
                elif trade_type == 'STOP_LOSS':
                    label = 'Stop Loss' if 'Stop Loss' not in labels_used else ''
                    labels_used.add('Stop Loss')
                    axes[0].scatter(step, price, marker='x', color='orange', s=50, label=label)
                elif trade_type == 'TAKE_PROFIT':
                    label = 'Take Profit' if 'Take Profit' not in labels_used else ''
                    labels_used.add('Take Profit')
                    axes[0].scatter(step, price, marker='*', color='purple', s=50, label=label)
        
        axes[0].set_title(f'Episode {result["episode"]}: Gold Price & Trades')
        axes[0].set_ylabel('Price')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Equity Curve
        axes[1].plot(result['equity_history'], label='Equity', color='blue')
        axes[1].axhline(y=10000, color='red', linestyle='--', alpha=0.5, label='Initial Balance')
        axes[1].set_title('Equity Curve')
        axes[1].set_ylabel('Equity ($)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Drawdown
        axes[2].fill_between(range(len(result['drawdown_history'])), 
                            result['drawdown_history'], 0, 
                            alpha=0.3, color='red')
        axes[2].plot(result['drawdown_history'], color='red', label='Drawdown')
        axes[2].set_title('Drawdown')
        axes[2].set_ylabel('Drawdown')
        axes[2].set_xlabel('Time Steps')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(save_dir, f'evaluation_episode_{result["episode"]}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plot saved: {plot_path}")


def print_summary(results):
    """Print summary statistics"""
    if not results:
        return
    
    avg_return = np.mean([r['total_return'] for r in results])
    avg_sharpe = np.mean([r['sharpe_ratio'] for r in results])
    avg_max_dd = np.mean([r['max_drawdown'] for r in results])
    avg_trades = np.mean([r['num_trades'] for r in results])
    
    print(f"\n=== EVALUATION SUMMARY ===")
    print(f"Episodes: {len(results)}")
    print(f"Average Return: {avg_return:.2%}")
    print(f"Average Sharpe Ratio: {avg_sharpe:.2f}")
    print(f"Average Max Drawdown: {avg_max_dd:.2%}")
    print(f"Average Number of Trades: {avg_trades:.0f}")
    
    # Best and worst episodes
    best_episode = max(results, key=lambda x: x['total_return'])
    worst_episode = min(results, key=lambda x: x['total_return'])
    
    print(f"\nBest Episode: {best_episode['episode']} (Return: {best_episode['total_return']:.2%})")
    print(f"Worst Episode: {worst_episode['episode']} (Return: {worst_episode['total_return']:.2%})")


def print_trade_history(results, max_trades=50):
    """Print trade history for quick inspection"""
    rows = []
    for result in results:
        for step, price, trade_type in result['trade_history']:
            rows.append((result['episode'], step, price, trade_type))

    if not rows:
        print("\nNo trades recorded.")
        return

    total = len(rows)
    rows = rows[:max_trades]
    print("\n=== TRADE HISTORY (first entries) ===")
    for episode, step, price, trade_type in rows:
        print(f"Episode {episode} | Step {step} | Price {price:.3f} | {trade_type}")

    if total > len(rows):
        print(f"... {total - len(rows)} more trades")


def save_trade_history(results, df, save_path):
    """Save trade history to CSV"""
    rows = []
    has_date_col = 'Date' in df.columns
    for result in results:
        for step, price, trade_type in result['trade_history']:
            row = {
                'episode': result['episode'],
                'step': step,
                'price': price,
                'trade_type': trade_type
            }
            if has_date_col and step < len(df):
                row['date'] = df.iloc[step]['Date']
            rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(save_path, index=False)
    print(f"Trade history saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained DQN agent')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model file')
    parser.add_argument('--data', type=str, default=None,
                       help='Path to CSV data file (if not provided, creates sample data)')
    parser.add_argument('--episodes', type=int, default=1,
                       help='Number of evaluation episodes')
    parser.add_argument('--plots', action='store_true',
                       help='Generate and save plots')
    parser.add_argument('--plot-dir', type=str, default='plots',
                       help='Directory to save plots')
    parser.add_argument('--print-trades', action='store_true',
                       help='Print trade history to console')
    parser.add_argument('--max-trades', type=int, default=50,
                       help='Max trades to print when using --print-trades')
    parser.add_argument('--trades-file', type=str, default=None,
                       help='Save trade history to CSV')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file {args.model} not found!")
        return
    
    # Load or create data
    if args.data and os.path.exists(args.data):
        print(f"Loading data from {args.data}")
        df = load_and_prepare_data(args.data)
    else:
        print("Creating sample data")
        df = create_sample_data(n_samples=1000)  # Smaller for evaluation
    
    # Create environment
    env = GoldTradingEnv(df)
    
    # Get state and action sizes
    state_size = env.reset().shape[0]
    action_size = 4
    
    # Create agent and load model
    agent = DQNAgent(state_size, action_size)
    agent.load(args.model)
    
    print(f"Model loaded: {args.model}")
    print(f"Evaluating for {args.episodes} episode(s)...")
    
    # Evaluate agent
    results = evaluate_agent(env, agent, args.episodes)
    
    # Print summary
    print_summary(results)

    if args.print_trades:
        print_trade_history(results, max_trades=args.max_trades)

    if args.trades_file:
        save_trade_history(results, df, args.trades_file)
    
    # Generate plots if requested
    if args.plots:
        print("\nGenerating plots...")
        plot_results(results, df, args.plot_dir)
    
    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()
