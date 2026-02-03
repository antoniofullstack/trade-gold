import argparse
import os
import time

import numpy as np
import pandas as pd

from env.gold_env import GoldTradingEnv
from agent.dqn_agent import DQNAgent
from utils.data_processor import load_and_prepare_data, create_sample_data


def train_agent(agent, env, episodes=100):
    """Train agent for given number of episodes"""
    for _ in range(episodes):
        state = env.reset()
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            
            state = next_state


def test_agent(agent, env):
    """Test agent without exploration"""
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    state = env.reset()
    done = False
    
    while not done:
        action = agent.act(state)
        state, reward, done = env.step(action)
    
    # Restore original epsilon
    agent.epsilon = original_epsilon
    
    return {
        'final_equity': env.net_worth,
        'total_return': (env.net_worth - env.initial_balance) / env.initial_balance,
        'max_drawdown': max(env.drawdown_history) if env.drawdown_history else 0,
        'num_trades': len(env.trade_history),
        'sharpe_ratio': calculate_sharpe(env.equity_history) if len(env.equity_history) > 1 else 0
    }


def calculate_sharpe(equity_history):
    """Calculate Sharpe ratio from equity history"""
    returns = np.diff(equity_history) / equity_history[0]
    return np.mean(returns) / (np.std(returns) + 1e-8)


def walk_forward_validation(df, train_window=2000, test_window=500, step_size=500, 
                          train_episodes=100, save_models=False, model_dir='walk_forward_models'):
    """
    Perform walk-forward validation
    
    Args:
        df: Complete dataframe
        train_window: Size of training window
        test_window: Size of test window
        step_size: Step size between windows
        train_episodes: Episodes to train per window
        save_models: Whether to save models for each window
        model_dir: Directory to save models
    
    Returns:
        list: Results for each window
    """
    if save_models:
        os.makedirs(model_dir, exist_ok=True)
    
    results = []
    total_windows = 0
    
    print(f"Starting walk-forward validation...")
    print(f"Train window: {train_window}, Test window: {test_window}, Step: {step_size}")
    print(f"Training episodes per window: {train_episodes}")
    
    start_time = time.time()
    
    for start in range(0, len(df) - train_window - test_window, step_size):
        total_windows += 1
        
        # Split data
        train_data = df.iloc[start : start + train_window].copy()
        test_data = df.iloc[start + train_window : start + train_window + test_window].copy()
        
        print(f"\n--- Window {total_windows} ---")
        print(f"Train: {len(train_data)} samples, Test: {len(test_data)} samples")
        
        # Create environments
        train_env = GoldTradingEnv(train_data)
        test_env = GoldTradingEnv(test_data)
        
        # Get state and action sizes
        state_size = train_env.reset().shape[0]
        action_size = 4
        
        # Create and train agent
        agent = DQNAgent(state_size, action_size)
        
        print("Training agent...")
        train_start = time.time()
        train_agent(agent, train_env, episodes=train_episodes)
        train_time = time.time() - train_start
        print(f"Training completed in {train_time:.1f}s")
        
        # Test agent
        print("Testing agent...")
        test_start = time.time()
        test_metrics = test_agent(agent, test_env)
        test_time = time.time() - test_start
        
        # Store results
        window_result = {
            'window': total_windows,
            'train_start': start,
            'train_end': start + train_window,
            'test_start': start + train_window,
            'test_end': start + train_window + test_window,
            'train_time': train_time,
            'test_time': test_time,
            **test_metrics
        }
        
        results.append(window_result)
        
        # Print results
        print(f"Results - Return: {test_metrics['total_return']:.2%}, "
              f"Sharpe: {test_metrics['sharpe_ratio']:.2f}, "
              f"Max DD: {test_metrics['max_drawdown']:.2%}, "
              f"Trades: {test_metrics['num_trades']}")
        
        # Save model if requested
        if save_models:
            model_path = os.path.join(model_dir, f"model_window_{total_windows}.pth")
            agent.save(model_path)
            print(f"Model saved: {model_path}")
    
    total_time = time.time() - start_time
    print(f"\nWalk-forward validation completed in {total_time:.1f}s")
    print(f"Total windows processed: {total_windows}")
    
    return results


def analyze_results(results):
    """Analyze and print walk-forward results"""
    if not results:
        return
    
    returns = [r['total_return'] for r in results]
    sharpe_ratios = [r['sharpe_ratio'] for r in results]
    max_drawdowns = [r['max_drawdown'] for r in results]
    num_trades = [r['num_trades'] for r in results]
    
    positive_returns = sum(1 for r in returns if r > 0)
    win_rate = positive_returns / len(returns)
    
    print(f"\n=== WALK-FORWARD ANALYSIS ===")
    print(f"Total Windows: {len(results)}")
    print(f"Win Rate: {win_rate:.1%} ({positive_returns}/{len(results)})")
    print(f"Average Return: {np.mean(returns):.2%}")
    print(f"Return Std Dev: {np.std(returns):.2%}")
    print(f"Best Return: {np.max(returns):.2%}")
    print(f"Worst Return: {np.min(returns):.2%}")
    print(f"Average Sharpe: {np.mean(sharpe_ratios):.2f}")
    print(f"Average Max Drawdown: {np.mean(max_drawdowns):.2%}")
    print(f"Average Trades per Window: {np.mean(num_trades):.0f}")
    
    # Consistency metrics
    consecutive_losses = 0
    max_consecutive_losses = 0
    
    for ret in returns:
        if ret <= 0:
            consecutive_losses += 1
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        else:
            consecutive_losses = 0
    
    print(f"Max Consecutive Losses: {max_consecutive_losses}")
    
    # Risk-adjusted metrics
    if np.std(returns) > 0:
        sortino_ratio = np.mean(returns) / np.std([r for r in returns if r < 0] or [0])
        print(f"Sortino Ratio: {sortino_ratio:.2f}")


def save_results(results, filepath):
    """Save results to CSV"""
    df_results = pd.DataFrame(results)
    df_results.to_csv(filepath, index=False)
    print(f"\nResults saved to {filepath}")


def main():
    parser = argparse.ArgumentParser(description='Walk-forward validation for DQN agent')
    parser.add_argument('--data', type=str, default=None,
                       help='Path to CSV data file (if not provided, creates sample data)')
    parser.add_argument('--train-window', type=int, default=2000,
                       help='Training window size')
    parser.add_argument('--test-window', type=int, default=500,
                       help='Test window size')
    parser.add_argument('--step-size', type=int, default=500,
                       help='Step size between windows')
    parser.add_argument('--train-episodes', type=int, default=100,
                       help='Training episodes per window')
    parser.add_argument('--save-models', action='store_true',
                       help='Save models for each window')
    parser.add_argument('--model-dir', type=str, default='walk_forward_models',
                       help='Directory to save models')
    parser.add_argument('--results-file', type=str, default='walk_forward_results.csv',
                       help='File to save results')
    
    args = parser.parse_args()
    
    # Load or create data
    if args.data and os.path.exists(args.data):
        print(f"Loading data from {args.data}")
        df = load_and_prepare_data(args.data)
    else:
        print("Creating sample data")
        df = create_sample_data(n_samples=10000)  # Larger for walk-forward
    
    # Check if we have enough data
    min_required = args.train_window + args.test_window
    if len(df) < min_required:
        print(f"Error: Not enough data. Need at least {min_required} samples, got {len(df)}")
        return
    
    # Perform walk-forward validation
    results = walk_forward_validation(
        df=df,
        train_window=args.train_window,
        test_window=args.test_window,
        step_size=args.step_size,
        train_episodes=args.train_episodes,
        save_models=args.save_models,
        model_dir=args.model_dir
    )
    
    # Analyze results
    analyze_results(results)
    
    # Save results
    save_results(results, args.results_file)
    
    print("\nWalk-forward validation completed!")


if __name__ == "__main__":
    main()
