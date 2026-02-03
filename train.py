import argparse
import os
import time

import pandas as pd
import torch

from env.gold_env import GoldTradingEnv
from agent.dqn_agent import DQNAgent
from utils.data_processor import load_and_prepare_data, create_sample_data, save_processed_data


def train_agent(
    env,
    agent,
    episodes=1000,
    save_interval=100,
    model_dir="models",
    train_every=1
):
    """
    Train the DQN agent
    
    Args:
        env: Trading environment
        agent: DQN agent
        episodes: Number of training episodes
        save_interval: Save model every N episodes
        model_dir: Directory to save models
    """
    # Create models directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    episode_rewards = []
    episode_lengths = []
    
    print(f"Starting training for {episodes} episodes...")
    print(f"Device: {agent.device}")
    print(f"Using PER: {agent.use_per}")
    
    start_time = time.time()
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            if steps % train_every == 0:
                agent.replay()
            
            state = next_state
            total_reward += reward
            steps += 1
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = sum(episode_rewards[-10:]) / 10
            avg_length = sum(episode_lengths[-10:]) / 10
            elapsed = time.time() - start_time
            
            print(f"Episode {episode+1}/{episodes} | "
                  f"Avg Reward (10): {avg_reward:.2f} | "
                  f"Avg Length: {avg_length:.0f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Time: {elapsed:.1f}s")
        
        # Save model
        if (episode + 1) % save_interval == 0:
            model_path = os.path.join(model_dir, f"model_episode_{episode+1}.pth")
            agent.save(model_path)
            print(f"Model saved: {model_path}")
    
    # Save final model
    final_model_path = os.path.join(model_dir, "model_final.pth")
    agent.save(final_model_path)
    print(f"Final model saved: {final_model_path}")
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f} seconds")
    print(f"Average reward (last 100): {sum(episode_rewards[-100:]) / 100:.2f}")
    print(f"Final epsilon: {agent.epsilon:.3f}")
    
    return episode_rewards, episode_lengths


def main():
    parser = argparse.ArgumentParser(description='Train DQN agent for gold trading')
    parser.add_argument('--data', type=str, default=None, 
                       help='Path to CSV data file (if not provided, creates sample data)')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of training episodes')
    parser.add_argument('--save-interval', type=int, default=100,
                       help='Save model every N episodes')
    parser.add_argument('--model-dir', type=str, default='models',
                       help='Directory to save models')
    parser.add_argument('--no-per', action='store_true',
                       help='Disable Prioritized Experience Replay')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--train-every', type=int, default=1,
                       help='Train every N steps')
    
    args = parser.parse_args()
    
    # Load or create data
    if args.data and os.path.exists(args.data):
        print(f"Loading data from {args.data}")
        df = load_and_prepare_data(args.data)
    else:
        print("Creating sample data")
        df = create_sample_data(n_samples=5000)
        
        # Save sample data for future use
        os.makedirs('data', exist_ok=True)
        sample_data_path = 'data/xauusd_sample.csv'
        save_processed_data(df, sample_data_path)
        print(f"Sample data saved to {sample_data_path}")
    
    # Create environment
    env = GoldTradingEnv(df)
    
    # Get state and action sizes
    state_size = env.reset().shape[0]
    action_size = 4  # hold, buy, sell, close
    
    print(f"State size: {state_size}, Action size: {action_size}")
    
    # Create agent
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        lr=args.lr,
        batch_size=args.batch_size,
        use_per=not args.no_per
    )
    
    # Train agent
    episode_rewards, episode_lengths = train_agent(
        env=env,
        agent=agent,
        episodes=args.episodes,
        save_interval=args.save_interval,
        model_dir=args.model_dir,
        train_every=args.train_every
    )
    
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
