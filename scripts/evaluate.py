#!/usr/bin/env python3
"""
Evaluation script for the trained line-following robot.
Loads a trained model and evaluates its performance.
"""

import os
import sys
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import gymnasium as gym

# Add the parent directory to the path to import our custom environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import line_following_env


def load_model(model_path):
    """Load a trained PPO model."""
    try:
        model = PPO.load(model_path)
        print(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None


def evaluate_model(model, env, n_episodes=10, render=False, deterministic=True):
    """Evaluate the model performance over multiple episodes."""
    episode_rewards = []
    episode_lengths = []
    episode_stats = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_errors = []
        
        done = False
        while not done:
            # Get action from the model
            action, _states = model.predict(obs, deterministic=deterministic)
            
            # Take action in environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Track statistics
            episode_reward += reward
            episode_length += 1
            episode_errors.append(abs(info['line_error']))
            
            # Render if requested
            if render:
                env.render()
                time.sleep(0.05)  # Slow down for visualization
        
        # Store episode results
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_stats.append({
            'reward': episode_reward,
            'length': episode_length,
            'mean_error': np.mean(episode_errors),
            'max_error': np.max(episode_errors),
            'final_error': abs(info['line_error'])
        })
        
        print(f"Episode {episode + 1:2d}: Reward={episode_reward:7.2f}, "
              f"Length={episode_length:4d}, Mean Error={np.mean(episode_errors):.3f}")
    
    return episode_rewards, episode_lengths, episode_stats


def analyze_performance(episode_stats):
    """Analyze and display performance statistics."""
    rewards = [stats['reward'] for stats in episode_stats]
    lengths = [stats['length'] for stats in episode_stats]
    mean_errors = [stats['mean_error'] for stats in episode_stats]
    max_errors = [stats['max_error'] for stats in episode_stats]
    final_errors = [stats['final_error'] for stats in episode_stats]
    
    print("\n" + "="*60)
    print("PERFORMANCE ANALYSIS")
    print("="*60)
    print(f"Average Reward:     {np.mean(rewards):8.2f} ± {np.std(rewards):.2f}")
    print(f"Average Length:     {np.mean(lengths):8.2f} ± {np.std(lengths):.2f}")
    print(f"Average Mean Error: {np.mean(mean_errors):8.3f} ± {np.std(mean_errors):.3f}")
    print(f"Average Max Error:  {np.mean(max_errors):8.3f} ± {np.std(max_errors):.3f}")
    print(f"Average Final Error:{np.mean(final_errors):8.3f} ± {np.std(final_errors):.3f}")
    print(f"Success Rate (Reward > 0): {sum(1 for r in rewards if r > 0) / len(rewards) * 100:.1f}%")
    print(f"High Performance (Reward > 100): {sum(1 for r in rewards if r > 100) / len(rewards) * 100:.1f}%")
    
    return {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'mean_length': np.mean(lengths),
        'mean_error': np.mean(mean_errors),
        'success_rate': sum(1 for r in rewards if r > 0) / len(rewards)
    }


def plot_evaluation_results(episode_stats, save_plot=False):
    """Plot evaluation results."""
    rewards = [stats['reward'] for stats in episode_stats]
    lengths = [stats['length'] for stats in episode_stats]
    mean_errors = [stats['mean_error'] for stats in episode_stats]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Model Evaluation Results', fontsize=16)
    
    # Episode rewards
    axes[0, 0].plot(rewards, 'b.-', markersize=8)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Episode lengths
    axes[0, 1].plot(lengths, 'g.-', markersize=8)
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Episode Length (steps)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Mean errors
    axes[1, 0].plot(mean_errors, 'r.-', markersize=8)
    axes[1, 0].set_title('Mean Line Error per Episode')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Mean Absolute Line Error')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Reward distribution
    axes[1, 1].hist(rewards, bins=10, alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(x=np.mean(rewards), color='r', linestyle='--', 
                      label=f'Mean: {np.mean(rewards):.1f}')
    axes[1, 1].set_title('Reward Distribution')
    axes[1, 1].set_xlabel('Total Reward')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig('evaluation_results.png', dpi=150, bbox_inches='tight')
        print("Evaluation plot saved as 'evaluation_results.png'")
    
    plt.show()


def run_single_episode_with_visualization(model, env):
    """Run a single episode with detailed visualization and step-by-step info."""
    print("\n" + "="*60)
    print("RUNNING SINGLE EPISODE WITH VISUALIZATION")
    print("="*60)
    
    obs = env.reset()
    episode_reward = 0
    step_count = 0
    
    print(f"{'Step':>4} {'Action':>8} {'Error':>8} {'Reward':>8} {'Total':>8}")
    print("-" * 44)
    
    done = False
    while not done and step_count < 50:  # Limit to 50 steps for demo
        # Get action from model
        action, _states = model.predict(obs, deterministic=True)
        action_names = ['LEFT', 'STRAIGHT', 'RIGHT']
        
        # Take step
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        step_count += 1
        
        # Print step info
        print(f"{step_count:4d} {action_names[action]:>8} {obs[0]:8.3f} {reward:8.2f} {episode_reward:8.2f}")
        
        # Render environment
        env.render()
        time.sleep(0.2)  # Pause for visualization
        
        if done:
            print(f"\nEpisode finished after {step_count} steps")
            print(f"Final reward: {episode_reward:.2f}")
            print(f"Final line error: {obs[0]:.3f}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate trained line-following robot')
    parser.add_argument('--model-path', type=str, required=True,
                      help='Path to the trained model')
    parser.add_argument('--episodes', type=int, default=10,
                      help='Number of evaluation episodes (default: 10)')
    parser.add_argument('--render', action='store_true',
                      help='Render episodes during evaluation')
    parser.add_argument('--plot', action='store_true',
                      help='Plot evaluation results')
    parser.add_argument('--save-plot', action='store_true',
                      help='Save evaluation plot to file')
    parser.add_argument('--demo', action='store_true',
                      help='Run single episode with step-by-step visualization')
    parser.add_argument('--deterministic', action='store_true', default=True,
                      help='Use deterministic policy (default: True)')
    
    args = parser.parse_args()
    
    # Load the model
    model = load_model(args.model_path)
    if model is None:
        return
    
    # Create environment
    env = gym.make('LineFollowing-v0')
    
    try:
        if args.demo:
            # Run single episode with visualization
            run_single_episode_with_visualization(model, env)
        else:
            # Run full evaluation
            print(f"Evaluating model over {args.episodes} episodes...")
            episode_rewards, episode_lengths, episode_stats = evaluate_model(
                model, env, 
                n_episodes=args.episodes,
                render=args.render,
                deterministic=args.deterministic
            )
            
            # Analyze performance
            performance_summary = analyze_performance(episode_stats)
            
            # Plot results if requested
            if args.plot or args.save_plot:
                plot_evaluation_results(episode_stats, save_plot=args.save_plot)
        
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
