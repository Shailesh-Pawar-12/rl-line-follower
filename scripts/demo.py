#!/usr/bin/env python3
"""
Demo script for the line-following robot.
Simple demonstration of how to load and use a trained model.
"""

import os
import sys
import argparse
import time
import numpy as np
from stable_baselines3 import PPO
import gymnasium as gym

# Add the parent directory to the path to import our custom environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import line_following_env


def run_demo(model_path, n_episodes=3, render=True):
    """Run a simple demo of the trained model."""
    
    # Load the trained model
    try:
        model = PPO.load(model_path)
        print(f"Loaded model from: {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Create environment
    env = gym.make('LineFollowing-v0')
    
    print(f"\nRunning demo with {n_episodes} episodes...")
    print("Press Ctrl+C to stop early\n")
    
    try:
        for episode in range(n_episodes):
            print(f"Episode {episode + 1}/{n_episodes}")
            print("-" * 40)
            
            obs, info = env.reset()
            episode_reward = 0
            step_count = 0
            
            done = False
            while not done:
                # Get action from model
                action, _states = model.predict(obs, deterministic=True)
                
                # Take action
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                step_count += 1
                
                # Print current state every 10 steps
                if step_count % 10 == 0 or done:
                    action_names = ['LEFT', 'STRAIGHT', 'RIGHT']
                    print(f"  Step {step_count:3d}: Action={action_names[action]:8s}, "
                          f"Error={obs[0]:6.3f}, Reward={reward:6.2f}")
                
                # Render if requested
                if render:
                    env.render()
                    time.sleep(0.1)
            
            print(f"  Final: Steps={step_count}, Total Reward={episode_reward:.2f}")
            print(f"  Line Error: {obs[0]:.3f}")
            
            if episode < n_episodes - 1:
                print("\nPress Enter to continue to next episode...")
                input()
                print()
    
    except KeyboardInterrupt:
        print("\nDemo stopped by user.")
    finally:
        env.close()


def interactive_demo(model_path):
    """Interactive demo where user can manually control some aspects."""
    
    # Load the trained model
    try:
        model = PPO.load(model_path)
        print(f"Loaded model from: {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Create environment
    env = gym.make('LineFollowing-v0')
    
    print("\nInteractive Demo Mode")
    print("The robot will follow the line automatically.")
    print("Press Enter to step through, 'q' to quit, 'r' to reset episode")
    print("-" * 60)
    
    try:
        obs, info = env.reset()
        episode_reward = 0
        step_count = 0
        
        while True:
            # Display current state
            print(f"\nStep {step_count}: Line Error = {obs[0]:.3f}")
            
            # Get action from model
            action, _states = model.predict(obs, deterministic=True)
            action_names = ['LEFT', 'STRAIGHT', 'RIGHT']
            print(f"Model chooses action: {action_names[action]}")
            
            # Render current state
            env.render()
            
            # Wait for user input
            user_input = input("Press Enter to step (or 'q' to quit, 'r' to reset): ").strip().lower()
            
            if user_input == 'q':
                break
            elif user_input == 'r':
                print("Resetting episode...")
                obs, info = env.reset()
                episode_reward = 0
                step_count = 0
                continue
            
            # Take action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            step_count += 1
            
            print(f"Reward: {reward:.2f}, Total: {episode_reward:.2f}")
            
            if done:
                print(f"\nEpisode finished! Total reward: {episode_reward:.2f}")
                print("Starting new episode...")
                obs, info = env.reset()
                episode_reward = 0
                step_count = 0
    
    except KeyboardInterrupt:
        print("\nInteractive demo stopped by user.")
    finally:
        env.close()


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description='Demo of trained line-following robot')
    parser.add_argument('--model-path', type=str, 
                      default='models/line_following_ppo_best/best_model.zip',
                      help='Path to the trained model')
    parser.add_argument('--episodes', type=int, default=3,
                      help='Number of demo episodes (default: 3)')
    parser.add_argument('--no-render', action='store_true',
                      help='Disable rendering')
    parser.add_argument('--interactive', action='store_true',
                      help='Run interactive demo')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Model not found at: {args.model_path}")
        print("Please train a model first using: python scripts/train.py")
        return
    
    if args.interactive:
        interactive_demo(args.model_path)
    else:
        run_demo(args.model_path, args.episodes, not args.no_render)


if __name__ == "__main__":
    main()
