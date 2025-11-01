#!/usr/bin/env python3
"""
Test script for the line-following environment.
Validates that the environment works correctly before training.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

# Add the parent directory to the path to import our custom environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import line_following_env


def test_environment_basic():
    """Test basic environment functionality."""
    print("Testing basic environment functionality...")
    
    # Create environment
    env = gym.make('LineFollowing-v0')
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Test reset
    obs, info = env.reset()
    print(f"Initial observation: {obs}")
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial observation type: {type(obs)}")
    
    # Test random actions
    print("\nTesting random actions:")
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        action_names = ['LEFT', 'STRAIGHT', 'RIGHT']
        print(f"Step {i+1}: Action={action_names[action]}, Obs={obs[0]:.3f}, "
              f"Reward={reward:.3f}, Done={done}")
    
    env.close()
    print("Basic environment test completed successfully!")


def test_environment_episode():
    """Test a full episode with random actions."""
    print("\nTesting full episode with random actions...")
    
    env = gym.make('LineFollowing-v0')
    obs, info = env.reset()
    
    episode_reward = 0
    step_count = 0
    max_steps = 100
    
    print(f"{'Step':>4} {'Action':>8} {'Error':>8} {'Reward':>8} {'Done':>6}")
    print("-" * 50)
    
    done = False
    while not done and step_count < max_steps:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        episode_reward += reward
        step_count += 1
        
        action_names = ['LEFT', 'STRAIGHT', 'RIGHT']
        if step_count % 10 == 0 or done:
            print(f"{step_count:4d} {action_names[action]:>8} {obs[0]:8.3f} "
                  f"{reward:8.3f} {done!s:>6}")
    
    print(f"\nEpisode completed:")
    print(f"  Steps: {step_count}")
    print(f"  Total reward: {episode_reward:.3f}")
    print(f"  Final error: {obs[0]:.3f}")
    print(f"  Episode ended: {'max steps reached' if step_count >= max_steps else 'done condition met'}")
    
    env.close()


def test_environment_visualization():
    """Test environment rendering."""
    print("\nTesting environment visualization...")
    
    env = gym.make('LineFollowing-v0')
    obs, info = env.reset()
    
    print("Rendering environment for 20 steps...")
    print("Close the matplotlib window to continue.")
    
    for i in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        env.render()
        plt.pause(0.5)  # Pause for half a second
        
        if done:
            print(f"Episode finished after {i+1} steps")
            break
    
    env.close()
    print("Visualization test completed!")


def test_environment_deterministic():
    """Test that environment behaves deterministically with same actions."""
    print("\nTesting environment determinism...")
    
    # Test sequence of actions
    actions = [1, 1, 0, 2, 1, 0, 0, 2]  # straight, straight, left, right, etc.
    
    results1 = []
    results2 = []
    
    # Run sequence twice
    for run in range(2):
        env = gym.make('LineFollowing-v0')
        # Set same random seed for deterministic test
        np.random.seed(42)
        obs, info = env.reset()
        
        results = []
        for action in actions:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            results.append((obs[0], reward, done))
            if done:
                break
        
        if run == 0:
            results1 = results
        else:
            results2 = results
        
        env.close()
    
    # Compare results
    if len(results1) == len(results2):
        differences = []
        for i, ((obs1, r1, d1), (obs2, r2, d2)) in enumerate(zip(results1, results2)):
            diff = abs(obs1 - obs2) + abs(r1 - r2)
            differences.append(diff)
            if diff > 1e-10:  # Allow for small numerical differences
                print(f"Step {i}: Difference detected - {diff}")
        
        if max(differences) < 1e-6:
            print("Environment behaves deterministically ✓")
        else:
            print(f"Environment shows non-deterministic behavior (max diff: {max(differences)})")
    else:
        print(f"Different episode lengths: {len(results1)} vs {len(results2)}")


def test_reward_function():
    """Test the reward function behavior."""
    print("\nTesting reward function...")
    
    env = gym.make('LineFollowing-v0')
    
    # Test rewards at different error levels
    test_errors = [-2.0, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 2.0]
    
    print(f"{'Error':>8} {'Action':>8} {'Reward':>8}")
    print("-" * 30)
    
    for error in test_errors:
        # Manually set the environment state for testing
        env.reset()
        env.line_error = error
        
        for action in range(3):
            # Test by actually taking the action
            env.reset()
            env.line_error = error
            obs, reward, terminated, truncated, info = env.step(action)
            action_names = ['LEFT', 'STRAIGHT', 'RIGHT']
            print(f"{error:8.1f} {action_names[action]:>8} {reward:8.3f}")
    
    env.close()


def main():
    """Run all environment tests."""
    print("Line Following Environment Test Suite")
    print("=" * 50)
    
    try:
        test_environment_basic()
        test_environment_episode()
        test_environment_deterministic()
        test_reward_function()
        
        # Ask user if they want to see visualization
        response = input("\nRun visualization test? (y/n): ").strip().lower()
        if response == 'y':
            test_environment_visualization()
        
        print("\n" + "=" * 50)
        print("All tests completed successfully!")
        print("Environment is ready for training.")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
