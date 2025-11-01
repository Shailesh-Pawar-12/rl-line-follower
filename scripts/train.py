#!/usr/bin/env python3
"""
Training script for the line-following robot using reinforcement learning.
Uses PPO (Proximal Policy Optimization) from stable-baselines3.
"""

import os
import sys
import argparse
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

# Add the parent directory to the path to import our custom environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import line_following_env


def create_environment(env_id="LineFollowing-v0", n_envs=1):
    """Create the training environment."""
    def _init():
        env = gym.make(env_id)
        return env
    
    if n_envs == 1:
        env = gym.make(env_id)
        env = Monitor(env)
        return env
    else:
        return make_vec_env(env_id, n_envs=n_envs)


def train_model(total_timesteps=100000, save_path="models/line_following_ppo", 
                learning_rate=3e-4, n_envs=4, eval_freq=10000):
    """Train the PPO model on the line-following environment."""
    
    print(f"Starting training with {total_timesteps} timesteps...")
    print(f"Learning rate: {learning_rate}")
    print(f"Number of environments: {n_envs}")
    
    # Create training environment
    env = create_environment(n_envs=n_envs)
    
    # Create evaluation environment
    eval_env = create_environment(n_envs=1)
    
    # Initialize PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=None,
        tensorboard_log="./tensorboard_logs/",
        policy_kwargs=None,
        verbose=1,
        seed=42,
        device='auto'
    )
    
    # Create callbacks
    # Stop training when the model reaches a reward threshold
    reward_threshold_callback = StopTrainingOnRewardThreshold(
        reward_threshold=250000.0,  # Conservative threshold for gradual correction system  
        verbose=1
    )
    
    # Evaluate the model periodically
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=reward_threshold_callback,
        eval_freq=eval_freq,
        best_model_save_path=save_path + "_best/",
        log_path=save_path + "_eval_logs/",
        verbose=1,
        deterministic=True,
        render=False
    )
    
    # Train the model
    print("Training started...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        log_interval=10,
        tb_log_name=f"PPO_LineFollowing_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        reset_num_timesteps=True,
        progress_bar=False
    )
    
    # Save the final model
    model.save(save_path + "_final")
    print(f"Model saved to {save_path}_final")
    
    # Close environments
    env.close()
    eval_env.close()
    
    return model


def plot_training_progress(log_path="models/line_following_ppo_eval_logs/"):
    """Plot training progress from evaluation logs."""
    try:
        from stable_baselines3.common.results_plotter import load_results, ts2xy
        
        if os.path.exists(log_path):
            results = load_results(log_path)
            x, y = ts2xy(results, 'timesteps')
            
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(x, y)
            plt.xlabel('Timesteps')
            plt.ylabel('Episode Reward')
            plt.title('Training Progress - Episode Rewards')
            plt.grid(True)
            
            plt.subplot(1, 2, 2)
            # Calculate moving average
            window = min(100, len(y) // 10)
            if window > 0:
                moving_avg = np.convolve(y, np.ones(window)/window, mode='valid')
                plt.plot(x[:len(moving_avg)], moving_avg)
                plt.xlabel('Timesteps')
                plt.ylabel('Moving Average Reward')
                plt.title(f'Moving Average ({window} episodes)')
                plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
            plt.show()
        else:
            print(f"Log path {log_path} not found. Cannot plot training progress.")
    except ImportError:
        print("Could not import results plotter. Skipping progress plot.")
    except Exception as e:
        print(f"Error plotting training progress: {e}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train line-following robot')
    parser.add_argument('--timesteps', type=int, default=100000,
                      help='Total training timesteps (default: 100000)')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                      help='Learning rate (default: 3e-4)')
    parser.add_argument('--n-envs', type=int, default=4,
                      help='Number of parallel environments (default: 4)')
    parser.add_argument('--save-path', type=str, default='models/line_following_ppo',
                      help='Path to save the model (default: models/line_following_ppo)')
    parser.add_argument('--eval-freq', type=int, default=10000,
                      help='Evaluation frequency (default: 10000)')
    parser.add_argument('--plot', action='store_true',
                      help='Plot training progress after training')
    
    args = parser.parse_args()
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    os.makedirs("tensorboard_logs", exist_ok=True)
    
    try:
        # Train the model
        model = train_model(
            total_timesteps=args.timesteps,
            save_path=args.save_path,
            learning_rate=args.learning_rate,
            n_envs=args.n_envs,
            eval_freq=args.eval_freq
        )
        
        print("Training completed successfully!")
        
        # Plot training progress if requested
        if args.plot:
            plot_training_progress()
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
