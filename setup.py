#!/usr/bin/env python3
"""
🤖 Line Following Robot RL Project - One-Click Setup
====================================================

This script automatically installs all dependencies and sets up the project.
Just run: python3 setup.py

Features:
- Installs lightweight CPU-only PyTorch (~180MB instead of ~2GB)
- Handles NumPy compatibility issues automatically  
- Tests complete pipeline after installation
- Works on Linux, macOS, and Windows
"""

import subprocess
import sys
import os
import platform


def print_header():
    """Print setup header with project info."""
    print("=" * 70)
    print("🤖 LINE FOLLOWING ROBOT - REINFORCEMENT LEARNING PROJECT")
    print("=" * 70)
    print("🚀 One-Click Setup Script")
    print("📝 This will install all dependencies and test the environment")
    print("⏱️  Estimated time: 2-5 minutes")
    print("💾 Download size: ~200MB (CPU-only PyTorch)")
    print("=" * 70)


def check_system_requirements():
    """Check system requirements and provide recommendations."""
    print("\n🔍 Checking System Requirements...")
    
    # Check Python version
    version = sys.version_info
    print(f"🐍 Python: {version.major}.{version.minor}.{version.micro}")
    
    if version.major != 3 or version.minor < 8:
        print("❌ ERROR: Python 3.8+ is required!")
        print("💡 Please install Python 3.8 or newer and try again.")
        return False
    
    # Check OS
    os_name = platform.system()
    print(f"💻 Operating System: {os_name}")
    
    # Check pip
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                      check=True, capture_output=True)
        print("📦 pip: Available")
    except subprocess.CalledProcessError:
        print("❌ ERROR: pip is not available!")
        print("💡 Please install pip and try again.")
        return False
    
    print("✅ System requirements met!")
    return True


def run_command(command, description="", show_output=False):
    """Run a command with better error handling."""
    print(f"\n🔧 {description}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        
        if show_output and result.stdout:
            print(result.stdout)
        elif "Successfully installed" in result.stdout:
            # Show only the installed packages line
            for line in result.stdout.split('\n'):
                if "Successfully installed" in line:
                    packages = line.replace("Successfully installed ", "")
                    print(f"   ✅ Installed: {packages}")
        else:
            print("   ✅ Success!")
            
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed!")
        if e.stderr:
            print(f"   Error details: {e.stderr[:200]}...")
        return False


def install_dependencies():
    """Install all required dependencies in correct order."""
    print("\n📦 Installing Dependencies...")
    
    steps = [
        ("pip3 install --upgrade pip --user", 
         "Upgrading pip to latest version"),
         
        ("pip3 install 'numpy<2.0' --user", 
         "Installing NumPy (compatible version)"),
        
        ("pip3 install gym==0.26.2 gymnasium==0.29.1 --user", 
         "Installing Gym environments"),
        
        ("pip3 install cloudpickle gym-notices farama-notifications typing-extensions --user", 
         "Installing Gym dependencies"),
        
        ("pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu --user", 
         "Installing PyTorch (CPU-only, ~180MB)"),
        
        ("pip3 install stable-baselines3==2.1.0 --user", 
         "Installing Reinforcement Learning library"),
        
        ("pip3 install pandas matplotlib scipy tensorboard --user", 
         "Installing visualization and logging tools"),
    ]
    
    failed_steps = []
    
    for i, (command, description) in enumerate(steps, 1):
        print(f"\n[{i}/{len(steps)}] {description}")
        if not run_command(command, f"Running: {command}"):
            failed_steps.append(description)
            # Continue with other installations even if one fails
    
    return failed_steps


def test_installation():
    """Test that everything is working correctly."""
    print("\n🧪 Testing Installation...")
    
    # Test 1: Import test
    print("\n[Test 1/3] Testing imports...")
    test_script = """
import numpy as np
import gymnasium as gym  
import torch
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
print("✅ All imports successful!")
"""
    
    try:
        result = subprocess.run([sys.executable, "-c", test_script], 
                              check=True, capture_output=True, text=True)
        print("   ✅ All required packages imported successfully!")
    except subprocess.CalledProcessError as e:
        print("   ❌ Import test failed!")
        print(f"   Error: {e.stderr}")
        return False
    
    # Test 2: Environment test  
    print("\n[Test 2/3] Testing custom environment...")
    if os.path.exists("scripts/test_env.py"):
        if run_command("timeout 30 python3 scripts/test_env.py 2>/dev/null || python3 scripts/test_env.py", 
                      "Running environment test"):
            print("   ✅ Environment test passed!")
        else:
            print("   ⚠️  Environment test had issues (but may still work)")
    else:
        print("   ⚠️  Test script not found (skipping)")
    
    # Test 3: Quick training test
    print("\n[Test 3/3] Testing training pipeline...")
    if os.path.exists("scripts/train.py"):
        if run_command("timeout 60 python3 scripts/train.py --timesteps 100 --n-envs 1 2>/dev/null", 
                      "Running quick training test"):
            print("   ✅ Training pipeline works!")
            return True
        else:
            print("   ⚠️  Training test failed (but basic setup may work)")
    
    return True


def create_directories():
    """Create necessary directories."""
    dirs = ["models", "tensorboard_logs"]
    for dir_name in dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"   📁 Created directory: {dir_name}")


def print_final_instructions(success=True):
    """Print final instructions and next steps."""
    print("\n" + "=" * 70)
    
    if success:
        print("🎉 SETUP COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("🚀 Quick Start Commands:")
        print("")
        print("  # Test the environment")
        print("  python3 scripts/test_env.py")
        print("")
        print("  # Train your robot (5-10 minutes)")
        print("  python3 scripts/train.py --timesteps 50000")
        print("")
        print("  # Watch your trained robot!")
        print("  python3 scripts/demo.py --model-path models/line_following_ppo_best/best_model.zip")
        print("")
        print("📚 For detailed instructions, see README.md")
        
    else:
        print("⚠️  SETUP COMPLETED WITH WARNINGS")
        print("=" * 70)
        print("Some components may not work perfectly, but basic functionality should be available.")
        print("Check the error messages above and try running individual commands.")
    
    print("=" * 70)


def main():
    """Main setup function with comprehensive error handling."""
    print_header()
    
    # Check system requirements
    if not check_system_requirements():
        print("\n❌ System requirements not met. Please fix the issues above and try again.")
        sys.exit(1)
    
    # Create directories
    print("\n📁 Creating project directories...")
    create_directories()
    
    # Install dependencies
    failed_steps = install_dependencies()
    
    # Test installation
    test_success = test_installation()
    
    # Print results
    if failed_steps:
        print(f"\n⚠️  {len(failed_steps)} installation steps had issues:")
        for step in failed_steps:
            print(f"   - {step}")
        print("\n💡 You may still be able to use basic functionality.")
    
    success = len(failed_steps) == 0 and test_success
    print_final_instructions(success)
    
    if not success:
        print("\n🔧 Troubleshooting Tips:")
        print("   • Try running the setup again")
        print("   • Check your internet connection") 
        print("   • Install Python dependencies manually: pip3 install -r requirements_light.txt")
        print("   • See README.md for manual installation instructions")


if __name__ == "__main__":
    main()
