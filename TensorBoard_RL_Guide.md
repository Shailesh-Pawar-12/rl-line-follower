# 📊 Complete TensorBoard Guide for Reinforcement Learning

*A comprehensive reference for understanding and interpreting TensorBoard metrics in any RL project*

---

## 🎯 Table of Contents

1. [Performance Metrics](#-performance-metrics)
2. [Policy Learning Metrics](#-policy-learning-metrics)  
3. [Value Function Metrics](#-value-function-metrics)
4. [Exploration vs Exploitation](#-exploration-vs-exploitation)
5. [Training Stability Metrics](#-training-stability-metrics)
6. [Algorithm-Specific Metrics](#-algorithm-specific-metrics)
7. [Debugging Guide](#-debugging-guide)
8. [Best Practices](#-best-practices)

---

## 🏆 Performance Metrics

### **eval/mean_reward** (Most Important!)
**What it measures:** Average total reward per episode during evaluation
**Range:** Depends on your environment (-∞ to +∞)
**Good trends:**
- ✅ Steady upward trend
- ✅ Reaches and maintains high values
- ✅ Low variance between episodes

**Bad trends:**
- ❌ Flat or decreasing
- ❌ High variance/instability
- ❌ Sudden drops after good performance

**How to use:**
- Primary indicator of learning success
- Use to decide when to stop training
- Compare different hyperparameters
- Set reward thresholds for early stopping

**Debugging:**
- If flat: Learning rate too low, or environment too hard
- If decreasing: Overfitting, learning rate too high
- If noisy: Increase evaluation episodes, check environment randomness

### **eval/mean_ep_length**
**What it measures:** Average number of steps per episode during evaluation
**Range:** 1 to max_episode_steps
**Good trends:**
- ✅ Increasing towards maximum (for survival tasks)
- ✅ Stable at expected length (for fixed-length tasks)
- ✅ Low variance

**Bad trends:**
- ❌ Decreasing over time (agent getting worse)
- ❌ High variance
- ❌ Stuck at very low values

**How to use:**
- Indicates if agent is failing early
- Shows task completion ability
- Helps identify if reward shaping is needed

### **train/mean_reward** 
**What it measures:** Average reward during training (often noisier than eval)
**Interpretation:** Similar to eval/mean_reward but with more variance
**Use case:** Real-time training progress monitoring

---

## 🧠 Policy Learning Metrics

### **train/policy_gradient_loss** (PPO/A3C/etc.)
**What it measures:** How much the policy is changing each update
**Range:** Usually negative values
**Good trends:**
- ✅ Decreasing magnitude (getting less negative)
- ✅ Smooth curve without large spikes
- ✅ Converges to small values

**Bad trends:**
- ❌ Large oscillations
- ❌ Continuously increasing magnitude
- ❌ Sudden spikes

**How to use:**
- Monitor learning progress
- Detect training instability
- Tune learning rate (high loss → lower LR)

**Algorithm notes:**
- PPO: Should be negative and decreasing
- TRPO: Similar behavior expected
- DQN: Uses different loss function (see Q-learning section)

### **train/approx_kl** (PPO/TRPO)
**What it measures:** KL divergence between old and new policies
**Range:** 0 to +∞ (typically 0.001 to 0.1)
**Target range:** Usually 0.01 to 0.05
**Good trends:**
- ✅ Small, stable values
- ✅ Occasional small spikes (normal)
- ✅ Not growing over time

**Bad trends:**
- ❌ Consistently high values (>0.1)
- ❌ Growing trend
- ❌ Large frequent spikes

**How to use:**
- Ensure policy updates aren't too aggressive
- Tune learning rate and PPO clip range
- Early stopping if KL gets too high

### **train/clip_fraction** (PPO)
**What it measures:** Fraction of policy updates that got clipped
**Range:** 0.0 to 1.0
**Target range:** 0.1 to 0.3 for healthy learning
**Good trends:**
- ✅ Moderate values (0.1-0.3)
- ✅ Gradually decreasing as learning stabilizes
- ✅ Stable during good performance

**Bad trends:**
- ❌ Always near 0 (updates too conservative)
- ❌ Always near 1 (updates too aggressive)
- ❌ Highly variable

**How to use:**
- Tune PPO clip range (default 0.2)
- Adjust learning rate
- Monitor training aggressiveness

---

## 💰 Value Function Metrics

### **train/value_loss**
**What it measures:** Error in predicting future rewards
**Range:** 0 to +∞ (lower is better)
**Good trends:**
- ✅ Decreasing over time
- ✅ Converging to low, stable value
- ✅ Smooth curve

**Bad trends:**
- ❌ Increasing over time
- ❌ Large oscillations
- ❌ Never converging

**How to use:**
- Monitor critic network learning
- Tune value function learning rate
- Detect overfitting in value estimation

**Debugging:**
- High loss: Value LR too high, or target too hard to predict
- Increasing loss: Overfitting, need regularization
- Oscillating: Unstable training, reduce learning rate

### **train/explained_variance**
**What it measures:** How well value function explains reward variance
**Range:** 0.0 to 1.0 (1.0 is perfect prediction)
**Good values:** > 0.5 for most tasks
**Good trends:**
- ✅ Increasing towards 1.0
- ✅ Stable at high values (>0.7)
- ✅ Smooth improvement

**Bad trends:**
- ❌ Decreasing over time
- ❌ Stuck at very low values (<0.1)
- ❌ Highly variable

**How to use:**
- Assess value function quality
- Debug reward prediction issues
- Tune value network architecture

### **train/value_function_error**
**What it measures:** Mean absolute error of value predictions
**Interpretation:** Similar to value_loss but in original reward units
**Use case:** More interpretable than loss functions

---

## 🎲 Exploration vs Exploitation

### **train/entropy_loss** 
**What it measures:** Randomness in policy decisions
**Range:** Negative values (more negative = more random)
**Good trends:**
- ✅ High entropy early (exploration)
- ✅ Gradually decreasing (focusing on good actions)
- ✅ Stabilizes at moderate level

**Bad trends:**
- ❌ Drops to zero too quickly (premature convergence)
- ❌ Stays too high (never learns to exploit)
- ❌ Highly variable

**Algorithm differences:**
- **PPO/A3C:** Entropy bonus encourages exploration
- **DQN:** Uses epsilon-greedy instead
- **SAC:** Automatically balances exploration

**How to tune:**
- Increase entropy coefficient for more exploration
- Decrease if agent never exploits good strategies
- Use entropy scheduling (high → low over training)

### **train/epsilon** (DQN family)
**What it measures:** Probability of random action selection
**Range:** 0.0 to 1.0
**Typical schedule:** 1.0 → 0.1 over training
**Good trends:**
- ✅ Starts high (1.0 or 0.9)
- ✅ Gradually decreases
- ✅ Stabilizes at low value (0.01-0.1)

**How to use:**
- Balance exploration vs exploitation
- Tune epsilon decay schedule
- Monitor if agent explores enough

---

## ⚙️ Training Stability Metrics

### **train/learning_rate**
**What it measures:** Current learning rate (if using scheduling)
**Common schedules:**
- Constant (flat line)
- Linear decay
- Exponential decay
- Cosine annealing

**How to use:**
- Monitor LR scheduling
- Tune initial learning rate
- Implement adaptive scheduling

### **train/clip_range** (PPO)
**What it measures:** Current PPO clipping parameter
**Typical values:** 0.1 to 0.3
**Usage:** Can be scheduled like learning rate

### **train/loss** (Total Loss)
**What it measures:** Combined loss from all components
**Components:** Policy loss + Value loss + Entropy loss
**Good trends:**
- ✅ Generally decreasing
- ✅ Stabilizes during good performance
- ✅ Smooth curve

### **train/grad_norm**
**What it measures:** Magnitude of gradients during training
**Range:** 0 to +∞
**Good values:** Usually 0.1 to 10.0
**Problems:**
- Very high (>100): Gradient explosion
- Very low (<0.001): Vanishing gradients
- Growing over time: Training instability

---

## 🔄 Algorithm-Specific Metrics

### **PPO (Proximal Policy Optimization)**
**Key metrics:**
- `policy_gradient_loss`: Policy improvement signal
- `value_loss`: Critic learning progress  
- `approx_kl`: Policy change magnitude
- `clip_fraction`: Update aggressiveness
- `entropy_loss`: Exploration level

**Healthy PPO training:**
- Policy loss: Decreasing and negative
- Value loss: Decreasing to stable level
- KL divergence: 0.01-0.05 range
- Clip fraction: 0.1-0.3 range

### **DQN (Deep Q-Network)**
**Key metrics:**
- `q_loss` or `td_error`: Temporal difference error
- `mean_q_value`: Average predicted Q-values
- `epsilon`: Exploration probability
- `target_network_update`: When target updates

**Healthy DQN training:**
- Q-loss: Decreasing over time
- Mean Q-value: Should increase as agent improves
- Epsilon: Gradual decrease from 1.0 to ~0.1

### **SAC (Soft Actor-Critic)**
**Key metrics:**
- `actor_loss`: Policy improvement
- `critic_loss`: Value function learning
- `alpha_loss`: Temperature parameter learning
- `entropy`: Policy entropy (automatic tuning)

### **A3C/A2C (Actor-Critic)**
**Key metrics:**
- `policy_loss`: Actor network loss
- `value_loss`: Critic network loss
- `entropy`: Policy randomness
- `advantage`: Advantage function values

---

## 🔧 Debugging Guide

### **Problem: Reward Not Improving**

**Symptoms:**
- Flat eval/mean_reward
- High variance in performance
- Policy loss not decreasing

**Potential causes & solutions:**
1. **Learning rate too low** → Increase LR
2. **Environment too hard** → Simplify task or add reward shaping
3. **Poor exploration** → Increase entropy coefficient or epsilon
4. **Network too small** → Increase network size
5. **Bad hyperparameters** → Grid search key parameters

### **Problem: Training Unstable**

**Symptoms:**
- Large oscillations in losses
- Performance keeps dropping
- High KL divergence or gradient norms

**Solutions:**
1. **Reduce learning rate** (most common fix)
2. **Increase batch size**
3. **Add gradient clipping**
4. **Reduce PPO clip range**
5. **Check environment for bugs**

### **Problem: Overfitting**

**Symptoms:**
- Training reward >> Evaluation reward
- Performance degrades after peak
- Value loss increasing while policy loss decreasing

**Solutions:**
1. **Early stopping** based on eval performance
2. **Regularization** (dropout, weight decay)
3. **Reduce network size**
4. **More diverse training environments**
5. **Shorter training episodes**

### **Problem: Premature Convergence**

**Symptoms:**
- Entropy drops to zero quickly
- Policy becomes deterministic too early
- Suboptimal final performance

**Solutions:**
1. **Increase entropy coefficient**
2. **Slower entropy decay**
3. **Higher exploration (epsilon/temperature)**
4. **Curriculum learning**

---

## 📈 Best Practices

### **Monitoring Strategy**
1. **Primary metrics:** eval/mean_reward, eval/mean_ep_length
2. **Secondary metrics:** policy_loss, value_loss, entropy
3. **Debug metrics:** approx_kl, clip_fraction, grad_norm
4. **Update frequency:** Every 1000-10000 steps for evaluation

### **Hyperparameter Tuning Priority**
1. **Learning rate** (most important)
2. **Network architecture**
3. **Batch size / n_steps**
4. **Entropy coefficient**
5. **Discount factor (gamma)**

### **When to Stop Training**
**Stop when:**
- ✅ Evaluation reward plateaus at satisfactory level
- ✅ Performance variance becomes very low
- ✅ Value loss starts increasing while policy plateaus

**Don't stop due to:**
- ❌ Training reward plateaus (check eval instead)
- ❌ Temporary performance drops
- ❌ High value loss if eval performance is good

### **Experiment Tracking**
```python
# Log custom metrics
writer.add_scalar('custom/success_rate', success_rate, step)
writer.add_scalar('custom/episode_length_std', length_std, step)
writer.add_scalar('custom/reward_per_step', reward/length, step)
```

### **TensorBoard Organization**
```
logs/
├── experiment_1_baseline/
├── experiment_2_higher_lr/
├── experiment_3_larger_network/
└── experiment_4_different_reward/
```

---

## 🎯 Quick Reference Cheat Sheet

| Metric | Good Range | Trend | Red Flags |
|--------|------------|-------|-----------|
| eval/mean_reward | Task-dependent | ↗️ Increasing | ↘️ Decreasing after peak |
| eval/mean_ep_length | Near maximum | ↗️ or → Stable | ↘️ Decreasing |
| policy_gradient_loss | Negative, small | ↘️ Decreasing magnitude | Large oscillations |
| value_loss | Low, positive | ↘️ Decreasing | ↗️ Increasing |
| approx_kl (PPO) | 0.01 - 0.05 | → Stable | > 0.1 consistently |
| clip_fraction (PPO) | 0.1 - 0.3 | ↘️ Gradually decreasing | Always 0 or 1 |
| entropy_loss | Moderate negative | ↘️ Gradual decrease | Drops to 0 quickly |
| explained_variance | > 0.5 | ↗️ Increasing | < 0.1 or decreasing |

---

## 📚 Additional Resources

- **Stable Baselines3 Docs:** [https://stable-baselines3.readthedocs.io/](https://stable-baselines3.readthedocs.io/)
- **OpenAI Spinning Up:** [https://spinningup.openai.com/](https://spinningup.openai.com/)
- **TensorBoard Guide:** [https://www.tensorflow.org/tensorboard](https://www.tensorflow.org/tensorboard)

---

*This guide covers the most common RL algorithms and metrics. For specialized algorithms or custom environments, additional metrics may be relevant.*
