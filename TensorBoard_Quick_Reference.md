# 🚀 TensorBoard Quick Reference - The Essential 5

*Focus on these 5 metrics for 90% of your RL debugging needs*

---

## 🏆 #1: eval/mean_reward (THE MOST IMPORTANT)
```
What: Average reward during testing
Goal: Should increase over time
Good: Steady upward trend
Bad: Flat line or decreasing
Action: If bad → check learning rate, environment difficulty
```

## 📏 #2: eval/mean_ep_length  
```
What: How long episodes last
Goal: Should reach maximum (for survival tasks)
Good: Increasing or stable at max
Bad: Decreasing or very short
Action: If bad → agent failing early, need better reward shaping
```

## 🧠 #3: train/policy_gradient_loss
```
What: How much AI's decisions are changing
Goal: Should decrease (getting less negative)
Good: Smooth decreasing curve  
Bad: Large spikes or increasing
Action: If bad → reduce learning rate
```

## 💰 #4: train/value_loss
```
What: How well AI predicts future rewards
Goal: Should decrease over time
Good: Decreasing to low stable value
Bad: Increasing or very high
Action: If bad → might still work (like your robot!), but consider tuning
```

## 🎲 #5: train/entropy_loss (exploration)
```
What: How random vs confident AI's decisions are
Goal: Start high (random), gradually decrease (confident)  
Good: Gradual decrease, not too fast
Bad: Drops to zero immediately OR stays too high
Action: Tune entropy coefficient to balance exploration
```

---

## 🚨 Emergency Debugging (When Things Go Wrong)

### **Training Exploding/Unstable:**
1. **Reduce learning rate** (try 10x smaller)
2. **Reduce PPO clip range** (from 0.2 to 0.1) 
3. **Add gradient clipping**

### **No Learning (Flat Performance):**
1. **Increase learning rate** (try 3x larger)
2. **Check if environment is too hard**
3. **Increase exploration** (entropy coefficient)

### **Good Training, Bad Evaluation:**
1. **Overfitting** - stop training earlier
2. **Environment mismatch** - check train vs eval environments
3. **Deterministic evaluation** - make sure eval uses same policy

---

## 🎯 The "Good Enough" Checklist

✅ **eval/mean_reward** is increasing  
✅ **eval/mean_ep_length** reaches maximum consistently  
✅ **policy_gradient_loss** is decreasing smoothly  
✅ Performance is stable (low variance)  
✅ Training doesn't crash or explode  

**If all 5 are checked → Your training is working!** 🎉

---

## 📱 Phone-Friendly Quick Check
*When you just want to peek at training progress*

**Look at these 3 graphs only:**
1. `eval/mean_reward` ← Going up? ✅  
2. `eval/mean_ep_length` ← At maximum? ✅
3. `train/policy_gradient_loss` ← Decreasing? ✅

**All good = Keep training**  
**Any bad = Check full guide for debugging**

---

*For complete details, see the full TensorBoard_RL_Guide.md*
