import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, Any


class LineFollowingEnv(gym.Env):
    """
    Custom Environment for a line-following two-wheeled robot.
    
    Observation:
        Type: Box(1)
        Num    Observation               Min                     Max
        0      Line position error      -max_line_error          max_line_error
               (negative = line left, positive = line right, zero = on line)
    
    Actions:
        Type: Discrete(3)
        Num   Action
        0     Turn left
        1     Go straight  
        2     Turn right
    
    Reward:
        +1.0 for staying on the line (error close to 0)
        -0.1 for small deviations
        -1.0 for large deviations
        +0.1 bonus for going straight when on line
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, max_line_error: float = 2.0, error_threshold: float = 0.05):
        super(LineFollowingEnv, self).__init__()
        
        # Environment parameters
        self.max_line_error = max_line_error
        self.error_threshold = error_threshold  # Threshold for "on line"
        
        # Action and observation space
        self.action_space = spaces.Discrete(3)  # [left, straight, right]
        self.observation_space = spaces.Box(
            low=-self.max_line_error, 
            high=self.max_line_error, 
            shape=(1,), 
            dtype=np.float32
        )
        
        # Robot state
        self.line_error = 0.0  # Current line position error
        self.robot_angle = 0.0  # Robot's orientation relative to line
        self.velocity = 1.0  # Constant forward velocity
        
        # Episode tracking
        self.steps = 0
        self.max_steps = 1000
        
        # Action history for oscillation detection
        self.recent_actions = []
        self.max_action_history = 10
        
        # For rendering
        self.fig = None
        self.ax = None
        
    def reset(self, seed=None, options=None) -> tuple:
        """Reset the environment to an initial state and return initial observation."""
        # Handle seeding
        if seed is not None:
            np.random.seed(seed)
            
        # Start with a random small error
        self.line_error = np.random.uniform(-0.5, 0.5)
        self.robot_angle = np.random.uniform(-0.2, 0.2)  # Small initial angle error
        self.steps = 0
        
        # Initialize previous error for stability calculation
        self.prev_error = abs(self.line_error)
        
        # Reset action history
        self.recent_actions = []
        
        # Reset drift detection
        self.error_history = []
        
        # Reset error trend tracking for ultra-precision
        self.error_trend_history = []  # Track recent errors for trend analysis
        self.correction_urgency = 0.0  # How urgently correction is needed
        
        # Reset over-correction detection
        self.correction_history = []  # Track recent correction directions for over-correction detection
        
        info = {
            'line_error': self.line_error,
            'robot_angle': self.robot_angle,
            'steps': self.steps
        }
        
        return self._get_observation(), info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one time step within the environment."""
        self.steps += 1
        
        # Convert action to integer if it's a numpy array
        if isinstance(action, np.ndarray):
            action = int(action.item())
        action = int(action)
        
        # Track action history for oscillation detection
        self.recent_actions.append(action)
        if len(self.recent_actions) > self.max_action_history:
            self.recent_actions.pop(0)
        
        # Action mapping: 0=left, 1=straight, 2=right - GENTLE to prevent overshoot oscillation
        action_effects = {
            0: -0.06,  # Turn left (very gentle to prevent overshoot and oscillation)
            1: 0.0,    # Go straight 
            2: 0.06,   # Turn right (very gentle to prevent overshoot and oscillation)
        }
        
        # Apply action to robot angle
        angle_change = action_effects[action]
        self.robot_angle += angle_change
        
        # Simulate robot dynamics
        # The robot moves forward with current angle, affecting line error
        dt = 0.1  # Time step
        
        # Update line error based on robot angle and some random disturbances
        self.line_error += self.robot_angle * dt * self.velocity
        
        # Add some random disturbances to make the problem more realistic  
        # Further reduced to match gentle steering for oscillation elimination
        disturbance = np.random.normal(0, 0.03)
        self.line_error += disturbance
        
        # Apply some damping to robot angle (friction/stability)
        self.robot_angle *= 0.95
        
        # Clip line error to maximum bounds
        self.line_error = np.clip(self.line_error, -self.max_line_error, self.max_line_error)
        
        # Calculate reward
        reward = self._calculate_reward(action)
        
        # Check if episode is done
        terminated = self._is_done()
        truncated = False  # We don't use truncation in this environment
        
        # Additional info
        info = {
            'line_error': self.line_error,
            'robot_angle': self.robot_angle,
            'steps': self.steps
        }
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Get the current observation."""
        # Ensure proper dtype compatibility
        obs = np.array([float(self.line_error)], dtype=np.float32)
        return obs.astype(np.float32)
    
    def _calculate_reward(self, action: int) -> float:
        """Calculate reward based on current state and action."""
        abs_error = abs(self.line_error)
        
        # EXTREME ULTRA-PRECISION reward system - exponential rewards for exact line following
        if abs_error <= 0.02:
            # ULTRA-PERFECT precision - ASTRONOMICAL rewards
            reward = 500.0  # Astronomical reward for being extremely close to line
            if action == 1:  # straight action when perfectly on line
                reward += 1000.0  # ASTRONOMICAL bonus = 1500 total for perfect behavior
            else:
                reward -= 200.0  # Heavy penalty for turning when perfectly positioned
                
        elif abs_error <= 0.03:  # 0.02 < error <= 0.03  
            # EXCEPTIONAL precision - MASSIVE rewards
            reward = 300.0  # Massive reward for exceptional precision
            if action == 1:  # straight action when exceptionally close
                reward += 500.0  # MASSIVE bonus = 800 total
            else:
                reward -= 100.0  # Penalty for unnecessary turning
                
        elif abs_error <= self.error_threshold:  # 0.03 < error <= 0.05
            # TARGET precision - VERY BIG rewards
            reward = 200.0  # Very big reward for hitting target precision
            if action == 1:  # straight action when at target precision
                reward += 300.0  # VERY BIG bonus = 500 total
            else:
                reward -= 50.0  # Moderate penalty for unnecessary turning
                
        elif abs_error <= 0.08:  # 0.05 < error <= 0.08
            # SMALL deviation - gentle approach encouraged
            if action == 1:  # going straight - good when close
                reward = 80.0  # Good reward for staying straight when close
            elif (self.line_error > 0 and action == 0) or (self.line_error < 0 and action == 2):
                reward = 60.0  # GENTLE reward for small corrections (avoid over-correction)
            else:
                reward = -30.0  # Penalty for wrong direction
                
        elif abs_error <= 0.15:  # 0.08 < error <= 0.15
            # MEDIUM deviation - moderate corrections
            if (self.line_error > 0 and action == 0) or (self.line_error < 0 and action == 2):
                reward = 80.0  # MODERATE reward for medium corrections
            elif action == 1:
                reward = 20.0  # Small reward for straight (not penalized, but correction preferred)
            else:
                reward = -50.0  # Penalty for wrong direction
                
        elif abs_error <= 0.25:  # 0.15 < error <= 0.25
            # LARGE deviation - stronger corrections needed
            if (self.line_error > 0 and action == 0) or (self.line_error < 0 and action == 2):
                reward = 100.0  # GOOD reward for larger corrections
            elif action == 1:
                reward = -20.0  # Mild penalty for ignoring medium-large error
            else:
                reward = -80.0  # Penalty for wrong direction
                
        else:
            # VERY LARGE deviation - correction is critical but still gradual
            if (self.line_error > 0 and action == 0) or (self.line_error < 0 and action == 2):
                reward = 120.0  # STRONG but not extreme reward for critical correction
            else:
                reward = -100.0  # Strong penalty for ignoring critical correction
        
        # Additional penalty for extreme errors
        if abs_error >= self.max_line_error * 0.8:
            reward -= 5.0
            
        # ULTRA-PRECISION: Error trend analysis and smart correction encouragement
        self.error_trend_history.append(abs_error)
        if len(self.error_trend_history) > 10:  # Keep last 10 steps for trend analysis
            self.error_trend_history.pop(0)
            
        # Calculate error trends and correction urgency
        correction_boost = 0.0
        if len(self.error_trend_history) >= 5:
            recent_errors = self.error_trend_history[-5:]
            older_errors = self.error_trend_history[-10:-5] if len(self.error_trend_history) >= 10 else []
            
            # Detect if error is increasing (needs correction)
            if len(older_errors) >= 3:
                recent_avg = sum(recent_errors) / len(recent_errors)
                older_avg = sum(older_errors) / len(older_errors)
                
                if recent_avg > older_avg + 0.02:  # Error is getting worse
                    self.correction_urgency = min(self.correction_urgency + 0.2, 1.0)
                elif recent_avg < older_avg - 0.02:  # Error is improving
                    self.correction_urgency = max(self.correction_urgency - 0.1, 0.0)
                    
            # Smart correction encouragement when urgency is high
            if self.correction_urgency > 0.5:  # High urgency for correction
                if (self.line_error > 0 and action == 0) or (self.line_error < 0 and action == 2):
                    correction_boost = 100.0 * self.correction_urgency  # MASSIVE boost for urgent corrections
                elif action == 1 and abs_error > 0.08:  # Penalize STRAIGHT when urgent correction needed
                    correction_boost = -80.0 * self.correction_urgency  # Penalty for ignoring urgent correction
        
        reward += correction_boost
        
        # OVER-CORRECTION DETECTION AND PREVENTION
        # Track correction direction for gradual approach
        correction_direction = None
        if action == 0:  # LEFT correction
            correction_direction = 'LEFT'
        elif action == 2:  # RIGHT correction  
            correction_direction = 'RIGHT'
        else:  # STRAIGHT
            correction_direction = 'STRAIGHT'
            
        # Add to correction history
        if not hasattr(self, 'correction_history'):
            self.correction_history = []
        self.correction_history.append(correction_direction)
        if len(self.correction_history) > 5:  # Keep last 5 corrections
            self.correction_history.pop(0)
            
        # Check for over-correction patterns (too many corrections in same direction)
        over_correction_penalty = 0.0
        if len(self.correction_history) >= 3:
            recent_corrections = self.correction_history[-3:]
            
            # If last 3 actions were all corrections in same direction, reduce reward
            if all(c == 'LEFT' for c in recent_corrections) and abs_error < 0.2:
                over_correction_penalty = 30.0  # Penalty for excessive LEFT corrections when not critically needed
            elif all(c == 'RIGHT' for c in recent_corrections) and abs_error < 0.2:
                over_correction_penalty = 30.0  # Penalty for excessive RIGHT corrections when not critically needed
                
            # If approaching center but still correcting aggressively
            if abs_error < 0.1:  # Close to center
                if ((self.line_error > 0 and correction_direction == 'LEFT') or 
                    (self.line_error < 0 and correction_direction == 'RIGHT')):
                    # Count recent corrections in this direction
                    same_direction_count = sum(1 for c in recent_corrections[-2:] if c == correction_direction)
                    if same_direction_count >= 2:  # Been correcting same way recently
                        over_correction_penalty += 20.0  # Additional penalty for potential overshoot
        
        reward -= over_correction_penalty
            
        # Stability bonus - reward for small changes in error AND gradual approach
        if hasattr(self, 'prev_error'):
            error_change = abs(abs_error - abs(self.prev_error))
            if error_change < 0.02:  # Very stable behavior (tightened threshold)
                reward += 10.0  # Bigger stability bonus for smooth approach
            elif error_change < 0.05:  # Moderately stable
                reward += 5.0  # Good stability bonus
                
            # Bonus for approaching center (error decreasing)
            if abs_error < abs(self.prev_error) - 0.01:  # Error significantly decreased
                reward += 15.0  # Big bonus for effective gradual approach to center
        
        # SMART PRECISION-FOCUSED anti-oscillation system
        oscillation_penalty = 0.0
        
        # Check if this is a NECESSARY correction (error increasing and correction in right direction)
        necessary_correction = False
        if len(self.error_trend_history) >= 3:
            recent_trend = sum(self.error_trend_history[-3:]) / 3
            if abs_error > 0.1 and recent_trend > abs_error - 0.02:  # Error was increasing
                if (self.line_error > 0.1 and action == 0) or (self.line_error < -0.1 and action == 2):
                    necessary_correction = True
        
        # Only penalize oscillation if it's NOT a necessary correction
        if not necessary_correction:
            # Check for 4-step oscillation patterns
            if len(self.recent_actions) >= 4:
                last_4 = self.recent_actions[-4:]
                if ((last_4[0] == 0 and last_4[1] == 2 and last_4[2] == 0 and last_4[3] == 2) or
                    (last_4[0] == 2 and last_4[1] == 0 and last_4[2] == 2 and last_4[3] == 0)):
                    oscillation_penalty += 300.0  # Heavy penalty for wasteful 4-step oscillation
                    
            # Check for 3-step oscillation patterns
            if len(self.recent_actions) >= 3:
                last_3 = self.recent_actions[-3:]
                if ((last_3[0] == 0 and last_3[1] == 2 and last_3[2] == 0) or
                    (last_3[0] == 2 and last_3[1] == 0 and last_3[2] == 2)):
                    oscillation_penalty += 250.0  # Heavy penalty for wasteful 3-step oscillation
                    
            # Check for 2-step rapid switching when error is small
            if len(self.recent_actions) >= 2 and abs_error < 0.15:
                last_2 = self.recent_actions[-2:]
                if ((last_2[0] == 0 and last_2[1] == 2) or (last_2[0] == 2 and last_2[1] == 0)):
                    oscillation_penalty += 200.0  # Heavy penalty for wasteful direction reversal
                    
        # Always penalize unnecessary micro-adjustments when very close to line
        if abs_error < 0.05 and action != 1 and not necessary_correction:
            oscillation_penalty += 150.0  # Penalty for micro-adjustments when very close
            
        # Apply oscillation penalties
        reward -= oscillation_penalty
        
        # ULTRA-PRECISION bonus system - adaptive thresholds based on recent performance
        recent_performance = sum(self.error_trend_history[-5:]) / 5 if len(self.error_trend_history) >= 5 else abs_error
        adaptive_threshold = max(0.03, min(0.08, recent_performance))  # Adaptive precision requirement
        
        # BALANCED bonus for STRAIGHT movement - attractive but not overwhelming
        if abs_error < 0.02 and action == 1:  # ULTRA-PRECISE straight movement
            reward += 500.0  # BIG bonus for ultra-precise STRAIGHT
        elif abs_error < 0.03 and action == 1:  # EXCEPTIONAL straight movement  
            reward += 400.0  # Good bonus for exceptional STRAIGHT precision
        elif abs_error < 0.05 and action == 1:  # TARGET straight movement
            reward += 300.0  # Good bonus for target STRAIGHT precision
        elif abs_error < 0.08 and action == 1:  # GOOD straight movement
            reward += 200.0  # Moderate bonus for good STRAIGHT precision
        elif abs_error < 0.12 and action == 1:  # DECENT straight movement
            reward += 150.0  # Small bonus for decent STRAIGHT precision
        elif abs_error < 0.15 and action == 1:  # BASIC straight movement
            reward += 100.0  # Basic bonus for STRAIGHT precision
            
        # Moderate bonus for consecutive STRAIGHT actions - balance stability with precision needs
        if len(self.recent_actions) >= 2 and abs_error < 0.08:
            if all(a == 1 for a in self.recent_actions[-2:]):  # Last 2 actions were STRAIGHT
                reward += 100.0  # Moderate bonus for sustained STRAIGHT movement
        if len(self.recent_actions) >= 3 and abs_error < 0.05:
            if all(a == 1 for a in self.recent_actions[-3:]):  # Last 3 actions were STRAIGHT  
                reward += 200.0  # Good bonus for sustained precise STRAIGHT
        if len(self.recent_actions) >= 4 and abs_error < 0.03:
            if all(a == 1 for a in self.recent_actions[-4:]):  # Last 4 actions were ultra-precise STRAIGHT
                reward += 300.0  # Big bonus for long ultra-precise STRAIGHT sequences
        
        # HUGE bonus for consecutive STRAIGHT actions ONLY when close to line
        if len(self.recent_actions) >= 5 and abs_error < 0.15:  # Only when very close to line
            if all(a == 1 for a in self.recent_actions[-5:]):  # Last 5 actions were STRAIGHT
                reward += 5.0  # Bonus for long straight sequences when on line
        elif len(self.recent_actions) >= 3 and abs_error < 0.1:  # Only when very close to line  
            if all(a == 1 for a in self.recent_actions[-3:]):  # Last 3 actions were STRAIGHT
                reward += 2.0
        
        # Drift detection and penalty
        self.error_history.append(self.line_error)
        if len(self.error_history) > 15:  # Keep last 15 steps
            self.error_history.pop(0)
            
        # STRONG penalties for drift - clear discouragement but learnable
        if len(self.error_history) >= 8:  # Check for sustained patterns
            recent_errors = self.error_history[-8:]
            
            # Strong penalties for sustained drift
            if all(e > 0.25 for e in recent_errors):  # Sustained right drift
                reward -= 40.0  # Strong penalty - robot should correct drift
            elif all(e < -0.25 for e in recent_errors):  # Sustained left drift  
                reward -= 40.0  # Strong penalty - robot should correct drift
                
            # Moderate penalties for moderate drift
            elif all(e > 0.12 for e in recent_errors):  # Moderate right drift
                reward -= 20.0  # Moderate penalty - drift should be corrected
            elif all(e < -0.12 for e in recent_errors):  # Moderate left drift
                reward -= 20.0  # Moderate penalty - drift should be corrected
                
            # Additional check for quick severe drift (5 steps)
            if len(self.error_history) >= 5:
                short_errors = self.error_history[-5:]
                if all(e > 0.4 for e in short_errors):  # Quick severe right drift
                    reward -= 60.0  # Heavy penalty for fast severe drift
                elif all(e < -0.4 for e in short_errors):  # Quick severe left drift
                    reward -= 60.0  # Heavy penalty for fast severe drift
        
        self.prev_error = abs_error
        return reward
    
    def _is_done(self) -> bool:
        """Check if episode should end."""
        # Episode ends if robot goes too far off line or max steps reached
        return (abs(self.line_error) >= self.max_line_error or 
                self.steps >= self.max_steps)
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Render the environment."""
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(10, 6))
            plt.ion()
        
        self.ax.clear()
        
        # Draw the line (centered at y=0)
        line_x = np.linspace(-5, 5, 100)
        line_y = np.zeros_like(line_x)
        self.ax.plot(line_x, line_y, 'k-', linewidth=5, label='Line to follow')
        
        # Draw the robot position
        robot_x = 0  # Robot is always at x=0 in this view
        robot_y = self.line_error
        
        # Robot represented as a small rectangle with orientation
        robot_width = 0.3
        robot_length = 0.5
        
        # Create robot shape
        robot_corners_x = np.array([-robot_length/2, robot_length/2, robot_length/2, -robot_length/2, -robot_length/2])
        robot_corners_y = np.array([-robot_width/2, -robot_width/2, robot_width/2, robot_width/2, -robot_width/2])
        
        # Rotate robot based on angle
        cos_angle = np.cos(self.robot_angle)
        sin_angle = np.sin(self.robot_angle)
        
        rotated_x = robot_corners_x * cos_angle - robot_corners_y * sin_angle + robot_x
        rotated_y = robot_corners_x * sin_angle + robot_corners_y * cos_angle + robot_y
        
        self.ax.plot(rotated_x, rotated_y, 'b-', linewidth=2)
        self.ax.fill(rotated_x, rotated_y, 'blue', alpha=0.7, label='Robot')
        
        # Add arrow to show robot direction
        arrow_length = 0.4
        arrow_x = robot_x + arrow_length * cos_angle
        arrow_y = robot_y + arrow_length * sin_angle
        self.ax.arrow(robot_x, robot_y, arrow_x - robot_x, arrow_y - robot_y, 
                     head_width=0.1, head_length=0.1, fc='red', ec='red')
        
        # Set plot properties
        self.ax.set_xlim(-3, 3)
        self.ax.set_ylim(-self.max_line_error * 1.2, self.max_line_error * 1.2)
        self.ax.set_xlabel('Forward Direction')
        self.ax.set_ylabel('Line Error (Left - / Right +)')
        self.ax.set_title(f'Line Following Robot - Error: {self.line_error:.3f}, Steps: {self.steps}')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        
        if mode == 'human':
            plt.draw()
            plt.pause(0.01)
            return None
        elif mode == 'rgb_array':
            self.fig.canvas.draw()
            buf = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            return buf
    
    def close(self):
        """Clean up rendering resources."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
