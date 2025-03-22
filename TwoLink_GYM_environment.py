import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TwoLinkManipulatorEnv(gym.Env):
    """
    Custom Environment for simulating a two-link manipulator with discrete actions
    using Gymnasium API.
    """

    def __init__(self, render_on=False):
        super(TwoLinkManipulatorEnv, self).__init__()

        # Action space: 9 discrete actions for joint angle control
        self.action_space = spaces.Discrete(9)

        # Store the render_on flag
        self.render_on = render_on

        # Observation space: [theta1, theta2, x, y, distance]
        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -np.pi, -2.0, -2.0, 0.0]),
            high=np.array([np.pi, np.pi, 2.0, 2.0, np.inf]),
            dtype=np.float32
        )

        # Step size for joint angles
        self.theta_step = np.pi / 18  # 10 degrees increment

        # Actions mapping: List of joint changes corresponding to action space
        self.actions = [
            [self.theta_step, self.theta_step],    # [+theta1, +theta2]
            [-self.theta_step, self.theta_step],   # [-theta1, +theta2]
            [-self.theta_step, -self.theta_step],  # [-theta1, -theta2]
            [self.theta_step, -self.theta_step],   # [+theta1, -theta2]
            [0, self.theta_step],                 # [0, +theta2]
            [0, -self.theta_step],                # [0, -theta2]
            [self.theta_step, 0],                 # [+theta1, 0]
            [-self.theta_step, 0],                # [-theta1, 0]
            [0, 0]                               # [0, 0]
        ]

        # Initialize state: [theta1, theta2, x, y, distance]
        self.state = np.array([0.0, 0.0, 2.0, 0.0, 2.112])  # Joint angles
        self.goal = np.array([0.8164, 1.7509])  # Predefined goal in x-y space
        self.max_steps = 100  # Limit episode length
        self.current_step = 0

    def reset(self, seed=None):
        """Resets the environment to its initial state."""
        super().reset(seed=seed)
        theta1, theta2 = 0.0, 0.0
        x = np.cos(theta1) + np.cos(theta1 + theta2)
        y = np.sin(theta1) + np.sin(theta1 + theta2)
        distance_to_goal = np.linalg.norm(np.array([x, y]) - self.goal)
        self.state = np.array([theta1, theta2, x, y, distance_to_goal], dtype=np.float32)
        self.current_step = 0
        return self.state, {}

    def step(self, action):
        """Performs one step in the environment based on the action."""
        assert self.action_space.contains(action), f"Invalid action: {action}"
        
        # Retrieve joint angle increments for the given action
        delta_theta1, delta_theta2 = self.actions[action]
        self.state[0] = np.clip(self.state[0] + delta_theta1, -np.pi, np.pi)
        self.state[1] = np.clip(self.state[1] + delta_theta2, -np.pi, np.pi)

        # Increment step count
        self.current_step += 1

        # Calculate the end-effector position and distance to goal
        x, y = self.calculate_end_effector_position()
        distance_to_goal = np.linalg.norm(np.array([x, y]) - self.goal)
        self.state[2:5] = [x, y, distance_to_goal]

        # Reward based on distance to goal
        reward = -distance_to_goal

        # Check if the goal is reached
        if distance_to_goal < 0.1:
            reward += 100

        # Termination condition: reaching goal or max steps
        done = distance_to_goal < 0.1 or self.current_step >= self.max_steps
        self.done_flag = done  # Store the termination flag
        return self.state, reward, done, False, {}

    def calculate_end_effector_position(self):
        """Calculates the end-effector position based on joint angles."""
        x = np.cos(self.state[0]) + np.cos(self.state[0] + self.state[1])
        y = np.sin(self.state[0]) + np.sin(self.state[0] + self.state[1])
        return x, y

    def render(self, mode='human'):
        """Renders the current state of the environment, only when the episode is done."""
        if hasattr(self, 'done_flag') and self.done_flag:
            x, y = self.calculate_end_effector_position()
            distance_to_goal = np.linalg.norm(np.array([x, y]) - self.goal)
            print(
                f"Episode Finished:\n"
                f"Joint Angles: theta1 = {self.state[0]:.2f}, theta2 = {self.state[1]:.2f}, "
                f"End-Effector: x = {x:.2f}, y = {y:.2f}, Distance to Goal = {distance_to_goal:.2f}"
            )

    def close(self):
        """Cleans up resources, if any."""
        pass
