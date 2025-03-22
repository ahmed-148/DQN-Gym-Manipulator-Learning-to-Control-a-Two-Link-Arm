# DQN-Gym-Manipulator-Learning-to-Control-a-Two-Link-Arm
Overview

This project implements a Deep Q-Network (DQN) for controlling a Two-Link Manipulator in a custom reinforcement learning environment (TwoLink_v0). The goal is to train the agent to move the manipulator towards a predefined target position efficiently.

Features

Custom Reinforcement Learning Environment: TwoLinkManipulatorEnv()

Deep Q-Network (DQN) Implementation: Uses a feedforward neural network to approximate Q-values.

Experience Replay: Stores past experiences to improve learning stability.

Epsilon-Greedy Policy: Balances exploration and exploitation.

Target Network: Stabilizes training by reducing variance in Q-value updates.

Performance Tracking: Logs rewards and success rates for analysis.

Requirements

Ensure you have the following dependencies installed:

pip install tensorflow numpy matplotlib

Environment Setup

The custom environment TwoLinkManipulatorEnv is imported from TwoLink_v0. It provides:

State Space: Continuous states representing the robot's joint positions and velocities.

Action Space: Discrete actions for joint movements.

Reward System: Encourages reaching the target while minimizing unnecessary movement.

Training Process

Initialize the DQN model with a 2-layer feedforward neural network.

Train for 200 episodes, updating Q-values using the Bellman equation.

Store and sample experiences from a replay buffer (deque(maxlen=10000)).

Update the target network every 5 episodes for stability.

Track performance with reward plots and success metrics.

Model Architecture

The DQN model is implemented using TensorFlow/Keras:

def build_model():
    model = Sequential([
        Dense(24, activation='relu', input_shape=(state_size,)),
        Dense(24, activation='relu'),
        Dense(action_size, activation='linear')  # Output Q-values for each action
    ])
    model.compile(optimizer=Adam(), loss='mse')
    return model

Results

Training performance is logged per episode, displaying total reward and steps taken.

After training, the model is tested for 10 episodes, measuring success rate and average reward.

Visualization

The training process is analyzed using two plots:

Success Rate per Episode: Tracks whether the agent successfully reaches the goal.

Total Reward per Episode: Shows the cumulative reward received in each episode.

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(success, marker='o', linestyle='-', color='green', label="Success")
plt.subplot(2, 1, 2)
plt.plot(Reward, marker='o', linestyle='-', color='blue', label="Total Reward")
plt.show()

Future Improvements

Implement Double DQN (DDQN) to reduce overestimation bias.

Use a Prioritized Experience Replay (PER) to prioritize valuable transitions.

Train with Continuous Action Spaces using DDPG instead of discrete actions.

Author

Developed by Ahmed Foaud as part of a Reinforcement Learning project for robotic control.
