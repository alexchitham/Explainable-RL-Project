import gymnasium

import highway_env
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor


# Function to create the gym environment using a recommended config
def create_gym_env():

    env_config = {
        "observation": {
            "type": "GrayscaleObservation",
            "observation_shape": (128, 64),
            "stack_size": 4,
            "weights": [0.2989, 0.5870, 0.1140],  # Weights for RGB conversion
            "scaling": 1.75,
        },
    }

    # Create the gym environment
    env = gymnasium.make('highway-fast-v0', render_mode='rgb_array', config=env_config)

    # The monitor wrapper allows important metrics to be tracked during training
    env = Monitor(env)
    env.reset()
    
    return env


# Function to create the SB3 vector environment from the gym environment
def create_vec_env():

    vec_env = DummyVecEnv([create_gym_env])
    vec_env.reset()
    return vec_env


# Load the trained agent from the zip file
def load_dqn_agent(filename):

    agent = DQN.load(filename)
    return agent


if __name__ == "__main__":

    # Create the environment
    vec_env = create_vec_env()
    obs = vec_env.reset()

    # Create the agent using a recommended config
    agent = DQN("CnnPolicy", vec_env, 
        learning_rate=5e-4,
        buffer_size=15000,
        learning_starts=200,
        batch_size=32,
        gamma=0.8,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=50,
        exploration_fraction=0.7,
        verbose=1,
        tensorboard_log="tensorboard"
    )

    # Train the agent on the environment
    agent.learn(total_timesteps=int(1e3), log_interval=100, progress_bar=True)

    # Save the trained agent to a zip file
    agent.save("dqn_highway")

