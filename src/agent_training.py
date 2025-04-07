import gymnasium
from enum import Enum
import torch
import numpy as np
import highway_env
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_schedule_fn


# Enum containing the actions and their corresponding index
class discrete_meta_actions(Enum):
    LANE_LEFT = 0
    IDLE = 1
    LANE_RIGHT = 2
    FASTER = 3
    SLOWER = 4

# Function to retrieve the name of the action from its index
def get_action_name(index):
    return discrete_meta_actions(index).name


# Function to retrieve the number of actions
def get_action_size():
    return len(discrete_meta_actions)


# Function to create the gym environment using a recommended config
def create_gym_env():

    render_scaling = 14
    obs_scaling = 1.75

    env_config = {
        "observation": {
            "type": "GrayscaleObservation",
            "observation_shape": (128, 64),
            "stack_size": 4,
            "weights": [0.2989, 0.5870, 0.1140],  # Weights for RGB conversion
            "scaling": obs_scaling,
        },
        "screen_width": 128 * (render_scaling / obs_scaling),
        "screen_height": 64 * (render_scaling / obs_scaling),
        "scaling" : render_scaling
    }

    # Create the gym environment
    env = gymnasium.make('highway-v0', render_mode='rgb_array', config=env_config)

    # The monitor wrapper allows important metrics to be tracked during training
    env = Monitor(env)
    env.reset()
    
    return env


# Function to retrieve the state size
def get_env_size():
    env = create_vec_env()
    state = env.reset()
    batchless_state = np.squeeze(state)
    return batchless_state.shape


# Function to create the SB3 vector environment from the gym environment
def create_vec_env():

    vec_env = DummyVecEnv([create_gym_env])
    vec_env.reset()
    return vec_env


# Load the trained agent from the zip file
def load_dqn_agent(filename, device="cpu"):

    agent = DQN.load(filename, device=device, custom_objects={"buffer_size": 1, "lr_schedule": get_schedule_fn(2.5e-4),
    "exploration_schedule": get_schedule_fn(0.05)})
    return agent


# Function to generate a steady state distribution through agent experience
def create_steady_state_dis(num_states = 100000):

    agent = load_dqn_agent("dqn_highway_norm_200K.zip")
    env = create_vec_env()
    state = env.reset()

    # Create a blank array to store the states
    steady_states = np.empty((num_states, state.shape[-3], state.shape[-2], state.shape[-1]), dtype=np.uint8)

    # Run episodes until we have the required number of states
    state_count = 0
    while state_count < num_states:

        # Reset for new episode
        done = False
        state = env.reset()
        step_counter = 0

        # Run until the episode is finished
        while not done:

            if state_count == num_states:
                break

            step_counter += 1

            # Extract the current state by removing the batch
            batchless_state = np.squeeze(state)

            # Save the current state
            if step_counter >= 4:
                steady_states[state_count] = batchless_state
                state_count += 1

            # Predict the most optimal action
            best_action, _ = agent.predict(state, deterministic=True)

            # Take the action in the environment
            state, _, done, _ = env.step(best_action)
            
            if state_count % 10000 == 0:
                print("Number of states saved: ", state_count, flush=True)

    env.close()

    # Save a compressed version of the states
    states_tensor = torch.tensor(steady_states, dtype=torch.uint8)
    file_name = "steady_states" + str(num_states) + ".pt"
    torch.save(states_tensor, file_name)

    return file_name


def agent_training_loop(): 

    # Create the environment
    vec_env = create_vec_env()
    obs = vec_env.reset()

    # Create the agent using a recommended config
    agent = DQN(
        "CnnPolicy",
        vec_env,
        learning_rate=2.5e-4,
        buffer_size=5000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.99,
        train_freq=(1, "step"),
        gradient_steps=2,
        target_update_interval=5000,
        exploration_fraction=0.3,
        verbose=1,
        tensorboard_log="tensorboard"
    )

    # Train the agent on the environment
    agent.learn(total_timesteps=int(1e5), log_interval=100, progress_bar=True)

    # Save the trained agent to a zip file
    agent.save("dqn_highway")

    vec_env.close()


if __name__ == "__main__":

    agent_training_loop()
    # filename = create_steady_state_dis(num_states=100000)

