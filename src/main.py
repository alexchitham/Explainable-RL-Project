
from collections import deque
from agent_training import load_dqn_agent, create_vec_env
import numpy as np
from utils import get_q_values
from sarfa import sarfa_saliency_map
from sverl import sverl_saliency_map


def test_env_loop(agent_directory):

    agent = load_dqn_agent(agent_directory)
    vec_env = create_vec_env()

        # Run a few episodes
    for _ in range(3):
        done = False
        obs = vec_env.reset()
        while not done:
            # Predict the action
            action, _states = agent.predict(obs, deterministic=True)
            # Take the action in the environment
            obs, reward, done, info = vec_env.step(action)

            # print(obs.shape)

            vec_env.render()
            # print(type(frame))
            # print(frame.shape)



def create_sarfa_saliency_map(agent_obs, rendered_states, q_values, agent):

    # Remove the batch dimension before passing through to the saliency map object
    batchless_state = np.squeeze(agent_obs)

    # Create a saliency map object
    saliency_map = sarfa_saliency_map(batchless_state, rendered_states, q_values)

    # Calculate the saliency for each pixel
    saliency_map.calculate_saliency(agent)

    # Upscale the saliency map to match the rendered image
    saliency_map.upscale_map()

    return saliency_map


def create_sverl_map(agent_obs, rendered_states, q_values, shapley_func_weights_dir):

    # Remove the batch dimension before passing through to the saliency map object
    batchless_state = np.squeeze(agent_obs)

    # Create the SVERL map object
    sverl_object = sverl_saliency_map(batchless_state, rendered_states, q_values, shapley_func_weights_dir)

    # Use the object to make predictions
    sverl_object.predict_shapley_values()

    # Upscale the generated map
    sverl_object.upscale_map()

    return sverl_object



def main_xrl_loop(agent_directory, images_directory, shapley_weights_directory):

    agent = load_dqn_agent(agent_directory)
    env = create_vec_env()

    # Run SARFA over a number of episodes
    for ep in range(1):

        # Reset for new episode
        done = False
        state = env.reset()
        step_count = 0
        recent_renders = deque(maxlen=4)

        # Run until the episode is finished
        while not done:

            # Predict the most optimal action
            best_action, _ = agent.predict(state, deterministic=True)

            # Extract the q-values from the network
            q_values = get_q_values(agent, state)

            # Create saliency map for the current state
            frame = env.get_images()[0]
            # frame = env.render(mode="rgb_array")

            # Store the most recent frames
            recent_renders.append(frame)

            if step_count == 10:
                sarfa_map_obj = create_sarfa_saliency_map(state, recent_renders, q_values, agent)
                sarfa_map_obj.create_overlay_img()
                sarfa_map_obj.save_heatmaps(images_directory + "/sarfa_heatmaps", str(ep) + "_" + str(step_count))

                sverl_map_obj = create_sverl_map(state, recent_renders, q_values, shapley_weights_directory)
                sverl_map_obj.create_overlay_img()
                sverl_map_obj.save_heatmaps(images_directory + "/sverl_heatmaps", str(ep) + "_" + str(step_count))

            # if len(recent_renders) >= 4: 

            # Take the action in the environment
            state, reward, done, info = env.step(best_action)
            step_count += 1


    env.close()



if __name__ == "__main__":

    agent_directory = "highway-env/agents/dqn_highway_norm_200K.zip"
    images_directory = "highway-env/images"
    shapley_weights_directory = "highway-env/sverl/shapley_func_weights/step100000_1"

    # test_env_loop(agent_directory)
    main_xrl_loop(agent_directory, images_directory, shapley_weights_directory)