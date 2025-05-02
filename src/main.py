
from collections import deque
from agent_training import load_dqn_agent, create_vec_env
import numpy as np
from utils import get_q_values, plot_loss_graph
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



def create_sarfa_saliency_map(agent_obs, rendered_states, q_values, agent, blur_sigma=5.0, sigma_sqr=25.0):

    # Remove the batch dimension before passing through to the saliency map object
    batchless_state = np.squeeze(agent_obs)

    # Create a saliency map object
    saliency_map = sarfa_saliency_map(batchless_state, rendered_states, q_values, blur_sigma, sigma_sqr)

    # Calculate the saliency for each pixel
    saliency_map.calculate_saliency(agent)

    return saliency_map


def create_sverl_map(agent_obs, rendered_states, q_values, shapley_func_weights_dir):

    # Remove the batch dimension before passing through to the saliency map object
    batchless_state = np.squeeze(agent_obs)

    # Create the SVERL map object
    sverl_object = sverl_saliency_map(batchless_state, rendered_states, q_values, shapley_func_weights_dir)

    # Use the object to make predictions
    sverl_object.predict_shapley_values()

    return sverl_object



def main_xrl_loop(agent_directory, images_directory):

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

            # if step_count == 10:
            if len(recent_renders) >= 4:

            # ------------------------------------------------ SARFA TESTS ---------------------------------------------------------------#

                # BASELINE / DEFAULT CONFIGURATION
                sarfa_map_obj_base = create_sarfa_saliency_map(state, recent_renders, q_values, agent, 5.0, 25.0)
                sarfa_map_obj_base.upscale_map(0.01)
                sarfa_map_obj_base.create_overlay_img()
                sarfa_map_obj_base.save_heatmaps(images_directory + "/sarfa_heatmaps", str(ep) + "_" + str(step_count), version="base-sarfa")
                # sarfa_map_obj_base.save_intermediate_blurs(images_directory + "/sarfa_heatmaps", str(ep) + "_" + str(step_count), version=1)

                # EXPERIMENT 1: GAUSSIAN STRENTH ABLATION TEST
                sarfa_map_obj_v1 = create_sarfa_saliency_map(state, recent_renders, q_values, agent, 8.0, 25.0)
                sarfa_map_obj_v1.upscale_map(0.01)
                sarfa_map_obj_v1.create_overlay_img()
                sarfa_map_obj_v1.save_heatmaps(images_directory + "/sarfa_heatmaps", str(ep) + "_" + str(step_count), version=1)
                # sarfa_map_obj_v1.save_intermediate_blurs(images_directory + "/sarfa_heatmaps", str(ep) + "_" + str(step_count), version=2)

                # EXPERIMENT 2: GAUSSIAN SPREAD ABLATION TEST
                sarfa_map_obj_v2 = create_sarfa_saliency_map(state, recent_renders, q_values, agent, 5.0, 49.0)
                sarfa_map_obj_v2.upscale_map(0.01)
                sarfa_map_obj_v2.create_overlay_img()
                sarfa_map_obj_v2.save_heatmaps(images_directory + "/sarfa_heatmaps", str(ep) + "_" + str(step_count), version=2)
                # sarfa_map_obj_v3.save_intermediate_blurs(images_directory + "/sarfa_heatmaps", str(ep) + "_" + str(step_count), version=3)


            # ------------------------------------------------ SVERL TESTS ---------------------------------------------------------------#

                # BASELINE / DEFAULT CONFIGURATION
                shapley_weights_directory = "highway-env/sverl/shapley_func_weights/step100k_char-v1_32_v1"
                sverl_map_obj_base = create_sverl_map(state, recent_renders, q_values, shapley_weights_directory)
                sverl_map_obj_base.upscale_map(0.01)
                sverl_map_obj_base.create_overlay_img()
                sverl_map_obj_base.save_heatmaps(images_directory + "/sverl_heatmaps", str(ep) + "_" + str(step_count), version="base-sverl")

                # EXPERIMENT 3: SHAPLEY FUNCTION BATCH SIZE TEST
                shapley_weights_directory = "highway-env/sverl/shapley_func_weights/step100k_char-v1_64_v3"
                sverl_map_obj_v3 = create_sverl_map(state, recent_renders, q_values, shapley_weights_directory)
                sverl_map_obj_v3.upscale_map(0.01)
                sverl_map_obj_v3.create_overlay_img()
                sverl_map_obj_v3.save_heatmaps(images_directory + "/sverl_heatmaps", str(ep) + "_" + str(step_count), version=3)

                # EXPERIMENT 4: MASK VALUE TEST
                shapley_weights_directory = "highway-env/sverl/shapley_func_weights/step100k_char-v3_32_v4"
                sverl_map_obj_v4 = create_sverl_map(state, recent_renders, q_values, shapley_weights_directory)
                sverl_map_obj_v4.upscale_map(0.01)
                sverl_map_obj_v4.create_overlay_img()
                sverl_map_obj_v4.save_heatmaps(images_directory + "/sverl_heatmaps", str(ep) + "_" + str(step_count), version=4)

                # EXPERIMENT 4: ACTION PROBABILITIES TEST
                shapley_weights_directory = "highway-env/sverl/shapley_func_weights/step100k_char-v2_32_v2"
                sverl_map_obj_v5 = create_sverl_map(state, recent_renders, q_values, shapley_weights_directory)
                sverl_map_obj_v5.upscale_map(0.01)
                sverl_map_obj_v5.create_overlay_img()
                sverl_map_obj_v5.save_heatmaps(images_directory + "/sverl_heatmaps", str(ep) + "_" + str(step_count), version=5)


            # ------------------------------------------------ JOINT TESTS ---------------------------------------------------------------#

                # EXPERIMENT 6: THRESHOLD FOR HIGHLIGHTING LOW TEST SARFA
                sarfa_map_obj_base.upscale_map(0.008)
                sarfa_map_obj_base.create_overlay_img()
                sarfa_map_obj_base.save_heatmaps(images_directory + "/sarfa_heatmaps", str(ep) + "_" + str(step_count), version="6_low")

                # EXPERIMENT 6: THRESHOLD FOR HIGHLIGHTING HIGH TEST SARFA
                sarfa_map_obj_base.upscale_map(0.015)
                sarfa_map_obj_base.create_overlay_img()
                sarfa_map_obj_base.save_heatmaps(images_directory + "/sarfa_heatmaps", str(ep) + "_" + str(step_count), version="6_high")

                # EXPERIMENT 6: THRESHOLD FOR HIGHLIGHTING LOW TEST SVERL
                sverl_map_obj_base.upscale_map(0.008)
                sverl_map_obj_base.create_overlay_img()
                sverl_map_obj_base.save_heatmaps(images_directory + "/sverl_heatmaps", str(ep) + "_" + str(step_count), version="6_low")
                
                # EXPERIMENT 6: THRESHOLD FOR HIGHLIGHTING HIGH TEST SVERL
                sverl_map_obj_base.upscale_map(0.015)
                sverl_map_obj_base.create_overlay_img()
                sverl_map_obj_base.save_heatmaps(images_directory + "/sverl_heatmaps", str(ep) + "_" + str(step_count), version="6_high")


            # Take the action in the environment
            state, reward, done, info = env.step(best_action)
            step_count += 1


    env.close()



if __name__ == "__main__":

    agent_directory = "highway-env/agents/dqn_highway_norm_200K.zip"
    images_directory = "highway-env/images"
    shapley_weights_directory = "highway-env/sverl/shapley_func_weights/step100k_char-v3_32_v4"

    # test_env_loop(agent_directory)
    main_xrl_loop(agent_directory, images_directory)
    # plot_loss_graph()