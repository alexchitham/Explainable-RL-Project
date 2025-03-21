
import cv2
import numpy as np
from PIL import Image
from agent_training import load_dqn_agent, create_vec_env
from xrl_method import xrl_method
from collections import deque


class sarfa_saliency_map(xrl_method):

    def __init__(self, agent_obs, rendered_states, q_values):

        super().__init__(agent_obs, rendered_states, q_values)        

        # Calculate probability used to find specific changes
        self.exp_q_values = np.exp(self.q_values)
        self.prob_s_a = self.exp_q_values[self.best_action] / np.sum(self.exp_q_values)

        # Calculate p_rem used to find relevant changes
        self.exp_q_values_no_best = np.delete(self.exp_q_values, self.best_action)
        self.min_value = 1e-10
        self.p_rem = self.exp_q_values_no_best / np.sum(self.exp_q_values_no_best)
        self.p_rem = np.clip(self.p_rem, self.min_value, 1)

        # Blur the whole image
        self.blur_sigma = 5.0
        self.blurred_whole_state = np.array([cv2.GaussianBlur(self.agent_obs[i], (0, 0), self.blur_sigma) for i in range(self.num_frames)])
        self.normalised_blurred_state = np.copy(self.blurred_whole_state)
        self.normalised_blurred_state = self.normalised_blurred_state.astype(np.float32) / 255.0
    

    def calculate_delta_p(self, exp_blurred_q_values):
        
        blurred_prob = exp_blurred_q_values[self.best_action] / np.sum(exp_blurred_q_values)

        delta_p = self.prob_s_a - blurred_prob
        return delta_p
        


    def calculate_kl_divergence(self, exp_blurred_q_values):
        
        exp_blurred_q_values_no_best = np.delete(exp_blurred_q_values, self.best_action)
        blurred_p_rem = exp_blurred_q_values_no_best / np.sum(exp_blurred_q_values_no_best)
        blurred_p_rem = np.clip(blurred_p_rem, self.min_value, 1)
        
        kl_divergence = np.sum(blurred_p_rem * np.log(blurred_p_rem / self.p_rem))
        return kl_divergence
    

    # Function to calculate the saliency for a pixel given the perturbed q-values 
    def calculate_feature_saliency(self, blurred_q_values):

        exp_blurred_q_values = np.exp(blurred_q_values)
        
        # Find delta_p which identifies specific changes
        delta_p = self.calculate_delta_p(exp_blurred_q_values)

        # Find kl_divergence which identifies relevant changes
        kl_divergence = self.calculate_kl_divergence(exp_blurred_q_values)

        k = 1 / (1 + kl_divergence)

        saliency = (2 * k * delta_p) / (k + delta_p)

        return saliency
    

    # Function to apply a blur around a pixel in the agent's observation
    def perturb_state(self, frame, pixel_row, pixel_col):

        # Define constants for use in the Gaussian mask
        sigma_sqr = 25.0
        denom_gaus = 2 * np.pi * sigma_sqr

        # Create the Gaussian mask using numpy broadcasting
        row, col = np.indices(self.map[0].shape)
        x_gaus_sqr = (row - pixel_row) ** 2
        y_gaus_sqr = (col - pixel_col) ** 2
        exponent_gaus = np.exp(-1 * ((x_gaus_sqr + y_gaus_sqr) / (2 * sigma_sqr)))
        
        # Set a minimum value in the mask and normalise to [0,1]
        mask = np.clip(exponent_gaus / denom_gaus, 1e-10, 1)
        mask = mask / np.max(mask)

        # Apply the localised blur to one of the observation frame
        perturbed_image = np.copy(self.normalised_agent_obs)
        perturbed_image[frame] = (self.normalised_agent_obs[frame] * (1 - mask)) + (self.normalised_blurred_state[frame] * mask)

        # Scale back up to [0,255]
        perturbed_image = np.clip(perturbed_image, 0, 1)
        perturbed_image = (perturbed_image * 255).astype(np.uint8)
        
        # Save an example of a perturbed image for each frame
        if (pixel_row == 64 and pixel_col == 32):
            self.example_blurs.append(np.copy(perturbed_image[frame]))
            
        # Add a batch dimension to match dimensions of q-network input
        batched_perturbed_img = np.expand_dims(perturbed_image, axis=0)

        return batched_perturbed_img


    def calculate_saliency(self, agent):

        # Could use CuPy to leverage GPUs to make this more efficient
        # Loop over every pixel in each frame of the observation
        for frame in range(self.map.shape[0]):
            for row in range(self.map.shape[1]):
                for col in range(self.map.shape[2]):

                    # Perturb / blur part of the agent's observation
                    perturbed_state = self.perturb_state(frame, row, col)

                    # Find the new q-values using the blurred state
                    blurred_q_values = get_q_values(agent, perturbed_state)

                    # Calculate the saliency and insert into the map
                    self.map[frame, row, col] = self.calculate_feature_saliency(blurred_q_values)


    
def create_saliency_map(agent_obs, rendered_states, q_values, agent):

    # Remove the batch dimension before passing through to the saliency map object
    batchless_state = np.squeeze(agent_obs)

    # Create a saliency map object
    saliency_map = sarfa_saliency_map(batchless_state, rendered_states, q_values)

    # Calculate the saliency for each pixel
    saliency_map.calculate_saliency(agent)

    # Upscale the saliency map to match the rendered image
    saliency_map.upscale_map()

    return saliency_map


def get_q_values(agent, state):

    # Convert the state to a PyTorch tensor
    obs_tensor, _ = agent.policy.obs_to_tensor(state)

    # Pass the observation through the Q-Network
    q_values = (agent.policy.q_net(obs_tensor))

    # Remove from backprop -> move to cpu -> convert to numpy -> remove batch dimension
    q_values = q_values.detach().cpu().numpy().squeeze()

    return q_values    



if __name__ == "__main__":

    agent = load_dqn_agent("dqn_highway_norm_500K.zip")
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
                saliency_map_obj = create_saliency_map(state, recent_renders, q_values, agent)
                saliency_map_obj.create_overlay_img(step_count)
                saliency_map_obj.save_heatmaps("sarfa_heatmaps", str(ep) + "_" + str(step_count))

            # if step_count == 11:
            #     saliency_map_obj = create_saliency_map(state, frame, q_values, agent)
            #     saliency_map_obj.create_overlay_img(step_count)
            #     saliency_map_obj.save_all_images("test_save_images", 2)

            # if len(recent_renders) >= 4: 

            # Take the action in the environment
            state, reward, done, info = env.step(best_action)
            step_count += 1


    env.close()

