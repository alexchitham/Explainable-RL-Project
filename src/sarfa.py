
import cv2
import numpy as np
from xrl_method import xrl_method
from utils import get_q_values


class sarfa_saliency_map(xrl_method):

    def __init__(self, agent_obs, rendered_states, q_values, blur_sigma, sigma_sqr):

        super().__init__(agent_obs, rendered_states, q_values)

        self.example_blurs = []    

        # Calculate probability used to find specific changes
        self.exp_q_values = np.exp(self.q_values)
        self.prob_s_a = self.exp_q_values[self.best_action] / np.sum(self.exp_q_values)

        # Calculate p_rem used to find relevant changes
        self.exp_q_values_no_best = np.delete(self.exp_q_values, self.best_action)
        self.min_value = 1e-10
        self.p_rem = self.exp_q_values_no_best / np.sum(self.exp_q_values_no_best)
        self.p_rem = np.clip(self.p_rem, self.min_value, 1)

        # Blur the whole image
        self.sigma_sqr = sigma_sqr
        self.blur_sigma = blur_sigma
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
        denom_gaus = 2 * np.pi * self.sigma_sqr

        # Create the Gaussian mask using numpy broadcasting
        row, col = np.indices(self.map[0].shape)
        x_gaus_sqr = (row - pixel_row) ** 2
        y_gaus_sqr = (col - pixel_col) ** 2
        exponent_gaus = np.exp(-1 * ((x_gaus_sqr + y_gaus_sqr) / (2 * self.sigma_sqr)))
        
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


    def save_intermediate_blurs(self, directory, suffix, version=""):

        new_path = directory + "/set" + str(suffix) + "_v" + str(version)

        # Save full blurred states
        self.save_grey_image(np.swapaxes(self.blurred_whole_state[0], 0, 1), new_path + "/blurred_whole_state_0.jpg")
        self.save_grey_image(np.swapaxes(self.blurred_whole_state[1], 0, 1), new_path + "/blurred_whole_state_1.jpg")
        self.save_grey_image(np.swapaxes(self.blurred_whole_state[2], 0, 1), new_path + "/blurred_whole_state_2.jpg")
        self.save_grey_image(np.swapaxes(self.blurred_whole_state[3], 0, 1), new_path + "/blurred_whole_state_3.jpg")

        # Save blurred observation
        self.save_grey_image(np.swapaxes(self.example_blurs[0], 0, 1), new_path + "/local_blur_0.jpg")
        self.save_grey_image(np.swapaxes(self.example_blurs[1], 0, 1), new_path + "/local_blur_1.jpg")
        self.save_grey_image(np.swapaxes(self.example_blurs[2], 0, 1), new_path + "/local_blur_2.jpg")
        self.save_grey_image(np.swapaxes(self.example_blurs[3], 0, 1), new_path + "/local_blur_3.jpg")

