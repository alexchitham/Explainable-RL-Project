
import cv2
import numpy as np
from PIL import Image
from agent_training import load_dqn_agent, create_vec_env, get_action_name


class sarfa_saliency_map:

    def __init__(self, agent_obs, rendered_state, q_values):

        # Create the template for the map and other map variables
        self.map = np.zeros(agent_obs[-1].shape)
        self.normalised_map = None
        self.upscaled_map = None
        self.blended_render = None
        self.example_blurs = []

        # Set up variables for agent observation
        self.agent_obs = agent_obs
        self.normalised_agent_obs = np.copy(self.agent_obs)
        self.normalised_agent_obs = (self.normalised_agent_obs).astype(np.float32) / 255.0
        self.num_frames = self.agent_obs.shape[0]
        self.rendered_state = rendered_state
        self.num_pixels = self.map.shape[-1] * self.map.shape[-2]

        # Set up variables for q-values and actions
        self.q_values = q_values
        self.num_actions = len(q_values)
        self.best_action = np.argmax(self.q_values)
        self.epsilon = 0.0
        self.best_action_val = 1.0
        self.other_action_val = 0.0

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


    def save_grey_image(self, img_array, filename):
        
        image = Image.fromarray(img_array, mode="L")
        image.save(filename)

    def save_rgb_image(self, img_array, filename):

        image = Image.fromarray(img_array, mode="RGB")
        image.save(filename)
    

    def calculate_delta_p(self, exp_blurred_q_values):
        
        blurred_prob = exp_blurred_q_values[self.best_action] / np.sum(exp_blurred_q_values)

        delta_p = self.prob_s_a - blurred_prob
        return delta_p
        


    def calculate_kl_divergence(self, exp_blurred_q_values):
        
        exp_blurred_q_values_no_best = np.delete(exp_blurred_q_values, self.best_action)
        blurred_p_rem = [exp_blurred_q_values_no_best[i] / np.sum(exp_blurred_q_values_no_best) for i in range(self.num_actions - 1)]
        
        kl_divergence = np.sum(blurred_p_rem * np.log(blurred_p_rem / self.p_rem))
        return kl_divergence
    

    # Function to calculate the saliency for a pixel given the perturbed q-values 
    def calculate_feature_saliency(self, blurred_q_values):

        exp_blurred_q_values = np.exp(blurred_q_values)
        
        delta_p = self.calculate_delta_p(exp_blurred_q_values)

        kl_divergence = self.calculate_kl_divergence(exp_blurred_q_values)

        k = 1 / (1 + kl_divergence)

        saliency = (2 * k * delta_p) / (k + delta_p)

        return saliency
    

    # Function to apply a blur around a pixel in the agent's observation
    def perturb_state(self, pixel_row, pixel_col):

        # Create the Gaussian mask
        mask = np.zeros(self.map.shape)
        sigma_sqr = 25.0
        denom_gaus = 2 * np.pi * sigma_sqr

        for row in range(mask.shape[0]):
            for col in range(mask.shape[1]):
    
                x_gaus_sqr = (row - pixel_row) ** 2
                y_gaus_sqr = (col - pixel_col) ** 2
                exponent_gaus = np.exp(-1 * ((x_gaus_sqr + y_gaus_sqr) / (2 * sigma_sqr)))
                
                mask[row,col] = np.clip(exponent_gaus / denom_gaus, 1e-10, 1)

        # Normalise the mask to [0,1]
        mask = mask / np.max(mask)

        # Apply the localised blur and scale back up to [0,255]
        perturbed_image = [(self.normalised_agent_obs[i] * (1 - mask)) + (self.normalised_blurred_state[i] * mask) for i in range(self.num_frames)]

        # Scale back up to [0,255]
        perturbed_image = np.clip(perturbed_image, 0, 1)
        perturbed_image = (perturbed_image * 255).astype(np.uint8)
        
        if (pixel_row == 64 and pixel_col == 32):
            self.example_blurs = np.copy(perturbed_image)
            

        # Add a batch dimension to match dimensions of q-network input
        batched_perturbed_img = np.expand_dims(perturbed_image, axis=0)

        return batched_perturbed_img


    def calculate_saliency(self, agent):
        
        # Loop over every pixel in the observation
        for row in range(self.map.shape[0]):
            for col in range(self.map.shape[1]):

                # Perturb / blur part of the agent's observation
                perturbed_state = self.perturb_state(row, col)

                # best_action_blur, _ = agent.predict(perturbed_state, deterministic=True)
                blurred_q_values = get_q_values(agent, perturbed_state)

                self.map[row, col] = self.calculate_feature_saliency(blurred_q_values)


    # Function to normalise the map to have values of either 255 (salient) or 0 (not salient)
    def normalise_map(self):

        # Normalise the saliency values to [0, 255]
        self.normalised_map = ((self.map - np.min(self.map)) / (np.max(self.map) - np.min(self.map))) * 255

        # Want to remove 98% of pixels, i.e. keep the top 2% most salient 
        percent = 0.98

        # Find the number of pixels to keep in the heatmap using `percent`
        cut_off = int(self.num_pixels * percent)

        # Sort the map and find the critical value where every pixel saliency above it is kept
        flat_map = (np.copy(self.normalised_map)).flatten()
        critical_val = (np.sort(flat_map))[cut_off]

        # Loop through the saliency map
        for i in range(self.normalised_map.shape[0]):
            for j in range(self.normalised_map.shape[1]):

                # If the saliency falls below the critical value, remove it
                if self.normalised_map[i,j] <= critical_val:
                    self.normalised_map[i,j] = 0

                # If above critical value, set to the max value of 255
                else:
                    self.normalised_map[i,j] = 255

        # Set the map to use 8-but integers so it can be saved as an image
        self.normalised_map = self.normalised_map.astype(np.uint8)


    # Function to upscale the normalised map to the same resolution as the rendered state
    def upscale_map(self):
        
        # Normalise the map
        self.normalise_map()

        # Swap the axes so the map matches the orientation of the rendered state
        self.normalised_map = np.swapaxes(self.normalised_map, 0, 1)

        # Perform the upscaling
        normalised_map_img = Image.fromarray(self.normalised_map, mode="L")
        upscaled_map_img = normalised_map_img.resize((1024, 512), resample=Image.Resampling.NEAREST)

        # Cast the image back to a numpy array
        self.upscaled_map = np.array(upscaled_map_img)


    # Function to overlay the upscaled saliency over the top of the rendered state
    def create_overlay_img(self, suffix):
        
        self.blended_render = np.copy(self.rendered_state)

        # The shape of the image is (H, W, 3), so self.blended_render[..., 0] is the red channel
        image_red_channel = self.rendered_state[..., 0]

        # Merge the rendered image and heatmap by adding to its red channel
        modified_red_channel = (image_red_channel * 0.95) + (self.upscaled_map * 0.2)

        # Clip the resulting image to stay between 0 and 255
        modified_red_channel = np.clip(modified_red_channel, 0, 255)

        # Reassign the modified channel
        self.blended_render[..., 0] = modified_red_channel
        
        # blended_render_img = Image.fromarray(self.blended_render, mode="RGB")
        # blended_render_img.save('test_images/blended_render' + str(suffix) + '.jpg')


    def save_all_images(self, directory, suffix):

        # Save agent observation
        self.save_grey_image(np.swapaxes(self.agent_obs[0], 0, 1), directory + str(suffix) + "/agent_obs_frame_0.jpg")
        self.save_grey_image(np.swapaxes(self.agent_obs[1], 0, 1), directory + str(suffix) + "/agent_obs_frame_1.jpg")
        self.save_grey_image(np.swapaxes(self.agent_obs[2], 0, 1), directory + str(suffix) + "/agent_obs_frame_2.jpg")
        self.save_grey_image(np.swapaxes(self.agent_obs[3], 0, 1), directory + str(suffix) + "/agent_obs_frame_3.jpg")

        # Save full blurred states
        self.save_grey_image(np.swapaxes(self.blurred_whole_state[0], 0, 1), directory + str(suffix) + "/blurred_whole_state_0.jpg")
        self.save_grey_image(np.swapaxes(self.blurred_whole_state[1], 0, 1), directory + str(suffix) + "/blurred_whole_state_1.jpg")
        self.save_grey_image(np.swapaxes(self.blurred_whole_state[2], 0, 1), directory + str(suffix) + "/blurred_whole_state_2.jpg")
        self.save_grey_image(np.swapaxes(self.blurred_whole_state[3], 0, 1), directory + str(suffix) + "/blurred_whole_state_3.jpg")

        # Save blurred observation
        self.save_grey_image(np.swapaxes(self.example_blurs[0], 0, 1), directory + str(suffix) + "/local_blur_0.jpg")
        self.save_grey_image(np.swapaxes(self.example_blurs[1], 0, 1), directory + str(suffix) + "/local_blur_1.jpg")
        self.save_grey_image(np.swapaxes(self.example_blurs[2], 0, 1), directory + str(suffix) + "/local_blur_2.jpg")
        self.save_grey_image(np.swapaxes(self.example_blurs[3], 0, 1), directory + str(suffix) + "/local_blur_3.jpg")

        # Save rendered frame
        self.save_rgb_image(self.rendered_state, directory + str(suffix) + "/rendered_state.jpg")

        # Save the original heat map
        self.save_grey_image(self.normalised_map, directory + str(suffix) + "/normalised_map.jpg")

        # Save upscaled heat map
        self.save_grey_image(self.upscaled_map, directory + str(suffix) + "/upscaled_map.jpg")

        # Save heatmap overlay
        self.save_rgb_image(self.blended_render, directory + str(suffix) + "/blended_render.jpg")

        # Print the best action
        print("Best action: ", get_action_name(self.best_action))



    
def create_saliency_map(agent_obs, rendered_state, q_values, agent):

    # Remove the batch dimension before passing through to the saliency map object
    batchless_state = np.squeeze(agent_obs)

    # Create a saliency map object
    saliency_map = sarfa_saliency_map(batchless_state, rendered_state, q_values)

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
    for _ in range(1):

        # Reset for new episode
        done = False
        state = env.reset()
        step_count = 0

        # Run until the episode is finished
        while not done:

            # Predict the most optimal action
            best_action, _ = agent.predict(state, deterministic=True)

            q_values = get_q_values(agent, state)

            # Create saliency map for the current state
            frame = env.get_images()[0]
            # frame = env.render(mode="rgb_array")

            if step_count == 10:
                saliency_map_obj = create_saliency_map(state, frame, q_values, agent)
                saliency_map_obj.create_overlay_img(step_count)
                saliency_map_obj.save_all_images("test_save_images", 1)

            # if step_count == 11:
            #     saliency_map_obj = create_saliency_map(state, frame, q_values, agent)
            #     saliency_map_obj.create_overlay_img(step_count)
            #     saliency_map_obj.save_all_images("test_save_images", 2)

            # Take the action in the environment
            state, reward, done, info = env.step(best_action)
            step_count += 1


    env.close()

