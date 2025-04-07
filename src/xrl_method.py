from PIL import Image
import numpy as np
import os
from agent_training import get_action_name


class xrl_method:

    def __init__(self, agent_obs, rendered_states, q_values):
        
        # Create the template for the map and other map variables
        self.map = np.zeros(agent_obs.shape)
        self.normalised_map = None
        self.upscaled_map = None
        self.blended_renders = None

        # Set up variables for agent observation
        self.agent_obs = agent_obs
        self.normalised_agent_obs = np.copy(self.agent_obs)
        self.normalised_agent_obs = (self.normalised_agent_obs).astype(np.float32) / 255.0
        self.num_frames = self.agent_obs.shape[0]
        self.rendered_states = np.array(rendered_states)
        self.num_pixels = self.map.shape[-1] * self.map.shape[-2]

        # Set up variables for q-values and actions
        self.q_values = q_values
        self.num_actions = len(q_values)
        self.best_action = np.argmax(self.q_values)
        self.epsilon = 0.0
        self.best_action_val = 1.0
        self.other_action_val = 0.0


    # Function to save a 2D numpy array as a greyscale image
    def save_grey_image(self, img_array, filename):
        
        image = Image.fromarray(img_array, mode="L")
        image.save(filename)

    # Function to save a 3D numpy array as an RGB image
    def save_rgb_image(self, img_array, filename):

        image = Image.fromarray(img_array, mode="RGB")
        image.save(filename)

    
    # Function to normalise the map to have values of either 255 (salient) or 0 (not salient)
    def normalise_map(self):

        # Normalise the saliency values to [0, 255]
        self.normalised_map = ((self.map - np.min(self.map)) / (np.max(self.map) - np.min(self.map))) * 255

        # Highlight the top 1% of most salient pixels
        percent = 0.01

        # Find the number of pixels to keep in the heatmap using `percent`
        cut_off = int(self.num_pixels * self.num_frames * (1 - percent))

        # Sort the map and find the critical value where every pixel saliency above it is kept
        flat_map = (np.copy(self.normalised_map)).flatten()
        critical_val = (np.sort(flat_map))[cut_off]

        # Loop through the saliency map
        for i in range(self.normalised_map.shape[0]):
            for j in range(self.normalised_map.shape[1]):
                for k in range(self.normalised_map.shape[2]):

                    # If the saliency falls below the critical value, remove it
                    if self.normalised_map[i,j,k] <= critical_val:
                        self.normalised_map[i,j,k] = 0

                    # If above critical value, set to the max value of 255
                    else:
                        self.normalised_map[i,j,k] = 255

        # Set the map to use 8-but integers so it can be saved as an image
        self.normalised_map = self.normalised_map.astype(np.uint8)


    # Function to upscale the normalised map to the same resolution as the rendered state
    def upscale_map(self):
        
        # Normalise the map to contain only values of 0 or 255
        self.normalise_map()

        # Swap the axes so the map matches the orientation of the rendered state
        self.normalised_map = np.swapaxes(self.normalised_map, 1, 2)

        # Perform the upscaling
        normalised_map_img = [Image.fromarray(self.normalised_map[i], mode="L") for i in range(self.num_frames)]
        upscaled_map_img = [(normalised_map_img[i]).resize((1024, 512), resample=Image.Resampling.NEAREST) for i in range(self.num_frames)]

        # Cast the images back to a numpy array
        self.upscaled_map = np.array([np.array(upscaled_map_img[i]) for i in range(self.num_frames)])


    # Function to overlay the upscaled saliency over the top of the rendered state
    def create_overlay_img(self):
        
        # Make a copy to put the overlay on
        self.blended_renders = np.copy(self.rendered_states)

        # The shape of the image is (num_frames, H, W, 3), so self.blended_renders[..., 0] is the red channel of each frame
        image_red_channel = self.rendered_states[..., 0]

        # Merge the rendered image and heatmap by adding to its red channel
        modified_red_channel = (image_red_channel * 0.95) + (self.upscaled_map * 0.2)

        # Clip the resulting image to stay between 0 and 255
        modified_red_channel = np.clip(modified_red_channel, 0, 255)

        # Reassign the modified channel
        self.blended_renders[..., 0] = modified_red_channel


    # Function to just save the final blended renders
    def save_heatmaps(self, directory, suffix):

        new_path = directory + "/set" + str(suffix)
        os.mkdir(new_path)

        # Print the best action
        print("Best action: ", get_action_name(self.best_action))

        # Save heatmap overlays
        self.save_rgb_image(self.blended_renders[0], new_path + "/blended_render0.jpg")
        self.save_rgb_image(self.blended_renders[1], new_path + "/blended_render1.jpg")
        self.save_rgb_image(self.blended_renders[2], new_path + "/blended_render2.jpg")
        self.save_rgb_image(self.blended_renders[3], new_path + "/blended_render3.jpg")


    # Function to save all the images relevant to the creation of the heatmaps
    def save_all_images(self, directory, suffix):

        # Save heatmap overlays
        self.save_heatmaps(directory, suffix) 
        new_path = directory + "/set" + str(suffix)

        # Save agent observation
        self.save_grey_image(np.swapaxes(self.agent_obs[0], 0, 1), new_path + "/agent_obs_frame_0.jpg")
        self.save_grey_image(np.swapaxes(self.agent_obs[1], 0, 1), new_path + "/agent_obs_frame_1.jpg")
        self.save_grey_image(np.swapaxes(self.agent_obs[2], 0, 1), new_path + "/agent_obs_frame_2.jpg")
        self.save_grey_image(np.swapaxes(self.agent_obs[3], 0, 1), new_path + "/agent_obs_frame_3.jpg")

        # Save upscaled heat maps
        self.save_grey_image(self.upscaled_map[0], new_path + "/upscaled_map0.jpg")
        self.save_grey_image(self.upscaled_map[1], new_path + "/upscaled_map1.jpg")
        self.save_grey_image(self.upscaled_map[2], new_path + "/upscaled_map2.jpg")
        self.save_grey_image(self.upscaled_map[3], new_path + "/upscaled_map3.jpg")    
