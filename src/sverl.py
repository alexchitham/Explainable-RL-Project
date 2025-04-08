
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import csv
from collections import deque
from xrl_method import xrl_method
from agent_training import load_dqn_agent, get_action_size, get_env_size
from utils import get_action_probabilities

from torch.distributions.categorical import Categorical



# Class taken from the FastSHAP implementation https://arxiv.org/pdf/2107.07436
# Find the code at https://github.com/iancovert/fastshap in fastshap/utils.py
class ShapleySampler:
    '''
    For sampling player subsets from the Shapley distribution.

    Args:
      num_players: number of players.
    '''

    def __init__(self, num_players):
        arange = torch.arange(1, num_players)
        w = 1 / (arange * (num_players - arange))
        w = w / torch.sum(w)
        self.categorical = Categorical(probs=w)
        self.num_players = num_players
        self.tril = np.tril(
            np.ones((num_players - 1, num_players), dtype=np.float32), k=0
        )
        self.rng = np.random.default_rng()

    def sample(self, batch_size, paired_sampling):
        '''
        Generate sample.

        Args:
          batch_size: number of samples.
          paired_sampling: whether to use paired sampling.
        '''
        num_included = 1 + self.categorical.sample([batch_size])
        S = self.tril[num_included - 1]
        S = self.rng.permuted(S, axis=1)  # Note: permutes each row.
        if paired_sampling:
            S[1::2] = 1 - S[0:(batch_size - 1):2]  # Note: allows batch_size % 2 == 1.
        return torch.from_numpy(S)



# Neural Network Architecture for both Characteristic Value and Shapley Value Functions
class ShapleyEstimator(nn.Module):

    def __init__(self, input_size, output_size, apply_softmax = True):

        super().__init__()

        self.apply_softmax = apply_softmax

        num_frames = input_size[0]

        # Define Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=num_frames, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        # Define Linear layers
        self.fc1 = nn.Linear(self.calculate_flattened_size(input_size), 256)
        self.fc2 = nn.Linear(256, output_size)

        # Define Activation function
        self.relu = nn.ReLU()
        
        # Define Softmax layer
        self.softmax = nn.Softmax(dim=1)


    def calculate_flattened_size(self, input_shape):

        # Simulate a dummy input and pass it through the three convolutional layers
        x = torch.zeros(input_shape)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Flatten the convolutional output and find the size
        x = torch.flatten(x)
        num_elements = x.size(0)
        return num_elements
    

    def forward(self, x):

        # Convert input to float32 as that's what the model needs to be trained on
        x = x.to(dtype=torch.float32)

        # First pass the input through the convolutional layers
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        # Resize to maintain the batch size but flatten the rest
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        if self.apply_softmax:
            x = self.softmax(x)

        return x
    

class sverl_saliency_map(xrl_method):
    
    def __init__(self, agent_obs, rendered_states, q_values, shapley_func_weights_dir, device="cpu"):

        super().__init__(agent_obs, rendered_states, q_values)

        # Convert the state to a PyTorch tensor if it isn't
        if not (isinstance(self.agent_obs, torch.Tensor)):
            self.torch_agent_obs = torch.tensor(self.agent_obs)

        # Add a batch back to the state so it can pass through the network
        self.batched_state = torch.unsqueeze(self.torch_agent_obs, 0)

        # Define the Shapley Value Approximator network
        self.shapley_val_network = ShapleyEstimator(self.agent_obs.shape, self.num_actions * self.num_pixels * self.num_frames, apply_softmax=False)
        
        # Load the pre-trained model weights in to the network
        self.shapley_val_network.load_state_dict(torch.load(shapley_func_weights_dir, weights_only=True, map_location=torch.device(device)))


    def predict_shapley_values(self):

        # Pass the state through the network and remove batch
        shapley_predictions = self.shapley_val_network(self.batched_state) # Shape: [1, A * F * H * W]
        shapley_predictions = torch.squeeze(shapley_predictions) # Shape: [A * F * H * W]

        # Reshape the flat output to be comparable to that of the masks and state shape
        shapley_predictions = torch.reshape(shapley_predictions, (self.num_actions, ) + self.agent_obs.shape) # Shape: [A, F, H, W]

        # Assign the Shapley Values for the best action to the object's map
        self.map = (shapley_predictions[self.best_action]).detach().numpy()
        print(self.map.shape)
    

# Function to create the object used to Shapley Kernel sample
def create_shapley_kernel(state_shape):

    # Find the "number of players"
    num_pixels =  np.prod(state_shape)

    # Create the return the object
    shapley_sampler = ShapleySampler(num_pixels)
    return shapley_sampler
    

# Function to perform the Shapley Kernel sampling
def shapley_kernel_sample(shapley_sampler, state_shape, batch_size):

    # Sample a set of masks
    masks = shapley_sampler.sample(batch_size, paired_sampling=False)

    # Reshape to match the batch size and state shape
    masks = torch.reshape(masks, (batch_size, ) + state_shape)
    return masks


def random_sample(state_shape, batch_size):

    # Random generate 0's and 1's for each feature in each state
    masks = torch.randint(0, 2, (batch_size, ) + state_shape, device=device)
    return masks
    

def sample_steady_states(steady_states, batch_size, device):

    # Extract the total number of states in the distribution
    num_states = steady_states.shape[0]

    # Randomly select indices from the number of states of batch length
    indices = torch.randint(0, num_states, (batch_size,), device='cpu')

    # Sample the states using the indices
    sampled_states = (steady_states[indices]).to(device=device)
    return sampled_states


def mask_states(batch_of_states, masks, device, mask_value=-100):    

    # Change the states to float32
    batch_of_states_float = batch_of_states.to(dtype=torch.float32, device=device)

    # Turn mask tensor from 1's and 0's, to True's and False's
    masks = masks.to(dtype=torch.bool, device=device)

    # Create the full mask tensor containing all mask values
    full_mask = torch.full_like(batch_of_states_float, mask_value, device=device)

    # Mask out the pixels specified in the mask
    masked_states = torch.where(masks, batch_of_states_float, full_mask)
    return masked_states


def calculate_mean_state(steady_states, device, batch_size=10000):

    # Find the number of states in the distribution
    num_states = steady_states.shape[0]

    # Create a blank tensor to store the sum
    sum_of_states = torch.zeros(steady_states.shape[1:], dtype=torch.float32, device=device)

    # Loop through each state in jumps of batch_size
    for i in range(0, num_states, batch_size):

        # Take a batch of the states and cast to float32
        batch = steady_states[i:i+batch_size].to(device=device, dtype=torch.float32)

        # Add to the sum of the batch to the ongoing sum
        sum_of_states += torch.sum(batch, dim=0)

    mean_state = sum_of_states / num_states

    # Add a batch dimension so it can be passed into the model
    mean_state = torch.unsqueeze(mean_state, 0)
    return mean_state




def train_char_val_network(model, optimiser, steady_states, agent, save_dir, device, num_batches, shapley_sample=True, batch_size=128):
    # Shape Guide: A = actions, B = batch (128), F = frames, H = height, W = width

    # Structures used to store and record loss
    loss_values = []
    recent_loss = deque(maxlen=50)

    # Put the model in training mode
    model.train()

    # Derive the shape of each state
    state_shape = steady_states.shape[1:] # Shape: [F, H, W]

    # If sampling using the Shapley Kernel
    if shapley_sample:
        shapley_sampler = create_shapley_kernel(state_shape)


    # Loop through all the batches
    for batch in range(num_batches):

        # Sample states from the distribution
        batch_of_states = (sample_steady_states(steady_states, batch_size, device)).to(device) # Shape: [B, F, H, W]

        # Use the trained agent to output the true action probabilities
        with torch.no_grad():
            true_values = (get_action_probabilities(agent, batch_of_states, softmax=True)).to(device) # Shape: [B, A]

        # Create the sampled masks
        if shapley_sample:
            masks = shapley_kernel_sample(shapley_sampler, state_shape, batch_size) # Shape: [B, F, H, W]
        else:
            masks = random_sample(state_shape, batch_size) # Shape: [B, F, H, W]        

        # Randomly mask the sampled states
        masked_states = (mask_states(batch_of_states, masks, device)).to(device) # Shape: [B, F, H, W]

        # Make predictions
        predicted_values = model(masked_states) # Shape: [B, A]

        # Calculate the loss
        loss = F.mse_loss(predicted_values, true_values, reduction='mean')

        recent_loss.append(loss.item())

        # Periodically save the model's weights
        if batch % 20000 == 0:
            filename = save_dir + "/characteristic_func_weights/step" + str(batch)
            torch.save(model.state_dict(), filename)
            most_recent_weights = filename

        # Periodically print and record the average loss
        if batch % 50 == 0:
            avg_loss = np.mean(recent_loss)
            loss_values.append((batch, np.round(avg_loss, decimals=7)))
        if batch % 1000 == 0:
            print(f"[Batch {batch}] Average Loss: {avg_loss:.6f}", flush=True)


        # Perform model updates
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()


    # Write the loss data to a CSV file
    csv_filename = save_dir + "/training_csv_outputs/char_val_loss.csv"
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["batch", "loss"])
        writer.writerows(loss_values)

    return most_recent_weights



def train_shapley_val_network(model, char_val_network, optimiser, steady_states, num_actions, save_dir, device, num_batches, shapley_sample=True, batch_size=32):
    # Shape Guide: A = actions, B = batch (128), F = frames, H = height, W = width

    # Structures used to store and record loss
    loss_values = []
    recent_loss = deque(maxlen=50)
    
    # Put the model in training mode
    model.train()
    
    # Derive the shape of each state
    state_shape = steady_states.shape[1:] # Shape: [F, H, W]

    # If sampling using the Shapley Kernel, create the object for it
    if shapley_sample:
        shapley_sampler = create_shapley_kernel(state_shape)

    # Calculate the mean state from the steady states
    mean_state = calculate_mean_state(steady_states, device) # Shape: [1, F, H, W]
    # mean_state_masked = torch.full((1, 4, 128, 64), -100.0)

    # Duplicate it to have an additional leading dimension of batch_size
    mean_state_batched = mean_state.repeat(batch_size, 1, 1, 1) # Shape: [B, F, H, W]

    # Pass it through the characteristic value function to get the average action probabilities
    with torch.no_grad():
        mean_state_probs = char_val_network(mean_state_batched) # Shape: [B, A]
        # mean_state_probs_masked = char_val_network(mean_state_masked) # Shape: [B, A]

    
    # Loop through all the batches
    for batch in range(num_batches):

        # Sample states from the distribution
        batch_of_states = (sample_steady_states(steady_states, batch_size, device)).to(device) # Shape: [B, F, H, W]

        # Create the sampled masks using one of the sampling methods
        if shapley_sample:
            masks = shapley_kernel_sample(shapley_sampler, state_shape, batch_size) # Shape: [B, F, H, W]
        else:
            masks = random_sample(state_shape, batch_size) # Shape: [B, F, H, W]

        # Mask the sampled states and pass them through the characteristic value function
        masked_states = (mask_states(batch_of_states, masks, device)).to(device) # Shape: [B, F, H, W]
        with torch.no_grad():
            masked_states_probs = char_val_network(masked_states) # Shape: [B, A]

        # Pass the batch of states through the Shapley Value network
        shapley_predictions = model(batch_of_states).to(device) # Shape: [B, A * F * H * W]

        # Reshape the flat output to be comparable to that of the masks and state shape
        shapley_predictions = torch.reshape(shapley_predictions, (batch_size, num_actions, ) + state_shape) # Shape: [B, A, F, H, W]

        # Add an additional dimension to the masks so it can be broadcast over all actions when performing further masking
        masks_expanded = (masks.unsqueeze(1)).to(device)  # [B, 1, F, H, W]

        # Cancel out features not part of the coaliton (mask contains just 1's and 0's)
        masked_shapley_vals = shapley_predictions * masks_expanded # Shape: [B, A, F, H, W]

        # Perform summation, summing over frames, height and width of the states
        shapley_sum = torch.sum(masked_shapley_vals, (2, 3, 4)) # Shape: [B, A]

        # Calculate the mean squared error loss
        loss = F.mse_loss(shapley_sum, masked_states_probs - mean_state_probs, reduction='mean')

        recent_loss.append(loss.item())

        # Periodically save the model's weights
        if batch % 10000 == 0:
            filename = save_dir + "/shapley_func_weights/step" + str(batch)
            torch.save(model.state_dict(), filename)
            most_recent_weights = filename

        # Periodically print and record the average loss
        if batch % 50 == 0:
            avg_loss = np.mean(recent_loss)
            loss_values.append((batch, np.round(avg_loss, decimals=7)))
        if batch % 500 == 0:
            print(f"[Batch {batch}] Average Loss: {avg_loss:.6f}", flush=True)

        # Perform model updates
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    
    # Write the loss data to a CSV file
    csv_filename = save_dir + "/training_csv_outputs/shap_val_loss.csv"
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["batch", "loss"])  # header
        writer.writerows(loss_values)

    return most_recent_weights


# Function to analyse the Steady State Distribution and test the Characteristic Value Function
def analyse_steady_state(steady_states, char_val_network, agent):

    from collections import Counter
    from agent_training import get_action_name

    best_moves = []

    for i in range(0, 30, 2):

        states = sample_steady_states(steady_states, 2, device)
        # states = torch.stack((steady_states[i], steady_states[i+1]))

        with torch.no_grad():
            true_values = (get_action_probabilities(agent, states, softmax=True))
            predicted_values = char_val_network(states)

        best_move_index = torch.argmax(true_values[0]).item()
        best_move = get_action_name(best_move_index)
        best_moves.append(best_move)

        best_move_index = torch.argmax(true_values[1]).item()
        best_move = get_action_name(best_move_index)
        best_moves.append(best_move)


        print("True values: ", true_values[0])
        print("Predictions: ", predicted_values[0])
        print("------------------------------")
        print("True values: ", true_values[1])
        print("Predictions: ", predicted_values[1])
        print("------------------------------")

    c = Counter(best_moves)
    print(c)







if __name__ == "__main__":    

    # DEBUGGING SUGGESTIONS:
    #   TEST with no coaltions to see if network can replicat eht DQN agent
    #   Could try flattening instead of CNN
    #   Check the masking is doing what I think it is


    num_actions = get_action_size()
    state_shape = get_env_size()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, flush=True)
    agent_directory = "highway-env/agents/dqn_highway_norm_200K.zip"
    states_directory = "highway-env/sverl/steady_states200k_1.pt"
    save_dir = "highway-env/sverl"

    agent = load_dqn_agent(agent_directory, device=device)
    steady_states = (torch.load(states_directory, weights_only=True))

    char_val_network = ShapleyEstimator(state_shape, num_actions, apply_softmax=True)
    shap_val_network = ShapleyEstimator(state_shape, num_actions * np.prod(state_shape), apply_softmax=False)

    # ----------------------------------------------------


    char_val_network = char_val_network.to(device)
    shap_val_network = shap_val_network.to(device)

    optimiser_char_val = optim.Adam(char_val_network.parameters(), lr=0.00025)
    optimiser_shap_val = optim.Adam(shap_val_network.parameters(), lr=0.00025)

    char_func_weights = train_char_val_network(char_val_network, optimiser_char_val, steady_states, agent, save_dir, device, num_batches=200005)

    char_val_network.load_state_dict(torch.load(char_func_weights, weights_only=True, map_location=torch.device(device)))

    # char_val_network.load_state_dict(torch.load(save_dir + "/characteristic_func_weights/step200k_-100_softmax_v1", weights_only=True, map_location=torch.device(device)))
    # shap_val_network.load_state_dict(torch.load(save_dir + "/shapley_func_weights/step100k_1", weights_only=True, map_location=torch.device(device)))

    char_val_network.eval()

    shap_func_weights = train_shapley_val_network(shap_val_network, char_val_network, optimiser_shap_val, steady_states, num_actions, save_dir, device, num_batches=100005)


    # ---------------------------------------------------------
    
    # char_val_network.load_state_dict(torch.load(save_dir + "/characteristic_func_weights/step60000_3", weights_only=True, map_location=torch.device(device)))
    # char_val_network.eval()
    # analyse_steady_state(steady_states, char_val_network, agent)

    # -----------------------------------------

    

    



    



        




