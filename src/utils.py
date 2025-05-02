
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

# Function to retrieve the q-values from an agent in a given state
def get_q_values(agent, states):

    # Convert the state to a PyTorch tensor if it isn't
    if not (isinstance(states, torch.Tensor)):
        states = torch.tensor(states)

    # Pass the observation through the Q-Network
    q_values = (agent.policy.q_net(states))

    # Remove from backprop and move to cpu
    q_values = q_values.detach().cpu()

    # Remove the batch dimension if there is just one state
    if q_values.shape[0] == 1:
        q_values = np.squeeze(q_values.numpy(), axis=0)

    return q_values


def get_action_probabilities(agent, states, softmax=True):

    # Get the q-values
    q_values = get_q_values(agent, states)

    # Use softmax so each batch of probabilities sums to 1
    if softmax:
        softmax = torch.nn.Softmax(dim=1)
        action_probs = softmax(q_values)

    # Use a one-hot style tensor to represent the most probable action
    else: 
        batch_size = q_values.shape[0]
        most_probable_action = torch.argmax(q_values, 1)
        action_probs = torch.zeros_like(q_values)
        action_probs[torch.arange(batch_size), most_probable_action] = 1.0

    return action_probs


def plot_loss_graph():

    loss_directories = [
    "highway-env/sverl/training_csv_outputs/char_val_loss200k_v1.csv",
    "highway-env/sverl/training_csv_outputs/char_val_loss200k_v3.csv",
    "highway-env/sverl/training_csv_outputs/char_val_loss200k_v2.csv",
    ]

    labels = [
    "Default Hyperparameters",
    "Experiment 4 (Mask Value changed)",
    "Experiment 5 (Action Probabilities changed)",
    ]

    # loss_directories = [
    # "highway-env/sverl/training_csv_outputs/shap_val_loss100k_v1.csv",
    # "highway-env/sverl/training_csv_outputs/shap_val_loss100k_v3.csv",
    # "highway-env/sverl/training_csv_outputs/shap_val_loss100k_v4.csv",
    # "highway-env/sverl/training_csv_outputs/shap_val_loss100k_v2.csv",
    # ]

    # labels = [
    # "Default Hyperparameters",
    # "Experiment 3 (Batch Size changed)",
    # "Experiment 4 (Mask Value changed)",
    # "Experiment 5 (Action Probabilities changed)",
    # ]

    plt.rcParams.update({'font.size': 14})

    plt.figure(figsize=(10, 6))  # Larger figure for clarity

    for dir_path, label in zip(loss_directories, labels):
        # Load CSV
        df = pd.read_csv(dir_path)
        starting_index = 2
        # starting_index = 15

        # Apply moving average smoothing
        # loss = df["loss"][starting_index:]
        loss = df["loss"][starting_index:].rolling(window=40).mean()
        # loss = df["loss"][starting_index:].rolling(window=100).mean()

        # Plot smoothed loss
        plt.plot(df["batch"][starting_index:], loss, label=label, alpha=0.6, linewidth=2.0)

    # Customize plot
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title("Characteristic Value Function Average Loss (Smoothed)")
    # plt.title("Shapley Value Function Average Loss (Smoothed)")
    plt.grid(True)
    plt.legend()
    plt.show()
