
import numpy as np
import torch

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