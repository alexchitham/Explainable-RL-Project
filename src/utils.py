

# Function to retrieve the q-values from an agent in a given state
def get_q_values(agent, state):

    # Convert the state to a PyTorch tensor
    obs_tensor, _ = agent.policy.obs_to_tensor(state)

    # Pass the observation through the Q-Network
    q_values = (agent.policy.q_net(obs_tensor))

    # Remove from backprop -> move to cpu -> convert to numpy -> remove batch dimension
    q_values = q_values.detach().cpu().numpy().squeeze()

    return q_values