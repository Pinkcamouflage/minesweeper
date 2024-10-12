import torch
import torch.nn.functional as F

from guessingNN.replayMemory import Transition

def optimizeModel(policyNet, targetNet, memory, optimizer, batch_size, gamma):
    if len(memory) < batch_size:
        return

    # Sample a batch of transitions from memory
    transitions = memory.sample(batch_size)
    
    # Unpack the transitions
    state_batch, action_batch, next_state_batch, reward_batch = zip(*transitions)
    
    # Convert to tensors
    state_batch = torch.stack(state_batch)
    action_batch = torch.stack(action_batch)
    next_state_batch = torch.stack(next_state_batch)
    reward_batch = torch.stack(reward_batch)

    # Compute predicted Q-values from the policy network
    predicted_q_values = policyNet(state_batch)

    # For each action in the batch, extract the Q-value

    predicted_q_values = predicted_q_values.gather(1, action_batch.long()).squeeze(1)
    # Compute the target Q-values using the target network and the reward
    with torch.no_grad():
        next_q_values = targetNet(next_state_batch)
        max_next_q_values = next_q_values.max(0).values
        target_q_values = reward_batch + gamma * max_next_q_values

    # Compute the loss between predicted and target Q-values
    loss = F.mse_loss(predicted_q_values, target_q_values)

    # Backpropagate the loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()