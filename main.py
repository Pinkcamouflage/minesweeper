


import torch
import torch.optim as optim
from guessingNN.neuralNetwork import DQN
from guessingNN.optimizer import optimizeModel
from guessingNN.replayMemory import ReplayMemory
from minesweeperCore.game import MineSweeper

AREA = 10
OBSERVATIONS = AREA**2
LR = 0.0001
EPISODES = 1000

if __name__ == "__main__":
    memory = ReplayMemory(10000)
    policyNet = DQN(OBSERVATIONS, 256, 2)
    targetNet = DQN(OBSERVATIONS, 256, 2)
    
    # Make the target network have the same parameters as the policy network
    targetNet.load_state_dict(policyNet.state_dict())
    
    # Set the target network in evaluation mode (no need for gradients here)
    targetNet.eval()

    optimizer = optim.AdamW(policyNet.parameters(), lr=LR, amsgrad=True)

    for episode in range(EPISODES):
        currentGame = MineSweeper(AREA)
        print(f"Episode {episode}")

        while not currentGame.done:
            state = currentGame.getState()
            action = policyNet.forward(torch.tensor(state).float())

            # Action scaling to the environment's area
            action_x = int(action[0] * AREA)
            action_y = int(action[1] * AREA)

            reward = currentGame.action(action_x, action_y)

            # Store the transition in memory
            next_state = currentGame.getState()
            memory.push(
                state.clone().detach().float(), 
                action, 
                next_state.clone().detach().float(), 
                reward.clone().detach().float()
            )

            # Perform optimization
            optimizeModel(policyNet, targetNet, memory, optimizer, 4, 0.99)

            # Update the target network slowly by copying from the policy network
            targetNetStateDict = targetNet.state_dict()
            policyNetStateDict = policyNet.state_dict()
            for key in targetNetStateDict.keys():
                targetNetStateDict[key] = 0.99 * targetNetStateDict[key] + 0.01 * policyNetStateDict[key]
            targetNet.load_state_dict(targetNetStateDict)

        print(f"Puzzle solved: {currentGame.success} with reward {reward}")