
import random
from typing import List
import torch

from minesweeperCore.tile import Tile


class MineSweeper():
    def __init__(self, area: int) -> None:
        self.field = [[Tile() for i in range(area)] for j in range(area)]
        self.mineCount = 16
        self.area = area
        self.started = False
        self.done = False
        self.success = False

    def getState(self) -> List[float]:
        state = []
        for i in range(len(self.field)):
            for j in range(len(self.field[i])):
                if self.field[i][j].show:
                    state.append(self.field[i][j].value + 1)
                else:
                    state.append(0)
        return torch.tensor(state)

    def action(self, x, y):
        if not self.started: # First action
            self.started = True
            self.generateField(x, y)
            reward = self.action(x, y)
            return reward

        if self.field[x][y].show: # Tile already revealed
            self.done = True
            self.success = False
            return torch.tensor([0])
        else:
            if self.field[x][y].mine: # Mine revealed
                self.done = True
                return torch.tensor([0])
            
            if self.field[x][y].value == 0: # Empty tile revealed
                reward = self.showEmpty(x, y)
                return torch.tensor([reward])
            
            else:
                if sum(1 for row in self.field for tile in row if not tile.show) == self.mineCount: # All non-mine tiles revealed
                    self.done = True
                    self.success = True
                    return torch.tensor([2000])
                else:       # Tile with value revealed
                    self.field[x][y].show = True
                    return torch.tensor([1])


    def generateField(self, clickX, clickY):
        for i in range(self.mineCount):
            x = random.randint(0, self.area - 1)
            y = random.randint(0, self.area - 1)
            while not (self.field[x][y].mine or (x != clickX and y != clickY)):
                x = random.randint(0, self.area - 1)
                y = random.randint(0, self.area - 1)
            self.field[x][y].mine = True
        for i in range(self.area):
            for j in range(self.area):
                if not self.field[i][j].mine:
                    self.field[i][j].value = self.countMines(i, j)
        
    def countMines(self, x, y):
        count = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                if x+i >= 0 and x+i < self.area and y+j >= 0 and y+j < self.area:
                    if self.field[x+i][y+j].mine:
                        count += 1
        return count
    
    def showEmpty(self, x, y):
        reward = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                if x+i >= 0 and x+i < self.area and y+j >= 0 and y+j < self.area:
                    if not self.field[x+i][y+j].show:
                        self.field[x+i][y+j].show = True
                        reward += 1
                        if self.field[x+i][y+j].value == 0:
                            reward += self.showEmpty(x+i, y+j)
        return reward
        