import random
import copy
from typing import Any
from dataclasses import dataclass

import torch
from torch import nn

from env import Stone, Board


def board_to_2ch(board: Board) -> tuple[list[list[float]], list[list[float]]]:
    board = board.get_board()
    return [
        [[1.0 if cell == Stone.BLACK else 0.0 for cell in row] for row in board],
        [[1.0 if cell == Stone.WHITE else 0.0 for cell in row] for row in board],
    ]


def action_to_1ch(pos: tuple[int, int]) -> list[list[float]]:
    action = [[0.0 for _1 in range(8)] for _2 in range(8)]
    action[pos[1]][pos[0]] = 1.0
    return action


def args_to_inputs(
    stone: Stone,
    board: Board,
    actions: list[tuple[int, int]],
):
    inputs = []
    black, white = board_to_2ch(board)
    if stone == Stone.BLACK:
        for action in actions:
            next_board = copy.deepcopy(board)
            next_board.act(stone, action)
            black_next, white_next = board_to_2ch(next_board)
            inputs.append(
                [
                    copy.deepcopy(black),
                    copy.deepcopy(white),
                    action_to_1ch(action),
                    black_next,
                    white_next,
                ]
            )
        return inputs
    else:
        for action in actions:
            next_board = copy.deepcopy(board)
            next_board.act(stone, action)
            black_next, white_next = board_to_2ch(next_board)
            inputs.append(
                [
                    copy.deepcopy(white),
                    copy.deepcopy(black),
                    action_to_1ch(action),
                    white_next,
                    black_next,
                ]
            )
        return inputs


@dataclass
class Experience:
    inputs: list[list[list[int]]]
    reward: float


class ExperienceBuffer:
    def __init__(self) -> None:
        self.buffer = []

    def add(self, inputs: Any):
        self.buffer.append(Experience(inputs, 0.0))

    def reward(self, reward: float) -> None:
        self.buffer[-1].reward = reward


class Agent:
    def __init__(self, stone):
        self.stone = stone
        self.buffer = ExperienceBuffer()

    def act(self, board: Board) -> tuple[int, int] | None:
        pass

    def reward(self, reward: float) -> None:
        self.buffer.reward(reward)

    def get_buffer(self) -> ExperienceBuffer:
        return self.buffer


class ModelAgent(Agent):
    def __init__(self, stone: Stone, model: nn.Module, epsilon: float) -> None:
        super().__init__(stone)

        self.model = model
        self.device = next(model.parameters()).device
        self.epsilon = epsilon

    def act(self, board: Board) -> tuple[int, int] | None:
        actions = board.get_actions(self.stone)
        if len(actions) == 0:
            return None

        inputs = args_to_inputs(self.stone, board, actions)
        outputs = self._predict(inputs)
        q_values = outputs.clone().flatten()

        index: int = q_values.argmax().item()
        if self.epsilon > random.random():
            index = random.randrange(0, len(actions))

        self.buffer.add(inputs[index])
        return actions[index]

    def _predict(self, inputs: list) -> torch.Tensor:
        inputs = torch.tensor(inputs).to(self.device)
        outputs = self.model(inputs).cpu()
        return outputs
