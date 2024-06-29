import os
import torch
from torch import nn
from torch.utils.data import DataLoader

from bitboard import Stone
from model import DQN
from agent import ModelAgent
from match import auto_match
from dataloader import QvalueDataset


def match_loop(learn_model: DQN, oppnt_model: DQN, loop: int, games: int):
    epsilon = 0.8 * (0.9 ** ((loop % 1000) / 20))

    exps = []
    results = [0, 0, 0]

    for a in range(games // 2):
        black_agent = ModelAgent(Stone.BLACK, learn_model, epsilon)
        white_agent = ModelAgent(Stone.WHITE, oppnt_model, epsilon)

        bexp, wexp, rwd = auto_match(black_agent, white_agent)
        exps.append(bexp)
        if rwd > 0:
            results[0] += 1
        elif rwd < 0:
            results[1] += 1
        else:
            results[2] += 1

        print(f"Loop: {loop}, Game: {a}, Reward: {rwd}")

    for a in range(games // 2):
        black_agent = ModelAgent(Stone.BLACK, oppnt_model, epsilon)
        white_agent = ModelAgent(Stone.WHITE, learn_model, epsilon)

        bexp, wexp, rwd = auto_match(black_agent, white_agent)
        exps.append(wexp)
        if rwd > 0:
            results[1] += 1
        elif rwd < 0:
            results[0] += 1
        else:
            results[2] += 1

        print(f"Loop: {loop}, Game: {a + games // 2}, Reward: {rwd}")

    print(
        f"Loop: {loop}, Results: learn {results[0]} - oppnt {results[1]} - draw {results[2]}"
    )

    return exps, results


def train(learn: dict, oppnt: dict, loop: int, epochs: int) -> list[int, int, int]:
    learn_model = learn["model"]
    learn_optimizer = learn["optimizer"]
    learn_criterion = learn["criterion"]

    oppnt_model = oppnt["model"]
    oppnt_optimizer = oppnt["optimizer"]
    oppnt_criterion = oppnt["criterion"]

    device = next(learn_model.parameters()).device
    learn_model.eval()
    oppnt_model.eval()
    exps, results = match_loop(learn_model, oppnt_model, loop, 1000)

    if loop % 5000 < 5:
        tail_sampling = (loop + 1) * 2
    else:
        tail_sampling = 64

    for epoch in range(epochs):
        learn_model.eval()
        dataset = QvalueDataset(exps, tail_sampling, learn_model)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

        loss = 0

        for x, y in dataloader:
            learn_model.train()

            x = x.to(device)
            y = y.to(device)

            learn_optimizer.zero_grad()
            output = learn_model(x)
            loss = learn_criterion(output, y)
            loss.backward()
            loss += loss.item()
            learn_optimizer.step()

        print(f"Loop : {loop}, Epoch: {epoch}, Loss: {loss / len(dataloader)}")

    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    learn_model = DQN().to(device)
    learn_optimizer = torch.optim.Adam(learn_model.parameters(), lr=0.02)
    learn_criterion = nn.MSELoss()

    oppnt_model = DQN().to(device)
    oppnt_optimizer = torch.optim.Adam(oppnt_model.parameters(), lr=0.02)
    oppnt_criterion = nn.MSELoss()

    switch_count = 0

    for loop in range(1000000):
        learner = {
            "model": learn_model,
            "optimizer": learn_optimizer,
            "criterion": learn_criterion,
        }
        oppnt = {
            "model": oppnt_model,
            "optimizer": oppnt_optimizer,
            "criterion": oppnt_criterion,
        }

        result = train(learner, oppnt, loop, 50)

        if result[0] > result[1]:
            print("switch models !")
            learn_model, oppnt_model = oppnt_model, learn_model
            learn_optimizer, oppnt_optimizer = oppnt_optimizer, learn_optimizer
            switch_count += 1

        if loop % 100 == 99:
            if os.path.exists("checkpoint") is False:
                os.makedirs("checkpoint")
            torch.save(
                {
                    "learning_model": learn_model.state_dict(),
                    "learning_optimizer": learn_optimizer.state_dict(),
                    "opponent_model": oppnt_model.state_dict(),
                    "opponent_optimizer": oppnt_optimizer.state_dict(),
                    "result": result,
                    "switch_count": switch_count,
                },
                f"checkpoint/point_{loop}.pth",
            )
    print("Training done.")


def gpu_accelerate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)


if __name__ == "__main__":
    print("GPU acceleration check...")
    gpu_accelerate()
    print("Start training...")
    main()
