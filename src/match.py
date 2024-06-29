from bitboard import Board, Stone, flip
from agent import Agent, ModelAgent, ExperienceBuffer


def auto_match(black_agent: Agent, white_agent: Agent) -> tuple[
    ExperienceBuffer,
    ExperienceBuffer,
    float,
]:
    board = Board()
    turn = Stone.BLACK
    passed = False

    while True:
        if turn == Stone.BLACK:
            action = black_agent.act(board)
        else:
            action = white_agent.act(board)

        if action == None:
            if passed:
                break
            else:
                passed = True
        else:
            passed = False
            board.act(turn, action)

        turn = flip(turn)
    b, w, e = board.get_count()
    reward = conut_to_reward(b, w, e)

    black_agent.reward(reward)
    white_agent.reward(reward)

    return (
        black_agent.get_buffer(),
        white_agent.get_buffer(),
        reward,
    )


def conut_to_reward(b, w, e):
    return (b - w) / (b + w)


def auto_match_random_start(
    black_agent: ModelAgent, white_agent: ModelAgent, start: int
) -> tuple[ExperienceBuffer, ExperienceBuffer, float]:
    board = Board()
    passed = False
    turn = Stone.BLACK
    turn_count = 0

    while True:
        if turn == Stone.BLACK:
            if turn_count < start:
                action = black_agent.act_random(board)
            else:
                action = black_agent.act(board)
        else:
            if turn_count < start:
                action = white_agent.act_random(board)
            else:
                action = white_agent.act(board)

        if action == None:
            if passed:
                break
            else:
                passed = True
        else:
            passed = False
            board.act(turn, action)

        turn = flip(turn)
        turn_count += 1
    b, w, e = board.get_count()
    reward = conut_to_reward(b, w, e)

    black_agent.reward(reward)
    white_agent.reward(reward)

    return (
        black_agent.get_buffer(),
        white_agent.get_buffer(),
        reward,
    )


if __name__ == "__main__":
    import time
    import torch
    from model import DQN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = DQN().to(device)

    tt = 0
    t = time.time()
    for i in range(100):
        black_agent = ModelAgent(Stone.BLACK, net, 0.5)
        white_agent = ModelAgent(Stone.WHITE, net, 0.5)
        _, _, r = auto_match(black_agent, white_agent)
        print(r)
    print(time.time() - t)
