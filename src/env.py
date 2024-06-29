from enum import IntEnum
from copy import deepcopy


class Stone(IntEnum):
    BLACK = 1
    WHITE = -1
    EMPTY = 0


def flip(stone: Stone):
    return Stone(stone * -1)


def out_of_board(x, y):
    return not (0 <= x < 8 and 0 <= y < 8)


class Board:
    def __init__(self) -> None:
        b = Stone.BLACK
        w = Stone.WHITE
        e = Stone.EMPTY

        self.board = [
            [e, e, e, e, e, e, e, e],
            [e, e, e, e, e, e, e, e],
            [e, e, e, e, e, e, e, e],
            [e, e, e, w, b, e, e, e],
            [e, e, e, b, w, e, e, e],
            [e, e, e, e, e, e, e, e],
            [e, e, e, e, e, e, e, e],
            [e, e, e, e, e, e, e, e],
        ]

        self.dirs = [
            (-1, -1),
            (0, -1),
            (1, -1),
            (-1, 0),
            (1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
        ]

    def __str__(self) -> str:
        c = {Stone.BLACK: "○", Stone.WHITE: "●", Stone.EMPTY: "*"}
        s = "\n".join([" ".join([c[stone] for stone in row]) for row in self.board])
        return s

    def get_board(self):
        return deepcopy(self.board)

    def get_actions(self, stone: Stone) -> list[tuple[int, int]]:
        actions = []
        for y in range(len(self.board)):
            for x in range(len(self.board[y])):
                if len(self.get_flip_dir(stone, (x, y))) > 0:
                    actions.append((x, y))
        return actions

    def get_flip_dir(
        self, stone: Stone, action: tuple[int, int]
    ) -> list[tuple[int, int]]:
        x, y = action
        dirs = []

        actor = stone
        opp = flip(stone)

        if self.board[y][x] != Stone.EMPTY:
            return dirs

        for dx, dy in self.dirs:
            flippable = False
            for i in range(1, 8):
                tx = x + dx * i
                ty = y + dy * i

                if out_of_board(tx, ty):
                    break

                s = self.board[ty][tx]
                if s == actor:
                    if flippable:
                        dirs.append((dx, dy))
                    break
                elif s == opp:
                    flippable = True
                else:
                    break
        return dirs

    def act(self, stone: Stone, action: tuple[int, int]):
        dirs = self.get_flip_dir(stone, action)
        x, y = action

        self.board[y][x] = stone
        for dx, dy in dirs:
            for i in range(1, 8):
                tx = x + dx * i
                ty = y + dy * i

                if out_of_board(tx, ty):
                    break

                if self.board[ty][tx] == stone:
                    break
                self.board[ty][tx] = stone

    def get_count(self) -> tuple[int, int, int]:
        b = 0
        w = 0
        e = 0
        for row in self.board:
            for stone in row:
                if stone == Stone.BLACK:
                    b += 1
                elif stone == Stone.WHITE:
                    w += 1
                else:
                    e += 1
        return b, w, e


if __name__ == "__main__":
    import random, time

    t = time.time()

    for game in range(10000):
        board = Board()
        turn = Stone.BLACK
        passed = False
        while True:
            # print(board)
            actions = board.get_actions(turn)
            if len(actions) == 0:
                turn = flip(turn)
                if passed:
                    break
                passed = True
                continue
            else:
                passed = False
            x, y = actions[random.randrange(0, len(actions))]
            board.act(turn, (x, y))
            turn = flip(turn)

    print(time.time() - t)
