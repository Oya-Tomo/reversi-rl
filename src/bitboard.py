from enum import IntEnum


class Stone(IntEnum):
    BLACK = 1
    WHITE = -1
    EMPTY = 0


def flip(stone: Stone):
    return Stone(stone * -1)


def bit_shift(target: int, shift: int) -> int:
    if shift > 0:
        return target << shift
    else:
        return target >> -shift


class Board:
    def __init__(self) -> None:
        self.black_board = 0
        self.white_board = 0

        self.black_board |= 1 << 28
        self.black_board |= 1 << 35
        self.white_board |= 1 << 27
        self.white_board |= 1 << 36

        self.shifts = [9, 8, 7, 1, -1, -7, -8, -9]

        self.V_TRIM_MASK = 0x00FFFFFFFFFFFF00  # vertical side
        self.H_TRIM_MASK = 0x7E7E7E7E7E7E7E7E  # horizontal side
        self.A_TRIM_MASK = 0x007E7E7E7E7E7E00  # all side

    def __str__(self) -> str:
        board = self.get_board()
        char = {Stone.BLACK: "○", Stone.WHITE: "●", Stone.EMPTY: "*"}
        s = "\n".join([" ".join([char[cell] for cell in row]) for row in board])
        return s

    def get_board(self) -> list[list[Stone]]:
        board = [[Stone.EMPTY for _1 in range(8)] for _2 in range(8)]
        for i in range(64):
            x = i % 8
            y = i // 8
            if self.black_board & (1 << i):
                board[y][x] = Stone.BLACK
            if self.white_board & (1 << i):
                board[y][x] = Stone.WHITE
        return board

    def get_actions(self, stone: Stone) -> list[tuple[int, int]]:
        board = self.get_legal_board(stone)
        actions = []
        for i in range(64):
            x = i % 8
            y = i // 8
            if board & (1 << i):
                actions.append((x, y))
        return actions

    def act(self, stone: Stone, action: tuple[int, int]):
        actor_board = self.black_board if stone == Stone.BLACK else self.white_board
        oppnt_board = self.white_board if stone == Stone.BLACK else self.black_board

        x, y = action
        action_board = 1 << (y * 8 + x)
        reverse_board = 0
        for shift in self.shifts:
            cr = 0
            for i in range(1, 8):
                mask = self._get_rev_mask(action_board, shift * i)
                if mask == 0:
                    break

                if oppnt_board & mask:
                    cr |= mask
                elif actor_board & mask:
                    reverse_board |= cr
                    break
                else:
                    break
        actor_board ^= reverse_board | action_board
        oppnt_board ^= reverse_board

        if stone == Stone.BLACK:
            self.black_board = actor_board
            self.white_board = oppnt_board
        else:
            self.black_board = oppnt_board
            self.white_board = actor_board

    def get_count(self):
        b = self.black_board.bit_count()
        w = self.white_board.bit_count()
        e = 64 - b - w
        return b, w, e

    def _get_rev_mask(self, a, shift) -> int:
        return bit_shift(a, shift) & 0xFFFFFFFFFFFFFFFF

    def get_legal_board(self, stone: Stone) -> int:
        actor_board = self.black_board if stone == Stone.BLACK else self.white_board
        oppnt_board = self.white_board if stone == Stone.BLACK else self.black_board

        v_trim_board = oppnt_board & self.V_TRIM_MASK
        h_trim_board = oppnt_board & self.H_TRIM_MASK
        a_trim_board = oppnt_board & self.A_TRIM_MASK
        blank_board = ~(self.black_board | self.white_board)

        legal_board = 0x0000000000000000

        for shift in self.shifts:
            if abs(shift) == 1:
                tmp = h_trim_board & bit_shift(actor_board, shift)
                tmp |= h_trim_board & bit_shift(tmp, shift)
                tmp |= h_trim_board & bit_shift(tmp, shift)
                tmp |= h_trim_board & bit_shift(tmp, shift)
                tmp |= h_trim_board & bit_shift(tmp, shift)
                tmp |= h_trim_board & bit_shift(tmp, shift)
                legal_board |= blank_board & bit_shift(tmp, shift)
            elif abs(shift) == 8:
                tmp = v_trim_board & bit_shift(actor_board, shift)
                tmp |= v_trim_board & bit_shift(tmp, shift)
                tmp |= v_trim_board & bit_shift(tmp, shift)
                tmp |= v_trim_board & bit_shift(tmp, shift)
                tmp |= v_trim_board & bit_shift(tmp, shift)
                tmp |= v_trim_board & bit_shift(tmp, shift)
                legal_board |= blank_board & bit_shift(tmp, shift)
            else:
                tmp = a_trim_board & bit_shift(actor_board, shift)
                tmp |= a_trim_board & bit_shift(tmp, shift)
                tmp |= a_trim_board & bit_shift(tmp, shift)
                tmp |= a_trim_board & bit_shift(tmp, shift)
                tmp |= a_trim_board & bit_shift(tmp, shift)
                tmp |= a_trim_board & bit_shift(tmp, shift)
                legal_board |= blank_board & bit_shift(tmp, shift)
        return legal_board


def print_legal_board(board: int):
    for r in range(0, 64, 8):
        for i in range(r, r + 8):
            if board & (1 << i):
                print("+ ", end="")
            else:
                print("☓ ", end="")
        print()


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
