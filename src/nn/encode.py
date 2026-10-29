"""
encode.py — board → tensor representation for the neural network.

Output shape: (3, ROWS, COLS) = (3, 6, 7), dtype float32, no batch dimension.
  Channel 0: 1.0 where *token* pieces are, 0.0 elsewhere
  Channel 1: 1.0 where opponent pieces are, 0.0 elsewhere
  Channel 2: all 1.0 if token == AI, all 0.0 if token == HUMAN
              (whose colour is to move; colour-blind nets, the default,
              ignore it: see network.py)
"""
import torch
from ..game.constants import ROWS, COLS, AI, HUMAN


def encode_board(board, token):
    """
    Encode `board` from `token`'s perspective into a (3, ROWS, COLS) float32 tensor.

    Parameters
    ----------
    board : 2-D list[list[int]]  — current board state (row 0 = top)
    token : int                  — AI or HUMAN; whose perspective to encode

    Returns
    -------
    torch.Tensor of shape (3, ROWS, COLS), dtype=torch.float32
    """
    opponent = HUMAN if token == AI else AI

    plane_token    = [[1.0 if board[r][c] == token    else 0.0 for c in range(COLS)]
                      for r in range(ROWS)]
    plane_opponent = [[1.0 if board[r][c] == opponent else 0.0 for c in range(COLS)]
                      for r in range(ROWS)]
    plane_turn     = [[1.0 if token == AI              else 0.0 for c in range(COLS)]
                      for r in range(ROWS)]

    tensor = torch.tensor(
        [plane_token, plane_opponent, plane_turn],
        dtype=torch.float32,
    )
    return tensor  # shape: (3, 6, 7)
