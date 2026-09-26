# from src.env import TicTacToe
# import numpy as np

# a = np.array([[
#     [1, 2, 3],
#     [4, 5, 6],
#     [7, 8, 9]
# ]])


# # print(np.swapaxes(a, -2, -1))

# b = np.array([[
#     [0, 0, 0],
#     [1, 1, 1],
#     [0, 0, 0]
# ],
# [
#     [0, 0, 1],
#     [1, 1, 1],
#     [0, 0, 0]
# ]
# ])


# print(np.sum(b, axis=(-2, -1)) == np.sum(b[..., 1, :], axis=-1))
# print()

from env import *


state = GameBoard([
    [1, 0, 0],
    [0, 0, 0],
    [2, 0, 1]
])