# import matplotlib.pyplot as plt
# import numpy as np
# import 开发版
# import 稳定版8
#
#
# def plot(board: np.ndarray):
#     h, w = board.shape
#     radius = 1. / h / 2
#     for ii in range(board.shape[0]):
#         for jj in range(board.shape[1]):
#             color = (1, 0, 0) if board[ii][jj] == 1 else (0, 0, 1) if board[ii][jj] == -1 else (1, 1, 1)
#             circle = plt.Circle((ii * radius * 2 + radius, jj * radius * 2 + radius), radius, color=color)
#             plt.gca().add_artist(circle)
#
#
# plt.ion()
#
# chessboard = np.array([
#     [0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, -1, 1, 0, 0, 0],
#     [0, 0, 0, 1, -1, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0]
# ])
# while True:
#     # do something
#
#     chessboard = 开发版.AI.go(chessboard)
#     plot(chessboard)
#     plt.pause(0.1)
#
# # end of all iterations
# plt.show()
