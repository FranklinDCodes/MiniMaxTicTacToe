
import numpy as np


# IF WANTED TO BATCH WOULD NEED TO CHANGE FOR LOOP IN MINIMAX
# WOULD HAVE TO BREAK DOWN RECURSIVE CALL INTO LIST COMP THO AND STACK RESULTS


# action object type
class Action(int):

    """
        Each action is represented by an integer
        The action object when indexed gives the board coordinates of the placement
        When iter is called, the board coordinates are passed
        When string is called, the name of the action is given
        The board looks as follows, with action numbers on the squares:

          | 0 | 1 | 2 |
        --+---+---+---+
        0 | 1 | 2 | 3 |
        --+---+---+---+
        1 | 4 | 5 | 6 |
        --+---+---+---+
        2 | 7 | 8 | 9 |
        --+---+---+---+

    """

    ACTION_NAMES = [
        "top left",
        "top center",
        "top right",
        "middle left",
        "middle center",
        "middle right",
        "bottom left",
        "bottom center",
        "bottom right"
    ]

    def __init__(self, arg):

        # has int init and coordinate init options
        if isinstance(arg, tuple):
            super().__init__(
                (arg[0] * 3) + (arg[1] + 1)
            )

        else:
            super().__init__(arg)

    def __iter__(self):

        return (
            (self - 1) // 3,    # row calc
            (self - 1) % 3      # col calc
        )

    def __getitiem__(self, idx):

        return self.__iter__()[idx]

    def __len__(self):
        return 2

    def __str__(self):
        return self.ACTION_NAMES[self - 1]


# state object type
class GameBoard(np.ndarray):

    """
        x's are 1, o's are 2, empty spaces are 0
        the x will always represent the agent
    """

    def __init__(self):

        # default init with empty board
        super().__init__([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype='uint8')

    def __str__(self):
        return ""
    

class TicTacToe:

    @staticmethod
    def state_transition(state: GameBoard, action: Action, player: int = 1) -> GameBoard:

        """
            state_transition(s, a) -> s'
            player is expected to be an int: 1 or 2
            this function does NOT check if a is possible action
        """

        state_prime = state.copy()

        state_prime[action[0], action[1]] = player

        return state_prime

    @staticmethod
    def _rotate_45(A):

        """
            intended to do the following transformation
            that rotates the diagonal to the center row:

            1 2 3    4 1 2
            4 5 6 -> 7 5 3
            7 8 9    8 9 6
        """

        R = A.copy()

        # top row
        R[..., 0, 0] = A[..., 1, 0]
        R[..., 0, 1:3] = A[..., 0, 0:2]

        # middle row
        R[..., 1, 0] = A[..., 2, 0]
        R[..., 1, 1] = A[..., 1, 1]
        R[..., 1, 2] = A[..., 0, 2]

        # bottom row
        R[..., 2, 0:2] = A[..., 2, 1:3]
        R[..., 2, 2] = A[..., 1, 2] 

        return R

    @staticmethod
    def _unrotate_45(A):

        """
            intended to do the following transformation
            that rotates the diagonal to the center row:

            1 2 3    4 1 2
            4 5 6 -> 7 5 3
            7 8 9    8 9 6
        """

        R = A.copy()

        # top row
        R[..., 1, 0] = A[..., 0, 0]
        R[..., 0, 0:2] = A[..., 0, 1:3]

        # middle row
        R[..., 2, 0] = A[..., 1, 0]
        R[..., 1, 1] = A[..., 1, 1]
        R[..., 0, 2] = A[..., 1, 2]

        # bottom row
        R[..., 2, 1:3] = A[..., 2, 0:2]
        R[..., 1, 2] = A[..., 2, 2]

        return R


    @staticmethod
    def possible_actions(state: np.ndarray[GameBoard]):

        # make 4 versions of each board:
        # standard, -45deg, 45deg, 90deg
        np_state_4 = np.repeat(state[..., np.newaxis, :, :], 4, axis=-3)  # (..., 4, 3, 3)

        # -45deg version puts the top-left to bottom-right diagonal as the center row
        np_state_4[..., 1, :, :] = TicTacToe._unrotate_45(np_state_4[..., 1, :, :])

        # 45deg puts the other diagonal as the center row
        np_state_4[..., 2, :, :] = TicTacToe._rotate_45(np_state_4[..., 2, :, :])

        # 90deg puts the columns as rows
        np_state_4[..., 3, :, :] = np.swapaxes(np_state_4, -2, -1)

        # check if any have all moves on middle of board
        np_on_middle = np.sum(np_state_4, axis=(-2, -1)) == np.sum(np_state_4[..., 1, :], axis=-1)



        # make sure to prune indistinguishable actions
        pass

    @staticmethod
    def reward():
        pass


