
import numpy as np


# action object type
class Action:

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

        if isinstance(arg, int):
            self._num = arg
        elif isinstance(arg, tuple) and len(arg) == 2:
            self._num = arg[0] * 3 + (arg[1] + 1)
        else:
            raise ValueError("Action must be initialized with 2-element tuple or int")

    def __iter__(self):

        return (
            (int(self - 1)) // 3,    # row calc
            (int(self - 1)) % 3      # col calc
        )

    def __getitem__(self, idx):

        return self.__iter__()[idx]

    def __len__(self):
        return 2

    def __int__(self):
        return self._num

    def __float__(self):
        return self._num
    
    def __str__(self):
        return self.ACTION_NAMES[int(self - 1)]

    def __add__(self, other):
        if isinstance(other, Action):
            return Action(self._num + other._num)
        elif isinstance(other, int):
            return Action(self._num + other)
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Action):
            return Action(self._num - other._num)
        elif isinstance(other, int):
            return Action(self._num - other)
        return NotImplemented
        
    def __mul__(self, other):
        if isinstance(other, Action):
            return Action(self._num * other._num)
        elif isinstance(other, int):
            return Action(self._num * other)
        return NotImplemented
        
    def __truediv__(self, other):
        if isinstance(other, Action):
            return Action(self._num / other._num)
        elif isinstance(other, int):
            return Action(self._num / other)
        return NotImplemented


class TicTacToe:

    @staticmethod
    def state_transition(state: np.ndarray, action: Action, player: int = 1) -> np.ndarray:

        """
            state_transition(s, a) -> s'
            player is expected to be an int: 1 or 2
            this function does NOT check if a is possible action
        """

        state_prime = state.copy()

        state_prime[action[0], action[1]] = player

        return state_prime

    @staticmethod
    def _rotate_45(A: np.ndarray) -> np.ndarray:

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
    def _unrotate_45(A: np.ndarray) -> np.ndarray:

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
    def unique_possible_actions(state: np.ndarray) -> list[Action]:

        """
            returns a list of possible actions

            It turns out, if all of the placed moves align (row, column, or diagonal),
            then all of the options on one side of the alignment are exactly like the other
            So this function rotates the board 4 different ways and if the middle row contains all
            the information, then returns all the options on just one side of the alignment
        """

        # make 4 versions of each board:
        # standard, -45deg, 45deg, 90deg
        np_state_4 = np.repeat(state[np.newaxis, :, :], 4, axis=-3)  # (4, 3, 3)

        # -45deg version puts the top-left to bottom-right diagonal as the center row
        np_state_4[1, :, :] = TicTacToe._unrotate_45(np_state_4[1, :, :])

        # 45deg puts the other diagonal as the center row
        np_state_4[2, :, :] = TicTacToe._rotate_45(np_state_4[2, :, :])

        # 90deg puts the columns as rows
        np_state_4[3, :, :] = np.swapaxes(np_state_4[3, :, :], -2, -1)

        # check if any have all moves on middle of board
        np_on_middle = np.sum(np_state_4, axis=(-2, -1)) == np.sum(np_state_4[:, 1, :], axis=-1)

        # if all are indistinguishable
        if np.all(np_on_middle):

            # it means middle is only occupied or emptied
            if state[1, 1] != 0:

                return [Action(1), Action(2)]

            else:

                return [Action(1), Action(2), Action(5)]

        # check if any are indistinguishable
        elif np.any(np_on_middle):

            # put all options on a board
            np_all_possible = np.arange(1, 10).reshape(3, 3)

            # repeat and rotate like the others
            np_all_possible = np.repeat(np_all_possible[np.newaxis, :, :], 4, axis=-3)  # (4, 3, 3)
            np_all_possible[1, :, :] = TicTacToe._unrotate_45(np_all_possible[1, :, :])
            np_all_possible[2, :, :] = TicTacToe._rotate_45(np_all_possible[2, :, :])
            np_all_possible[3, :, :] = np.swapaxes(np_all_possible[3, :, :], -2, -1)

            # grab the rotation that reveals the redundancy
            # if multiple rotations do, just grab the first one
            np_rotated_actions = np_all_possible[np_on_middle][0]
            np_rotated_state = np_state_4[np_on_middle][0]

            # grab any missing actions on the centerline
            np_empty_on_centerline = (np_rotated_state[1, :] == 0)
            np_add_from_centerline = np_rotated_actions[1, np_empty_on_centerline]

            # take the top row as possible actions
            return list(map(Action, np.concatenate((np_rotated_actions[0], np_add_from_centerline)).tolist()))

        else:

            # mask out taken spaces
            np_occupied = state != 0

            # get available
            np_all_possible = np.arange(1, 10).reshape(3, 3)
            np_possible_actions = np_all_possible[~np_occupied]

            return list(map(Action, np_possible_actions.tolist()))

    @staticmethod
    def possible_actions(state: np.ndarray) -> list[Action]:

        """
            returns a list of possible actions

        """

        # mask out taken spaces
        np_occupied = state != 0

        # get available
        np_all_possible = np.arange(1, 10).reshape(3, 3)
        np_possible_actions = np_all_possible[~np_occupied]

        return [Action(i) for i in np_possible_actions.tolist()]

    @staticmethod
    def _did_win(state: np.array, player: int) -> bool:

        np_player_state = (state == player)

        # make 4 versions of each board:
        # standard, -45deg, 45deg, 90deg
        np_state_4 = np.repeat(np_player_state[np.newaxis, :, :], 4, axis=-3)  # (4, 3, 3)
        
        # 90deg puts the columns as rows
        np_state_4[1, :, :] = np.swapaxes(np_state_4[1, :, :], -2, -1)

        # -45deg version puts the top-left to bottom-right diagonal as the center row
        np_state_4[2, :, :] = TicTacToe._unrotate_45(np_state_4[2, :, :])

        # 45deg puts the other diagonal as the center row
        np_state_4[3, :, :] = TicTacToe._rotate_45(np_state_4[3, :, :])

        # check for a straight win
        if np.any(np_state_4[0:2, :, :].sum(axis=-1) == 3):

            return True

        # check for diagonal win
        # sum middle rows for the last 2 rotations and see if they add up to 3
        if np.any(np_state_4[2:, 1, :].sum(axis=-1) == 3):

            return True

        return False

    @staticmethod
    def reward(state: np.ndarray, player: int = 1) -> int:

        # check if player won
        if TicTacToe._did_win(state, player):

            return 1

        # check if other player won
        if TicTacToe._did_win(state, [2, 1][player - 1]):

            return -1

        return 0

    @staticmethod
    def game_over(state: np.ndarray) -> bool:

        return TicTacToe._did_win(state, 1) or TicTacToe._did_win(state, 2) or len(TicTacToe.possible_actions(state)) == 0


def print_state(state: np.ndarray) -> None:

    print("+---+---+---+")

    for row in state:

        print("| ", end="")
        print(*[[' ','X','O'][i] for i in row], sep=" | ", end="")
        print(" |")
        print("+---+---+---+")

