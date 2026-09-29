
from env import *
from minimax import MinimaxSearch
import random


def main():

    random.seed(42)

    searcher = MinimaxSearch(
        TicTacToe.state_transition,
        TicTacToe.unique_possible_actions,
        TicTacToe.reward,
        TicTacToe.game_over
    )

    # play game
    game_num = 0
    while True:

        np_state = np.array([[0 for i in range(3)] for i in range(3)])

        # if game num is odd, agent goes first
        if game_num % 2 == 1:

            print_state(np_state)
            print()

            # apply agent choice
            agent_choice, _ = searcher(np_state)
            np_state[agent_choice[0], agent_choice[1]] = 1

        while not TicTacToe.game_over(np_state):

            print_state(np_state)
            print()

            # user input loop
            valid_input = False
            while not valid_input:

                # input
                user_choice_1 = input("Select row: ")
                user_choice_2 = input("Select col: ")

                # apply user choice
                try:

                    # try to convert to int
                    int_choice_1 = int(user_choice_1) - 1
                    int_choice_2 = int(user_choice_2) - 1

                    # check if possible
                    if not int(Action((int_choice_1, int_choice_2))) in [int(i) for i in TicTacToe.possible_actions(np_state)]:
                        valid_input = False
                        print("Please enter values for an empty cell.\n")
                    else:
                        valid_input = True
                        np_state[int_choice_1, int_choice_2] = 2
                except (ValueError, IndexError):
                    print("Please enter values for an empty cell in rows 1-3 and cols 1-3.\n")

            print()
            print_state(np_state)
            print()

            if not TicTacToe.game_over(np_state):

                # apply agent choice
                agent_choice, _ = searcher(np_state)
                np_state[agent_choice[0], agent_choice[1]] = 1

            print()

        print_state(np_state)
        print("GAME OVER")
        if TicTacToe._did_win(np_state, 1):
            print("AGENT WINS")
        elif TicTacToe._did_win(np_state, 2):
            print("PLAYER WINS")
        else:
            print("CATS GAME")
        print()

        game_num += 1


if __name__ == "__main__":
    main()
