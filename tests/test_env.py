
import unittest

import numpy as np
import sys
from pathlib import Path


# get the directory
project_root = Path(__file__).resolve().parent.parent

# add the parent directory to Python's module search path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


from src.env import *


class TestEnv(unittest.TestCase):

    def test_possible_actions(self):

        # vanilla cases

        state = np.array([
            [1, 0, 0],
            [0, 0, 0],
            [2, 0, 1]
        ])
        expected = np.array([2, 3, 4, 5, 6, 8])
        actual = TicTacToe.possible_actions(state)
        np.testing.assert_equal(actual, expected)

        state = np.array([
            [1, 0, 1],
            [0, 1, 0],
            [2, 0, 1]
        ])
        expected = np.array([2, 4, 6, 8])
        actual = TicTacToe.possible_actions(state)
        np.testing.assert_equal(actual, expected)

        state = np.array([
            [1, 1, 1],
            [2, 1, 2],
            [2, 0, 1]
        ])
        expected = np.array([8])
        actual = TicTacToe.possible_actions(state)
        np.testing.assert_equal(actual, expected)

        state = np.array([
            [1, 1, 1],
            [2, 1, 2],
            [2, 2, 1]
        ])
        expected = np.array([])
        actual = TicTacToe.possible_actions(state)
        np.testing.assert_equal(actual, expected)

        # cases with redundancies

        state = np.array([
            [0, 1, 0],
            [0, 1, 0],
            [0, 2, 0]
        ])
        expected = np.array([1, 4, 7])
        actual = TicTacToe.possible_actions(state)
        np.testing.assert_equal(actual, expected)
        
        state = np.array([
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 0]
        ])
        expected = np.array([1, 4, 7, 8])
        actual = TicTacToe.possible_actions(state)
        np.testing.assert_equal(actual, expected)

        state = np.array([
            [0, 0, 0],
            [2, 1, 1],
            [0, 0, 0]
        ])
        expected = np.array([1, 2, 3])
        actual = TicTacToe.possible_actions(state)
        np.testing.assert_equal(actual, expected)

        state = np.array([
            [0, 0, 0],
            [2, 1, 0],
            [0, 0, 0]
        ])
        expected = np.array([1, 2, 3, 6])
        actual = TicTacToe.possible_actions(state)
        np.testing.assert_equal(actual, expected)
        
        state = np.array([
            [0, 0, 2],
            [0, 1, 0],
            [2, 0, 0]
        ])
        expected = np.array([4, 1, 2])
        actual = TicTacToe.possible_actions(state)
        np.testing.assert_equal(actual, expected)

        state = np.array([
            [0, 0, 2],
            [0, 1, 0],
            [0, 0, 0]
        ])
        expected = np.array([4, 1, 2, 7])
        actual = TicTacToe.possible_actions(state)
        np.testing.assert_equal(actual, expected)
                
        state = np.array([
            [2, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        expected = np.array([2, 3, 6])
        actual = TicTacToe.possible_actions(state)
        np.testing.assert_equal(actual, expected)

        state = np.array([
            [2, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ])
        expected = np.array([2, 3, 6, 9])
        actual = TicTacToe.possible_actions(state)
        np.testing.assert_equal(actual, expected)

        state = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ])
        expected = np.array([1, 2])
        actual = TicTacToe.possible_actions(state)
        np.testing.assert_equal(actual, expected)

        state = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ])
        expected = np.array([1, 2, 5])
        actual = TicTacToe.possible_actions(state)
        np.testing.assert_equal(actual, expected)

    def test_did_win(self):

        state = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [2, 0, 1]
        ])
        expected = True
        actual = TicTacToe._did_win(state, 1)
        self.assertIs(expected, actual)

        state = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [2, 0, 1]
        ])
        expected = False
        actual = TicTacToe._did_win(state, 2)
        self.assertIs(expected, actual)
        
        state = np.array([
            [1, 1, 0],
            [0, 1, 0],
            [2, 1, 2]
        ])
        expected = True
        actual = TicTacToe._did_win(state, 1)
        self.assertIs(expected, actual)
        
        state = np.array([
            [1, 1, 0],
            [0, 1, 0],
            [2, 2, 2]
        ])
        expected = True
        actual = TicTacToe._did_win(state, 2)
        self.assertIs(expected, actual)
        
        state = np.array([
            [1, 0, 0],
            [0, 0, 0],
            [2, 0, 2]
        ])
        expected = False
        actual = TicTacToe._did_win(state, 2)
        self.assertIs(expected, actual)
        
        state = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [2, 0, 2]
        ])
        expected = False
        actual = TicTacToe._did_win(state, 1)
        self.assertIs(expected, actual)
        
        state = np.array([
            [1, 2, 2],
            [0, 1, 2],
            [2, 0, 0]
        ])
        expected = False
        actual = TicTacToe._did_win(state, 2)
        self.assertIs(expected, actual)
        
        state = np.array([
            [1, 2, 1],
            [2, 1, 2],
            [2, 1, 2]
        ])
        expected = False
        actual = TicTacToe._did_win(state, 1)
        self.assertIs(expected, actual)
                
        state = np.array([
            [1, 1, 1],
            [2, 1, 2],
            [2, 1, 2]
        ])
        expected = True
        actual = TicTacToe._did_win(state, 1)
        self.assertIs(expected, actual)
                
        state = np.array([
            [2, 0, 1],
            [2, 0, 1],
            [2, 0, 1]
        ])
        expected = True
        actual = TicTacToe._did_win(state, 1)
        self.assertIs(expected, actual)
                        
        state = np.array([
            [2, 0, 1],
            [2, 0, 1],
            [2, 0, 1]
        ])
        expected = True
        actual = TicTacToe._did_win(state, 2)
        self.assertIs(expected, actual)

    def test_reward(self):

        state = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [2, 0, 1]
        ])
        expected = 1
        actual = TicTacToe.reward(state, 1)
        self.assertEqual(expected, actual)

        state = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [2, 0, 1]
        ])
        expected = -1
        actual = TicTacToe.reward(state, 2)
        self.assertEqual(expected, actual)
        
        state = np.array([
            [1, 0, 0],
            [0, 0, 0],
            [2, 2, 1]
        ])
        expected = 0
        actual = TicTacToe.reward(state, 1)
        self.assertEqual(expected, actual)

        state = np.array([
            [1, 0, 0],
            [0, 0, 0],
            [2, 2, 1]
        ])
        expected = 0
        actual = TicTacToe.reward(state, 2)
        self.assertEqual(expected, actual)

        state = np.array([
            [1, 0, 0],
            [0, 0, 0],
            [2, 2, 2]
        ])
        expected = 1
        actual = TicTacToe.reward(state, 2)
        self.assertEqual(expected, actual)

        state = np.array([
            [1, 0, 1],
            [0, 0, 1],
            [2, 2, 1]
        ])
        expected = 1
        actual = TicTacToe.reward(state, 1)
        self.assertEqual(expected, actual)
        
        state = np.array([
            [1, 0, 1],
            [0, 0, 1],
            [2, 2, 1]
        ])
        expected = -1
        actual = TicTacToe.reward(state, 2)
        self.assertEqual(expected, actual)
        
        state = np.array([
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ])
        expected = 0
        actual = TicTacToe.reward(state, 1)
        self.assertEqual(expected, actual)
        
        state = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ])
        expected = 0
        actual = TicTacToe.reward(state, 2)
        self.assertEqual(expected, actual)

if __name__ == "__main__":

    unittest.main()
