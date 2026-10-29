import unittest

from src.game.board import create_board, place
from src.game.constants import AI, HUMAN, ROWS, COLS
from src.game.rules import valid_cols


def board_from(moves):
    board, token = create_board(), AI
    for col in moves:
        place(board, col, token)
        token = HUMAN if token == AI else AI
    return board, token


def pieces(board):
    return sum(cell != 0 for row in board for cell in row)


class RedValueTests(unittest.TestCase):
    def test_red_value_terminal_and_perspective(self):
        from src.viewer.analysis import red_value

        def refuse(_board, _player):
            raise AssertionError("terminal positions must not query the model")

        red_won, _ = board_from([0, 6, 1, 6, 2, 6, 3])
        yellow_won, _ = board_from([0, 6, 1, 6, 2, 6, 4, 6])
        self.assertEqual(red_value(red_won, HUMAN, refuse), 1.0)
        self.assertEqual(red_value(yellow_won, AI, refuse), -1.0)

        board = create_board()
        self.assertEqual(red_value(board, AI, lambda _b, _p: 0.25), 0.25)
        self.assertEqual(red_value(board, HUMAN, lambda _b, _p: 0.25), -0.25)


class CellHeatmapTests(unittest.TestCase):
    def test_cell_heatmap_shape_and_null_cells(self):
        from src.viewer.analysis import cell_heatmap

        board, to_move = board_from([3, 3])
        snapshot = [row[:] for row in board]
        cells = cell_heatmap(board, to_move, lambda b, _p: 0.01 * pieces(b))
        self.assertEqual(board, snapshot)
        self.assertEqual((len(cells), len(cells[0])), (ROWS, COLS))
        for row in range(ROWS):
            for col in range(COLS):
                occupied = board[row][col] != 0
                playable = row == (5 if col != 3 else 3)
                with self.subTest(row=row, col=col):
                    if occupied or playable:
                        self.assertIsInstance(cells[row][col], float)
                    else:
                        self.assertIsNone(cells[row][col])

    def test_cell_heatmap_signs_for_each_side_to_move(self):
        from src.viewer.analysis import cell_heatmap

        def value(board, _player):
            return 0.01 * pieces(board)

        # Red to move: base +0.02; removing a piece -> +0.01; playing -> Yellow to move at -0.03.
        board, to_move = board_from([3, 3])
        cells = cell_heatmap(board, to_move, value)
        self.assertAlmostEqual(cells[5][3], 0.01)
        self.assertAlmostEqual(cells[4][3], 0.01)
        self.assertAlmostEqual(cells[5][0], -0.05)
        self.assertAlmostEqual(cells[3][3], -0.05)

        # Yellow to move: base -0.03; removing a piece -> -0.02; playing -> Red to move at +0.04.
        board, to_move = board_from([3, 3, 4])
        cells = cell_heatmap(board, to_move, value)
        self.assertAlmostEqual(cells[5][4], -0.01)
        self.assertAlmostEqual(cells[5][0], 0.07)


def scripted(targets):
    """value_fn whose Red-perspective value after n pieces is targets[n]."""
    def value(board, player):
        red = targets[pieces(board)]
        return red if player == AI else -red
    return value


class AnalyzeGameTests(unittest.TestCase):
    def test_pivot_is_largest_red_swing(self):
        from src.viewer.analysis import analyze_game

        result = analyze_game([3, 4, 3, 4], scripted([0.0, 0.1, 0.2, -0.6, -0.5]))
        self.assertEqual(len(result["values"]), 5)
        self.assertEqual(result["pivot"], 2)
        self.assertEqual([p["token"] for p in result["plies"]], [AI, HUMAN, AI, HUMAN])
        self.assertAlmostEqual(result["plies"][2]["delta"], -0.8)
        self.assertIsNone(result["heatmaps"])
        self.assertIsNone(result["winner"])

    def test_side_to_move_bias_does_not_pick_the_pivot(self):
        from src.viewer.analysis import analyze_game

        # True Red value is flat, then jumps by 0.6 at move 5; the evaluator adds +0.4
        # with Red to move and -0.4 with Yellow to move, so raw deltas alternate ±0.8.
        biased = [0.4, -0.4, 0.4, -0.4, 0.4, 0.2, 1.0]
        result = analyze_game([0, 1, 2, 3, 4, 5], scripted(biased))
        self.assertIsNone(result["plies"][0]["swing"])
        self.assertAlmostEqual(result["plies"][4]["swing"], 0.6)
        self.assertEqual(result["pivot"], 4)

    def test_analyze_game_includes_heatmaps(self):
        from src.viewer.analysis import analyze_game

        result = analyze_game([3, 4], scripted([0.0, 0.1, 0.2, 0.3]), heatmaps=True)
        self.assertEqual(len(result["heatmaps"]), 3)
        self.assertTrue(all(len(h) == ROWS and len(h[0]) == COLS for h in result["heatmaps"]))

    def test_illegal_moves_raise(self):
        from src.viewer.analysis import analyze_game

        with self.assertRaises(ValueError):
            analyze_game([0] * 7, scripted([0.0] * 43))
        with self.assertRaises(ValueError):
            analyze_game([0, 6, 1, 6, 2, 6, 3, 5], scripted([0.0] * 43))
        with self.assertRaises(ValueError):
            analyze_game([9], scripted([0.0] * 43))

    def test_winner_reported(self):
        from src.viewer.analysis import analyze_game

        result = analyze_game([0, 6, 1, 6, 2, 6, 3], scripted([0.0] * 43))
        self.assertEqual(result["winner"], AI)
        self.assertEqual(result["values"][-1], 1.0)


class SelfPlayReplayTests(unittest.TestCase):
    def test_self_play_replay_is_legal_and_terminal(self):
        from src.viewer.analysis import self_play_replay
        from src.game.rules import check_win, is_full

        replay = self_play_replay(lambda _b, _p: 0.0)
        board, token = create_board(), AI
        for col in replay["moves"]:
            self.assertIn(col, valid_cols(board))
            place(board, col, token)
            token = HUMAN if token == AI else AI
        self.assertTrue(check_win(board, AI) or check_win(board, HUMAN) or is_full(board))
        self.assertEqual(len(replay["values"]), len(replay["moves"]) + 1)

    def test_heuristic_value_fn_is_bounded(self):
        from src.viewer.analysis import heuristic_value_fn

        board, to_move = board_from([3, 3, 4])
        value = heuristic_value_fn()
        self.assertEqual(to_move, HUMAN)
        # Red holds two connected bottom pieces, so Yellow-to-move is worse than Red-to-move.
        self.assertTrue(-1.0 < value(board, HUMAN) < 0.0)
        self.assertTrue(0.0 < value(board, AI) < 1.0)
