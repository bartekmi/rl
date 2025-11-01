from c4.ttt_board import TttBoard, Color

def test_board_legal_moves():
  board: TttBoard = TttBoard()

  assert board.legal_moves() == [0, 1, 2, 3, 4, 5, 6, 7, 8]

  board.make_move(Color.O, 0)
  board.make_move(Color.X, 4)
  board.make_move(Color.O, 8)

  assert board.legal_moves() == [1, 2, 3, 5, 6, 7]

def test_board_is_winning_blank():
  board: TttBoard = TttBoard()
  assert not board.is_winning(Color.O)

def test_board_is_winning_horizontal():
  input: str = """
. . .
O O O
. . .
"""  
  assert TttBoard.from_string(input).is_winning(Color.O)

def test_board_is_winning_vertical():
  input: str = """
. . X
. . X
. . X
"""  
  assert TttBoard.from_string(input).is_winning(Color.X)

def test_board_is_winning_diag_1():
  input: str = """
X . .
. X .
. . X
"""  
  assert TttBoard.from_string(input).is_winning(Color.X)

def test_board_is_winning_diag_2():
  input: str = """
. . X
. X .
X . .
"""  
  assert TttBoard.from_string(input).is_winning(Color.X)

def test_missed_win_singe():
  input: str = """
. . .
O O X
X O X
"""  
  board: TttBoard = TttBoard.from_string(input)
  assert board.missed_win(0)
  assert board.missed_win(2)

  assert not board.missed_win(1)


def test_missed_win_multiple():
  input: str = """
. . .
X O X
X O O
"""  
  board: TttBoard = TttBoard.from_string(input)
  assert board.missed_win(2)

  assert not board.missed_win(0)
  assert not board.missed_win(1)

def test_failed_to_block_single_threat():
  input: str = """
. . .
O X X
X O O
"""  
  board: TttBoard = TttBoard.from_string(input)
  assert board.failed_to_block(0)
  assert board.failed_to_block(1)

  assert not board.failed_to_block(2)

def test_failed_to_block_multi_threat():
  input: str = """
. . .
X X .
X . .
"""  
  board: TttBoard = TttBoard.from_string(input)
  for ii in range(9):
    assert not board.failed_to_block(ii)
