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
