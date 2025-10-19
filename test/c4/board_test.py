from c4.board import Board, Color

def test_board_legal_moves():
  board: Board = Board()

  assert board.legal_moves() == [0, 1, 2, 3, 4, 5, 6]

  board.make_move(Color.O, 0)
  board.make_move(Color.X, 0)
  board.make_move(Color.O, 0)
  board.make_move(Color.X, 0)
  board.make_move(Color.O, 0)
  board.make_move(Color.X, 0)

  assert board.legal_moves() == [1, 2, 3, 4, 5, 6]

def test_board_is_winning_blank():
  board: Board = Board()
  assert not board.is_winning(Color.O)

def test_board_is_winning_horizontal():
  input: str = """
. . . . . . .
. . . . . . .
. . . . . . .
. . . . . . .
. . . O O O O
"""  
  assert Board.from_string(input).is_winning(Color.O)
  
  input: str = """
. . . . . . .
. . . . . . .
. . . . . . .
. . . . . . .
O O O O . . .
"""  
  assert Board.from_string(input).is_winning(Color.O)

  
  input: str = """
. . . O O O O
. . . . . . .
. . . . . . .
. . . . . . .
. . . . . . .
"""  
  assert Board.from_string(input).is_winning(Color.O)

  
  input: str = """
O O O O . . .
. . . . . . .
. . . . . . .
. . . . . . .
. . . . . . .
"""  
  assert Board.from_string(input).is_winning(Color.O)


def test_board_is_winning_vertical():
  input: str = """
. . . . . . .
. . . . . . O
. . . . . . O
. . . . . . O
. . . . . . O
"""  
  assert Board.from_string(input).is_winning(Color.O)
  
  input: str = """
O . . . . . .
O . . . . . .
O . . . . . .
O . . . . . .
. . . . . . .
"""  
  assert Board.from_string(input).is_winning(Color.O)


def test_board_is_winning_diagonal_1():
  input: str = """
. . . . . . .
O . . . . . .
. O . . . . .
. . O . . . .
. . . O . . .
"""  
  assert Board.from_string(input).is_winning(Color.O)
  
  input: str = """
. . . O . . .
. . . . O . .
. . . . . O .
. . . . . . O
. . . . . . .
"""  
  assert Board.from_string(input).is_winning(Color.O)

def test_board_is_winning_diagonal_2():
  input: str = """
. . . O . . .
. . O . . . .
. O . . . . .
O . . . . . .
. . . . . . .
"""  
  assert Board.from_string(input).is_winning(Color.O)
  
  input: str = """
. . . . . . .
. . . . . . O
. . . . . O .
. . . . O . .
. . . O . . .
"""  
  assert Board.from_string(input).is_winning(Color.O)
  

def test_to_from_string():
  input: str = """
. . . . . . .
. . . . . . .
. . . . . . .
. . . . . O .
. . . X O X .
"""

  board: Board = Board.from_string(input)
  output: str = board.to_string()

  assert output.strip() == input.strip()
