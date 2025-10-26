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


def test_needs_blocking_no_room():
  input: str = """
. . . . . O .
. . . . . O .
. . . . . O .
. . . O . O .
. . . O . X .
"""

  board: Board = Board.from_string(input)
  assert board.needs_blocking(5, Color.X) == False
  assert board.needs_blocking(3, Color.X) == False
  assert board.failing_to_block_column(0, Color.X) == False

def test_needs_blocking_single():
  input: str = """
. . . . . . .
. . . . . . .
. . . . . O .
. . . . . O .
. . . . . O .
"""

  board: Board = Board.from_string(input)
  assert board.needs_blocking(5, Color.X) == True
  assert board.failing_to_block_column(0, Color.X) == True

def test_needs_blocking_multiple():
  input: str = """
. . . . . . .
. . . . . . O
. . . . . O O
. . . . . O O
. . . . . O X
"""

  board: Board = Board.from_string(input)
  assert board.needs_blocking(5, Color.X) == True
  assert board.needs_blocking(6, Color.X) == True

  assert board.failing_to_block_column(0, Color.X) == True
  assert board.failing_to_block_column(5, Color.X) == False
  assert board.failing_to_block_column(6, Color.X) == False
  
