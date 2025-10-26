import numpy as np
from enum import Enum
from typing import List

class Color(Enum):
  NONE = 0
  O = 1
  X = -1

  @staticmethod
  def opposite(color: "Color") -> "Color":
      if color == Color.O:
          return Color.X
      elif color == Color.X:
          return Color.O
      else:
          return Color.NONE

WINNING_LENGTH: int = 4

DEFAULT_ROWS: int = 6
DEFAULT_COLUMNS: int = 7

class C4Board:
  def __init__(self, rows: int=DEFAULT_ROWS, columns: int=DEFAULT_COLUMNS):
    self.rows = rows
    self.columns = columns
    self.board = np.zeros((rows, columns), dtype=int)
    self.expected_next_move_color: Color = Color.O
    self.move_count = 0

  def legal_moves(self) -> list[int]:
    legal: list[int] = []

    for ii in range(self.columns):
      if self.board[0, ii] == Color.NONE.value:
        legal.append(ii)

    return legal
  
  def make_move(self, color: Color, column: int) -> None:
    if color != self.expected_next_move_color:
      raise Exception(f"Expected {self.expected_next_move_color}, but move is for {color}")
    
    if column not in self.legal_moves():
      raise Exception(f"Illegal move - column: {column}")
    
    self.expected_next_move_color = Color.opposite(color)
    self.move_count += 1

    for row in range(self.rows + 1):
      if row == self.rows or self.board[row, column] != Color.NONE.value:
        self.board[row - 1, column] = color.value
        return
      
  def is_tie(self) -> bool:
    return self.move_count == self.rows * self.columns
  
  def is_winning(self, colorEnum: Color) -> bool:
    color: int = colorEnum.value

    # Horizontal
    for row in range(self.rows):
      count: int = 0
      for col in range(self.columns):
        if self.board[row,col] == color:
          count += 1
          if count == WINNING_LENGTH:
            return True
        else:
          count = 0 # Reset consecutive count

    # Vertical
    for row in range(self.columns):
      count: int = 0
      for col in range(self.rows):
        if self.board[col,row] == color:
          count += 1
          if count == WINNING_LENGTH:
            return True
        else:
          count = 0 # Reset consecutive count

    # Diagonal - \
    start_incl: int = -(self.rows - WINNING_LENGTH)
    end_excl: int = self.columns - WINNING_LENGTH + 1
    for row in range(start_incl, end_excl):  # Iterate column on grid shifted like /
      count: int = 0
      for col in range(self.rows):
        column: int = row + col
        if column >= 0 and column < self.columns and self.board[col,column] == color:
          count += 1
          if count == WINNING_LENGTH:
            return True
        else:
          count = 0 # Reset consecutive count

    # Diagonal - /
    start_incl: int = WINNING_LENGTH - 1
    end_excl: int = self.columns + self.rows - WINNING_LENGTH
    for row in range(start_incl, end_excl):  # Iterate column on grid shifted like \
      count: int = 0
      for col in range(self.rows):
        column: int = row - col
        if column >= 0 and column < self.columns and self.board[col,column] == color:
          count += 1
          if count == WINNING_LENGTH:
            return True
        else:
          count = 0 # Reset consecutive count

    return False
  
  def to_string(self, include_headers: bool=False) -> str:
    symbol_map = {
        0: ".",
        1: "O",
        -1: "X"
    }

    lines = [" ".join(symbol_map[val] for val in row) for row in self.board]
    if include_headers:
      header = " ".join(str(i) for i in range(self.columns))
      lines.insert(0, header)  # prepend header line
    
    return "\n".join(lines)
  
  def print(self, include_headers: bool=True) -> None:
    print(self.to_string(include_headers))

  @staticmethod
  def from_string(string: str):
    mapping = {
      'O': 1,
      'X': -1,
      '.': 0
    }

    string = string.strip()

    # Split lines, remove empty ones, and parse tokens
    rows = [
        [mapping[ch] for ch in line.strip().split()]
        for line in string.strip().splitlines()
        if line.strip()
    ]

    rows_count = len(rows)
    cols_count = len(rows[0]) if rows_count > 0 else 0

    board: C4Board = C4Board(columns=cols_count, rows=rows_count)
    board.board = np.array(rows, dtype=int)
    return board

  def copy(self) -> "C4Board":
    b: C4Board = C4Board(rows=self.rows, columns=self.columns)

    b.board = self.board.copy()
    b.expected_next_move_color = self.expected_next_move_color
    b.move_count = self.move_count

    return b
  
  def failing_to_block_column(self, move: int, color: Color) -> bool:
    threatened_columns: List[int] = []
    for ii in range(self.columns):
      if self.needs_blocking(ii, color):
        threatened_columns.append(ii)

    if len(threatened_columns) > 0 and move not in threatened_columns:
      return True
    
    return False
  
  def needs_blocking(self, column: int, color: Color):
      # A column needs blocking if there is at least one air gap on top, and the
      # top 3 tokens are of the opposite color
      if self.board[0, column] != Color.NONE.value:
        return False # Moot point - no room to block
      
      count: int = 0
      opposite: int = Color.opposite(color).value
      for row in range(self.rows):
        token: int = self.board[row, column]
        if token == Color.NONE.value:
          continue
        elif token == opposite:
          count += 1
          if count == 3:
            return True # 3-in-a-row reached - this should be blocked
        else:
          return False # Encountered non-opposite color - no need to block
        
      return False # Reach bottom of board without getting 3 in a row - no need to block
          

  

