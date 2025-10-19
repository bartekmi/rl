import numpy as np
from enum import Enum

class Color(Enum):
  NONE = 0
  O = 1
  X = -1

WINNING_LENGTH: int = 4

DEFAULT_ROWS: int = 6
DEFAULT_COLUMNS: int = 7

class Board:
  def __init__(self, rows: int=DEFAULT_ROWS, columns: int=DEFAULT_COLUMNS):
    self.rows = rows
    self.columns = columns
    self.board = np.zeros((rows, columns), dtype=int)
    self.expected_next_move: Color = Color.O
    self.move_count = 0

  def legal_moves(self) -> list[int]:
    legal: list[int] = []

    for ii in range(self.columns):
      if self.board[0, ii] == Color.NONE.value:
        legal.append(ii)

    return legal
  
  def make_move(self, color: Color, column: int) -> None:
    if color != self.expected_next_move:
      raise Exception(f"Expected {self.expected_next_move}, but move is for {color}")
    
    if column not in self.legal_moves():
      raise Exception(f"Illegal move - column: {column}")
    
    self.expected_next_move = Color.X if color == Color.O else Color.O
    self.move_count += 1

    for ii in range(self.rows + 1):
      if ii == self.rows or self.board[ii, column] != Color.NONE.value:
        self.board[ii - 1, column] = color.value
        return
      
  def is_tie(self):
    return self.move_count == self.rows * self.columns
  
  def is_winning(self, colorEnum: Color) -> bool:
    color: int = colorEnum.value

    # Horizontal
    for ii in range(self.rows):
      count: int = 0
      for jj in range(self.columns):
        if self.board[ii,jj] == color:
          count += 1
          if count == WINNING_LENGTH:
            return True
        else:
          count = 0 # Reset consecutive count

    # Vertical
    for ii in range(self.columns):
      count: int = 0
      for jj in range(self.rows):
        if self.board[jj,ii] == color:
          count += 1
          if count == WINNING_LENGTH:
            return True
        else:
          count = 0 # Reset consecutive count

    # Diagonal - \
    start_incl: int = -(self.rows - WINNING_LENGTH)
    end_excl: int = self.columns - WINNING_LENGTH + 1
    for ii in range(start_incl, end_excl):  # Iterate column on grid shifted like /
      count: int = 0
      for jj in range(self.rows):
        column: int = ii + jj
        if column >= 0 and column < self.columns and self.board[jj,column] == color:
          count += 1
          if count == WINNING_LENGTH:
            return True
        else:
          count = 0 # Reset consecutive count

    # Diagonal - /
    start_incl: int = WINNING_LENGTH - 1
    end_excl: int = self.columns + self.rows - WINNING_LENGTH
    for ii in range(start_incl, end_excl):  # Iterate column on grid shifted like \
      count: int = 0
      for jj in range(self.rows):
        column: int = ii - jj
        if column >= 0 and column < self.columns and self.board[jj,column] == color:
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

    board: Board = Board(columns=cols_count, rows=rows_count)
    board.board = np.array(rows, dtype=int)
    return board

