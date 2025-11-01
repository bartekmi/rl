from typing import List
import random
import numpy as np
from c4.c4_board import Color

class TttBoard:
  def __init__(self):
    self.board = np.zeros((3, 3), dtype=int)
    self.expected_next_move_color: Color = Color.O
    self.move_count = 0

  def legal_moves(self) -> list[int]:
    legal: list[int] = []

    for ii in range(9):
      if self.get_at(ii) == Color.NONE.value:
        legal.append(ii)

    return legal
  
  def get_at(self, index: int) -> int:
    row: int = index // 3
    col: int = index % 3
    return self.board[row, col]
  
  def set_at(self, index: int, color: Color) -> None:
    row: int = index // 3
    col: int = index % 3
    self.board[row, col] = color.value

  def make_move(self, color: Color, move: int) -> None:
    if color != self.expected_next_move_color:
      raise Exception(f"Expected {self.expected_next_move_color}, but move is for {color}")
    
    if move not in self.legal_moves():
      raise Exception(f"Illegal move - column: {move}")
    
    self.expected_next_move_color = Color.opposite(color)
    self.move_count += 1

    self.set_at(move, color)
  
  def is_illegal(self, action: int) -> bool:
    return self.get_at(action) != 0
      
  def is_tie(self) -> bool:
    return self.move_count == 9
  
  def is_winning(self, colorEnum: Color) -> bool:
    color: int = colorEnum.value

    # Horizontal
    for row in range(3):
      failed: bool = False
      for col in range(3):
        if self.board[row,col] != color:
          failed = True
          break
      if not failed:
        return True  

    # Vertical
    for col in range(3):
      failed: bool = False
      for row in range(3):
        if self.board[row,col] != color:
          failed = True
          break
      if not failed:
        return True  

    # Diagonal
    if (self.board[0,0] == color and
        self.board[1,1] == color and
        self.board[2,2] == color or
        self.board[0,2] == color and
        self.board[1,1] == color and
        self.board[2,0] == color):
        return True

    return False
  


  # ----------------------------------------------------------------------------
  # --------------- Helpers, etc -----------------------------
  def to_string(self) -> str:
    symbol_map = {
        0: ".",
        1: "O",
        -1: "X"
    }

    lines = [" ".join(symbol_map[val] for val in row) for row in self.board]
    return "\n".join(lines)
  
  def print(self) -> None:
    print(self.to_string())

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

    board: TttBoard = TttBoard()
    board.board = np.array(rows, dtype=int)
    return board

  def copy(self) -> "TttBoard":
    b: TttBoard = TttBoard()

    b.board = self.board.copy()
    b.expected_next_move_color = self.expected_next_move_color
    b.move_count = self.move_count

    return b

  def suggest_random_legal_move(self) -> int:
    legal: List[int] = self.legal_moves()
    if len(legal) == 0:
      raise Exception("No legal moves")
    
    return random.choice(legal)
  
  def missed_win(self, move: int) -> bool:
    possible_wins: List[int] = []
    for legal_move in self.legal_moves():
      copy: "TttBoard" = self.copy()
      copy.make_move(self.expected_next_move_color, legal_move)
      if copy.is_winning(self.expected_next_move_color):
        possible_wins.append(legal_move)

    return len(possible_wins) > 0 and move not in possible_wins


  def failed_to_block(self, move: int) -> bool:
    opponent_color: Color = Color.opposite(self.expected_next_move_color)

    # Are any moves immediately winning for opponent?
    threats: List[int] = []
    for legal_move in self.legal_moves():
      copy: "TttBoard" = self.copy()
      copy.expected_next_move_color = opponent_color
      copy.make_move(opponent_color, legal_move)
      if copy.is_winning(opponent_color):
        threats.append(legal_move)

    # If there are multiple threats, we can't really blame the player for 
    # what happens next, since no matter what they play, the will lose
    # on the opponent's next move
    if len(threats) > 1:
      return False
    
    # If "move" is not preventing the single threat, return True
    return len(threats) == 1 and move != threats[0]
