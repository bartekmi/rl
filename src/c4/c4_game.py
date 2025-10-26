from typing import Tuple
from c4.c4_board import C4Board, Color

class C4Game:
  def __init__(self) -> None:
    self.board = C4Board()

  def start(self):
    while True:
      self.board.print(True)

      expected_color: Color = self.board.expected_next_move_color

      column: int
      while True:
        column_str: str = input(f"Enter Column: (0-6) for {expected_color.name}: ")
        success, column = C4Game.try_parse(column_str) 
        if not success or column < 0 or column > 6 or column not in self.board.legal_moves():
          print("Invalid input " + column_str)
        else:
          break

      self.board.make_move(expected_color, column)

      if self.board.is_winning(expected_color):
        print(f"{expected_color.name} WINS!!!")
        break

      if self.board.is_tie():
        print(f"It's a TIE - No more moves!")
        break

    # Print final board state
    self.board.print(True)
      
  @staticmethod
  def try_parse(s: str) -> Tuple[bool, int]:
    try:
      return (True, int(s))
    except ValueError:
      return (False, 0)
      

