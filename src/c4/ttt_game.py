from typing import Tuple
from c4.c4_board import Color
from c4.ttt_board import TttBoard

class TttGame:
  def __init__(self) -> None:
    self.board = TttBoard()

  def start(self):
    while True:
      self.board.print()

      expected_color: Color = self.board.expected_next_move_color

      value: int
      while True:
        column_str: str = input(f"Enter Value: (0-8) for {expected_color.name}: ")
        success, value = TttGame.try_parse(column_str) 
        if not success or value < 0 or value > 8 or value not in self.board.legal_moves():
          print("Invalid input " + column_str)
        else:
          break

      self.board.make_move(expected_color, value)

      if self.board.is_winning(expected_color):
        print(f"{expected_color.name} WINS!!!")
        break

      if self.board.is_tie():
        print(f"It's a TIE - No more moves!")
        break

    # Print final board state
    self.board.print()
      
  @staticmethod
  def try_parse(s: str) -> Tuple[bool, int]:
    try:
      return (True, int(s))
    except ValueError:
      return (False, 0)
      

