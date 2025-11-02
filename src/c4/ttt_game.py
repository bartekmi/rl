from typing import Tuple
from c4.c4_board import Color
from c4.ttt_1_play_env import Ttt1PlayEnv
from c4.ttt_board import TttBoard
from c4.ttt_optimal_player import TttOptimalPlayer
from stable_baselines3.common.base_class import BaseAlgorithm

from c4.ttt_q_learning import TttQLearning

class TttGame:
  def __init__(self, opponent: TttOptimalPlayer | BaseAlgorithm | TttQLearning) -> None:
    self.board = TttBoard()
    self.opponent = opponent

  def start(self, human_player: Color):
    print()
    print(">>>>>>>>>>>>>>> NEW GAME....")
    while True:
      self.board.print()

      expected_color: Color = self.board.expected_next_move_color

      move: int
      if self.board.expected_next_move_color == human_player:
        while True:
          column_str: str = input(f"Enter Value: (1-9) for {expected_color.name}: ")
          success, move = TttGame.try_parse(column_str) 
          move -= 1 # Conveert range 1-9 to 0-8

          if not success or move < 0 or move > 8 or move not in self.board.legal_moves():
            print("Invalid input " + column_str)
          else:
            break
      else:
        if isinstance(self.opponent, TttOptimalPlayer):
          move = self.opponent.get_optimal_move_for_X(self.board)
        elif isinstance(self.opponent, TttQLearning):
          move = self.opponent.best_move(self.board)
        else:
          move_arr, _ = self.opponent.predict(Ttt1PlayEnv.obs(self.board))
          move = int(move_arr)

      self.board.make_move(expected_color, move)
      print("-----")

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
      

