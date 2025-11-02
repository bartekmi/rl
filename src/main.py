# from c4.c4_game import C4Game
from c4.ttt_game import TttGame
# from c4.ttt_optimal_player import TttOptimalPlayer
from stable_baselines3 import DQN
from c4.ttt_1_play_reinforcement_learning import model_path
from c4.ttt_board import Color

# game: C4Game = C4Game()
# opponent: TttOptimalPlayer = TttOptimalPlayer()
opponent = player1 = DQN.load(model_path)
while True:
  game: TttGame = TttGame(opponent)
  game.start(Color.X)