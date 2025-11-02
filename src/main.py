# from c4.c4_game import C4Game
from c4.ttt_game import TttGame
from c4.ttt_optimal_player import TttOptimalPlayer

# game: C4Game = C4Game()
opponent: TttOptimalPlayer = TttOptimalPlayer()
while True:
  game: TttGame = TttGame(opponent)
  game.start()