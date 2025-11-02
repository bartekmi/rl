from c4.c4_board import Color
from c4.ttt_board import TttBoard


class TttOptimalPlayer:
  def __init__(self):
    self.cache: dict[tuple[int, ...], tuple[int, int]] = {}
    self.solve(TttBoard(), Color.O)  # Precompute from empty board

  def solve(self, board: TttBoard, player: Color) -> int:
    key = self._hash(board)

    if key in self.cache:
      return self.cache[key][0]

    legal = board.legal_moves()
    if not legal:
      self.cache[key] = (0, -1)
      return 0

    best_score = -2 if player == Color.X else 2
    best_move = -1

    for move in legal:
      b = board.copy()
      b.make_move(player, move)

      if b.is_winning(player):
        score = +1 if player == Color.X else -1
      else:
        next_player = Color.O if player == Color.X else Color.X
        score = self.solve(b, next_player)

      if player == Color.X:
        if score > best_score:
          best_score = score
          best_move = move
      else:
        if score < best_score:
          best_score = score
          best_move = move

    self.cache[key] = (best_score, best_move)
    return best_score

  def get_optimal_move_for_X(self, board: TttBoard) -> int:
    return self.cache[self._hash(board)][1]

  @staticmethod
  def _hash(board: TttBoard) -> tuple[int, ...]:
    return tuple(board.board.reshape(-1))
