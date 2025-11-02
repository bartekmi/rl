import random
from typing import Tuple, Dict
from c4.ttt_board import TttBoard, TttBoardState
from c4.c4_board import Color

TRAINING_RUNS: int = 100_000
REWARD_WIN = +1.0
REWARD_LOSS = -1.0
REWARD_TIE = 0.0
REWARD_CONTINUE = 0.0

QKey = Tuple[TttBoardState, int]

class TttQLearning:
  def __init__(self) -> None:
    self.q_table: Dict[QKey, float] = {}
    self.epsilon: float = 0.2
    self.learning_rate = 0.1
    self.gamma = 0.95

  def train_multiple_games(self, iterations: int=TRAINING_RUNS) -> None:
    for ii in range(iterations):
      self.train_one_game()
      if ii % 1000 == 0:
        print(f"Iteration {ii}: Dict size: {len(self.q_table)}")

  
  def train_one_game(self) -> None:
    board: TttBoard = TttBoard()
    while True:
      color: Color = board.expected_next_move_color
      move: int = self.select_move(board)

      state_before = board.state()
      board.make_move(color, move)

      if board.is_winning(color):
        self.update_q_table(state_before, board, move, REWARD_WIN, True)
        break
      elif board.is_tie():
        self.update_q_table(state_before, board, move, REWARD_TIE, True)
        break
      else:
        self.update_q_table(state_before, board, move, REWARD_CONTINUE, False)

  def update_q_table(
    self,
    state: TttBoardState,
    board_after: TttBoard,
    action: int,
    reward: float,
    terminal: bool
  ) -> None:
    key: QKey = (state, action)
    next_state: TttBoardState = board_after.state()
    old_q: float = self.q_table[key] if key in self.q_table else 0.0

    if terminal:
      target: float = reward
    else:
      next_q_max: float = float("-inf")
      legal_moves: list[int] = board_after.legal_moves()

      for a in legal_moves:
        k: QKey = (next_state, a)
        val: float = self.q_table[k] if k in self.q_table else 0.0
        next_q_max = val if val > next_q_max else next_q_max

      # alternate perspective
      target = reward + self.gamma * (-next_q_max)

    self.q_table[key] = old_q + self.learning_rate * (target - old_q)


  def select_move(self, board: TttBoard) -> int:
    legal_moves: list[int] = board.legal_moves()
    assert len(legal_moves) > 0

    # Exploration
    if random.random() < self.epsilon:
      return random.choice(legal_moves)
    
    # Exploitation
    best_score: float = -1
    best_move: int = legal_moves[0]

    # TODO: By picking the first legal move if there are no previous table entries,
    # we are under-exploring the state-space
    for move in legal_moves:
      key: QKey = (board.state(), move)
      score: float = 0.0
      if key in self.q_table:
        score = self.q_table[key]

      if score > best_score:
        best_score = score
        best_move = move

    return best_move
  
  def best_move(self, board: TttBoard) -> int:
    q_max: float = float("-inf")
    best_move: int = -100
    state: TttBoardState = board.state()

    legal_moves: list[int] = board.legal_moves()
    assert len(legal_moves) > 0

    for a in legal_moves:
      k: QKey = (state, a)
      val: float = self.q_table[k] if k in self.q_table else 0.0
      if val > q_max:
        q_max = val
        best_move = a

    if best_move == -100:
      return random.choice(legal_moves)

    return best_move


# learn: TttQLearning = TttQLearning()
# learn.train_multiple_games()
