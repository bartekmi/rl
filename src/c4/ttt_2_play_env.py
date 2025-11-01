from typing import Any, Tuple, Dict
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from c4.ttt_board import Color, TttBoard
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback

class Ttt2PlayEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": ["human"]}

    def __init__(self, opponent: BaseAlgorithm, agent_color: Color):
        super().__init__()
        assert agent_color in (Color.O, Color.X)
        self.agent_color = agent_color
        self.opponent_color = Color.opposite(agent_color)
        self.opponent: BaseAlgorithm = opponent

        self.move_count = 0
        self.illegal_count = 0
        self.missed_block_count = 0

        # Obs: (rows, col) -> board plane
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3, 3),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(9) # type: ignore

    def reset(
            self, 
            *, 
            seed: int | None=None, 
            options: dict[str, Any] | None=None
            ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.board = TttBoard()

        # If X is being trained, make an opening move for O
        if self.agent_color == Color.X:
            assert self.board.expected_next_move_color == Color.O # Invariant
            move, _ = self.opponent.predict(self._obs())
            self.board.make_move(self.opponent_color, int(move))
            assert self.board.expected_next_move_color == self.agent_color
            
        return self._obs(), {}

    def _obs(self) -> np.ndarray:
        return Ttt2PlayEnv.obs(self.board)
    
    @staticmethod
    def obs(board: TttBoard) -> np.ndarray:
        return board.board.astype(np.float32)

    def step(self, action: int):
        # print(f"--- Step Start: Agent color={self.agent_color}, Hero action={hero_action} ---")
        assert self.opponent is not None

        # Make agent move
        result = self._make_move(action, -2, +1)
        if result[2]:
            return result

        # Make opponent move
        opponent_action_arr, _ = self.opponent.predict(self._obs())
        opponent_action: int = int(opponent_action_arr)
        return self._make_move(opponent_action, 0, -1)

    def _make_move(
            self, action: int, 
            illegal_penalty: int,
            win_reward: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str,str]]:
        
        self.move_count += 1
        color: Color = self.board.expected_next_move_color

        # Check for illegal moves
        if action not in self.board.legal_moves():
            self.illegal_count += 1
            return self._obs(), illegal_penalty, True, False, {f"illegal_move_by_{color}": "True"}
        
        # if self.move_count % 1000 == 0:
        #     print(f"Illegal: {self.illegal_count} / {self.move_count} => {self.illegal_count / self.move_count:.4f}")

        self.board.make_move(color, action)

        if self.board.is_winning(color):
            return self._obs(), win_reward, True, False, {"winner": str(color)}

        if self.board.is_tie():
            return self._obs(), 0.0, True, False, {"tie": "True"}

        return self._obs(), 0.0, False, False, {}  # No reward or punishment

    def render(self):
        if self.board:
            self.board.print()

    def close(self):
        pass


class EvaluateCallback(BaseCallback):
    def __init__(self, opponent: BaseAlgorithm, color: Color, verbose: int = 0):
        super().__init__(verbose)
        self.opponent = opponent
        self.agent_color = color
        self.illegal: int = 0
        self.steps: int = 0
    
    def _on_step(self) -> bool:

        if self.num_timesteps % 500 == 0:
            hero: BaseAlgorithm = self.model
            is_O_learning: bool = self.agent_color == Color.O

            playerO: BaseAlgorithm = hero if is_O_learning else self.opponent
            playerX: BaseAlgorithm = self.opponent if is_O_learning else hero

            # Play 100 games to see the rate of illegal moves
            for _ in range(100):
                board: TttBoard = TttBoard()

                while True:
                    if self.make_move(board, playerO):
                        break
                    if self.make_move(board, playerX):
                        break

            # Log illegal move rate
            illegal_rate: float = self.illegal / self.steps
            print(f"Evaluation at step {self.num_timesteps}: "
                    f"Illegal moves: {self.illegal}/{self.steps} ({illegal_rate:.4f})")
            
        return True
    
    def make_move(self, board: TttBoard, player: BaseAlgorithm) -> bool:
        obs: np.ndarray = Ttt2PlayEnv.obs(board)
        action, _ = player.predict(obs, deterministic=True)
        action = int(action)

        self.steps += 1
        if board.is_illegal(action):
            self.illegal += 1
            return True   # Breaking to prevent infinite deterministic loops

        color: Color = board.expected_next_move_color
        board.make_move(color, action)
        
        return board.is_tie() or board.is_winning(color)
