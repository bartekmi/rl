from typing import Any, Tuple, Dict
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from c4.ttt_board import Color, TttBoard
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback

from c4.ttt_optimal_player import TttOptimalPlayer

class Ttt1PlayEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": ["human"]}

    def __init__(self, opponent: TttOptimalPlayer, agent_color: Color):
        super().__init__()
        assert agent_color in (Color.O, Color.X)
        self.agent_color = agent_color
        self.opponent: TttOptimalPlayer = opponent

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
            
        return self._obs(), {}

    def _obs(self) -> np.ndarray:
        return Ttt1PlayEnv.obs(self.board)
    
    @staticmethod
    def obs(board: TttBoard) -> np.ndarray:
        return board.board.astype(np.float32)

    def step(self, action: int):
        assert self.opponent is not None

        # Make agent move
        result = self._make_move(action, -2, +1)
        if result[2]:
            return result

        # Make opponent move
        opponent_action: int = self.opponent.get_optimal_move_for_X(self.board)
        return self._make_move(opponent_action, 0, -1)

    def _make_move(
            self, action: int, 
            illegal_penalty: int,
            win_reward: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str,str]]:
        
        color: Color = self.board.expected_next_move_color

        # Check for illegal moves
        if action not in self.board.legal_moves():
            return self._obs(), illegal_penalty, True, False, {f"illegal_move_by_{color}": "True"}

        self.board.make_move(color, action)

        if self.board.is_winning(color):
            return self._obs(), win_reward, True, False, {"winner": str(color)}

        if self.board.is_tie():
            return self._obs(), 1, True, False, {"tie": "True"}

        return self._obs(), +0.1, False, False, {}  # No reward or punishment

    def render(self):
        if self.board:
            self.board.print()

    def close(self):
        pass


class EvaluateCallback(BaseCallback):
    def __init__(self, opponent: TttOptimalPlayer, verbose: int = 0):
        super().__init__(verbose)
        self.opponent = opponent

        # Modified during game-play
        self.steps: int = 0
        self.illegal: int = 0
        self.missed_win: int = 0
        self.fail_to_block: int = 0
    
    def _on_step(self) -> bool:

        if self.num_timesteps % 250 == 0:
            self.steps = 0
            self.illegal = 0
            self.missed_win = 0
            self.fail_to_block = 0

            board: TttBoard = TttBoard()

            while True:
                # O's move
                self.steps += 1
                if self.make_move(board, self.model):
                    break

                # X's move
                self.steps += 1
                move_x: int = self.opponent.get_optimal_move_for_X(board)
                board.make_move(board.expected_next_move_color, move_x)
                if board.is_tie() or board.is_winning(board.expected_next_move_color):
                    break

            # Print stats
            print(f"TS {self.num_timesteps}: "
                    f"Ill / M.W. / no-block: {self.illegal} / {self.missed_win} / {self.fail_to_block} / {self.steps}")
            
        return True
    
    def make_move(self, board: TttBoard, player: BaseAlgorithm) -> bool:
        obs: np.ndarray = Ttt1PlayEnv.obs(board)
        action, _ = player.predict(obs, deterministic=True)
        action = int(action)

        if board.is_illegal(action):
            self.illegal += 1
            return True   # Breaking to prevent infinite deterministic loops
        
        if board.missed_win(action):
            self.missed_win += 1
        
        if board.failed_to_block(action):
            self.fail_to_block += 1

        color: Color = board.expected_next_move_color
        board.make_move(color, action)
        
        return board.is_tie() or board.is_winning(color)
