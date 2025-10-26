from typing import Any, Tuple, Dict
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from c4.board import Color, Board, DEFAULT_COLUMNS, DEFAULT_ROWS
from stable_baselines3.common.base_class import BaseAlgorithm

class ConnectFourEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": ["human"]}

    def __init__(self, opponent: BaseAlgorithm | None, agent_color: Color):
        super().__init__()
        assert agent_color in (Color.O, Color.X)
        self.agent_color = agent_color
        self.opponent_color = Color.opposite(agent_color)
        self.opponent = opponent

        self.move_count = 0
        self.illegal_count = 0
        self.missed_block_count = 0

        # Obs: (rows, col) -> board plane
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(DEFAULT_ROWS, DEFAULT_COLUMNS),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(DEFAULT_COLUMNS) # type: ignore

    def reset(
            self, 
            *, 
            seed: int | None=None, 
            options: dict[str, Any] | None=None
            ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.board = Board()

        # If X is being trained, make an opening move for O
        if self.agent_color == Color.X:
            assert self.board.expected_next_move_color == Color.O # Invariant
            move, _ = self.opponent.predict(self._obs())
            self.board.make_move(self.opponent_color, int(move))
            assert self.board.expected_next_move_color == self.agent_color
            
        return self._obs(), {}

    def _obs(self) -> np.ndarray:
        return ConnectFourEnv.obs(self.board)
    
    @staticmethod
    def obs(board: Board) -> np.ndarray:
        return board.board.astype(np.float32)

    def step(self, hero_action: int):
        # print(f"--- Step Start: Agent color={self.agent_color}, Hero action={hero_action} ---")
        assert self.opponent is not None

        # Make agent move
        result = self._make_move(hero_action, -10, +1)
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
        # print(f"Playing {action} by {color}")

        # Check for illegal moves
        if action not in self.board.legal_moves():
            self.illegal_count += 1
            # print(f"Illegal action {action} by {color}")
            return self._obs(), illegal_penalty, True, False, {f"illegal_move_by_{color}": "True"}
        
        # Check for failing to block 3-token column
        if self.board.failing_to_block_column(action, color):
            self.missed_block_count += 1
        
        if self.move_count % 100 == 0:
            print(f"Illegal/No Block: {self.illegal_count} / {self.missed_block_count} / {self.move_count} => {self.illegal_count / self.move_count:.4f} / {self.missed_block_count / self.move_count:.4f}")

        self.board.make_move(color, action)
        # print(f"Board after {color} move:")
        # self.board.print()
        # print(self._obs())

        if self.board.is_winning(color):
            # print(f"{color} wins!")
            return self._obs(), win_reward, True, False, {"winner": str(color)}

        if self.board.is_tie():
            # print(f"Tie after {color} move")
            return self._obs(), 0.0, True, False, {"tie": "True"}

        return self._obs(), 0.0, False, False, {}  # No reward or punishment


    def render(self):
        if self.board:
            self.board.print()

    def close(self):
        pass
