import random
from typing import Any, Tuple, Dict
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from c4.c4_board import Color

class NoClobberEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()
        
        self.total_move_count = 0
        self.game_move_count = 0
        self.illegal_count = 0

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
        self.board = np.zeros((3, 3), dtype=int)
        self.game_move_count = 0
        return self.obs(), {}

    def obs(self) -> np.ndarray:
        return self.board.astype(np.float32)

    def step(self, hero_action: int):
        # Make agent move
        result = self._make_move(hero_action, -10, +1, Color.O)
        if result[2]:
            return result

        opponent_action: int = self.suggest_random_legal_move()
        self.game_move_count += 1
        self.set_at(opponent_action, Color.X)
        return self.obs(), 0, False, False, {}

    def suggest_random_legal_move(self) -> int:
        legal: list[int] = self.legal_moves()

        if len(legal) == 0:
            raise Exception("No legal moves")
        
        return random.choice(legal)

    def legal_moves(self) -> list[int]:
        legal: list[int] = []

        for ii in range(9):
            if self.get_at(ii) == Color.NONE.value:
                legal.append(ii)

        return legal

    def get_at(self, index: int) -> int:
        row: int = index // 3
        col: int = index % 3
        return self.board[row, col]
  
    def set_at(self, index: int, color: Color) -> None:
        row: int = index // 3
        col: int = index % 3
        self.board[row, col] = color.value

    def _make_move(
            self, action: int, 
            illegal_penalty: int,
            win_reward: int,
            color: Color
            ) -> Tuple[np.ndarray, float, bool, bool, Dict[str,str]]:
        
        self.total_move_count += 1
        self.game_move_count += 1
        
        if self.total_move_count % 100 == 0:
            print(f"Illegal: {self.illegal_count} / {self.total_move_count} => {self.illegal_count / self.total_move_count:.4f}")

        # Check for illegal moves
        if action not in self.legal_moves():
            self.illegal_count += 1
            return self.obs(), illegal_penalty, True, False, {f"illegal_move_by_{color}": "True"}

        self.set_at(action, color)

        if self.o_wins():
            return self.obs(), win_reward, True, False, {"winner": str(color)}

        return self.obs(), 0.0, False, False, {}  # No reward or punishment

    def o_wins(self) -> bool:
        return self.game_move_count == 9

    def render(self):
        print(self.to_string())

    def to_string(self) -> str:
        symbol_map = {
            0: ".",
            1: "O",
            -1: "X"
        }

        lines = [" ".join(symbol_map[val] for val in row) for row in self.board]
        return "\n".join(lines)

    def close(self):
        pass
