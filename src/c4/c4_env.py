from typing import Any
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from c4.board import Board, DEFAULT_COLUMNS, DEFAULT_ROWS

class ConnectFourEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()

        # Define Gym spaces once
        self.action_space = spaces.Discrete(DEFAULT_COLUMNS)  # type: ignore[assignment]
        self.observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=(DEFAULT_ROWS, DEFAULT_COLUMNS),
            dtype=np.int8,
        )

        # Initialize environment state
        self.done: bool = False
        self.reset()

    def reset(
            self, 
            *, 
            seed: int | None=None, 
            options: dict[str, Any] | None=None
            ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.board = Board()
        self.done = False

        obs: np.ndarray = self._obs()
        return obs, {}

    def _obs(self) -> np.ndarray:
        """
        Flip the board in such a way so the agent always sees their tokens as +1,
        even when playing as X
        """
        me = self.board.expected_next_move.value          # +1 for O, -1 for X
        obs = (self.board.board.astype(np.int8) * me).astype(np.int8)
        return obs

    def step(self, action: int):
        if self.done:
            raise RuntimeError("Game already ended. Call reset().")

        # Extract current player directly from board
        player = self.board.expected_next_move

        if action not in self.board.legal_moves():
            # Illegal move → penalty
            reward = -5.0
            self.done = True
            info = {"illegal_move": True}
            return self._obs(), reward, self.done, False, info

        # Apply the move
        self.board.make_move(player, action)

        # Evaluate outcome
        if self.board.is_winning(player):
            reward = 1.0
            self.done = True
            info = {"winner": player}
        elif self.board.is_tie():
            reward = 0.0
            self.done = True
            info = {"tie": True}
        else:
            reward = 0.0
            self.done = False
            info = {}

        return self._obs(), reward, self.done, False, info

    def render(self):
        if self.board:
            self.board.print()

    def close(self):
        pass
