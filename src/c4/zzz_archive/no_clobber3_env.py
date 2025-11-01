from typing import Any
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class NoClobber3Env(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()
        
        self.total_move_count = 0
        self.illegal_count = 0

        # Obs: (rows, col) -> board plane
        self.observation_space = spaces.Box(
            low=0,
            high=1.0,
            shape=(9,),
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
        self.board = np.zeros((9,), dtype=int)
        self.game_move_count = 0
        return self.obs(), {}

    def obs(self) -> np.ndarray:
        return self.board.astype(np.float32)

    def step(self, action: int):
        self.total_move_count += 1
        self.game_move_count += 1
        
        # if self.total_move_count % 1000 == 0:
        #     print(f"Illegal: {self.illegal_count} / {self.total_move_count} => {self.illegal_count / self.total_move_count:.4f}")
        #     self.illegal_count = 0
        #     self.total_move_count = 0

        # Check for illegal moves
        if self.is_illegal(action):
            self.illegal_count += 1
            return self.obs(), -1, True, False, {f"illegal_move": "True"}

        self.board[action] = 1

        if self.o_wins():
            return self.obs(), +1, True, False, {"won": "True"}

        return self.obs(), 0.1, False, False, {}  # No reward or punishment
        
    def is_illegal(self, move: int) -> bool:
        return self.board[move] == 1

    def o_wins(self) -> bool:
        return bool(np.all(self.board == 1))

    def render(self):
        print(self.to_string())

    def to_string(self) -> str:
        symbol_map = {
            0: ".",
            1: "O",
        }

        return " ".join(symbol_map[val] for val in self.board)

    def close(self):
        pass


class EvaluateCallback(BaseCallback):
    def _on_step(self) -> bool:

        if self.num_timesteps % 100 == 0:
            steps: int = 0
            illegal: int = 0
            env: NoClobber3Env = NoClobber3Env()

            for _ in range(100):
                env.reset()

                while True:
                    action, _ = self.model.predict(env.obs(), deterministic=True)
                    action = int(action)

                    steps += 1
                    if env.is_illegal(action):
                        illegal += 1
                        break   # Breaking to prevent infinite deterministic loops

                    env.step(action)
                    
                    if env.o_wins():
                        break
                
            # Log deterministic illegal move rate
            illegal_rate: float = illegal / steps
            print(f"Evaluation at step {self.num_timesteps}: "
                    f"Illegal moves: {illegal}/{steps} ({illegal_rate:.4f})")
            
        return True