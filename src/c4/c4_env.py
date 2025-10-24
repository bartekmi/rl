from typing import Any
import gymnasium as gym
import random
from gymnasium import spaces
import numpy as np
from c4.board import Board, DEFAULT_COLUMNS, DEFAULT_ROWS, Color

class ConnectFourEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": ["human"]}

    def __init__(self, opponent: str = "heuristic", agent_color: Color = Color.O):
        super().__init__()
        assert agent_color in (Color.O, Color.X)
        self.agent_color = agent_color
        self.opponent_color = Color.opposite(agent_color)
        self.opponent_type = opponent

        self.threat_seen = 0
        self.threat_blocked = 0

        # Obs: (rows, cols, 2) -> board plane, turn plane (agent=+1, opponent=-1)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(DEFAULT_ROWS, DEFAULT_COLUMNS),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(DEFAULT_COLUMNS) # type: ignore
        self.reset()

    def reset(
            self, 
            *, 
            seed: int | None=None, 
            options: dict[str, Any] | None=None
            ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.board = Board()

        # If the opponent is to move first, make its opening move.
        if self.board.expected_next_move != self.agent_color:
            self._opponent_move()

        if self.threat_seen > 0 and self.threat_seen % 20 == 0:
            rate = self.threat_blocked / self.threat_seen
            print(f"[diag] threat_seen={self.threat_seen}, threat_blocked={self.threat_blocked}, rate={rate:.3f}")
            
        obs: np.ndarray = self._obs()
        return obs, {}

    def _obs(self) -> np.ndarray:
        return self.board.board.astype(np.float32)

    def step(self, action: int):
        # Detect "threat" state: 3 vertical opponent tokens in column 3 with space above
        array = self.board.board
        if (array[5, 3] == -1 and  # bottom row
            array[4, 3] == -1 and  # one above bottom
            array[3, 3] == -1 and  # two above bottom
            array[2, 3] == 0):     # space above is empty
            self.threat_seen += 1
            if action == 3:
                self.threat_blocked += 1

        # --- Agent move ---
        if action not in self.board.legal_moves():
            # Penalize illegal action and terminate (simple but effective)
            return self._obs(), -10.0, True, False, {"illegal_move": True}

        self.board.make_move(self.agent_color, action)

        # Check for terminal after agent move
        if self.board.is_winning(self.agent_color):
            return self._obs(), 1.0, True, False, {"winner": "agent"}

        if self.board.is_tie():
            return self._obs(), 0.0, True, False, {"tie": True}

        # --- Opponent reply ---
        self._opponent_move()

        # Check terminal after opponent move
        if self.board.is_winning(self.opponent_color):
            return self._obs(), -1.0, True, False, {"winner": "opponent"}

        if self.board.is_tie():
            return self._obs(), 0.0, True, False, {"tie": True}

        # Non-terminal transition
        return self._obs(), 0.0, False, False, {}  # No reward or punishment

    def _opponent_move(self):
        """Choose and apply an opponent move."""
        assert self.board is not None
        legal = self.board.legal_moves()
        if not legal:
            return
        
        if random.random() < 0.99999999:
            col = self._heuristic_move()
        else:  # random
            col = random.choice(legal)

        self.board.make_move(self.opponent_color, col)

    def _heuristic_move(self) -> int:
        """
        Smarter heuristic opponent:
        1. Win immediately if possible.
        2. Block opponent's immediate win.
        3. Avoid moves that let the agent win next turn.
        4. Prefer center columns for positional advantage.
        """
        assert self.board is not None
        legal = self.board.legal_moves()
        if not legal:
            raise RuntimeError("No legal moves available")

        # 1️⃣ Try to win immediately
        for c in legal:
            b2: Board = self._sim_after(self.board, self.opponent_color, c)
            if b2.is_winning(self.opponent_color):
                return c

        # 2️⃣ Block agent's immediate win
        for c in legal:
            b2 = self._sim_after(self.board, self.agent_color, c)
            if b2.is_winning(self.agent_color):
                return c

        # 3️⃣ Avoid moves that let the agent win next turn
        safe_moves: list[int] = []
        for c in legal:
            # Simulate opponent's move
            b2 = self._sim_after(self.board, self.opponent_color, c)
            opp_can_win_next = False

            for oc in b2.legal_moves():
                b3 = self._sim_after(b2, self.agent_color, oc)
                if b3.is_winning(self.agent_color):
                    opp_can_win_next = True
                    break

            if not opp_can_win_next:
                safe_moves.append(c)

        # If all moves are unsafe, just fall back to all legal moves
        candidates = safe_moves if safe_moves else legal

        # 4️⃣ Prefer center columns
        center = self.board.columns // 2
        return min(candidates, key=lambda x: abs(x - center))
    
    @staticmethod
    def _sim_after(b_init: Board, color: Color, column: int) -> Board:
        b = b_init.copy()
        b.expected_next_move = color
        b.make_move(color, column)
        return b    

    def render(self):
        if self.board:
            self.board.print()

    def close(self):
        pass
