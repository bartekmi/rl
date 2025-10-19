import numpy as np
from c4.c4_env import ConnectFourEnv

if __name__ == "__main__":
    env = ConnectFourEnv()
    obs, info = env.reset()
    done = False
    step_num = 0

    while not done:
        env.render()
        legal = env.board.legal_moves()
        action = np.random.choice(legal)
        print(f"\nStep {step_num}: playing column {action}")
        obs, reward, done, truncated, info = env.step(action)
        print(f"Reward: {reward}, Done: {done}, Info: {info}")
        step_num += 1

    env.render()
    print("Game ended.")
