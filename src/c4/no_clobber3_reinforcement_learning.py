import os
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.base_class import BaseAlgorithm

from c4.no_clobber3_env import EvaluateCallback, NoClobber3Env

agent = DQN(policy="MlpPolicy",
    env=NoClobber3Env(),
    learning_rate=0.001,
    buffer_size=10000,
    learning_starts=500,
    batch_size=64,
    tau=1.0,
    gamma=0.95,
    train_freq=1,
    target_update_interval=100,
    exploration_fraction=0.5,
    exploration_final_eps=0.1,
    verbose=0)

# agent = PPO(
#     policy="MlpPolicy",
#     env=NoClobber3Env(),
#     learning_rate=0.001,  # Same as DQN
#     n_steps=2048,         # Number of steps per rollout (controls update frequency)
#     batch_size=64,        # Same as DQN
#     n_epochs=10,          # Number of optimization epochs per update
#     gamma=0.95,           # Same as DQN
#     gae_lambda=0.95,      # GAE lambda for advantage estimation (standard value)
#     clip_range=0.2,       # PPO clip range (standard value)
#     ent_coef=0.01,        # Entropy coefficient to encourage exploration
#     verbose=0             # Same as DQN
# )

# agent = A2C(
#     policy="MlpPolicy",
#     env=NoClobber3Env(),
#     learning_rate=0.001,  # Same as DQN and PPO
#     n_steps=5,            # Number of steps per rollout (small for frequent updates)
#     gamma=0.95,           # Same as DQN and PPO
#     gae_lambda=0.95,      # GAE lambda, same as PPO
#     ent_coef=0.01,        # Entropy coefficient for exploration, same as PPO
#     vf_coef=0.5,          # Value function coefficient (standard value)
#     normalize_advantage=True,  # Normalize advantages for stable training
#     verbose=0             # Same as DQN and PPO
# )

callback = EvaluateCallback()
agent.learn(total_timesteps=10000, callback=callback)
#agent.learn(total_timesteps=100_000)

# Execute a sample game
def execute_game(player1: BaseAlgorithm, deterministic: bool):
    board: NoClobber3Env = NoClobber3Env()
    board.reset()
    count: int = 0

    while True:
        move_arr, _ = player1.predict(board.obs(), deterministic=deterministic)
        move: int = int(move_arr)
        count += 1

        print(f"About to make move {move} for O")
        board.board[move] = 1
        board.render()

        if board.o_wins():
            print(f"O Won!!!")
            break

    print(f"Won after {count} moves.")

while True:
    print()
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>> NEW GAME <<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    execute_game(agent, True)

    command: str = input("ENTER Q to quit: ")
    if command == 'q' or command == 'Q':
        break



