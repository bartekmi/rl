import os
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.base_class import BaseAlgorithm

from c4.no_clobber2_env import NoClobber2Env, Color

model_path_1 = "nc_player1_dqn.zip"

if os.path.exists(model_path_1):
    print("Loading existing models...")
    player1 = DQN.load(model_path_1)
else:
    dummy_env: NoClobber2Env = NoClobber2Env()
    common_params = dict(
        policy="MlpPolicy",
        env=dummy_env,
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
        verbose=0,
    )

    # Two agents using the same settings
    player1 = DQN(**common_params)
    # player2 = DQN(**common_params)

    LEARNING_ITERATIONS: int = 1
    TIME_STEPS: int = 10000

    for iteration in range(LEARNING_ITERATIONS):
        # Learn for Player 1 (O's)
        print(f"Iteration {iteration} for Player O")
        env1: NoClobber2Env = NoClobber2Env()
        player1.set_env(env1)
        player1.learn(TIME_STEPS)

        # Learn for Player 2 (X's)
        # print(f"Iteration {iteration} for Player X")
        # env2: TttEnv = TttEnv(player1, Color.X)
        # player2.set_env(env2)
        # player2.learn(TIME_STEPS)

    player1.save("nc_player1_dqn")
    # player2.save("nc_player2_dqn")

# Execute a sample game
def execute_game(player1: BaseAlgorithm, deterministic: bool):
    board: NoClobber2Env = NoClobber2Env()
    board.reset()

    while True:
        # Hero Move
        move_arr, _ = player1.predict(board.obs(), deterministic=deterministic)
        move: int = int(move_arr)

        if move not in board.legal_moves():
            print(f"Terminating due to invalid move by O: {move}")
            break

        print(f"About to make move {move} for O")
        board.board[move] = Color.O.value
        board.render()

        if board.o_wins():
            print(f"O Won!!!")
            break

        # Opponent Move
        move = board.suggest_random_legal_move()
        print(f"About to make move {move} for X")
        board.board[move] = Color.X.value
        board.render()

while True:
    print()
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>> NEW GAME <<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    execute_game(player1, True)

    command: str = input("ENTER Q to quit: ")
    if command == 'q' or command == 'Q':
        break



