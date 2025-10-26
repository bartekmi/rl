import os
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.base_class import BaseAlgorithm

from c4.ttt_board import Color, TttBoard
from c4.ttt_env import TttEnv

model_path_1 = "ttt_player1_dqn.zip"
model_path_2 = "ttt_player2_dqn.zip"

if os.path.exists(model_path_1) and os.path.exists(model_path_2):
    print("Loading existing models...")
    player1 = DQN.load(model_path_1)
    player2 = DQN.load(model_path_2)
else:
    dummy_env: TttEnv = TttEnv(None, Color.O)
    common_params = dict(
        policy="MlpPolicy",
        env=dummy_env,
        learning_rate=0.1,
        buffer_size=2000,
        learning_starts=500,
        batch_size=64,
        tau=1.0,
        gamma=0.9,
        train_freq=1,
        target_update_interval=500,
        exploration_fraction=0.4,
        exploration_final_eps=0.1,
        verbose=0,
    )

    # Two agents using the same settings
    player1 = DQN(**common_params)
    player2 = DQN(**common_params)

    LEARNING_ITERATIONS: int = 10
    TIME_STEPS: int = 2000

    for iteration in range(LEARNING_ITERATIONS):
        # Learn for Player 1 (O's)
        print(f"Iteration {iteration} for Player O")
        env1: TttEnv = TttEnv(player2, Color.O)
        player1.set_env(env1)
        player1.learn(TIME_STEPS)

        # Learn for Player 2 (X's)
        print(f"Iteration {iteration} for Player X")
        env2: TttEnv = TttEnv(player1, Color.X)
        player2.set_env(env2)
        player2.learn(TIME_STEPS)

    player1.save("ttt_player1_dqn")
    player2.save("ttt_player2_dqn")

# Execute a sample game
def execute_game(player1: BaseAlgorithm, player2: BaseAlgorithm, deterministic: bool):
    board: TttBoard = TttBoard()
    while True:
        color: Color = board.expected_next_move_color
        player: BaseAlgorithm = player1 if color == Color.O else player2
        move_arr, _ = player.predict(TttEnv.obs(board), deterministic=deterministic)
        move: int = int(move_arr)

        if move not in board.legal_moves():
            print(f"Terminating due to invalid move by {color}: {move}")
            break

        print(f"About to make move {move} by {color}")
        board.make_move(board.expected_next_move_color, move)
        board.print()

        if board.is_tie():
            print("TIED GAME!")
            break

        if board.is_winning(color):
            print(f"{color}'s WIN!!!")
            break


while True:
    print()
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>> NEW GAME <<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    execute_game(player1, player2, False)

    command: str = input("ENTER Q to quit: ")
    if command == 'q' or command == 'Q':
        break



