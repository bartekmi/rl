import os
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.base_class import BaseAlgorithm

from c4.ttt_board import Color, TttBoard
from c4.ttt_1_play_env import EvaluateCallback, Ttt1PlayEnv
from c4.ttt_optimal_player import TttOptimalPlayer

model_path_1 = "ttt_player1_dqn.zip"
model_path_2 = "ttt_player2_dqn.zip"

# Execute a sample game
def execute_game(agent: BaseAlgorithm, opponent: TttOptimalPlayer):
    board: TttBoard = TttBoard()
    while True:
        move: int

        color: Color = board.expected_next_move_color
        if color == Color.O:
            move_arr, _ = agent.predict(Ttt1PlayEnv.obs(board), deterministic=True)
            move = int(move_arr)
        else:
            move = opponent.get_optimal_move_for_X(board)

        if move not in board.legal_moves():
            print(f"Terminating due to invalid move by {color}: {move}")
            break

        print(f"About to make move {move} by {color}")
        board.make_move(board.expected_next_move_color, move)
        board.print()

        if board.is_winning(color):
            print(f"{color}'s WIN!!!")
            break

        if board.is_tie():
            print("TIED GAME!")
            break

if os.path.exists(model_path_1) and os.path.exists(model_path_2):
    print("Loading existing models...")
    player1 = DQN.load(model_path_1)
    player2 = DQN.load(model_path_2)
else:
    common_params = dict(
        policy="MlpPolicy",
        env=Ttt1PlayEnv(None, Color.O),
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

    TIME_STEPS: int = 25000
    # TIME_STEPS: int = 1_000

    opponent: TttOptimalPlayer = TttOptimalPlayer()
    env1: Ttt1PlayEnv = Ttt1PlayEnv(opponent, Color.O)
    player1.set_env(env1)
    player1.learn(TIME_STEPS, callback=EvaluateCallback(opponent, Color.O))

    player1.save("ttt_1_play_dqn")

    execute_game(player1, opponent)





