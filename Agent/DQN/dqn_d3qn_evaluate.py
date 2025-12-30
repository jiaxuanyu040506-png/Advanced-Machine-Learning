from __future__ import print_function

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

sys.path.append("game/")
import wrapped_flappy_bird as game

# =============== 配置区域 ===============
# 选要评估哪一个模型： "dqn" / "double" / "d3qn"
MODE = "d3qn"        # 改成 "dqn" / "double" / "d3qn"

NUM_EVAL_EPISODES = 20      # 每个模型评估 20 局
MAX_EPISODE_STEPS = 10000   # 每局最多 10000 步

# baseline DQN：deep_q_network.py
if MODE == "dqn":
    from deep_q_network import createNetwork, ACTIONS, GAME
    CKPT_DIR = "saved_networks"
    CKPT_PREFIX = "bird-dqn"          # -> saved_networks/bird-dqn-5000000

# Double DQN：deep_q_network_double.py
elif MODE == "double":
    from deep_q_network_double import createNetwork as createNetwork_double, ACTIONS, GAME
    CKPT_DIR = "saved_networks_double"
    CKPT_PREFIX = "bird-double-dqn"   # -> saved_networks_double/bird-double-dqn-5000000

# D3QN：deep_q_network_d3qn.py
elif MODE == "d3qn":
    from deep_q_network_d3qn import createNetwork as createNetwork_d3qn, ACTIONS, GAME
    CKPT_DIR = "saved_networks_d3qn"
    CKPT_PREFIX = "bird-d3qn"         # -> saved_networks_d3qn/bird-d3qn-5000000

else:
    raise ValueError("MODE must be one of: 'dqn', 'double', 'd3qn'")

CKPT_STEP = 5000000   # 只评估 5M 这个 checkpoint
# =======================================


def preprocess(obs):
    """和训练时一样的预处理：80x80 灰度 + 二值化"""
    x = cv2.cvtColor(cv2.resize(obs, (80, 80)), cv2.COLOR_BGR2GRAY)
    _, x = cv2.threshold(x, 1, 255, cv2.THRESH_BINARY)
    return x


def build_network_for_eval():
    """
    根据 MODE 构建评估用的网络，返回 (s_placeholder, readout_op)
    """
    if MODE == "dqn":
        s, readout, _ = createNetwork()
        return s, readout
    elif MODE == "double":
        s_online, readout_online, _, _ = createNetwork_double("online")
        return s_online, readout_online
    elif MODE == "d3qn":
        s_online, readout_online, _, _ = createNetwork_d3qn("online")
        return s_online, readout_online
    else:
        raise ValueError("Unsupported MODE")


def eval_checkpoint_5m(ckpt_path):
    """对 5M checkpoint 做 NUM_EVAL_EPISODES 局 greedy evaluation，返回 reward 数组"""
    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    s, readout = build_network_for_eval()

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, ckpt_path)

    env = game.GameState()
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1

    episode_rewards = []

    for ep in range(NUM_EVAL_EPISODES):
        # 初始化一局
        obs, r0, terminal = env.frame_step(do_nothing)
        x_t = preprocess(obs)
        s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
        total_r = 0.0
        steps = 0

        while True:
            # greedy 策略（epsilon=0）
            readout_t = readout.eval(feed_dict={s: [s_t]})[0]
            action_index = int(np.argmax(readout_t))
            a_t = np.zeros([ACTIONS])
            a_t[action_index] = 1

            obs1, r_t, terminal = env.frame_step(a_t)
            x_t1 = preprocess(obs1)
            x_t1 = np.reshape(x_t1, (80, 80, 1))
            s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

            s_t = s_t1
            total_r += r_t
            steps += 1

            if terminal or steps >= MAX_EPISODE_STEPS:
                break

        episode_rewards.append(total_r)
        print("Episode {:2d}: reward = {:.4f}, steps = {}".format(ep + 1, total_r, steps))

    sess.close()
    return np.array(episode_rewards, dtype=float)


def main():
    ckpt_path = os.path.join(CKPT_DIR, f"{CKPT_PREFIX}-{CKPT_STEP}")
    print("MODE =", MODE)
    print("Evaluating checkpoint:", ckpt_path)

    if not (os.path.exists(ckpt_path + ".index") or os.path.exists(ckpt_path)):
        print("Checkpoint not found for step", CKPT_STEP, "under", CKPT_DIR)
        return

    rewards = eval_checkpoint_5m(ckpt_path)

    mean_r = rewards.mean()
    min_r = rewards.min()
    max_r = rewards.max()
    std_r = rewards.std()

    print("\n========== {} 5M Checkpoint Evaluation ==========".format(MODE.upper()))
    print(f"Num episodes     : {len(rewards)}")
    print(f"Mean reward      : {mean_r:.4f}")
    print(f"Std  reward      : {std_r:.4f}")
    print(f"Min  reward      : {min_r:.4f}")
    print(f"Max  reward      : {max_r:.4f}")
    print("Episode rewards  :", rewards)
    print("==================================================")

    # ---------- Plot 1: Episode Reward + mean + ±1 std ----------
    episodes = np.arange(1, len(rewards) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(episodes, rewards, marker='o', label='Episode reward')
    plt.axhline(mean_r, color='red', linestyle='--', label=f'Mean = {mean_r:.2f}')
    plt.fill_between(episodes,
                     mean_r - std_r,
                     mean_r + std_r,
                     color='red',
                     alpha=0.15,
                     label='Mean ± 1 SD')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'{MODE.upper()} @ 5M: Episode rewards')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # ---------- Plot 2: Reward distribution (histogram) ----------
    plt.figure(figsize=(6, 4))
    plt.hist(rewards, bins='auto', alpha=0.7)
    plt.xlabel('Reward')
    plt.ylabel('Count')
    plt.title(f'{MODE.upper()} @ 5M: Reward distribution (N={len(rewards)})')
    plt.grid(True)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()

