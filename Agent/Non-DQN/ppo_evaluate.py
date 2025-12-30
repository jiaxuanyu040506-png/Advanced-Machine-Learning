import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import cv2
import numpy as np
import sys

# ---------------- Paths ----------------
sys.path.append(".")
sys.path.append("game/")

import wrapped_flappy_bird as game
from PPO import createPPONetwork

# ---------------- Config ----------------
NUM_EVAL_EPISODES = 20
MAX_EPISODE_STEPS = 10000

CHECKPOINT_DIR = "saved_networks/ppo"
CHECKPOINT_PREFIX = "bird-ppo"
CHECKPOINT_STEP = 5000000

def preprocess(obs):
    x = cv2.cvtColor(cv2.resize(obs, (80, 80)), cv2.COLOR_BGR2GRAY)
    _, x = cv2.threshold(x, 1, 255, cv2.THRESH_BINARY)
    return x

def sample_action(sess, policy_op, s_ph, state, is_logits=False):
    """从 policy 输出采样动作，自动处理 logits 或概率"""
    pi = sess.run(policy_op, feed_dict={s_ph: [state]})[0]
    if is_logits or not np.allclose(pi.sum(), 1.0):
        pi = np.exp(pi - np.max(pi))
        pi /= pi.sum()
    return np.random.choice(len(pi), p=pi)

def evaluate_model(sess, s_ph, policy_op, game_env, is_logits=False, episodes=NUM_EVAL_EPISODES, max_steps=MAX_EPISODE_STEPS):
    rewards = []
    for ep in range(episodes):
        env = game_env()
        do_nothing = np.zeros(2); do_nothing[0] = 1
        obs, _, terminal = env.frame_step(do_nothing)
        x = preprocess(obs)
        state = np.stack([x]*4, axis=2)

        ep_reward = 0
        steps = 0
        while not terminal and steps < max_steps:
            action = sample_action(sess, policy_op, s_ph, state, is_logits)
            a_vec = np.zeros(2); a_vec[action] = 1
            obs, r, terminal = env.frame_step(a_vec)
            x_new = preprocess(obs)
            state = np.append(state[:, :, 1:], np.expand_dims(x_new, 2), axis=2)
            ep_reward += r
            steps += 1

        rewards.append(ep_reward)
    return np.array(rewards)

def load_checkpoint(sess, ckpt_dir, ckpt_prefix, step):
    import os
    import tensorflow.compat.v1 as tf
    saver = tf.train.Saver()
    ckpt = os.path.join(ckpt_dir, f"{ckpt_prefix}-{step}")
    saver.restore(sess, ckpt)
    print(f"✔ Restored checkpoint from {ckpt}")

def main():
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    s_ph, policy, value, logits = createPPONetwork()
    load_checkpoint(sess, CHECKPOINT_DIR, CHECKPOINT_PREFIX, CHECKPOINT_STEP)
    rewards = evaluate_model(sess, s_ph, policy, game.GameState, is_logits=True)
    sess.close()

    print("\n====== PPO Evaluation ======")
    print(f"Average Reward : {rewards.mean():.2f}")
    print(f"Max Reward     : {rewards.max():.2f}")
    print(f"Min Reward     : {rewards.min():.2f}")
    print(f"Std Reward     : {rewards.std():.2f}")

if __name__ == "__main__":
    main()