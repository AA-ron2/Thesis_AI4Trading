import numpy as np

def generate_trajectory(env, agent, episodes=1):
    """
    Works for both return_vectorized=False (RL single traj)
    and return_vectorized=True (batch).
    """
    all_rewards = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_rewards = []
        while not done if isinstance(done, bool) else not done.all():
            action = agent.get_action(obs)
            obs, r, done, truncated, info = env.step(action)
            ep_rewards.append(r)
            if isinstance(done, bool) and done:
                break
            if not isinstance(done, bool) and done.all():
                break
        all_rewards.append(np.array(ep_rewards, dtype=float))
    return all_rewards
