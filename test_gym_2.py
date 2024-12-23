import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
env.reset()

for i in range(100):
    done = False
    game_rew = 0

    while not done:
        action = env.action_space.sample()
        obs, rew, terminated, truncated, info = env.step(action)
        game_rew += rew

        # Combine `terminated` (episode ended in a terminal state) 
        # and `truncated` (time limit or another truncation condition)
        done = terminated or truncated

        if done:
            print(f"Episode {i} finished, reward: {game_rew}")
            env.reset()

