import numpy as np
from Breakout import BreakoutGame
from QN import DQNAgent
import matplotlib.pyplot as plt


env = BreakoutGame()

state_size = 5
action_size = 3

agent = DQNAgent(state_size, action_size, device="cpu")

# Load the model
agent.load_model("good/Breakout_3954_record_600.pth")

n_episodes = 10
max_t = 100000
scores = []

for i_episode in range(1, n_episodes + 1):
    state = env.reset()
    total_reward = 0
    for t in range(max_t):
        action = agent.act(state)
        next_state, reward, done, score = env.step(action, draw_game=True)
        state = next_state
        if done:
            break
    scores.append(score)


print(f"Average Score over {n_episodes} episodes: {np.mean(scores)}")
plt.plot(scores)
plt.xlabel("Episode")
plt.ylabel("Score")
plt.title("Scores over episodes")
plt.savefig("breakout_scores_test.png")
plt.show()
