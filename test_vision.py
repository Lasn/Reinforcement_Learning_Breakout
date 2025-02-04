import numpy as np
from Breakout_vision import BreakoutGame
from QN_breakout_vision import DQNAgent, ReplayBuffer, CNNQNetwork
import matplotlib.pyplot as plt


env = BreakoutGame()

state_size = (1, 84, 84)
action_size = 3

agent = DQNAgent(state_size, action_size, device="cpu")

# Load the model
agent.load_model("good/Breakout_vision_1087_record_220.pth")

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
plt.savefig("breakout_vision_scores_test.png")
plt.show()
