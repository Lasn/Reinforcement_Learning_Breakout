import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os
import matplotlib.pyplot as plt

from Breakout import BreakoutGame


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64, device="cpu"):
        super(QNetwork, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x).to(self.device)


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, device="cpu"):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.device = device

    def __len__(self):
        return len(self.memory)

    def add(self, experience):
        self.memory.append(experience)

    def sample(self):
        experiences = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)

        states = torch.tensor(np.vstack(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.vstack(actions), dtype=torch.int64).to(self.device)
        rewards = torch.tensor(np.vstack(rewards), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.vstack(next_states), dtype=torch.float32).to(
            self.device
        )
        dones = torch.tensor(np.vstack(dones).astype(np.uint8), dtype=torch.float32).to(
            self.device
        )

        return (states, actions, rewards, next_states, dones)


class DQNAgent:
    def __init__(
        self,
        state_size,
        action_size,
        hidden_size=64,
        buffer_size=10000,
        batch_size=64,
        gamma=0.99,
        lr=0.001,
        update_every=4,
        tau=0.001,  # Add tau for soft update
        device="cpu",
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.update_every = update_every
        self.device = device
        self.tau = tau  # Soft update parameter

        self.qnetwork_local = QNetwork(state_size, action_size, hidden_size, device).to(
            device
        )
        self.qnetwork_target = QNetwork(
            state_size, action_size, hidden_size, device
        ).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        self.memory = ReplayBuffer(buffer_size, batch_size, device)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done))
        self.t_step = (self.t_step + 1) % self.update_every

        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences)

    def act(self, state, eps=0.0):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        Q_targets_next = (
            self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        )
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        Q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = nn.MSELoss()(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target)

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )

    def save_model(self, model, file_name="model.pth"):
        model_folder_path = "./model"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        model.to("cpu")
        torch.save(model.state_dict(), file_name)
        model.to(self.device)

    def load_model(self, file_name="model.pth"):
        model_folder_path = "./model"
        file_name = os.path.join(model_folder_path, file_name)
        if os.path.exists(file_name):
            self.qnetwork_local.load_state_dict(
                torch.load(file_name, map_location=self.device)
            )
            self.qnetwork_target.load_state_dict(
                torch.load(file_name, map_location=self.device)
            )
            print(f"Loaded model from {file_name}")
        else:
            print(f"No model found at {file_name}, training from scratch")


def train(DQNAgent, device="cpu"):
    env = BreakoutGame()

    state_size = 5
    action_size = 3

    agent = DQNAgent(state_size, action_size, device=device)

    # load model if needed to continue training from a checkpoint
    # agent.load_model("good/Breakout_3954_record_600.pth")

    n_episodes = 50000
    max_t = 30000
    eps_start = 0.10
    eps_end = 0.01
    eps_decay = 0.995

    best_score = 0
    scores = []
    eps_s = []
    model_name = "Breakout"
    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        eps = max(eps_end, eps_decay * eps_start)
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, score = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        scores.append(score)
        eps_start *= eps_decay
        print(f"Episode {i_episode}\tScore: {score}")
        if score > best_score:
            best_score = score
            agent.save_model(
                model=agent.qnetwork_local,
                file_name=f"{model_name}_{i_episode}_record_{int(best_score)}.pth",
            )
    agent.save_model(
        model=agent.qnetwork_local,
        file_name=f"{model_name}_{i_episode}_record_{int(best_score)}.pth",
    )

    plt.plot(scores)
    plt.plot(eps_s)
    plt.ylabel("Score")
    plt.xlabel("Episode #")
    plt.savefig("cartpole_scores.png")
    plt.show()


if __name__ == "__main__":
    # if torch.cuda.is_available():
    #     DEVICE = torch.device("cuda")
    # elif torch.backends.mps.is_available():
    #     DEVICE = torch.device("mps")
    # else:
    #     DEVICE = torch.device("cpu")
    # print("Device:", DEVICE)
    train(DQNAgent)
