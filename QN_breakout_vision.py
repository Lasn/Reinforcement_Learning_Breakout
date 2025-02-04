import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os
import matplotlib.pyplot as plt
from PIL import Image

from Breakout_vision import BreakoutGame


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


class CNNQNetwork(nn.Module):
    def __init__(self, input_shape, action_size, hidden_size=512, device="cpu"):
        super(CNNQNetwork, self).__init__()
        self.input_shape = input_shape
        self.action_size = action_size
        self.device = device

        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        def conv2d_size_out(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(
            conv2d_size_out(conv2d_size_out(input_shape[1], 8, 4), 4, 2), 3, 1
        )
        convh = conv2d_size_out(
            conv2d_size_out(conv2d_size_out(input_shape[2], 8, 4), 4, 2), 3, 1
        )
        linear_input_size = convw * convh * 64

        self.fc1 = nn.Linear(linear_input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = torch.relu(self.conv1(state))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.flatten(start_dim=1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x).to(self.device)


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

        states = torch.tensor(np.stack(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.vstack(actions), dtype=torch.int64).to(self.device)
        rewards = torch.tensor(np.vstack(rewards), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.stack(next_states), dtype=torch.float32).to(
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

        self.qnetwork_local = CNNQNetwork(
            state_size, action_size, hidden_size, device
        ).to(device)
        self.qnetwork_target = CNNQNetwork(
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
        state = state.view(1, *self.state_size)
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
        # Reshape states to [batch_size, channels, height, width]
        batch_size = states.size(0)
        states = states.view(batch_size, *self.state_size)
        next_states = next_states.view(batch_size, *self.state_size)

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

    state_size = (1, 84, 84)
    action_size = 3

    agent = DQNAgent(state_size, action_size, device=device)

    agent.load_model("Breakout_vision_275_record_210.pth")

    n_episodes = 100000
    max_t = 30000
    eps_start = 1.0
    eps_end = 0.01
    eps_decay = 0.995

    best_score = 0
    scores = []
    avg_scores = []
    model_name = "Breakout_vision"
    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        eps = max(eps_end, eps_decay * eps_start)
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, score = env.step(action, draw_game=True)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        scores.append(score)
        avg_score = np.mean(scores)
        eps_start *= eps_decay
        print(f"Episode {i_episode}\tScore: {score}")
        if score > best_score:
            best_score = score
            agent.save_model(
                model=agent.qnetwork_local,
                file_name=f"{model_name}_{i_episode}_record_{int(best_score)}.pth",
            )
        if i_episode % 1000 == 0:
            plt.plot(scores)
            plt.plot(avg_scores)
            plt.ylabel("Score")
            plt.xlabel("Episode #")
            plt.savefig("breakout_scores.png")
            plt.show()
    agent.save_model(
        model=agent.qnetwork_local,
        file_name=f"{model_name}_{i_episode}_record_{int(best_score)}.pth",
    )

    plt.plot(scores)
    plt.plot(avg_scores)
    plt.ylabel("Score")
    plt.xlabel("Episode #")
    plt.savefig("breakout_scores.png")
    plt.show()


if __name__ == "__main__":
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")
    print("Device:", DEVICE)
    train(DQNAgent)
