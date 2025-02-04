# Breakout Reinforcement Learning

This repository contains a reinforcement learning project for training an agent to play the classic game Breakout using Deep Q-Networks (DQN).

## Demo



https://github.com/user-attachments/assets/9c5cda9d-3049-4b12-bae4-2d79afaceca6



## Project Structure

- `QN_breakout_vision.py`: Contains the implementation of the DQN agent with a Convolutional Neural Network (CNN) for processing the screen as visual input.
- `Breakout_vision.py`: Contains the implementation of the Breakout game with visual input.
- `QN.py`: Contains the implementation of the DQN agent with a simple feedforward neural network.
- `Breakout.py`: Contains the implementation of the Breakout game with a simple state representation.
- `test_vision.py`: Script for testing the trained DQN agent with the screen as visual input.
- `test.py`: Script for testing the trained DQN agent with simple state representation.
- `model/`: Directory for saving and loading trained models.
- `README.md`: This file.

## Requirements

- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- Pygame
- PIL (Python Imaging Library)

## Installation

1. Clone the repository:

   ```sh
   git clone https://github.com/yourusername/breakout-rl.git
   cd breakout-rl
   ```

2. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

## Training

To train the DQN agent with visual input, run:

```sh
python QN_breakout_vision.py
```

To train the DQN agent with simple state representation, run:

```sh
python QN.py
```

## Testing

To test the trained DQN agent with visual input (This is not fully trained), run:

```sh
python test_vision.py
```

To test the trained DQN agent with simple state representation, run:

```sh
python test.py
```

## License

This project is licensed under the MIT License.

## Acknowledgments

- The implementation of the DQN algorithm is based on the original paper "Playing Atari with Deep Reinforcement Learning" by Mnih et al.
- The Breakout game implementation is inspired by various open-source projects and tutorials.
