import numpy as np
from typing import Tuple

class QLearningAgent:
    """
    Q-Learning 알고리즘을 수행하는 에이전트 클래스입니다.
    """
    
    def __init__(self, state_size: int, action_size: int, 
                 learning_rate: float = 0.1, gamma: float = 0.99, 
                 epsilon: float = 1.0, epsilon_decay: float = 0.995, min_epsilon: float = 0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table = np.zeros((state_size, action_size)) 

    def choose_action(self, state: int) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        else:
            return int(np.argmax(self.q_table[state]))

    def learn(self, state: int, action: int, reward: float, next_state: int, done: bool):
        future_q = 0.0 if done else np.max(self.q_table[next_state])
        td_target = reward + self.gamma * future_q
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.lr * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
