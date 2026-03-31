import numpy as np
from typing import Tuple

class QLearningAgent:
    """
    Q-Learning 알고리즘을 수행하는 에이전트 클래스입니다.
    상태 공간과 행동 공간을 바탕으로 Q-Table을 생성하고, 환경과의 상호작용을 통해 가치를 업데이트합니다.
    """
    
    def __init__(self, state_size: int, action_size: int, 
                 learning_rate: float = 0.1, gamma: float = 0.99, 
                 epsilon: float = 1.0, epsilon_decay: float = 0.995, min_epsilon: float = 0.01):
        """
        초기 하이퍼파라미터 및 Q-Table을 세팅합니다.
        """
        self.state_size = state_size
        self.action_size = action_size
        
        self.lr = learning_rate        # 새 정보를 받아들이는 속도 [cite: 261]
        self.gamma = gamma             # 미래 보상에 대한 할인율 [cite: 263]
        self.epsilon = epsilon         # 초기 탐험 확률
        self.epsilon_decay = epsilon_decay # 탐험 축소 비율 [cite: 385]
        self.min_epsilon = min_epsilon
        
        # Q-Table을 모두 0으로 초기화 (상태 수 x 행동 수) [cite: 395, 396]
        self.q_table = np.zeros((state_size, action_size)) 

    def choose_action(self, state: int) -> int:
        """
        Epsilon-Greedy 정책에 따라 탐험(Exploration) 또는 활용(Exploitation)을 선택합니다.
        """
        if np.random.random() < self.epsilon:
            # 탐험: 랜덤 행동 선택 [cite: 374, 376]
            return np.random.randint(self.action_size) [cite: 377]
        else:
            # 활용: 현재 상태에서 가장 Q값이 높은 행동 선택 [cite: 378]
            return int(np.argmax(self.q_table[state])) [cite: 379]

    def learn(self, state: int, action: int, reward: float, next_state: int, done: bool):
        """
        벨만 방정식을 사용하여 Q-Table을 업데이트합니다.
        공식: Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a(Q(s',a')) - Q(s,a)] 
        """
        # 목표 도달 또는 에피소드 종료 시 미래 가치는 0으로 처리 [cite: 406]
        future_q = 0.0 if done else np.max(self.q_table[next_state])
        
        # TD Target 및 오차 계산 [cite: 406, 407]
        td_target = reward + self.gamma * future_q
        td_error = td_target - self.q_table[state, action]
        
        # Q-Table 업데이트 [cite: 408]
        self.q_table[state, action] += self.lr * td_error

    def decay_epsilon(self):
        """
        매 에피소드 종료 후 epsilon 값을 감소시켜 점진적으로 탐험을 줄입니다. [cite: 285]
        """
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
