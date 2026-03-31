import numpy as np
from q_agent import QLearningAgent
from environment import FrozenLakeManager
from utils import plot_learning_curve  # 새롭게 추가된 시각화 모듈 임포트

def train_agent(episodes: int = 5000, is_slippery: bool = False):
    """
    지정된 에피소드만큼 Q-Learning 에이전트를 학습시킵니다.
    """
    print(f"--- 학습 시작 (is_slippery={is_slippery}, 에피소드={episodes}) ---")
    
    # 1. 환경 및 에이전트 초기화
    env_manager = FrozenLakeManager(map_name="4x4", is_slippery=is_slippery)
    agent = QLearningAgent(state_size=env_manager.state_size, 
                           action_size=env_manager.action_size,
                           learning_rate=0.1, gamma=0.99, epsilon=1.0)
    
    # 성과 추적을 위한 리스트
    rewards_history = []
    
    # 2. 메인 학습 루프
    for episode in range(episodes):
        state = env_manager.reset()
        done = False
        total_reward = 0
        
        while not done:
            # a. 행동 선택 (Epsilon-Greedy)
            action = agent.choose_action(state)
            
            # b. 환경에서 행동 실행
            next_state, reward, done, _ = env_manager.step(action)
            
            # c. Q-Table 업데이트 (벨만 방정식)
            agent.learn(state, action, reward, next_state, done)
            
            # d. 상태 전환
            state = next_state
            total_reward += reward
            
        # 에피소드 종료 후 Epsilon 감소 및 보상 기록
        agent.decay_epsilon()
        rewards_history.append(total_reward)
        
        # 1000 에피소드마다 진행 상황 출력
        if (episode + 1) % 1000 == 0:
            recent_success_rate = np.mean(rewards_history[-1000:]) * 100
            print(f"Episode {episode + 1}/{episodes} | 최근 1000번 성공률: {recent_success_rate:.1f}% | Epsilon: {agent.epsilon:.3f}")
            
    env_manager.close()
    print("--- 학습 종료 ---")
    return rewards_history, agent.q_table

if __name__ == "__main__":
    # 1. 결정적 환경(미끄러지지 않음)에서 5000번 학습 실행
    history, final_q_table = train_agent(episodes=5000, is_slippery=False)
    
    # 2. 새롭게 추가된 부분: 학습이 끝난 후 utils.py의 시각화 함수 호출
    plot_learning_curve(history, window=100, save_path="learning_curve.png")
