import numpy as np
from q_agent import QLearningAgent
from environment import FrozenLakeManager
from utils import plot_learning_curve, plot_policy

def train_agent(episodes: int = 2000, alpha: float = 0.1, gamma: float = 0.99, is_slippery: bool = False):
    """
    단일 파라미터 세팅으로 에이전트를 학습시킵니다.
    """
    env_manager = FrozenLakeManager(map_name="4x4", is_slippery=is_slippery)
    agent = QLearningAgent(state_size=env_manager.state_size, 
                           action_size=env_manager.action_size,
                           learning_rate=alpha, gamma=gamma)
    
    rewards_history = []
    
    for episode in range(episodes):
        state = env_manager.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env_manager.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
        agent.decay_epsilon()
        rewards_history.append(total_reward)
            
    env_manager.close()
    return rewards_history, agent.q_table

if __name__ == "__main__":
    alphas = [0.01, 0.1, 0.5]
    gammas = [0.9, 0.99]
    episodes = 2000
    
    best_reward = -float('inf')
    best_q_table = None
    best_params = {}
    
    print("--- 🔬 하이퍼파라미터 튜닝 시작 ---")
    
    # 1. 파라미터 조합 실험 루프
    for a in alphas:
        for g in gammas:
            history, q_table = train_agent(episodes=episodes, alpha=a, gamma=g)
            avg_reward = np.mean(history[-500:]) # 최근 500 에피소드 평균 보상
            print(f"Alpha: {a}, Gamma: {g} | 최근 500번 평균 보상: {avg_reward:.2f}")
            
            if avg_reward > best_reward:
                best_reward = avg_reward
                best_q_table = q_table
                best_params = {'alpha': a, 'gamma': g}
                best_history = history
                
    print(f"\n🏆 최적 파라미터 발견: Alpha={best_params['alpha']}, Gamma={best_params['gamma']}")
    
    # 2. 최적 결과물 시각화
    plot_learning_curve(best_history, save_path="learning_curve.png")
    plot_policy(best_q_table, save_path="policy_arrows.png")
