import os
import numpy as np
from environment import FrozenLakeManager
from q_agent import QLearningAgent
from utils import plot_learning_curve, visualize_optimal_policy

def main():
    # ==========================================
    # Phase 1: 환경 및 에이전트 초기화
    # ==========================================
    is_slippery = False
    
    # 1. 환경 매니저 인스턴스화
    manager = FrozenLakeManager(map_name="4x4", is_slippery=is_slippery)
    
    # 2. 에이전트 인스턴스화 (최적화 실험으로 도출된 Alpha=0.1, Gamma=0.99 적용)
    agent = QLearningAgent(
        state_size=manager.state_size,
        action_size=manager.action_size, 
        learning_rate=0.1, 
        gamma=0.99, 
        epsilon=1.0
    )
    
    episodes = 5000
    rewards_history = []
    
    print(f"🚀 [Training Pipeline Started] Environment: FrozenLake-v1 (Slippery={is_slippery})")
    print("-" * 50)
    
    # ==========================================
    # Phase 2: 강화학습 메인 루프
    # ==========================================
    for ep in range(episodes):
        state = manager.reset()
        done = False
        total_reward = 0
        
        while not done:
            # a. 현재 상태에서 행동 선택 (Epsilon-Greedy)
            action = agent.choose_action(state)
            
            # b. 환경에서 한 스텝 진행 후 피드백 수신 (info 포함 4개 리턴값 정확히 반영)
            next_state, reward, done, info = manager.step(action)
            
            # c. 벨만 방정식을 통한 Q-Table 업데이트
            agent.learn(state, action, reward, next_state, done)
            
            # d. 상태 전이
            state = next_state
            total_reward += reward
            
        # 에피소드 종료 시 Epsilon 감소 및 보상 기록
        agent.decay_epsilon()
        rewards_history.append(total_reward)
        
        # 1000 에피소드마다 진행 상황 로깅
        if (ep + 1) % 1000 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(f"[*] Episode: {ep + 1}/{episodes} | Avg Reward (Last 100): {avg_reward:.2f} | Epsilon: {agent.epsilon:.3f}")
            
    print("-" * 50)
    print("✅ 모든 에피소드 학습 완료!")
    manager.close()

    # ==========================================
    # Phase 3: 결과 시각화 및 데이터 추출
    # ==========================================
    os.makedirs("results", exist_ok=True)
    
    # 1. 학습 곡선 시각화
    print("[System] 학습 곡선(Learning Curve) 생성 중...")
    plot_learning_curve(rewards_history, save_path="results/learning_curve.png")
    
    # 2. 동적 정책 맵(Policy Map) 시각화 
    print("[System] 최적 정책 맵(Dynamic Policy Map) 생성 중...")
    
    # 환경 매니저를 통해 실제 환경(env) 객체에서 원본 맵 데이터를 동적으로 추출
    current_map = manager.env.unwrapped.desc 
    policy_output_path = "results/frozenlake_optimal_policy.png"
    
    # 추출한 맵 데이터와 학습된 Q-Table을 범용 시각화 함수에 주입
    visualize_optimal_policy(agent.q_table, current_map, policy_output_path)
    
    print(f"✅ 최적 정책 시각화 이미지 저장 완료: {policy_output_path}")

if __name__ == "__main__":
    main()
