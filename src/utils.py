import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_learning_curve(rewards_history: list, window: int = 100, save_path: str = "learning_curve.png"):
    moving_avg = [np.mean(rewards_history[max(0, i - window):i + 1]) for i in range(len(rewards_history))]
    plt.figure(figsize=(10, 5))
    plt.plot(moving_avg, color='blue', label=f'Moving Average (Window={window})')
    plt.title('Q-Learning Learning Curve (FrozenLake)')
    plt.xlabel('Episodes')
    plt.ylabel('Success Rate / Reward')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    print(f"📊 학습 곡선 그래프가 '{save_path}'로 저장되었습니다!")

def plot_policy(q_table: np.ndarray, map_size: int = 4, save_path: str = "policy_arrows.png"):
    """
    Q-Table을 바탕으로 에이전트의 최적 정책(방향 화살표)을 4x4 그리드 위에 시각화합니다.
    """
    directions = {0: '←', 1: '↓', 2: '→', 3: '↑'}
    best_actions = np.argmax(q_table, axis=1).reshape(map_size, map_size)
    
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(0, map_size)
    ax.set_ylim(0, map_size)
    
    for i in range(map_size):
        for j in range(map_size):
            action = best_actions[i, j]
            # y좌표는 위에서 아래로 그려지도록 뒤집음
            ax.text(j + 0.5, map_size - i - 0.5, directions[action],
                    ha='center', va='center', fontsize=20)
            
    ax.set_xticks(np.arange(map_size))
    ax.set_yticks(np.arange(map_size))
    ax.grid(color='black', linestyle='-', linewidth=1)
    plt.title('Learned Policy (Q-Table)')
    plt.savefig(save_path)
    plt.close()
    print(f"🧭 정책 시각화 이미지가 '{save_path}'로 저장되었습니다!")
