import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_learning_curve(rewards, save_path="results/learning_curve.png", window=100):
    """
    학습 진행에 따른 에피소드별 보상을 이동 평균(Moving Average)으로 시각화.
    """
    plt.figure(figsize=(10, 5))
    
    # 윈도우 사이즈를 기반으로 이동 평균 계산
    if len(rewards) >= window:
        moving_avg = [np.mean(rewards[max(0, i - window):i + 1]) for i in range(len(rewards))]
        plt.plot(moving_avg, color='teal', linewidth=2, label=f'Moving Avg (Window={window})')
    else:
        plt.plot(rewards, color='teal', linewidth=2, label='Reward')
        
    plt.title("Training Learning Curve", fontsize=15, fontweight='bold')
    plt.xlabel("Episodes", fontsize=12)
    plt.ylabel("Reward (Moving Avg)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_optimal_policy(q_table, env_map, save_path):
    """
    환경에서 동적으로 추출한 맵 데이터(env_map)를 바탕으로 최적 정책을 시각화.
    """
    # 맵의 동적 크기 측정 (4x4, 8x8 등 모두 호환)
    grid_size = env_map.shape[0]
    
    # 행동(Action) 값을 화살표 기호로 매핑 (FrozenLake 기준: 0:Left, 1:Down, 2:Right, 3:Up)
    arrow_map = {0: '←', 1: '↓', 2: '→', 3: '↑'}
    
    fig, ax = plt.subplots(figsize=(grid_size * 1.5, grid_size * 1.5))
    
    # 배경 그리드 생성
    grid = np.zeros((grid_size, grid_size))
    sns.heatmap(grid, annot=False, cmap="Blues", cbar=False, linewidths=1, linecolor='black', ax=ax)
    
    # 동적 맵 데이터를 순회하며 화살표 텍스트 삽입
    for i in range(grid_size):
        for j in range(grid_size):
            state = i * grid_size + j
            char = env_map[i][j].decode('utf-8')  # 바이트 문자열 디코딩
            
            text = char
            color = "black"
            
            if char == 'H': 
                color = "red"
            elif char == 'G':
                color = "green"
            else: 
                best_action = np.argmax(q_table[state])
                text = f"{char}\n{arrow_map[best_action]}"
                
            ax.text(j + 0.5, i + 0.5, text, ha='center', va='center', 
                    color=color, fontsize=16, fontweight='bold')
    
    plt.title("Agent's Optimal Policy Navigation", fontsize=16, fontweight='bold', pad=15)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
