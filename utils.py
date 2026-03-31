import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curve(rewards_history: list, window: int = 100, save_path: str = "learning_curve.png"):
    """
    에피소드별 보상 기록을 바탕으로 이동 평균(Moving Average) 학습 곡선을 그리고 저장합니다.
    """
    # 이동 평균 계산 (최근 window 개의 에피소드 평균)
    moving_avg = [np.mean(rewards_history[max(0, i - window):i + 1]) for i in range(len(rewards_history))]
    
    plt.figure(figsize=(10, 5))
    plt.plot(moving_avg, color='blue', label=f'Moving Average (Window={window})')
    plt.title('Q-Learning Learning Curve (FrozenLake)')
    plt.xlabel('Episodes')
    plt.ylabel('Success Rate / Reward')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # 그래프를 이미지 파일로 저장
    plt.savefig(save_path)
    plt.close()
    print(f"📊 학습 곡선 그래프가 '{save_path}'로 저장되었습니다!")
