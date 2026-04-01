import gymnasium as gym
from typing import Tuple, Any

class FrozenLakeManager:
    """
    Gymnasium의 FrozenLake 환경을 관리하며, 커스텀 보상 설계를 적용하는 래퍼 클래스입니다.
    """
    
    def __init__(self, map_name: str = "4x4", is_slippery: bool = False, render_mode: str = None):
        self.env = gym.make("FrozenLake-v1", map_name=map_name, is_slippery=is_slippery, render_mode=render_mode)
        self.state_size = self.env.observation_space.n
        self.action_size = self.env.action_space.n

    def reset(self) -> int:
        state, _ = self.env.reset()
        return state

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        
        # --- 커스텀 보상 설계 (Reward Shaping) ---
        if terminated and reward == 0:
            # 목표 도달 실패(구멍에 빠짐): 강력한 패널티 부여
            reward = -1.0
        elif not done:
            # 1보 이동할 때마다 시간 지연 패널티: 최단 거리 유도
            reward = -0.01
        
        return next_state, reward, done, info

    def close(self):
        self.env.close()
