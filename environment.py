import gymnasium as gym
from typing import Tuple, Any

class FrozenLakeManager:
    """
    Gymnasium의 FrozenLake 환경을 관리하는 래퍼(Wrapper) 클래스입니다.
    """
    
    def __init__(self, map_name: str = "4x4", is_slippery: bool = False, render_mode: str = None):
        """
        FrozenLake 환경을 초기화합니다.
        """
        
        self.env = gym.make("FrozenLake-v1", map_name=map_name, is_slippery=is_slippery, render_mode=render_mode)
        
        self.state_size = self.env.observation_space.n
        self.action_size = self.env.action_space.n

    def reset(self) -> int:
        state, _ = self.env.reset()
        return state

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return next_state, reward, done, info

    def close(self):
        self.env.close()
