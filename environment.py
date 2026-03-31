import gymnasium as gym
from typing import Tuple, Any

class FrozenLakeManager:
    """
    Gymnasium의 FrozenLake 환경을 관리하는 래퍼(Wrapper) 클래스입니다.
    """
    
    def __init__(self, map_name: str = "4x4", is_slippery: bool = False, render_mode: str = None):
        """
        FrozenLake 환경을 초기화합니다.
        is_slippery가 True이면 확률적(Stochastic) 환경으로 동작합니다. [cite: 341, 342]
        """
        self.env = gym.make("FrozenLake-v1", map_name=map_name, is_slippery=is_slippery, render_mode=render_mode) [cite: 316]
        
        # 에이전트가 사용할 상태 공간과 행동 공간의 크기 추출
        self.state_size = self.env.observation_space.n
        self.action_size = self.env.action_space.n

    def reset(self) -> int:
        """
        환경을 초기화하고 시작 상태를 반환합니다. [cite: 194, 195]
        """
        state, _ = self.env.reset() [cite: 192]
        return state

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        """
        행동을 환경에 전달하고 결과(다음 상태, 보상, 종료 여부 등)를 반환합니다. [cite: 196, 198]
        """
        next_state, reward, terminated, truncated, info = self.env.step(action)
        # terminated(목표도달/실패) 또는 truncated(시간초과) 시 에피소드 종료
        done = terminated or truncated [cite: 404]
        return next_state, reward, done, info

    def close(self):
        """
        환경을 안전하게 종료합니다. [cite: 199, 201]
        """
        self.env.close()
