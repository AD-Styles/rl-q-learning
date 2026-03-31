# 🤖 Q-Learning for FrozenLake Environment

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29.1-lightgrey.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 1. 프로젝트 요약 (Project Overview)
본 프로젝트는 강화학습의 핵심 기초 알고리즘인 **Q-Learning**을 활용하여, 에이전트가 살얼음판(FrozenLake) 미로를 무사히 건너 목표 지점에 도달하도록 학습시키는 파이썬(Python) 기반의 구현체입니다. 단일 스크립트 형태를 벗어나, 에이전트(Agent)와 환경(Environment)을 분리한 **객체지향(OOP) 구조**로 설계하여 코드의 재사용성과 가독성을 높였습니다.

## 2. 핵심 목표 (Motivation)
* **강화학습(RL) 기초 구현:** 정답이 없는 환경에서 시행착오를 통해 최적의 행동을 스스로 찾아내는 에이전트의 학습 과정을 코드로 구현합니다.
* **마르코프 결정 과정(MDP) 이해:** 탐험(Exploration)과 활용(Exploitation)의 균형을 맞추는 $\epsilon$-Greedy 전략을 적용합니다.
* **확률적 환경 분석:** 미끄러짐(Slippery) 유무에 따른 환경의 난이도 변화와 Q-Table의 수렴 차이를 분석합니다.

## 3. 환경 소개 (Environment - FrozenLake)
Gymnasium 라이브러리에서 제공하는 `FrozenLake-v1` 환경을 사용했습니다. 에이전트는 4x4 그리드 위에서 구멍(H)을 피해 시작점(S)에서 목표 지점(G)으로 이동해야 합니다.

| 상태 기호 | 설명 | 보상 (Reward) |
| :---: | :--- | :--- |
| **S** | 시작 지점 (Start) | 0 |
| **F** | 안전한 얼음판 (Frozen) | 0 |
| **H** | 구멍 (Hole) - 빠지면 에피소드 종료 | 0 |
| **G** | 목표 지점 (Goal) - 도달 시 에피소드 성공 | **+1** |



### 🧠 사용된 알고리즘: 벨만 방정식 기반 Q-Learning
에이전트는 다음 업데이트 공식을 사용하여 상태-행동 가치함수인 $Q(s,a)$ 테이블을 지속적으로 갱신합니다.
$$Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$$

## 4. 프로젝트 구조 (Repository Structure)
유지보수와 확장이 용이하도록 역할을 분리하여 모듈화했습니다.

```text
├── environment.py   # FrozenLake 환경 셋업 및 래퍼 클래스
├── q_agent.py       # Q-Table 관리 및 학습 알고리즘(두뇌) 클래스
├── train.py         # 하이퍼파라미터 설정 및 메인 학습 루프 실행
├── utils.py         # 학습 결과(성공률, 보상 등) 시각화 도구
├── README.md        # 프로젝트 설명 문서
└── .gitignore
