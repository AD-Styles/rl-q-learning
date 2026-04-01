# 🤖 Q-Learning for FrozenLake Environment

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29.1-lightgrey.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 1. 프로젝트 요약 (Project Overview)
본 프로젝트는 강화학습의 핵심 기초 알고리즘인 **Q-Learning**을 활용하여, 에이전트가 살얼음판(FrozenLake) 미로를 무사히 건너 목표 지점에 도달하도록 학습시키는 파이썬(Python) 기반의 구현체입니다. 단일 스크립트 형태를 벗어나, 에이전트(Agent)와 환경(Environment)을 분리한 **객체지향(OOP) 구조**로 설계하여 코드의 재사용성과 가독성을 높였습니다.

## 2. 핵심 목표 (Motivation)
* **강화학습(RL) 기초 구현:** 정답이 없는 환경에서 시행착오를 통해 최적의 행동을 스스로 찾아내는 에이전트의 학습 과정을 코드로 구현합니다.
* **마르코프 결정 과정(MDP) 이해:** 탐험(Exploration)과 활용(Exploitation)의 균형을 맞추는 epsilon-Greedy 전략을 적용합니다.
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
에이전트는 다음 업데이트 공식을 사용하여 상태-행동 가치함수인 Q(s,a) 테이블을 지속적으로 갱신합니다.

$$Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$$

## 4. 프로젝트 구조 (Repository Structure)
유지보수와 확장이 용이하도록 역할을 분리하여 모듈화했습니다.

```text
├── environment.py   # FrozenLake 환경 셋업 및 래퍼 클래스
├── q_agent.py       # Q-Table 관리 및 학습 알고리즘(두뇌) 클래스
├── train.py         # 하이퍼파라미터 설정 및 메인 학습 루프 실행
├── utils.py         # 학습 결과(성공률, 보상 등) 시각화 도구
├── requirements.txt # 의존성 라이브러리 목록 (추가)
├── README.md        # 프로젝트 설명 문서
└── .gitignore   
```

## 5. 실행 방법 (Quick Start)

### Requirements 설치
```bash
pip install -r requirements.txt
```

### 학습 파이프라인 실행
결정적 환경(`is_slippery=False`)에서 에이전트를 5,000 에피소드 동안 학습시킵니다. 
* **적용된 하이퍼파라미터:** 학습률($\alpha$)=0.1, 할인율($\gamma$)=0.99, 탐험률($\epsilon$)=1.0 (감소율 적용)

```bash
python train.py
```

## 6. 실험 결과 및 분석 (Results & Analysis)
환경의 불확실성(`is_slippery`) 여부에 따른 학습 성능을 비교 분석했습니다.

![Learning Curve](learning_curve.png)

| 환경 난이도 | 랜덤 에이전트 성공률 | Q-Learning 성공률 | 분석 포인트 |
| :--- | :--- | :--- | :--- |
| **결정적 (Deterministic)**<br>`is_slippery=False` | ~1-5% | **~95-100%** | 행동의 결과가 100% 보장되므로, 최적 경로(Q-Table)가 빠르게 수렴하여 완벽한 주행이 가능함. |
| **확률적 (Stochastic)**<br>`is_slippery=True` | ~1-2% | **~70-80%** | 의도한 방향으로 이동할 확률이 1/3로 감소하여 미끄러짐 발생. 100% 성공은 불가능하나, 구멍을 피하는 안전한 우회 경로를 학습하는 것을 확인. |

## 7. 향후 과제 (Future Work)
본 프로젝트에서 구현한 전통적인 Q-Table 방식은 상태(State)가 한정된 FrozenLake(16개)와 같은 소규모 환경에서는 강력합니다. 

하지만 연속적인 상태 공간을 가지거나 복잡한 화면 이미지를 입력으로 받는 환경에서는 **상태 폭발(State Explosion)** 문제가 발생하여 표(Table)로 저장하는 것이 불가능합니다. 향후 본 프로젝트의 아키텍처를 기반으로, Q-Table을 딥러닝 신경망으로 대체하여 가치를 예측하는 **DQN(Deep Q-Network)** 알고리즘으로 모델을 확장해 나갈 계획입니다.

## 8. 회고 (Retrospective)
이번 프로젝트를 진행하며 단순히 기능이 동작하는 수준을 넘어, 엔지니어링 역량과 분석력을 키울 수 있었습니다.

| 영역 | 학습 및 성장 포인트 |
| :--- | :--- |
| **아키텍처 설계** | 주피터 노트북 스크립트를 객체지향(OOP) 기반의 모듈화 코드로 전환하며, 유지보수성과 가독성을 고려한 코드 작성법을 체득했습니다. |
| **알고리즘 분석** | 결정적/확률적 환경 비교 실험을 통해 현실 세계의 불확실성 속에서 알고리즘이 맞이하는 한계를 분석하고, 마르코프 결정 과정(MDP)의 실질적 의미를 도출했습니다. |
| **실무 대비** | 올여름 본격적인 취업 시장 진입을 앞두고, 작동하는 코드를 넘어 논리적인 디렉토리 구조와 체계적인 문서화(README) 역량을 갖춘 개발자로 도약하는 계기가 되었습니다. |
