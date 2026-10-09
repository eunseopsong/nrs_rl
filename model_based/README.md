# 모델 기반 정책 최적화 — 비-PPO

기존 가공량 checkpoint는 OSQP/Powell/CEM 등의 최적화 결과를 TorchScript 정책으로
저장한 모델이다. 이 알고리즘을 PPO/정책 경사 강화학습으로 표현하지 않는다.
단순 수작업 규칙만 있는 것이 아니므로 `rule_based` 대신 `model_based`를 사용한다.
새 코드와 코드 수정은 Git 루트 `tasks/manager_based/nrs_rl/` 안에서만
진행한다. 새 진입점은 패키지에 두며, 기존 `scripts/` 호환 링크는 추가하지 않는다.
`runtime/fixed_force_12p59_source/`는 해시가 고정된 배포 의존 사본이므로 수정하지 않는다.

| 폴더 | 역할 |
| --- | --- |
| `policies/` | residence/speed profile과 소수 파라미터의 정책·export |
| `training/` | QP, Powell, CEM 및 bounded-parameter 탐색 |
| `runtime/` | 비-PPO 정책의 Isaac/ROS 추론과 action adapter |
| `evaluation/` | 기존 후보 선정·heldout·실증 연동 검증 |
| `../utils/reporting/` | 두 계열 공통 실험 비교, 그래프·보고서 |
| `deployment/` | 보존 배포본 생성 도구 |
| `tests/` | 기존 정책·런타임·최적화 회귀 검사 |

현재 고정 모델의 핵심은 `policies/fixed_force_dynamics.py`의
`PreviewResidenceActor`와 `training/train_fixed_force_dynamics.py`다.
433-knot residence profile을 보간하며 학습 가능한 신경망 actor가 아니다.
고정 배포본은 `deployment/fixed_force_12p59/`, 고정 기록은
`deployment/FIXED_MODEL.json`에 있다. 이 이동으로 기존 checkpoint를 교체하지 않는다.

고정 배포본의 Python 계산 의존 코드는 `runtime/fixed_force_12p59_source/`에
별도로 보존했다. 배포본의 import와 해시 검사 경로만 이 사본으로 연결했다.
mode 5·6 각각 1,250개 입력의 observation/action/depth가 변경 전과 정확히 일치하고,
전체 경로 측정 재적분과 10개 launch 구성 검사도 통과했다. 기존 checkpoint와
두 C++ 제어 바이너리는 동일하다. 이 고정 사본은 새 정책 개발용 코드와 구분한다.

공통 로봇·가공량 계산은 `../mdp/`와 `../utils/`를 사용한다.
실제 PPO 구현은 `../mdp/agents/`에 따로 있다.
과거의 가변 힘/역주행 연구 코드도 이 폴더에 보존되어 있으나 현재 고정 힘 PPO의
허용 action이나 기본 실행 설정이 아니다.

기존 `scripts/` 실행 경로는 상대 심볼릭 링크로 호환되며, 원본은
`logs/experiments/ppo_fixed_force_20261009/source_before_reorganization/`에 보존했다.
