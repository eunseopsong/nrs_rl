# 고정 힘 가공량 균일도 제어

이 디렉터리 `tasks/manager_based/nrs_rl/`가 Git 저장소의 루트다.
새 소스와 수정은 이 저장소 안에서 관리한다. 실제 신경망 강화학습은 `mdp/`,
비강화학습 정책 최적화는 `model_based/`, 공통 계산·시각화·디버깅은 `utils/`에 둔다.

```text
nrs_rl/                         # Git 루트
├── mdp/
│   ├── env_cfg.py             # Isaac 환경 구성
│   ├── observation.py, action.py, rewards.py, terminations.py
│   ├── removal_*.py           # 고정 힘 공간 제거량 PPO MDP
│   ├── agents/                # PPO actor/critic, 학습, SKRL 설정
│   ├── evaluation/            # PPO 후보 선정, Isaac 검증, save 관리
│   ├── runtime/               # PPO ROS 추론
│   └── tests/
├── model_based/
│   ├── policies/              # OSQP/Powell/CEM 기반 정책·export
│   ├── training/              # 비강화학습 최적화
│   ├── runtime/               # 비강화학습 추론과 고정 배포 의존 소스
│   ├── evaluation/
│   ├── deployment/            # 배포 산출물 생성·경로 관리
│   └── tests/
├── utils/
│   ├── preston_surface.py, rotary_geometry.py, removal_metrics.py
│   ├── visualization.py, debug.py, adaptive_velocity_debug.py
│   ├── reporting/             # 두 계열이 공유하는 분석·그래프·PDF
│   ├── diagnostics/           # ROS·측정·시각화 진단
│   └── tests/
├── assets/, datasets/          # 로봇과 기준 경로 입력
├── y2_control_pybind/           # 공통 하위 힘 제어기
├── paths.py
└── .gitignore
```

이전 `agents/`, `evaluation/`, `runtime/`, `tests/` 및 일부 단일 파일 경로는
같은 저장소 내부의 새 소스를 가리키는 호환 링크다. 새 코드는 위 정식 경로에 작성한다.
바깥 `tasks/model_based` 경로를 새 코드나 import에서 사용하지 않는다.

## 공정 조건과 성능 지표

- 목표 접촉력 **20 N 고정**, 제거량 계산에서만 **가상 1,000 RPM 고정**.
- PPO action은 전진 속도 1개, 이동속도 범위 1.5–12 mm/s. 속도에 따른 체류시간만 변한다.
- 역주행, 두 번째 전체 패스, 별도 정지시간 action은 사용하지 않는다.
- 주 지표는 완주 후 **공간 제거 깊이 CV**. 첫 활성 2초 기여와 초기 18 mm 경로 footprint를 제외한다.
- 동일 0.5 mm 격자·고정 ROI에서 깊이 0인 셀도 포함하며 모든 성능 백분율은 동일 조건 **mode 6 대비**다.
- processing ROI, full ROI, whole grid 제거량이 각각 mode 6의 50% 이상이어야 한다.
- 제거량 단위는 미보정 Preston 모델의 `h/K`, `V/K`다. 100 N 초과 즉시 정지 조건을 유지한다.

PPO는 회전 제거량·이송 명령 응답 근사 환경에서 stochastic PPO-Clip으로 학습하고,
별도 Isaac 로봇·접촉 물리 실행으로 검증한다. Isaac에서 직접 학습했다고 표현하지 않는다.
16D 관측과 progress Fourier 특징, 128×128 actor/critic을 사용하며 actor 파라미터는 20,865개다.
비-PPO profile을 teacher로 사용하지 않는다. MDP 정의는 `mdp/removal_config.py`,
`removal_observation.py`, `removal_action.py`, `removal_rewards.py`에서 확인한다.

## 실행

`~/nrs_rl` 작업 공간과 `env_isaaclab` 환경에서 실행한다. 이미 사용한 결과 폴더는 덮어쓰지 않는다.

```bash
cd ~/nrs_rl
conda activate env_isaaclab
python -m nrs_rl.tasks.manager_based.nrs_rl.mdp.agents.train_fixed_force_ppo \
  --output logs/experiments/ppo_new/training_seed3201 --seed 3201 --updates 192
```

사용자가 지정한 mode 6 대비 CV **8.896%** 개선 세이브본의 actor, 학습 재개 상태,
검증 자료 위치는 `logs/experiments/uniformity_saves/PPO_FIXED.json`에 기록되어 있다.
학습을 이어갈 때는 `--resume <ppo_training_state.pt>`를 지정한다.
`mdp/evaluation/refine_fixed_force_ppo.py`는 세이브본 보존, 추가 학습, 후보 선정,
독립 Isaac 검증을 수행한다. `utils/reporting/export_ppo_refinement_results.py`는 수치 감사와
공통 그래프를 출력한다. 근사 환경의 결과만으로 최종 개선율을 주장하지 않는다.

2026-10-09 추가 학습은 세 설정을 각각 192 update 진행했다. 후보 선정 seed 3241에서
seed 3203의 추가 update 176 정책을 고정한 뒤, 독립 Isaac seed 3251·3252로 검증했다.
두 검증의 평균은 다음과 같으며 모든 백분율의 기준은 같은 batch의 mode 6이다.

| 정책 | 공간 CV 개선 | 총 제거량 유지 |
| --- | ---: | ---: |
| 기존 8.90% 세이브본 재검증 | 9.10% | 97.45% |
| 추가 학습 PPO | **10.81%** | **104.35%** |

새 후보는 `logs/experiments/ppo_fixed_force_refinement_20261009_1606/candidate/`에 별도로
보존하며 기존 세이브본을 교체하지 않는다. 두 seed 검증은 이 기준 경로의 결과이며,
다른 경로나 실제 재료 제거량까지 검증했다는 의미는 아니다.
폴더 이동 후 160개 테스트, 16 update 학습 재현, 1,000 step Isaac 재현 검사를 통과했다.
학습 파라미터와 Isaac 26개 기록 항목은 이동 전후 오차 0이었다.

```bash
python -m unittest discover -s source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/utils/tests -p 'test_*.py'
python -m unittest discover -s source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/model_based/tests -p 'test_*.py'
```

## Git과 산출물

`logs/`, `outputs/`, checkpoint, NumPy 배열 덤프, TensorBoard, 녹화, 빌드·캐시 등은
`.gitignore`로 제외한다. 실험 결과는 작업 공간의 `logs/experiments/`,
그래프는 `logs/polishing_results/`, PDF는 `reports/`에 보관하며 Git에 올리지 않는다.
필수 소형 기준 경로 HDF5와 기존 하위 힘 제어기 가중치 1개만 명시적으로 예외 처리한다.
이미 추적 중인 대용량 로그가 없는지도 확인한다. 과거 실험의 소스 스냅샷은 결과 자료로 보존한다.

기존 GUI 고정 정책은 **비-PPO**다. 새 PPO 연구 모델로 자동 교체하지 않는다.
`model_based/runtime/fixed_force_12p59_source/`는 해시를 보존하는 배포 의존 사본이다.
새 학습은 별도 checkpoint를 생성하고, 원래 세이브본 파일을 덮어쓰지 않는다.
