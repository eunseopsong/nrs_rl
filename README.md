# 가공량 균일도 강화학습

이 폴더는 신경망 PPO와 Isaac 로봇 환경, 공통 가공량 계산의 유지보수 위치다.
현재 고정 힘 PPO는 **목표 20 N, 가상 1,000 RPM, 전진 속도 1개 action**으로
가공 완료 후 표면 깊이의 **공간 CV**를 줄인다. 속도에 따른 체류시간이 가변하며
별도 정지시간 action, 역주행, 두 번째 전체 패스는 사용하지 않는다.

기존 OSQP/Powell/CEM 정책은 `../../model_based/`로 분리했다. 기존 고정 배포
checkpoint는 `deployment/FIXED_MODEL.json`에 기록되어 있으며 자동 교체하지 않는다.
`scripts/`의 기존 명령 경로는 이 소스에 연결되는 상대 심볼릭 링크로 유지한다.
기존 배포본의 Python 의존 코드는 `model_based/runtime/fixed_force_12p59_source/`에
별도로 고정하여, 유지보수 코드의 이동이 기존 mode 5·6 실행을 막지 않도록 했다.

## 현재 PPO의 파일별 역할

| 폴더 / 파일 | 역할 |
| --- | --- |
| `agents/ppo_network.py` | Gaussian actor, critic, 결정론적 TorchScript export |
| `agents/train_fixed_force_ppo.py` | rollout, GAE, PPO clipped update, 평가와 학습 상태 저장 |
| `agents/fixed_force_ppo.py` | 위 구현을 모은 공개 import API |
| `mdp/removal_env.py` | 회전 제거량·이송 응답 근사 환경의 상태와 transition |
| `mdp/removal_observation.py` | 16D 관측: 힘·속도·경로·깊이 부족량·체류시간 |
| `mdp/removal_action.py` | 전진 속도, 명령 지연·필터·제한기 |
| `mdp/removal_rewards.py` | 예상 최종 CV² 감소와 mode 6 제거량 50% 하한 penalty |
| `mdp/removal_terminations.py` | 경로 완주, 336초 제한 |
| `mdp/removal_config.py` | 고정 힘·RPM·주기·action 권한 계약 |
| `mdp/fixed_force_ppo_action.py` | 기존 하위 힘 제어기에 PPO 이송 출력 연결 |
| `utils/spatial_geometry.py`, `rotary_geometry.py` | 공통 경로/footprint/가상 회전 특징 계산 |
| `utils/preston_surface.py` | 0.5 mm 최종 깊이 지도와 제거량 적분 |
| `utils/removal_metrics.py`, `assess_fixed_force_velocity.py` | ROI, 원시 trace 재적분, 동등한 mode 6 비교 |
| `evaluation/` | Isaac 검증, numerical audit, 그래프·PDF 출력 |
| `tests/` | PPO gradient, export, 관측/속도/제거량 회귀 검사 |
| `assets/`, `datasets/`, `y2_control_pybind/` | 기존 로봇·경로·센서·C++ 제어기 |

## 학습과 검증의 범위

PPO **학습은 로봇 접촉 물리를 생략한 회전 제거량/명령 응답 근사 환경**에서 한다.
Isaac는 학습 이후 독립된 전체 로봇·접촉 물리 검증에 사용한다. 현재 PPO를
Isaac에서 직접 학습한 모델이라고 부르지 않는다. 두 seed를 각각 64 update,
2,097,152 transition 학습했고 actor는 20,865개의 신경망 파라미터를 가진다.
기존 residence profile을 teacher나 초기값으로 넣지 않았다.

모든 성능 백분율은 동등한 mode 6 대비다. 주 지표는 첫 활성 2초 기여와 첫 18 mm
기준 경로 footprint를 제외한 고정 ROI의 공간 CV다. 깊이 0인 셀도 포함한다.
제거량은 회전 Preston 모델의 `h/K`, `V/K`이며 실제 재료 제거 계수 보정값은 아니다.
processing ROI, full ROI, whole grid 각각 mode 6 제거량의 50% 이상인지 확인한다.

## 실행

저장된 검증 결과: `logs/experiments/ppo_fixed_force_20261009/final_report.json`.
PPO 후보는 같은 실험 폴더의 `candidate/checkpoints/actor.pt`에 따로 저장한다.
GUI mode 5 배포는 기존 고정 모델을 유지한다.

새 학습은 사용하지 않은 출력 폴더를 지정한다.

```bash
cd ~/nrs_rl
conda activate env_isaaclab
python scripts/train_fixed_force_ppo.py \
  --output logs/experiments/ppo_new/training_seed3101 --seed 3101 --updates 64
```

동일 구현을 패키지로 실행할 수도 있다.

```bash
python -m nrs_rl.tasks.manager_based.nrs_rl.agents.train_fixed_force_ppo \
  --output logs/experiments/ppo_new_second/training_seed3101 --seed 3101
```

학습 직후 `selection_surrogate.json`은 근사 환경 결과다. Isaac의 후보 선정 및
고정 checkpoint 독립 검증을 통과하기 전까지 최종 성능이나 배포본으로 사용하지 않는다.
`probe_fixed_force_ppo.py` / `finish_fixed_force_ppo.py`가 Isaac 검증 진입점이며,
원본 실험은 selection seed 3141, 독립 seed 3151·3152를 사용했다.

```bash
python -m unittest discover -s scripts -p 'test_*.py'
```

## 보존한 이전 강화학습

`nrs_rl_env_cfg.py`, `mdp/action.py`, `mdp/observation.py`, `mdp/rewards.py`,
`mdp/terminations.py`는 기존 Isaac/14D adaptive-velocity 환경 및 하위 제어기다.
현재 16D 공간 CV PPO의 reward는 `mdp/removal_rewards.py`다.
`agents/skrl/`, `agents/velocity_actor.py`, `agents/train_velocity_gsde.py`는 이전의
실제 SKRL/SB3 PPO 구현이며 비-PPO 최적화 코드와 구분해 이 경로에 보존했다.
Gym 등록은 명시적으로 수행하며 import 시 학습·Isaac·ROS 프로세스를 자동 실행하지 않는다.

코드 이동 전 원본은 실험 폴더 `source_before_reorganization/`에 보존했다.
이후 새 PPO 학습은 전체 Python/C++/설정 소스와 해시를 `package_source_snapshot/`에 남긴다.
