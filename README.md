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

보상 격자는 `--reward-cell-mm`으로 2 mm(기존), 1 mm, 0.5 mm를 선택할 수 있다.
`mdp/removal_reward_surface.py`가 별도 보상 표면의 CV와 제거량을 계산한다.
정책 관측은 기존 2 mm 표면에서 계산하므로 배포용 16D 입력 의미를 유지한다.
촘촘한 보상 격자도 80 ms·기준 경로의 근사 환경이며, 실제 TCP·접촉력으로
8 ms마다 적분하는 별도 Isaac 검증을 대체하지 않는다.

## 실행

`~/nrs_rl` 작업 공간과 `env_isaaclab` 환경에서 실행한다. 이미 사용한 결과 폴더는 덮어쓰지 않는다.

```bash
cd ~/nrs_rl
conda activate env_isaaclab
python -m nrs_rl.tasks.manager_based.nrs_rl.mdp.agents.train_fixed_force_ppo \
  --output logs/experiments/ppo_new/training_seed3201 --seed 3201 --updates 192
```

사용자가 지정한 mode 6 대비 CV **11.135%** 개선 세이브본의 actor, 학습 재개 상태,
검증 자료 위치는 `logs/experiments/uniformity_saves/PPO_FIXED.json`에 기록되어 있다.
이전 10.81%, 8.90% 세이브본도 별도 디렉터리에 보존한다.
학습을 이어갈 때는 `--resume <ppo_training_state.pt>`를 지정한다.
`mdp/evaluation/refine_fixed_force_ppo.py`는 세이브본 보존, 추가 학습, 후보 선정,
독립 Isaac 검증을 수행한다. `utils/reporting/export_ppo_refinement_results.py`는 수치 감사와
공통 그래프를 출력한다. 근사 환경의 결과만으로 최종 개선율을 주장하지 않는다.

반복 학습은 `--plan <JSON>`으로 설정별 seed, GAE lambda, 학습률, update 수,
환경 수, rollout 길이, entropy 계수를 지정한다. 학습·후보 선정·독립 검증 seed를
서로 다르게 지정해야 하며, 시작한 실험의 설정과 기준 세이브본은 변경할 수 없다.
실험은 가변 registry 대신 해당 세이브본의 `SAVE.json`과 actor·학습 상태 해시를 고정한다.

```bash
python -m nrs_rl.tasks.manager_based.nrs_rl.mdp.evaluation.refine_fixed_force_ppo \
  --root logs/experiments/ppo_next \
  --plan logs/experiments/plans/ppo_refinement_20261009_1700.json
python -m nrs_rl.tasks.manager_based.nrs_rl.utils.reporting.export_ppo_refinement_results \
  --root logs/experiments/ppo_next
```

실험 계획과 결과는 Git에 포함하지 않는다. 재실행에는 해당 계획 JSON이 필요하다.
다른 실험을 시작할 때는 새 결과 디렉터리와 새 검증 seed를 지정한다.

2026-10-09 추가 학습은 세 설정을 각각 192 update 진행했다. 후보 선정 seed 3241에서
seed 3203의 추가 update 176 정책을 고정한 뒤, 독립 Isaac seed 3251·3252로 검증했다.
두 검증의 평균은 다음과 같으며 모든 백분율의 기준은 같은 batch의 mode 6이다.

| 정책 | 공간 CV 개선 | 총 제거량 유지 |
| --- | ---: | ---: |
| 기존 8.90% 세이브본 재검증 | 9.10% | 97.45% |
| 추가 학습 PPO | **10.81%** | **104.35%** |

사용자 요청으로 이 후보를
`logs/experiments/uniformity_saves/ppo_fixed20_virtual1000_cv10p81_20261009_1700/`에
새 세이브본으로 보존하고 PPO registry를 갱신했다. actor SHA-256은
`9010dad6550bfc337b4f7afcd6f67950c82112418cd9ff8a9f8de5ef61d14a2c`다.
두 seed 검증은 이 기준 경로의 결과이며,
다른 경로나 실제 재료 제거량까지 검증했다는 의미는 아니다.
폴더 이동 후 160개 테스트, 16 update 학습 재현, 1,000 step Isaac 재현 검사를 통과했다.
학습 파라미터와 Isaac 26개 기록 항목은 이동 전후 오차 0이었다.

10.81% 세이브본에서 추가로 세 설정을 각각 320 update 학습했다
(총 31,457,280 전이). 학습률 5e-5 또는 7.5e-5, GAE lambda .998/.999/.9995와
동일 batch 크기의 64환경×512 step / 32환경×1,024 step을 비교했다.
근사 환경의 최대 CV 개선은 mode 6 대비 14.08%였지만,
Isaac 후보 선정 seed 3341에서는 seed 3301의 추가 update 304가 선정됐다.
이 checkpoint를 고정한 뒤 seed 3351·3352로 검증한 결과는 다음과 같다.

| 정책 | 공간 CV 개선, mode 6 대비 | 총 제거량 유지, mode 6 대비 |
| --- | ---: | ---: |
| 10.81% 세이브본의 이번 재검증 | 10.74% | 104.38% |
| 추가 학습 PPO 후보 | 11.13% | 104.06% |

새 후보의 시드별 CV 개선은 mode 6 대비 11.20%, 11.07%였다.
두 시드 모두 제거량 50% 하한을 통과했고, 고정 힘·전진 이동·속도 명령 제한·
제거량 수치 적분 검증도 통과했다. 다만 같은 실행의 세이브본과 비교한
mode 6 기준 개선율 차이가 사전에 정한 최소 0.5%p에 못 미쳐 채택하지 않았다.
이 0.5%p는 교체 판단 기준이며 통계적 유의성을 뜻하지 않는다.
당시에는 10.81% 세이브본을 유지했으며, 추가 후보와 전체 검증 자료는
`logs/experiments/ppo_fixed_force_refinement_20261009_1700/`, 그래프는
`logs/polishing_results/20261009_1938/`에 보존한다.

후속 사용자 지시로 11.13% 후보를 새 세이브본으로 지정했다.
저장 위치는 `logs/experiments/uniformity_saves/ppo_fixed20_virtual1000_cv11p13_20261009_2200/`,
actor SHA-256은 `363abffb65b4a6078a63baffa9ac864f0609ede73a55605df09e7e1862aa1d0c`다.
원래 자동 교체 판정은 `accepted: false`로 보존하고 `SAVE.json`에 사용자 선택 근거를 기록했다.
`save_ppo_checkpoint --user-selected`는 사용자가 명시적으로 후보를 선택했을 때 사용한다.
이 옵션도 수치 감사, 전체 경로 완주 조건과 mode 6 대비 제거량 50% 하한을 요구한다.

11.13% 세이브본에서 이어 학습한 다음 실험은 보상 격자를 비교했다.
2 mm·1 mm 설정은 각각 320 update(32환경×512 step), 0.5 mm 설정은
192 update(16환경×512 step)로 총 12,058,624 전이를 추가 학습했다.
계산 비용에 따라 학습량과 학습률도 달라 순수한 격자 효과의 인과 비교는 아니다.
관측·힘·궤적이 원래 환경과 동일하고, 0.5 mm 보상 적분이 독립 표면 계산과
일치하는 것을 확인했다. 모든 후보의 actor·critic 초기값은 세이브본과 정확히 같았다.

후보 선정 seed 3541에서 mode 6 대비 CV 개선은 세이브본 11.21%,
추가 2 mm 후보 11.20%, 1 mm 후보 11.10%, 0.5 mm 후보 11.04%였다.
추가 후보 중 최고인 seed 3501의 update 272를 고정한 뒤 seed 3551·3552로 검증했다.

| 정책 | 공간 CV 개선, mode 6 대비 | 총 제거량 유지, mode 6 대비 |
| --- | ---: | ---: |
| 11.13% 세이브본의 이번 재검증 | 10.94% | 103.94% |
| 추가 학습 선정 후보 | 10.82% | 103.44% |

이번 추가 후보는 채택하지 않았고 11.13% 세이브본을 유지했다.
고정 목표 힘 20 N, action 범위·전진·가속도·jerk와 제거량 재적분 검증은 통과했다.
실험 자료는 `logs/experiments/ppo_fixed_force_refinement_20261009_2216/`,
최종 그래프는 `logs/polishing_results/20261010_0121/`에 보존한다.

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
