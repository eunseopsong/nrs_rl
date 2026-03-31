# nrs_rl

Isaac Lab 기반의 `nrs_rl` 태스크 저장소입니다.  
UR10e + spindle 환경에서 HDF5 trajectory를 따라가도록 구성된 RL/환경 코드와, C++ 기구학 함수를 Python에서 사용할 수 있도록 만든 바인딩 코드가 함께 정리되어 있습니다.

---

## 1. 실행 방법

### 1-1. 환경 활성화
```bash
conda activate env_isaaclab
```

### 1-2. 학습 실행
```bash
~/IsaacLab/isaaclab.sh -p ~/nrs_rl/scripts/skrl/train.py --task Template-Nrs-Rl-v0
```

### 1-3. 학습 후 play 실행
```bash
~/IsaacLab/isaaclab.sh -p ~/nrs_rl/scripts/skrl/play.py --task Template-Nrs-Rl-v0
```

---

## 2. GitHub 반영 방법

```bash
cd ~/nrs_rl/source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl

git init

git add .

git commit -m "commit message"

git push origin main


```


---

## 3. 저장소 개요

이 폴더는 Isaac Lab의 manager-based task 구조 안에서 동작하는 `nrs_rl` 태스크의 핵심 코드 모음입니다.

현재 구조의 핵심 목적은 다음과 같습니다.

- **환경 설정**: `nrs_rl_env_cfg.py`
- **MDP 관련 로직**: `mdp/`
- **디버그/시각화 유틸리티**: `utils/`
- **로봇/센서 asset 설정**: `assets/`
- **학습 데이터(HDF5/TXT)**: `datasets/`
- **C++ 기구학/제어 함수를 Python에서 호출하기 위한 바인딩 코드**: `nrs_ik_py_bind/`
- **학습 에이전트 설정**: `agents/`

---

## 4. 폴더 구조

```text
nrs_rl/
├── agents/
├── assets/
├── datasets/
├── mdp/
├── nrs_ik_py_bind/
├── utils/
├── nrs_rl_env_cfg.py
├── __init__.py
└── README.md
```

---

## 5. 폴더별 기능 정리

### 5-1. `agents/`
학습기 설정 파일을 보관하는 폴더입니다.

- `skrl_ppo_cfg.yaml`  
  SKRL 기반 PPO 학습 설정 파일입니다. 학습 하이퍼파라미터, rollout, batch 관련 설정을 관리합니다.

---

### 5-2. `assets/`
환경에서 사용하는 로봇 및 센서 asset 설정을 담는 폴더입니다.

현재 구조는 한 단계 더 중첩되어 있으며, 실제 설정 코드는 `assets/assets/` 아래에 있습니다.

#### `assets/assets/robots/`
- `ur10e_w_spindle.py`  
  UR10e + spindle 로봇 asset 정의 파일입니다.

#### `assets/assets/sensors/`
- `ur10e_contact_sensors.py`  
  UR10e와 관련된 contact sensor 설정 코드입니다.

---

### 5-3. `datasets/`
Trajectory 및 reference data를 저장하는 폴더입니다.

예시:
- `flat_g_recording.h5`
- `concave_g_recording.h5`
- `joint_recording.h5`
- `joint_recording_filtered.h5`
- `flat_g_recording.txt`

또한 TXT → HDF5 변환용 스크립트가 포함되어 있습니다.

- `convert_txt_to_h5.py`
- `convert_hand_g_to_h5.py`

이 데이터들은 주로 다음 용도로 사용됩니다.

- action target trajectory 로딩
- observation target reference 생성
- play / tracking 실험용 reference 제공

---

### 5-4. `mdp/`
환경의 핵심 MDP 로직이 들어 있는 폴더입니다.

#### `action.py`
ActionTerm 정의 파일입니다.

현재 구조에서는 RL action 자체보다, **HDF5에 저장된 target path를 따라가는 action logic**이 핵심입니다.

주요 역할:
- HDF5 target trajectory 로드
- end-effector target pose 생성
- 현재 EE pose와 target pose 비교
- Jacobian 기반 IK / DLS 계산
- joint position target 생성 및 robot에 전달
- debug 출력 함수 호출

#### `observation.py`
Observation 함수 정의 파일입니다.

주요 역할:
- end-effector pose 계산
- HDF5 target position horizon 생성
- fixed joint 기준 6-axis FT sensor 값 읽기
- camera / sensor 관련 observation 제공
- debug cache 업데이트

#### `rewards.py`
현재는 **placeholder 파일**입니다.

기존 reward 함수들을 정리한 뒤, 현재는 기본 import/header만 남겨둔 상태입니다.  
즉, reward 계산 로직은 사실상 비워져 있으며, reward plotting 기능은 `utils/visualization.py`로 이동되었습니다.

#### `terminations.py`
Episode 종료 조건을 정의하는 파일입니다.

예:
- trajectory finished
- 필요 시 timeout / safety termination 추가 가능

---

### 5-5. `utils/`
공통 유틸리티 폴더입니다.

MDP 핵심 로직과 분리해 두어, 코드 구조를 더 명확하게 유지하기 위한 목적입니다.

#### `debug.py`
디버그 출력 유틸리티입니다.

주요 역할:
- action debug 출력
- 6-axis FT sensor debug cache 및 출력
- camera debug 출력
- fixed joint FT 관련 초기화/에러 메시지 출력

즉, 실제 제어/관측 로직은 `mdp/`에 두고, 사람이 읽는 로그 출력은 `utils/debug.py`가 담당합니다.

#### `visualization.py`
학습/실험 결과 시각화용 유틸리티입니다.

주요 역할:
- episode별 tracking plot 저장
- reward plot 저장
- best episode 기록
- run timestamp 기반 출력 폴더 분리

---

### 5-6. `nrs_ik_py_bind/`
**C++ 제어/기구학 함수를 Python 형태로 불러오기 위해 존재하는 코드**가 들어 있는 폴더입니다.

즉, Python 환경(`action.py`, `observation.py` 등)에서 사용할 수 있도록,
C++로 구현한 FK/IK 관련 기능을 바인딩해 둔 영역입니다.

구성 요소:
- `nrs_fk_core.cpython-311-...so`
- `nrs_ik_core.cpython-311-...so`
- `nrs_ik_py/` 내부 C++ 소스 및 헤더
- `setup.py`, `pyproject.toml`
- FK/IK 테스트/검증 스크립트

주요 역할:
- C++ 구현의 FK solver 제공
- C++ 구현의 IK solver 제공
- Python 코드에서 빠르게 호출 가능하도록 extension module 제공
- Python만으로 처리하기 어려운 기구학 계산을 효율적으로 수행

---

### 5-7. `nrs_rl_env_cfg.py`
이 저장소의 **최상위 환경 설정 파일**입니다.

주요 역할:
- scene 구성
- robot / workpiece / sensor 배치
- action cfg 연결
- observation cfg 연결
- rewards / terminations / events 연결
- episode length, physics, viewer, GPU buffer 설정

즉, `mdp/`, `assets/`, `utils/`의 요소들을 한데 묶어 실제 Isaac Lab task 환경으로 만드는 진입점 역할을 합니다.

---

## 6. 현재 코드 흐름

현재 구조를 기능 흐름으로 보면 아래와 같습니다.

1. `nrs_rl_env_cfg.py` 에서 scene / action / observation / termination 설정
2. `mdp/action.py` 에서 HDF5 target trajectory 기반 target pose 생성
3. `mdp/observation.py` 에서 current EE pose, target horizon, FT sensor 값 등 계산
4. 필요 시 `utils/debug.py` 에서 action / FT / camera debug 출력
5. 필요 시 `utils/visualization.py` 에서 결과 plot 저장
6. `nrs_ik_py_bind/` 의 FK/IK 바인딩 모듈을 통해 Python 코드에서 C++ 기구학 함수 호출

---

## 7. 실행 시 참고 사항

### 학습 실행 전
- `conda activate env_isaaclab` 로 환경을 활성화해야 합니다.
- Isaac Lab 실행 스크립트 경로가 `~/IsaacLab/isaaclab.sh` 기준으로 맞아야 합니다.
- `Template-Nrs-Rl-v0` task registration 이 정상이어야 합니다.

### 데이터 관련
- `datasets/` 내부 HDF5 경로가 `action.py`, `observation.py`, `nrs_rl_env_cfg.py`에서 참조하는 경로와 일치해야 합니다.

### 바인딩 관련
- `nrs_ik_py_bind/` 내부 `.so` 모듈이 현재 Python/conda 환경과 호환되어야 합니다.
- FK/IK 바인딩이 import되지 않으면 `observation.py`의 EE pose 계산이나 관련 기구학 로직이 실패할 수 있습니다.

---

## 8. 유지보수 관점에서의 현재 정리 상태

현재 구조는 다음 기준으로 정리되어 있습니다.

- **환경 구성**: `nrs_rl_env_cfg.py`
- **MDP 로직**: `mdp/`
- **유틸리티(log/plot)**: `utils/`
- **로봇/센서 asset**: `assets/`
- **데이터셋**: `datasets/`
- **C++ ↔ Python 바인딩**: `nrs_ik_py_bind/`
- **에이전트 설정**: `agents/`

즉, 역할별 분리가 비교적 명확하게 되어 있어,
앞으로 디버깅/확장/리팩토링할 때도 파일 책임이 잘 나뉘는 구조입니다.

---

## 9. 권장 수정 원칙

향후 코드 수정 시에는 아래 원칙을 유지하는 것을 권장합니다.

- `mdp/`에는 **환경 동작에 직접 필요한 로직**만 둔다.
- `utils/`에는 **출력/시각화/보조 유틸리티**만 둔다.
- `nrs_ik_py_bind/`는 **C++ 성능 의존 기구학 코드 전용 영역**으로 유지한다.
- `nrs_rl_env_cfg.py`는 **최상위 wiring/config 역할**만 담당하게 유지한다.

---

## 10. 요약

이 저장소는 다음 세 축으로 이해하면 됩니다.

1. **Isaac Lab 환경 설정 및 실행**  
   → `nrs_rl_env_cfg.py`, `mdp/`, `assets/`

2. **디버그 및 결과 저장**  
   → `utils/`

3. **C++ 기구학/제어 함수를 Python에서 사용하기 위한 바인딩**  
   → `nrs_ik_py_bind/`

현재 구조상, `debug.py`와 `visualization.py`는 `utils/`로 분리되어 있어 역할이 명확하며,  
`action.py` / `observation.py`는 실제 환경 동작 로직에 집중하도록 정리되어 있습니다.
