힘 스파이크에 대한 국소 제거량 보상 설계 — 2026-10-14

상태
학습 보상 모듈 및 Isaac PPO wrapper 연결 완료. 새 checkpoint 학습/GUI 실증 결과가
아니다. validation.json은 합성 입력과 기존 GUI 로그를 이용한 오프라인 검증이다.
배포된 actor, ROS 제어기, USD, TCP 146 mm 설정, 속도 제한은 수정하지 않았다.

물리적 목표
1000 RPM = 104.719755 rad/s. 현재 30 mm 회전 패드 Preston 모델에서는 이송을
6->12 mm/s로 올려도 초당 V/K 변화는 약 0.0034%다. 이송으로 조절할 대상은
체류 시간과 단위 경로 길이당 V/K다. 이를 일정하게 하려면 국소적으로 v ∝ F:
20 N / 6 mm/s를 기준으로 23 N에서는 약 6.9 mm/s, 17 N에서는 약 5.1 mm/s.
이 수치는 실제 깊이/부피의 교정 결과가 아니라 현재 제거 모델에서의 근사다.
회전속도와 힘이 그대로이면 이송만으로 초당 전체 제거율을 즉시 상쇄할 수 없다.

보상식
W: 최근 80 ms, Q*: 스파이크 직전 측정 제거율 V/K/s,
v*: 같은 구간의 측정 전진속도, lambda*=Q*/v*.
e = [sum(delta_V/K) - lambda* sum(delta_s_parallel)] / (Q* W)
delta_s_parallel은 실제 TCP 변위를 경로 접선에 부호 있게 투영한 값이다.
명령 속도, 예정 cursor 이동량, 변위 절댓값 합산을 보상 분모로 쓰지 않는다.
현재 속도로 나누지 않으므로 정지 중에도 회전 접촉에 의한 과가공 벌점이 남는다.
접촉 이탈 후 버퍼가 다시 채워지는 동안에는 실제 관측된 길이의 W를 사용한다.

A = clip(max(1, |filtered_F-F*|/1 N, |d(filtered_F)/dt|/(8 N/s)), 1, 4)
r_spike = -dt * min(20, A * [20 e^2 + 80 max(|e|-0.10, 0)^2])
r_total = 기존 CV potential/제거량/시간 보상 + sum(r_spike) - 미완주 벌점
새 보상 활성화 시 미완주 벌점을 20->100으로 강화해 스파이크 직전 중단을 억제한다.
이는 모든 가능한 조기 중단을 수학적으로 금지하는 값은 아니므로 완주는 별도 필수
선정 조건으로도 유지한다. 스파이크 보상 비활성화 시 미완주 벌점은 기존 20이다.
행동 변화 벌점 계수는 평상시 1e-5, 이벤트를 포함한 80 ms 전이에서 1e-6.
큰 오차와 그 지속 시간이 모두 불리하며, 빠른 실제 보상이 지연 보상보다 유리하다.
단순히 action을 크게 바꾸는 것만으로 양의 보상을 주지 않는다.
이 국소 지표는 footprint 중첩을 모두 표현하지 않으므로 공간 depth map CV도 유지한다.

감지 및 기준 유지
- 제어 주기 8 ms, 보상 윈도 80 ms, 추가 힘 필터 시정수 24 ms.
- 기준 힘 추정 시정수 1 s. 안정된 전진 접촉을 240 ms 관측한 뒤 활성화한다.
  기존 CV용 최초 2 s/18 mm 마스크를 이 스파이크 항에는 적용하지 않는다.
- 기준과 1 N 이상 차이 또는 8 N/s 이상 변화가 16 ms 지속되면 이벤트 시작.
- 최초 의심 시점부터 기준 제거량/속도를 고정한다. 지속된 힘 오프셋을 기다리기만
  해서 기준을 새 힘에 맞춰 이동시키는 우회 행동을 방지한다.
- 힘 차이 0.5 N 이하, 변화율 3 N/s 이하가 400 ms 유지되면 종료한다.
- 접촉 이탈/안전 shield는 기준을 지우지 않으며, 기준 설정 후에는 25/s 벌점.
  이벤트 벌점 최대 20/s보다 커서 접촉을 잃는 것으로 순간 벌점을 줄일 수 없다.
- 에피소드 종료 후 반복 action 안에 들어오는 자동 reset 표본을 이전 보상에
  합치지 않는다. 각 환경의 기준과 누적 통계는 독립적으로 초기화한다.

CV 범위 및 검증 기준
기존 CV 학습 항은 processing ROI와 최초 2 s/18 mm 제외 조건을 그대로 사용한다.
spike 항은 안정 전진 기준이 만들어진 시점부터 적용한다. 전체 active depth map CV도
full_active_cv로 별도 기록한다. 최초 측정 접촉부터의 GUI 전체 CV는 기존 비교기에서
pre-control 접촉까지 포함해 평가해야 한다. 위 지표들을 같은 CV라고 혼용하지 않는다.
새 모델 선정 시 기존 actor 및 Mode 6와 같은 경로/시드로 공간 CV, 국소 제거량 오차,
오차 지속 시간, 완주, 총 제거량, 힘/추종/안전 상태를 함께 비교한다.
학습 return 하나만으로 모델을 교체하지 않는다. 재학습 시 가중치 균형을 확인해야 한다.

검증 결과
- 보상/실측 적분/반복 action 종료 처리 테스트 11개와 TCP 설정 회귀 테스트 3개 통과.
- 합성 20->23 N, 20->17 N 계단 입력을 각각 2 s 적용했을 때 추가 보상:
  빠른 물리 응답 -0.36422, 추가 320 ms 지연 -0.98817, 속도 유지 -3.79084.
  두 부호 모두 올바른 방향이 유리하고, 반대 방향의 속도 변화는 더 불리했다.
  합성 Q/K=1000 F이며, 실제 회전 표면 모델에서도 v ∝ F 근사를 별도 검사했다.
- 실제 기존 GUI 로그 두 모드 모두 경로 5~80%에는 이벤트 벌점이 없었고,
  85~90% 및 종료 직전의 스파이크를 감지했다. 모드별 기준 속도가 다르므로
  이 보상 합계만으로 두 모드의 절대 가공 품질 우열을 판단할 수 없다.
- 기존 Mode 5는 스파이크 인근에서 원래 CV 정책의 속도 변화도 크므로
  고정된 직전 기준과의 오차가 크다. 새 목적과 전체 CV의 균형은 아직 미검증이다.

응답 속도 한계
신경망 관측은 기존 16D이며 힘 변화율/직전 기준을 새 입력으로 추가하지 않았다.
reward만으로 미래 힘을 보거나 미분 신호를 actor에 직접 제공하지 않는다.
기존 80 ms 추론, 80 ms action 필터, 18 mm/s^2 action slew,
16 mm/s^2 / 160 mm/s^3 물리 feed 제한을 유지한다.
합성 +0.9 mm/s 보상에서 변화량의 50% 도달은 약 0.296 s, 90%는 약 0.624 s다.
이는 제한기 계산이며 실제 Isaac/로봇 응답 보장이 아니다. 더 짧은 힘 스파이크의
완전한 상쇄에는 이 보상 외에 제어 대역폭/관측/지연 검토도 필요할 수 있다.

사용법
기존 mdp/agents/train_latest_force_ppo.py 학습 명령에 다음 옵션을 추가한다:
  --spike-reward-config
또는 해당 옵션 뒤에 별도 JSON 경로를 지정한다.
기본 설정: mdp/config/spike_dose_reward.json
옵션을 생략하면 기존 학습 보상이다. 새로운 출력 디렉터리를 사용한다.
보상 설정/소스 해시가 바뀌면 critic 출력과 critic Adam 상태를 자동 초기화하며,
actor 가중치는 부모 checkpoint에서 이어받는다. 설정/소스는 학습 폴더에 복사되고
protocol 및 actor metadata에도 기록된다. 긴 episode를 담는 rollout과 critic warmup을
사용하는 것이 권장되며, 기존 CV만을 보는 자동 후보 선택기를 그대로 쓰지 않는다.

재현 명령 (env_isaaclab Python, 작업 위치는 task 저장소 루트)
python -m unittest nrs_rl.tasks.manager_based.nrs_rl.mdp.tests.test_spike_dose_reward nrs_rl.tasks.manager_based.nrs_rl.mdp.tests.test_robot_geometry -v
python -m nrs_rl.tasks.manager_based.nrs_rl.mdp.evaluation.validate_spike_dose_reward --archive logs/adaptive_velocity/20261014/gui_test --output validation/spike_dose_reward_20261014/validation.json
