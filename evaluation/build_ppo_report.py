"""Build the source-grounded neural PPO MDP and independent Isaac validation PDF."""
from pathlib import Path
import hashlib
import json

if not __package__:
    import sys
    _root = next(p / 'source/nrs_rl' for p in Path(__file__).resolve().parents
                 if (p / 'source/nrs_rl/nrs_rl').is_dir())
    sys.path.insert(0, str(_root))

from nrs_rl.tasks.manager_based.nrs_rl.paths import ROOT
from nrs_rl.tasks.model_based.reporting.build_fixed_force_reports import (
    REPORTS, WIDTH, p, table, page, picture, build, Spacer,
)


def main():
    root=ROOT/'logs/experiments/ppo_fixed_force_20261009'
    report=json.loads((root/'final_report.json').read_text())
    result=json.loads((root/'result.json').read_text())
    contract=json.loads((root/'candidate/fixed_force_policy.json').read_text())
    audit=json.loads((root/'ppo_training_audit.json').read_text())
    assert audit['passed'] and report['numerical_audit_passed']
    metrics=report['mean_heldout_metrics'];plots=Path(report['plots_directory'])
    story=[p('고정 힘 · 고정 가상 RPM\n신경망 PPO 가공량 균일도','title'),
        p('2026-10-09 | 실제 PPO-Clip 학습과 독립 Isaac 물리 검증','sub'),
        p('신경망 actor·critic을 확률적 on-policy rollout과 GAE, PPO clipped objective로 학습했다. 기존 OSQP/Powell 프로파일을 teacher로 사용하거나 프로파일 값을 신경망 대신 저장하지 않았다. 기존 고정 모델은 별도로 보존하고 이번 PPO를 자동 배포하지 않았다.'),
        table([['항목','결과 / 조건'],['독립 Isaac 공간 CV',f"mode 6 대비 평균 {metrics['mode6_cv_improvement_percent']:.4f}% 개선"],
            ['전체 ROI 제거량',f"mode 6 대비 평균 {metrics['mode6_full_volume_retention_percent']:.4f}% 유지"],
            ['가공 구간 제거량',f"mode 6 대비 평균 {metrics['mode6_processing_volume_retention_percent']:.4f}% 유지"],
            ['조건','목표 힘 20 N, 가상 1,000 RPM 고정; 전진 속도와 연동 체류시간 가변'],
            ['학습량','초기값 2개, 각각 2,097,152 transition / 64 PPO update'],
            ['선택 정책',f"seed {contract['seed']}, update {contract['ppo_updates']}, actor 20,865 parameters"]], [140,WIDTH-140]),
        Spacer(1,13),p('검증된 PPO actor SHA-256','sub'),p(report['checkpoint_sha256'],'small'),
        p('학습 환경은 회전 제거량·디지털 이송 응답을 근사한 모델이다. Isaac 로봇·접촉 물리는 학습 후 독립 검증에 사용했다. 이 결과는 GUI 로봇 실증 또는 하드웨어 검증 결과가 아니다.','small')]

    page(story,'1. 시스템 상태와 학습·검증 환경')
    story += [p('기준 경로 길이는 약 697.533 mm다. 상태에는 경로 cursor, 누적 깊이 지도, 누적 체류시간 지도, 명령 지연 이력, 필터·이송 속도 상태, 경과시간, 측정 힘이 포함된다. 정책은 이 전체 상태를 16개 특징으로 요약해 받는다. 정보가 압축되고 지연 이력을 직접 주지 않으므로 완전 관측 MDP라고 단정하지 않는다.'),
        table([['구분','학습용 근사 환경','Isaac 검증'],
            ['이송 / 물리','기준 경로 위 cursor; 디지털 필터와 속도 응답','전체 로봇/접촉 물리와 기존 힘 제어기'],
            ['힘','목표 20 N; 측정값을 episode별 17.25~22.75 N 범위로 무작위화','목표 20 N; 실제 센서 법선 힘으로 누적'],
            ['제거량 적분','2 mm grid; 80 ms마다 기준 경로 중점에 회전 커널 누적','0.5 mm grid; 실제 TCP와 실측 속도로 8 ms 적분'],
            ['정책 / 제어','80 ms / 8 ms, 64개 병렬 환경','80 ms / 8 ms, 조건별 병렬 robot'],
            ['접촉 / 추종 오차','별도 접촉 동역학 없음; 추종 오차 특징 0','Isaac 힘 변화·접촉·추종 오차 반영']], [75,215,WIDTH-290],small=True),
        p('episode마다 측정 힘 scale 0.9~1.1, bias −0.75~0.75 N, 명령 지연 0~3 tick을 무작위화한다. 이는 목표 접촉력의 action 권한을 추가하는 것이 아니다. 모든 정책 action은 이송 속도 1개다.'),
        p('근사 모델에서 학습한 정책이 Isaac에서 동일한 개선율을 낼 것이라고 가정하지 않는다. 후보 선정 seed 3141을 거친 뒤, checkpoint를 고정하고 3151·3152에서 새로 완주 검증했다.')]

    page(story,'2. Observation — 16차원 공간 특징')
    rows=[['0','s/L','경로 진행률'],['1','F/20','측정 법선 힘'],['2','v/6','실측 접선 속도'],
        ['3','nominal depth deficit','현재 위치 nominal 깊이 부족량 − 경로 평균 부족량'],
        ['4','predicted-final deficit','누적 깊이 + nominal 잔여 깊이의 예상 부족량'],
        ['5','behind minus current','5 mm 뒤쪽과 현재 위치의 예상 부족량 차이'],
        ['6','local residence / 5 s','현재 footprint의 누적 체류시간'],
        ['7','tracking error / 10 mm','추종 오차; 근사 환경 0, Isaac 실측'],
        ['8','elapsed / (L/6)','nominal 완료시간 대비 진행 시간'],['9','reverse / (0.05 L)','현재 조건 0'],
        ['10','contact','측정 힘 ≥ 1.5 N'],['11','shield','이송 보호 정지 여부'],
        ['12','target force / 20','항상 1'],['13','F×|v|/120 − 1','기존 비회전 TCP rate proxy'],
        ['14','return allowed','항상 0'],['15','remaining depth / target','현재 영역의 nominal 잔여 깊이']]
    story += [table([['idx','표현','의미'],*rows],[27,163,WIDTH-190],small=True),
        p('float32, 각 특징 [−5, 5] clipping. local 값은 footprint의 압력 가중 평균이며, 기준 깊이는 20 N / 6 mm/s / 1,000 RPM nominal 지도 평균이다. idx 13은 회전을 포함하지 않는 보조 특징이며 reward의 가공량 계산에는 아래 회전 모델을 사용한다.','small'),
        p('신경망에는 이 16D 입력 외에 progress의 sin/cos 주파수 1~8을 추가한다. 총 32개 입력이며 특정 체류시간 profile이나 최적화된 lookup table을 주입하지 않았다.','small')]

    page(story,'3. Action과 체류시간의 관계')
    story += [p('PPO는 스칼라 잠재 Gaussian에서 action을 샘플링한다. 학습 후 결정론적 평가에서는 신경망 평균 출력을 변환해 사용한다.'),
        p('z ~ Normal(μθ(o), σ);    a = 0.125 + 0.875 tanh(z)','formula'),
        p('v_request = 6 + 6a;    v_request ∈ [1.5, 12] mm/s','formula'),
        p('residence per distance = 1/v;    q = 6/v ∈ [0.5, 4]','formula'),
        p('감속하면 같은 위치의 체류시간이 길어진다. 별도의 정지시간 action, 역주행, 추가 전체 패스는 없다. “가변 체류시간”은 이 전진 속도 조절로 생기는 체류시간이다.'),
        table([['항목','값'],['목표 힘 / 회전','20 N / 가상 1,000 RPM 고정'],['정책 판단 주기','80 ms; 제어 tick 10개 동안 action 유지'],
            ['명령 필터','시정수 0.08초, 물리 이송 slew 18 mm/s²'],['속도 제한','forward 0~12 mm/s; 가속도 16 mm/s², jerk 160 mm/s³'],
            ['Isaac 추종 보정','gain 0.6, 시정수 0.08초, 보정 상한 4 mm'],['종료','경로 완주 또는 336초; Isaac 힘 >100 N 즉시 fault']], [150,WIDTH-150]),
        p('요청 최소 속도는 1.5 mm/s이지만, 시작 램프·접촉 대기·보호 정지에서는 실제 적용 이송 속도가 0에 접근할 수 있다. 검증에서는 전체 요청·적용 속도와 가속도/jerk, 목표 힘을 trace로 확인했다.')]

    page(story,'4. Reward — 최종 공간 CV 감소')
    story += [p('주 목적은 시간축 순간 제거율의 변동이 아니라, 경로 완료 후 가공 구간 전체에 누적된 깊이의 공간 CV를 줄이는 것이다. 현재까지의 깊이에 남은 nominal 깊이를 더해 예상 최종 지도를 만든다.'),
        p('D̂_final,t = D_processing,t + D_nominal,remaining,t','formula'),
        p('Φ_t = mean(D̂_final,t²) / mean(D̂_final,t)² − 1','formula'),
        p('r_t = 100 (Φ_(t−1) − Φ_t) − 10⁻⁵ (a_t−a_(t−1))² − 10⁻⁴ Δt','formula'),
        p('episode 종료 시 추가 penalty','sub'),
        p('−100 max(0, 0.5−V_processing/V6, 0.5−V_full/V6)²','formula'),
        p('−20 if path incomplete at 336 s','formula'),
        p('Δt=0.08 s, discount γ=1.0. 경로를 완주하면 잔여 nominal 지도는 0이므로, potential 차이의 누적합이 최종 공간 CV² 감소에 연결된다. 작은 action 변화 penalty와 시간 penalty는 과도한 변화·지연을 억제하는 보조 항이다.'),
        p('첫 활성 2초의 깊이 기여를 processing 지도에서 제외하고, 첫 18 mm 기준 경로 footprint를 평가 mask에서 제외한다. 학습에서 제거량 하한은 soft penalty이며, 독립 Isaac 채택에서는 processing ROI·full ROI·whole grid 각각 mode 6의 50% 이상인지를 별도로 확인한다.'),
        p('힘 목표 추종이나 RPM은 PPO action으로 조정하지 않는다. 회전이 없는 F×v만으로 체류시간을 평가하지 않고, 회전 포함 깊이 누적을 reward의 기반으로 사용한다.')]

    page(story,'5. PPO 구현과 학습 증거')
    story += [p('Actor와 critic은 각각 32 → 128 tanh → 128 tanh → 1의 독립 신경망이다. 초기 actor 출력은 nominal 6 mm/s, hidden feature는 무작위 초기화다. 학습 가능한 Gaussian log standard deviation을 포함해 stochastic rollout을 수집하고, 학습 후 actor만 TorchScript로 내보냈다.'),
        p('ρ_t = exp(log πθ(z_t|o_t) − log πold(z_t|o_t))','formula'),
        p('L_policy = mean(max(−Â_t ρ_t, −Â_t clip(ρ_t, 0.8, 1.2)))','formula'),
        p('L_total = L_policy + 0.25 mean((V−return)²) − 0.0002 H(Normal)','formula'),
        table([['설정','값'],['Rollout / 병렬 환경','512 decisions × 64 environments/update'],['GAE / discount','λ=0.98 / γ=1.0'],
            ['Optimizer','Adam lr=2×10⁻⁴, eps=10⁻⁵'],['Minibatch / epoch','2,048 / 최대 4'],
            ['PPO clip / KL early stop','0.2 / 0.025'],['Gradient norm cap','0.5'],
            ['log std','초기 −0.55, 학습 중 [−2.5, −0.3] 제한'],['Seeds / total','3101, 3102; 각각 64 updates / 2,097,152 transitions']], [170,WIDTH-170]),
        p('tanh 변환의 Jacobian은 동일 latent action에 대한 old/new density 비율에서 상쇄된다. entropy 보너스는 변환 전 Normal entropy이며 squashed action의 정확한 entropy는 아니다. advantage는 minibatch에서 정규화한다.','small'),
        p('학습 감사: 64회 모두 actor weight가 실제로 변경됐고 critic과 Adam state도 저장됐다. export 4,096개 입력에서 학습 신경망과 일치하며 residence_profile buffer가 없다. 원본 학습 코드, protocol, metric CSV와 full training state를 보존했다.','small'),
        p('알고리즘 원전: Schulman et al., Proximal Policy Optimization Algorithms (2017), https://arxiv.org/abs/1707.06347','small')]

    page(story,'6. 회전 공구의 제거 깊이 계산')
    story += [p('공구 직경 30 mm, 균일 압력의 원형 접촉 footprint를 가정한다. 재료 상수 K를 보정하지 않았으므로 깊이 h/K와 체적 V/K를 비교한다. 실제 μm, mm³ 제거량 측정값으로 해석하지 않는다.'),
        p('ω = 2π × 1000 / 60;    R = 15 mm','formula'),
        p('p_j = F_n w_j / (Σ_k w_k ΔA),    w=1 inside disk','formula'),
        p('v_rel,j = √[(v_x−ωy_j)² + (v_y+ωx_j)²]','formula'),
        p('Δ(h_j/K) = p_j v_rel,j Δt;    V/K = Σ_j (h_j/K) ΔA','formula'),
        p('TCP 전진 속도가 낮아져도 회전 기여 ωr가 남으므로 체류시간 증가가 누적 제거량 증가로 이어질 수 있다. 이는 제거량 계산기의 가상 회전이며 URDF 회전이나 스핀들 관성 동역학을 시뮬레이션한 것은 아니다.'),
        p('검증 누적 조건: 활성 가공, 측정 법선 힘 ≥1.5 N, fault 없음. actual TCP 샘플 중점에 실측 sliding velocity와 측정 힘을 사용한다. 격자 경계 밖의 공구 영역은 압력 정규화에 포함하고, 격자 밖 제거량을 ROI 안으로 재분배하지 않는다.'),
        p('독립 감사에서는 모든 원시 trace의 제거 지도를 재적분하고, 여러 시점의 회전 커널을 별도의 직접 격자 계산과 비교했다. 이 감사는 수치 구현의 일치를 확인하며 실제 재료 제거 계수의 타당성을 보증하지 않는다.')]

    page(story,'7. 독립 Isaac 결과 — 모든 수치는 mode 6 대비')
    rows=[]
    for seed,holdout in result['heldouts'].items():
        for name,label in [('saved','고정 비-PPO'),('selected','PPO')]:
            c=holdout['candidates'][name]
            rows.append([seed,label,f"{100*c['mode6_cv_gain']:.4f}%",f"{100*c['mode6_volume_ratios']['full_roi_volume_over_k']:.4f}%"])
    story += [table([['seed','모델','공간 CV 개선','전체 ROI 제거량 유지'],*rows],[60,110,170,WIDTH-340]),
        p('평가 grid 0.5 mm, 모든 정책에 동일한 기준 경로 ROI를 사용하고 깊이 0인 셀도 포함한다. 첫 2초 기여와 첫 18 mm 기준 footprint를 제외한 processing 공간 CV가 주 지표다. 전체 ROI/whole-grid 제거량 하한도 함께 확인한다.'),
        picture(plots/'02_isaac_mode6_comparison.png'),
        p('각 seed에서 mode 6를 두 번 병렬 실행해 계산·물리 반복 변동을 확인했다. 채택은 CV 개선이 max(0.5%, mode 6 반복 변동의 2배) 이상이고 제거량 50% 하한과 완주·fault 검사를 통과하는 조건이다. 이는 실용적 채택 규칙이며 통계적 유의성 검정은 아니다.','small'),
        p('독립 seed 두 개는 반복성의 제한된 증거다. GUI robot에서 새 PPO를 실행한 결과는 아직 없으며, 기존 profile보다 나은 모델이라고 자동 결론내리지 않는다. 현재 고정 배포 모델은 그대로다.','small')]

    page(story,'8. 학습 곡선과 공간 분포')
    story += [picture(plots/'01_ppo_learning_curves.png'),Spacer(1,12),
        p('위 CV 곡선은 근사 학습 환경의 deterministic 평가다. Isaac 수치와 구분한다. 아래는 독립 Isaac의 최종 processing 지도이며, 각 지도 평균으로 정규화하고 색 범위를 맞췄다.','small'),
        picture(plots/'03_isaac_processing_heatmaps.png')]

    page(story,'9. 저장 위치와 소스 분리')
    rows=[['PPO 신경망','manager_based/nrs_rl/agents/ppo_network.py'],['PPO 학습','manager_based/nrs_rl/agents/train_fixed_force_ppo.py'],
        ['환경 / 관측','manager_based/nrs_rl/mdp/removal_env.py, removal_observation.py'],
        ['행동 / 보상 / 종료','manager_based/nrs_rl/mdp/removal_action.py, removal_rewards.py, removal_terminations.py'],
        ['Isaac 검증','manager_based/nrs_rl/evaluation/probe_fixed_force_ppo.py, finish_fixed_force_ppo.py'],
        ['공통 계산','manager_based/nrs_rl/utils/preston_surface.py, rotary_geometry.py'],
        ['기존 비-PPO','model_based/policies/, training/, runtime/, evaluation/']]
    story += [p('아래 소스 경로는 source/nrs_rl/nrs_rl/tasks/ 기준이다. model_based는 기존 OSQP/Powell/CEM 최적화 모델을 뜻하며 강화학습과 구분했다. scripts/ 경로는 호환 실행 진입점으로 유지한다.'),
        table([['구분','경로'],*rows],[100,WIDTH-100],small=True),Spacer(1,12),
        p('PPO checkpoint','sub'),p(report['checkpoint'],'small'),
        p('학습 상태','sub'),p(report['ppo_training_state'],'small'),
        p('근거 파일','sub'),p(str(root/'final_report.json'),'small'),p(str(root/'ppo_training_audit.json'),'small'),
        p(str(root/'independent_numerical_audit.json'),'small'),p(str(root/'source_reorganization.json'),'small'),
        p('원본 학습 버전은 training_seed*/source_snapshot과 source_before_reorganization에 남겨 두었다. 이동 후 159개 테스트를 통과했고, PPO 업데이트 8회의 가중치·학습 지표 및 Isaac 1,000 step의 기록이 변경 전과 정확히 일치했다.','small'),
        p('기존 고정 모델의 Python 의존 코드는 model_based/runtime/fixed_force_12p59_source에 고정했다. import·해시 검사 경로를 변경했으며, checkpoint와 C++ 제어 바이너리는 동일하다. mode 5·6의 각 1,250개 입력 및 전체 경로 제거량 재적분 결과가 일치하고 10개 launch 구성 검사를 통과했다.','small')]
    output=REPORTS/'fixed_force_ppo_20261009.pdf'
    build(output,story,'고정 힘 회전 가공량 신경망 PPO 학습 및 검증')
    manifest={'report':str(output),'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
              'actor_sha256':report['checkpoint_sha256'],'source_result':str(root/'final_report.json'),
              'algorithm':'PPO','isaac_used_for':'independent validation, not training'}
    (REPORTS/'fixed_force_ppo_20261009_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
