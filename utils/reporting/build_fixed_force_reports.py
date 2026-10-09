"""Build Korean PDF reports from frozen model metadata and measured result files."""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

from nrs_rl.tasks.manager_based.nrs_rl.paths import ROOT as _REPO_ROOT
from datetime import datetime
import hashlib
import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image

ROOT=_REPO_ROOT
REPORTS=ROOT/'reports'
pdfmetrics.registerFont(TTFont('Nanum','/usr/share/fonts/truetype/nanum/NanumGothic.ttf'))
pdfmetrics.registerFont(TTFont('NanumBold','/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf'))
pdfmetrics.registerFontFamily('Nanum',normal='Nanum',bold='NanumBold')
pdfmetrics.registerFont(TTFont('Formula','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
STYLE={
 'title':ParagraphStyle('title',fontName='NanumBold',fontSize=24,leading=34,textColor=colors.HexColor('#16334b'),spaceAfter=20,wordWrap='CJK'),
 'h':ParagraphStyle('h',fontName='NanumBold',fontSize=16,leading=23,textColor=colors.HexColor('#16334b'),spaceAfter=14,wordWrap='CJK'),
 'sub':ParagraphStyle('sub',fontName='NanumBold',fontSize=11.5,leading=17,spaceBefore=10,spaceAfter=7,wordWrap='CJK'),
 'body':ParagraphStyle('body',fontName='Nanum',fontSize=10,leading=16,spaceAfter=8,wordWrap='CJK'),
 'small':ParagraphStyle('small',fontName='Nanum',fontSize=8.2,leading=12,spaceAfter=5,wordWrap='CJK'),
 'formula':ParagraphStyle('formula',fontName='Formula',fontSize=10.2,leading=17,spaceAfter=10,wordWrap='CJK',leftIndent=8),
}
WIDTH=A4[0]-86


def p(text,kind='body'):
    primary=pdfmetrics.getFont(STYLE[kind].fontName).face.charToGlyph
    fallback=pdfmetrics.getFont('Formula').face.charToGlyph
    rendered=[]
    for char in str(text):
        if char=='\n':rendered.append('<br/>')
        elif ord(char) not in primary and ord(char) in fallback:
            rendered.append('<font name="Formula">'+escape(char)+'</font>')
        else:rendered.append(escape(char))
    return Paragraph(''.join(rendered),STYLE[kind])


def table(rows,widths=None,small=False):
    data=[[p(x,'small' if small else 'body') for x in row] for row in rows]
    result=Table(data,colWidths=widths or [WIDTH/len(rows[0])]*len(rows[0]),repeatRows=1,hAlign='LEFT')
    result.setStyle(TableStyle([
       ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eaf0f5')),
       ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f7f9fb')]),
       ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),
       ('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6),
       ('BOTTOMPADDING',(0,0),(-1,-1),4),
       ('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#afbdca'))]))
    return result


def picture(path,width=WIDTH):
    from PIL import Image as PILImage
    with PILImage.open(path) as image:size=image.size
    return Image(str(path),width=width,height=width*size[1]/size[0])


def page(story,title):
    if story:story.append(PageBreak())
    story.append(p(title,'h'))


def build(path,story,title):
    def decorate(canvas,doc):
        canvas.setStrokeColor(colors.HexColor('#ced7df'));canvas.line(43,43,A4[0]-43,43)
        canvas.setFont('Nanum',8);canvas.setFillColor(colors.HexColor('#596a78'))
        canvas.drawString(43,29,'NRS RL | 고정 힘 · 가상 회전 · 가공량 균일도')
        canvas.drawRightString(A4[0]-43,29,str(doc.page))
    doc=SimpleDocTemplate(str(path),pagesize=A4,rightMargin=43,leftMargin=43,topMargin=43,bottomMargin=57,
                         title=title,author='NRS RL project',subject='Code and checkpoint audit')
    doc.build(story,onFirstPage=decorate,onLaterPages=decorate)


def baseline():
    REPORTS.mkdir(exist_ok=True)
    lock=json.loads((ROOT/'deployment/FIXED_MODEL.json').read_text())
    c=json.loads((ROOT/'deployment/fixed_force_12p59/fixed_force_policy.json').read_text())
    gui=json.loads((ROOT/'logs/polishing_results/20261009_1433/comparison.json').read_text())
    story=[]
    story += [p('고정 힘 가공량 균일도\n최종 기준 모델 보고서','title'),
        p('사용자 고정 모델 · compensated433 · 2026-10-09','sub'),
        p('이 문서는 실제 저장 파일과 학습 소스를 기준으로 작성했다. 현재 고정 모델은 PPO 모델이 아니라, 제약이 있는 체류시간 프로파일 최적화 결과를 TorchScript actor로 저장한 정책이다. 신규 PPO 연구는 별도 실험이며 이 기준 모델을 덮어쓰지 않는다.'),
        table([['항목','고정 사양'],['목표 접촉력','20 N 고정; 정책의 힘 변경 권한 없음'],
            ['회전','가공량 추정기에만 일정 1,000 RPM 가정; URDF 회전 없음'],
            ['정책 출력','전진 이송 속도 1개; 속도로 단위 길이 체류시간 조절'],
            ['경로','697.533257 mm; 역주행·추가 패스 없음'],
            ['학습 / 실증 결과','독립 headless 검증 CV 평균 12.59% 개선; GUI 주 비교 CV 13.99% 개선. 모두 해당 mode 6 대비.']], [115,WIDTH-115]),
        Spacer(1,16),p('고정 checkpoint SHA-256','sub'),p(lock['checkpoint_sha256'],'small'),
        p(lock['checkpoint'],'small'),p('고정 기록: deployment/FIXED_MODEL.json. 자동 교체 허용: false. 기존 12.15% 세이브와 원본 학습 아카이브도 보존했다.','small')]

    page(story,'1. MDP 관점의 인터페이스와 실제 학습 방식')
    story += [p('로봇과 힘 제어기, 누적 가공량 지도, 경로 진행률, 필터 상태가 시스템 상태를 이룬다. 정책 인터페이스는 16차원 observation과 1차원 action이다. 다만 이 모델은 환경 rollout의 step reward로 policy gradient를 업데이트하지 않았다.'),
        table([['구분','실제 구현'],['Observation','16D 공간 특징. 최종 actor는 obs[0], 경로 진행률만 사용한다.'],
            ['Action','스칼라 이송 명령. 고정 법선 힘 제어기는 별도 계층에서 20 N을 추종한다.'],
            ['Transition','Isaac 로봇/접촉 물리와 기존 제어기; 학습용 근사 모델은 가공량 커널과 속도 필터 응답.'],
            ['Reward','PPO용 step reward를 이용한 업데이트 없음. 공간 CV² 기반 목적함수를 OSQP/Powell로 최소화.'],
            ['Episode','기준 경로 cursor 완주. 검증 최대 42,000 제어 tick, 336초; 과도한 힘 100 N 초과 시 fault.']], [105,WIDTH-105]),
        p('실제 actor 구성','sub'),p('학습 가능한 신경망 weight는 없고, 길이 433의 residence_profile 버퍼가 저장되어 있다. 프로그램은 이 프로파일을 경로 진행률에 따라 보간한다. 측정 힘·깊이 부족량·실측 속도 등의 특징은 생성·기록되지만, 이 최종 actor의 출력에는 직접 사용되지 않는다.'),
        p('별도의 저수준 힘 제어기에는 기존 NAF-MDGradi 가중치가 사용된다. 그 가중치와 본 가공량 균일도 actor는 서로 다른 모델이다. 본 실험은 그 힘 제어기를 재학습하지 않았다.'),
        p('근거: scripts/fixed_force_dynamics.py:16, scripts/train_fixed_force_dynamics.py:41, scripts/probe_fixed_force_refinement.py:183. env.step이 반환한 기존 reward는 유한성 확인에만 쓰였다.','small')]

    page(story,'2. Observation 정의 — 16차원')
    obs=[
        ['0','progress','s / L; 이 actor가 실제 사용하는 입력'],
        ['1','force_over_20','측정 법선 힘 F / 20'],['2','speed_over_6','실측 접선 속도 v / 6'],
        ['3','centered_nominal_depth_deficit','1 − local(D_nom) / D* − 경로 평균 부족량'],
        ['4','centered_predicted_final_deficit','1 − local(D_now + D_remaining) / D* − 경로 평균 부족량'],
        ['5','behind_minus_current_deficit','5 mm 뒤쪽의 예상 부족량 − 현재 위치 예상 부족량'],
        ['6','local_residence_over_5s','현재 접촉 영역의 누적 체류시간 평균 / 5초'],
        ['7','tracking_error_over_10mm','경로 추종 위치 오차 / 10 mm'],
        ['8','elapsed_over_nominal','활성 가공 시간 / (L / 6)'],
        ['9','reverse_budget_fraction','누적 역방향 거리 / (0.05 L); 현재 모델 0'],
        ['10','contact','측정 법선 힘 ≥ 1.5 N'],['11','shield','제어기의 속도 보호 정지 상태'],
        ['12','target_force_over_20','목표 힘 / 20; 현재 모델 1'],
        ['13','rate_error','F × |v| / 120 − 1; 회전을 포함하지 않는 기존 proxy'],
        ['14','return_allowed','부분 역주행 허용 여부; 현재 모델 0'],
        ['15','remaining_nominal_depth_over_target','현재 영역에 앞으로 추가될 nominal 깊이 / D*']]
    story += [table([['idx','이름','계산 / 의미'],*obs],[28,190,WIDTH-218],small=True),
        Spacer(1,10),p('모든 특징은 float32, [−5, 5] clipping. 특징용 격자는 2 mm이며 최종 평가 격자 0.5 mm와 다르다. D*는 20 N / 6 mm/s / 1,000 RPM nominal 최종 지도 평균이다. local(·)은 현재 공구 footprint의 압력 가중 평균이다.','small'),
        p('특히 idx 13은 회전 포함 순간 제거율이 아니다. 현재 actor가 idx 13을 사용하지 않으므로 최종 정책은 이 proxy에 반응하지 않는다. 근거: spatial_uniformity.py:172, fixed_force_velocity.py:91.','small')]

    page(story,'3. Action, 체류시간과 제어 계층')
    story += [p('저장된 q는 nominal 6 mm/s 대비 단위 길이 체류시간의 배율이다. 정해진 경로 위에서 q가 큰 위치는 느리게, q가 작은 위치는 빠르게 진행한다.'),
        p('q = 6 / v,    residence per distance = q / 6  [s/mm]','formula'),
        p('phase = clip(s/L, 0, 1);    q(phase) = linear interpolation of 433 knots','formula'),
        p('v0 = clip(6/q(phase), 1.5, 12)','formula'),
        p('preview = min(v0 × 0.357834, 2.0) [mm]','formula'),
        p('v_request = clip(6/q(phase + preview/L), 1.5, 12)','formula'),
        p('a = v_request/6 − 1;    controller mapping: v = 6 + 6a','formula'),
        table([['파라미터','값'],['최종 q 최소 / 최대',f'{min(c["residence_profile"]):.9f} / {max(c["residence_profile"]):.1f}'],
            ['정책 판단 / 제어 주기','80 ms (12.5 Hz) / 8 ms (125 Hz)'],['행동 저역통과 필터','시정수 0.08초'],
            ['행동 slew / 속도 가속도 / jerk','18 mm/s² / 16 mm/s² / 160 mm/s³'],
            ['경로 추종 보정','gain 0.6, 시정수 0.08초, 보정 길이 상한 4 mm'],
            ['하위 법선 힘','항상 20 N; 순간 실제 측정 힘과 목표 힘은 다름']], [190,WIDTH-190]),
        p('preview 속도는 잡음이 있는 실측 TCP 속도가 아니라 현재 프로파일에서 계산한 v0다. 필터와 가속도 제한의 지연을 보상한다. 체류시간 증가는 감속의 결과이며 별도 정지 시간 출력, 주기적 정지, 역주행 출력은 없다. 가공 완료 후 홈 복귀는 정책의 추가 가공 패스에 포함하지 않는다.')]

    page(story,'4. Reward에 해당하는 실제 최적화 목적')
    story += [p('학습 변수는 residence 프로파일 q와 그 scale·contrast·preview 파라미터다. 경로 기하와 이전 고정 힘 Isaac trace로 제거 깊이 응답 행렬 A를 만들고, 두 응답 모델의 예측 공간 CV²를 줄인다.'),
        p('D ≈ A q,    CV² = mean(D²) / mean(D)² − 1','formula'),
        p('초기 QP: 두 모델의 평균 제거량을 각 nominal mode 6 대비 1.25로 고정한다. q ∈ [0.5, 4], 인접 q 차이 ≤ 0.24. 두 모델 가중치 {0.375, 0.5, 0.625}를 각각 풀고 아래 점수가 가장 작은 해를 선택한다.'),
        p('J_fit = 0.75 mean(CV²) + 0.25 max(CV²) + 10⁻⁵ ‖Δq‖² / 433','formula'),
        p('OSQP는 평균 고정으로 분모를 상수화한 이차식을 푼다. 선택 가중치는 0.375이며, 0.24의 knot 차이 제한은 초기 QP의 제약이다. 후속 변환 후에는 실제 제어기의 속도·가속도·jerk 제한을 별도로 적용한다.'),
        p('명령 응답 보정 단계','sub'),
        p('q_final = clip(1.043058216 × q_initial^1.014137394, 0.5, 4)','formula'),
        p('J_response = 0.75 mean(CV²) + 0.25 max(CV²) + max(P)','formula'),
        p('P = 20 max(0, 0.75−V_ratio)² + 20 max(0, V_ratio−1.50)²\n     + 10 max(0, T−190)²','formula'),
        p('Powell 최적화로 scale, contrast, preview 시간(0~0.4초)을 구한다. 두 응답 모델 × 0/3 tick 지연 조합에서 위 점수를 계산한다. 0.75~1.50은 근사 모델의 soft penalty 범위이며, 실제 채택 조건은 각 독립 Isaac 검증에서 mode 6 제거량의 0.50 이상이다.'),
        p('선정 seed 2801, 독립 검증 seed 2811·2812. 독립 검증 trace는 프로파일 피팅에 쓰지 않았다. PPO clipped loss, GAE, value network, PPO optimizer state는 이 기준 모델에 존재하지 않는다.','small')]

    page(story,'5. 가공량 계산 — 회전 공구 Preston 적분')
    story += [p('고정된 원형 접촉 footprint 안에서 압력과 국소 상대 미끄럼 속도를 곱해 깊이를 적분한다. 실제 제거 계수 K를 보정하지 않았으므로 출력 단위는 깊이 h/K와 체적 V/K다.'),
        p('ω = 2π × RPM / 60;    RPM = 1000,    R = 15 mm','formula'),
        p('p_j = F_n w_j / (Σ_k w_k ΔA),    w_j = 1 inside disk, else 0','formula'),
        p('v_rel,j = √[(v_x − ω y_j)² + (v_y + ω x_j)²]','formula'),
        p('Δ(h_j/K) = p_j × v_rel,j × Δt;    Δt = 0.008 s','formula'),
        p('V_ROI/K = Σ_(j in ROI) (h_j/K) ΔA;    ΔA = 0.25 mm²','formula'),
        p('TCP가 정지해도 가상 회전 때문에 v_rel = ωr가 남는다. 따라서 정지 또는 감속으로 체류시간이 늘면 제거량이 증가할 수 있다. 회전 자체는 URDF 관성·접촉 물리·스핀들 모터 상태에는 적용하지 않는다.'),
        table([['요소','구현'],['공구 / 접촉 직경','30 mm / 30 mm; 평면 전면 접촉, 균일 압력 가정'],
            ['누적 조건','가공 active, 법선 힘 ≥ 1.5 N, fault 없음'],
            ['속도 방향 / 크기','TCP 위치 차분의 접선 방향 / 제어 snapshot의 실측 sliding speed'],
            ['footprint 위치','앞뒤 샘플의 중점. float32 위치 연산을 학습·실증 기록기에 맞춤'],
            ['격자 밖 접촉','압력 정규화는 전체 공구 footprint 기준; 격자 밖 양을 ROI 안으로 재분배하지 않음']], [115,WIDTH-115]),
        p('모델에 포함되지 않은 요소: 실제 K 보정, 마모, 온도, 재료 비선형성, 탄성 접촉면적 변화, 실제 RPM 변동. 그러므로 CV와 상대 체적은 모델 내 비교값이며 실제 μm 또는 mm³ 제거량을 확정하지 않는다.','small')]

    page(story,'6. 공정한 평가 구간과 검증 결과')
    story += [p('주 지표는 시간축 순간 제거율의 CV가 아니라, 완료 후 표면에 남은 누적 제거 깊이의 공간 CV다. 동일한 기준 경로에서 ROI를 미리 만들며, 깊이가 0인 셀도 포함한다.'),
        p('CV_spatial = std(h/K over fixed ROI) / mean(h/K over fixed ROI)','formula'),
        p('Improvement_vs_mode6 = 100 × (1 − CV_mode5 / CV_mode6)','formula'),
        p('① 활성 가공 시작 후 첫 2초의 깊이 기여를 뺀다. ② 기준 경로 첫 18 mm가 공구 footprint로 덮는 셀을 고정 제외한다. 최종 평가 격자 0.5 mm, 전체 66,254셀 / 가공 구간 61,277셀.'),
        table([['검증','공간 CV 개선','전체 ROI 제거량 유지'],['Headless seed 2811','12.54%','128.64%'],
            ['Headless seed 2812','12.64%','128.44%'],['GUI 주 비교','13.99%','128.11%']], [190,150,WIDTH-340]),
        p('모든 백분율은 각 검증의 mode 6 대비다. Headless 검증 평균이 기존 “12.59 모델” 이름의 근거이며, GUI 13.99%는 별도 실행의 관측 결과다.'),
        table([['GUI 주 비교의 추가 지표','mode 6 대비'],['가공 구간 제거량','130.10% 유지'],
            ['가공 구간 시간','29.06% 증가; 115.584 → 149.176초'],['시간축 순간 제거율 CV','34.42% 증가']], [260,WIDTH-260]),
        p('GUI는 각 모드 1회의 주 비교다. 저장된 홈 위치는 0.537 mm 차이가 있지만 가공 진입 TCP 위치는 일치했다. 전체 물리 초기 상태의 완전한 일치는 로그로 증명하지 못한다. 제거량 저장·홈 복귀·샘플 누락 0·fault 0을 확인했고 원시 snapshot 재적분도 저장 지도와 일치했다.','small')]

    page(story,'7. GUI 결과 시각화')
    images=ROOT/'logs/polishing_results/20261009_1433'
    story += [picture(images/'00_mode6_relative_comparison.png'),Spacer(1,15),
        picture(images/'02_normalized_depth_maps.png'),Spacer(1,12),
        p('상단: mode 6를 100으로 정규화한 비교. 하단: 각 지도의 평균 깊이로 나눈 공간 분포이며 두 그림은 같은 색 범위를 사용한다. 평균값 차이와 공간 균일도 차이를 구분하기 위한 표시다.','small'),
        p('고정 모델은 처리 시간을 더 사용해 공간 CV를 줄였다. 시간축 순간 제거율 변동은 별도 지표이므로 공간 CV 개선을 순간 제거율 CV 개선으로 해석하지 않는다.','small')]

    page(story,'8. 고정 산출물과 재현 근거')
    sources=[('고정 manifest','deployment/FIXED_MODEL.json'),('배포 checkpoint','deployment/fixed_force_12p59/checkpoints/actor.pt'),
        ('배포 설정','deployment/fixed_force_12p59/fixed_force_policy.json'),('학습 actor / 알고리즘','scripts/fixed_force_dynamics.py; scripts/train_fixed_force_dynamics.py'),
        ('관측 / 회전 명목 지도','scripts/spatial_uniformity.py; scripts/fixed_force_velocity.py'),
        ('회전 Preston 계산','source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/utils/preston_surface.py'),
        ('GUI 측정기','deployment/fixed_force_12p59/paired_measurement.py'),
        ('학습 보존본','logs/experiments/uniformity_saves/fixed20_virtual1000_dynamics_20261009_1311/'),
        ('학습 검증 결과','logs/experiments/fixed_force_dynamics_20261009/result.json'),
        ('실증 비교 / 재적분 감사','logs/polishing_results/20261009_1433/comparison.json'),
        ('추가 보존 사본','logs/experiments/ppo_fixed_force_20261009/frozen_baseline/actor.pt')]
    story += [p('아래 경로는 모두 /home/eunseop/nrs_rl 기준이다. 현재 배포 실행 파일·제어 파라미터·checkpoint는 이 보고서를 위해 수정하지 않았다.'),
        table([['내용','경로'],*sources],[110,WIDTH-110],small=True),Spacer(1,12),
        p('별도 PPO 연구와의 경계','sub'),p('새 PPO는 신경망 actor·critic을 정책 경사로 업데이트하는 별도 모델이다. 이 보고서의 433-knot 기준 모델을 PPO라고 명명하지 않는다. 새 PPO의 채택 여부와 상관없이 현재 고정 checkpoint를 자동 교체하지 않는다.'),
        p('작성 기준일 2026-10-09. checkpoint 해시·source manifest·실행별 deployment.json·원시 CSV/NPZ가 이 문서의 근거다. source manifest에 포함된 기존 학습 코드 20개 파일의 일치를 확인했다.','small')]
    output=REPORTS/'fixed_force_baseline_20261009.pdf'
    build(output,story,'고정 힘 가공량 균일도 최종 기준 모델')
    return output


if __name__=='__main__':
    output=baseline()
    print(output)
    print('sha256',hashlib.sha256(output.read_bytes()).hexdigest())
