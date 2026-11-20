# 드론 파지 RL 설계 (`DroneGraspRl-Grasp-UR5`)

UR5 + ROBOTIS RH-P12-RN(A) 그리퍼가 공중에 떠 있는 Iris 드론의 몸체 허리를 **아래에서 올라가 잡고**, 모터가 꺼진 뒤에도
버티는 태스크다. 이 문서는 정책 구조, 관측, 행동, 보상, 종료, 판정, 학습 방법과 그렇게 정한 이유를 정리한다.

현재 정책은 **특권 정보 선생(privileged teacher)**이다. 시뮬레이터 안에서만 쓰며 (1) VLA 학습용 시범 데이터를 만들고
(2) 나중에 카메라 영상 기반 학생 정책으로 증류해 실물에 배포하는 본보기가 된다.
숫자는 코드(`src/drone_grasp_rl/tasks/grasp/`)와 같고, 바꿀 때는 이 문서도 같이 고친다.

- 기준 엔진: **PhysX** (기존 data collection 레포의 게인이 PhysX 기준). Newton(MJWarp)은 에셋·태스크 검사만 통과, 파지 학습은 아직
- 주기: 물리 120 Hz, 정책 30 Hz (decimation 4), 에피소드 최대 12 s
- 코드 위치

| 파일 | 내용 |
|---|---|
| `tasks/grasp/grasp_env_cfg.py` | 장면, 관측, 이벤트, 보상, 종료 (로봇 공통) |
| `tasks/grasp/config/ur5/env_cfg.py` | UR5 장면·센서·행동, 커리큘럼, 판정 센서 이름 |
| `tasks/grasp/config/ur5/agents/rsl_rl_ppo_cfg.py` | PPO 설정 |
| `tasks/grasp/mdp/` | `actions.py`, `events.py`, `observations.py`, `rewards.py`, `judge.py`, `terminations.py` |

<br>

## 1. 에피소드 흐름

```
리셋 (under_drone 자세, 드론 정지)
  → 접근: 드론 바로 아래 수직선으로 와서 위로 올라감
  → 닫기: 허리에서 닫기 명령 (한 번 닫으면 에피소드 끝까지 닫힘)
  → 잡힘 (GRASPED)            : 닫는 중 0.2 s 정지 + 62° 미만 + 양쪽 손가락 > 1 N
  → 1 s 뒤 모터 정지 (KILLED) : 드론을 띄우던 힘 제거 → 드론 무게가 그리퍼로
  → 2.5 s 버팀 → 성공 (HELD)  : 내려앉음 ≤ 15 mm, 마지막 0.5 s 변화 ≤ 0.5 mm,
                                  드론 회전(TCP 기준) ≤ 10°, 그리퍼 기울기(위 기준) ≤ 10°
```

- 판정기(`mdp/judge.py`)는 기존 레포의 `grasp_judge.py`를 옮긴 것이다. 마지막 조건(그리퍼 기울기)만 새로 추가했다.
  사람 텔레옵은 그리퍼를 똑바로 들었지만 학습된 정책은 드론과 함께 기울였기 때문이다.
- 판정기는 **첫 번째 종료 조건**(`grasp_success`)으로 돈다. Isaac Lab은 종료 조건을 보상보다 먼저 계산하므로, 같은 스텝의
  다른 종료 조건과 보상이 판정 결과를 읽는다(`mdp.get_judge(env)`).
- 정책은 모터 정지를 관측하지 않는다. 실제 드론 제어기도 자신이 잡혔는지 모른다.

<br>

## 2. 장면과 리셋

| 항목 | 값 | 출처 |
|---|---|---|
| 테이블 | 2 × 1 m, 상판 z 0.762 m = 로봇 베이스 높이 | `scene_drone.yaml` |
| 로봇 | env 원점, `under_drone` 자세에서 시작 (TCP가 드론보다 0.305 m 아래), 그리퍼 열림 | 기존 레포 |
| 드론 | Iris 0.75배, 0.8 kg. 리셋마다 25곳 기준 위치 중 하나 ±3 cm (로봇 베이스 기준), 정지 상태 | `scene_drone.yaml` |
| 드론 비행 | 기하 제어기(geometric controller)가 hover 목표를 따라감. 사인파 프리셋 static/small/mid/big. 모터 정지 뒤 추력 0 | 6-1장 |
| 접촉 재질 | PhysX: 손가락·드론 정지 0.6 / 운동 0.5 (각각 ×0.7~1.3 랜덤화). Newton: 기본값 1.0 유지 | `drive_gains.yaml` |
| 드론 충돌체 | PhysX: 원래 `convexDecomposition`. Newton: 미리 분해한 30조각 | `assets/data/drones/iris/README.md` |

**학습 커리큘럼 (초기 상태):** 학습 리셋의 50%는 드론을 **시작 TCP 바로 위 0~30 cm**(수직선 위, 옆 ±3 mm)에 놓는다.
나머지 50%는 원래 25곳이다. `play`와 평가 스크립트는 커리큘럼 없이 25곳만 쓴다.

<br>

## 3. 관측

비대칭 actor-critic이다. 선생 actor(`policy`, 52차원)는 실물에서 얻을 수 있는 30개 뒤에 **특권 정보 22개**를 더 본다.
critic은 학습 때만 쓴다.

### actor 앞부분 (`policy` 0~29, 실물에서도 얻을 수 있는 값)
| term | 차원 | 실물에서 얻는 방법 |
|---|---|---|
| 팔 관절 위치 | 6 | UR5 엔코더 (RTDE) |
| 팔 관절 속도 | 6 | UR5 (RTDE) |
| 그리퍼 관절 위치 (`rh_r1_joint`) | 1 | 그리퍼 현재 위치 |
| TCP 위치 (베이스 기준) | 3 | 관절 각도로 FK |
| **드론 자세 (모션캡처 모델)** | 7 | 모션캡처 |
| 직전 행동 | 7 | 정책 자신의 출력 |

**모션캡처 모델** (`mdp.mocap_drone_pose_b`, 기존 레포의 가정):
- 지연 20 ms + 샘플 나이 0~10 ms (100 Hz 모션캡처를 30 Hz로 읽음). 현재 속도로 과거 자세를 계산
- 에피소드마다 고정 보정 오프셋: 위치 ±2 mm, 회전 최대 0.5° (추정값)
- 매 샘플 잡음: 위치 1 mm, 회전 0.3° (정규분포)
- 마커 가림: 초당 0.2번, 0.1~0.5 s 동안 마지막 값 유지 (OpenAI in-hand manipulation 방식)

팔 관절 위치(±0.002 rad), 속도(±0.02 rad/s), 그리퍼(±0.005 rad), TCP(±1 mm)에도 작은 균등 잡음을 넣는다.

### actor 뒷부분 (`policy` 30~51, 특권 정보, 선생 전용)
| term | 차원 | 비고 |
|---|---|---|
| 드론 정확한 위치·자세 (베이스 기준) | 7 | |
| 드론 선속도·각속도 (×5) | 6 | 크기를 1 근처로 맞춤 |
| TCP 기준 드론 위치 (×10) | 3 | |
| 손가락 링크 4개 접촉력 (×0.05, 0~50 N) | 4 | |
| 그리퍼 닫힘 (래치), 잡힘 감지 | 2 | 판정기 단계. **모터 정지 여부는 넣지 않음**: 카메라 학생이 알 수 없는 정보라 증류를 어렵게 함 |

뒤에 붙인 이유: 30차원으로 학습된 정책을 `scripts/expand_policy_inputs.py`로 52차원으로 늘려 이어서 학습하기 위해서다.
새 입력 가중치는 0, 관측 정규화는 평균 0·표준편차 1, Adam 모멘트도 같은 크기로 늘린다(변환 직후 동작은 원래 정책과 같다).
**주의:** 변환하면서 행동 잡음(std)을 0.86 → 0.3으로 바꾸면 10 iteration 만에 정책이 접근을 멈췄다(보상과 무관함을 분리 실험으로 확인).
std는 그대로 둔다.

### critic (`critic`, 17차원)
드론의 정확한 위치(3)·자세(4), 드론 선속도(3)·각속도(3), 손가락 링크 4개의 드론 접촉력(4).

### 자세 계산 방식
TCP와 드론 자세는 `ee_frame`(FrameTransformer) **설정**으로 정의하지만, 값은 관절체 몸체 자세에서 직접 계산한다
(`mdp.frame_pose_w`). Newton에서는 FrameTransformer 데이터가 첫 물리 스텝 전까지 0이라 NaN이 났기 때문이다.
센서는 시각화(`debug_vis`)용으로 남아 있다.

<br>

## 4. 행동 (7차원)

| term | 차원 | 동작 |
|---|---|---|
| `arm_action` (`InterpolatedRelativeJointPositionAction`) | 6 | 행동 [−1, 1] × **0.05 rad**만큼 관절 목표(goal)를 옮기고, 그 스텝의 물리 4번 동안 선형 보간. 관절 한계 − 0.05 rad로 제한. 에피소드마다 0~1 스텝 명령 지연 |
| `gripper_action` (`ProfiledBinaryJointPositionAction`) | 1 | 행동 < 0이면 닫기. 목표가 실물 속도 0.516 rad/s로 이동(완전히 닫는 데 2.2 s). **한 번 닫으면 에피소드 끝까지 유지(latch)** |

- 팔: 스텝당 0.05 rad = 최대 1.5 rad/s(기존 IK의 관절 목표 속도 한계). 보간은 기존 레포의 33 ms 보간·실물 UR servoJ와 같다.
  절대 목표가 한 스텝에 0.5 rad씩 바뀌는 방식은 보호 정지(추종 오차 5°)에 계속 걸렸다.
- 데이터셋에 기록할 행동은 `processed_actions` = 절대 관절 목표(기존 LeRobot 스키마와 같음).
- **팔 동작 다듬기 (선생 시범을 차분하게):**
  - 저역 필터(EMA): 적용 행동 = 0.5 × 새 행동 + 0.5 × 이전 적용 행동
  - 근처 감속: 잡는 점에서 25 cm 안에서는 스텝 상한을 거리에 비례해 줄임 (25 cm 100% → 잡는 점 10%)
  - 잡은 뒤 정지: 판정기가 잡힘을 감지하면 팔 목표를 고정 (스크립트 전문가와 같은 동작). 측정해 보니 잡은 뒤 팔을 멈춰도
    성공률은 유지되고(98 → 95%) 기울기는 절반(6.6° → 3.2°)이었다. 정책이 잡은 뒤 움직이던 것은 불필요한 동작이었다
  - 효과: 잡기 직전 5 cm 최고 속도 267 → 137 mm/s, 잡은 뒤 팔 움직임 7°/s → 0.1°/s
- 그리퍼 래치: 닫는 데 0.6~2.2 s 동안 계속 명령해야 하는데, 스텝마다 독립인 탐색 잡음은 33 ms마다 열기/닫기를 뒤집어
  학습 중에 그리퍼가 **실제로 한 번도 닫히지 않았다.** 기존 데이터 수집 규칙(에피소드당 한 번 잡기, 실패 후 재시도 시연 없음)과도 맞는다.
  실물에서도 같은 래치를 제어기에 둔다.

<br>

## 5. 보상

Isaac Lab의 보상 관리자는 매 스텝 보상에 **dt(1/30 s)를 곱한다**. 성공·실패 보상(`terminal_bonus`)만 dt로 나눠서
가중치가 그대로 한 번 지급된다(Isaac Lab `terminated_penalty`와 같은 방식).

| term | 식 | 켜지는 구간 | 가중치 | 출처 / 이유 |
|---|---|---|---|---|
| `approach` | 접근 경로 남은 길이가 에피소드 최고 기록보다 5 mm 줄면 1 (진행 보상) | 잡기 전 | +5 | Isaac Lab core lift `position_command_progress`, DextrAH-G `minimize`. 물러났다 다시 오는 꼼수 방지 |
| `approach_fine` | `1 − tanh(d / 2 cm)` (TCP–잡는 점 거리) | 잡기 전, 드론 아래 수직선 위(옆 2 cm 이내)일 때만 | +2 | contrib lift의 거친+정밀 보상. 진행 보상만으로는 허리 4 cm 아래에서 멈춤 |
| `misalign` | `(1 − z_tcp·z_drone) + (1 − \|y_tcp·y_drone\|)` | 항상 | −1 | cabinet `align_ee_handle`. 시작 자세가 이미 정렬이라 벌점 형태 |
| `gripper_tilt` | `1 − cos(TCP z와 위쪽 사이 각)` | 항상 | −25 | 잡은 뒤 드론과 같이 기울어 `misalign`이 0이 되는 빈틈. −15에서는 랜덤화 학습 중 기울기가 15° 한계 가까이 커짐 |
| `drone_tilt` | `1 − cos(드론 z와 위쪽 사이 각)` | 항상 | −10 | 잡은 드론을 똑바로 |
| `close_near` | 허리 4 mm 이내·옆 1 cm 이내에서 닫기 명령이면 1 | 잡기 전 | +2 | cabinet `grasp_handle`. 15 mm였을 때 올라가는 중에 닫아 드론을 짜고 기울임 |
| `drone_slip` | 잡은 뒤 드론이 TCP 기준으로 2 mm/s 넘게 움직인 속도 [m/s] (위치 변화 기준) | 잡힘~모터 정지 뒤 | −1000 | 남은 실패의 대부분이 모터 정지 뒤 slip. 잡은 뒤 팔은 멈추므로 잡는 위치·방식으로만 대응 |
| `finger_contact` | 양쪽 손가락 접촉력 > 1 N이면 1 | 모터 정지 전까지 | +2 | core lift 손가락 접촉 보상, 1 N은 기존 판정기 |
| `hold` | 모터 정지 뒤 양쪽 접촉 + 내려앉음 ≤ 15 mm면 1 | 모터 정지 뒤 | +4 | 기존 판정기 기준 |
| `arm_still` | 팔 행동 **절댓값(L1)** 합 | 잡기 확인 뒤 | −1.5 (커리큘럼) | 비행 물체 잡기(Hu et al. 2023)의 "접촉 뒤 손 멈추기". 제곱(L2)은 0.1 크기의 미세 떨림에 거의 벌점이 없었음. 닫기 명령부터 걸면 기울기를 못 바로잡아 성공률 82→45% |
| `approach_speed` | 드론 기준 TCP 속도가 `5 cm/s + 거리 × 1/s`를 넘은 양 [m/s] | 잡기 전, 잡는 점 10 cm 이내 | −10 (커리큘럼) | 잡기 직전 돌진(중앙값 27 cm/s) 억제. 접근 전체에 걸었더니 정책이 움직이지 않게 됨 |
| `joint_acc` | 팔 관절 가속도² 합 | 항상 | −1e-4 (커리큘럼) | 급출발·급정지 억제 |
| `success` | 판정 held | 그 순간 한 번 | **+20** | |
| `failure` | 실패 종료 (놓침·slip, 드론 쳐 냄, 보호 정지) | 그 순간 한 번 | **−5** | core lift `early_termination` |
| `drone_push` | 잡기 전 드론이 hover 목표에서 4 cm 넘게 벗어난 거리 [m] | 잡기 전 | −20 | 근거 없음. 4 cm는 드론 자체 흔들림(약 19 mm RMS)보다 크게 |
| `action_rate` | `‖aₜ − aₜ₋₁‖²` | 항상 | −0.001 → −0.01 (커리큘럼) | contrib lift |
| `joint_vel` | 팔 관절 속도² 합 | 항상 | −0.0001 | contrib lift. MJWarp는 관절 속도 한계를 적용하지 않음 |
| `joint_limits` | 관절 한계 초과량 | 항상 | −1 | Isaac Lab 기본 |

스크립트 전문가(`scripts/scripted_grasp.py`)의 성공 에피소드 총 보상은 약 45다(접근 ~10, 접촉 ~2, 버팀 ~10, 성공 20).

**뺀 것:** `empty_close`(멀리서 닫기 벌점)는 "절대 닫지 않기"를 가르쳐서 뺐다. 판정기 지표(`Judge/empty_close`)는 남아 있다.

<br>

## 6. 종료 조건

| term | 종류 | 조건 |
|---|---|---|
| `grasp_success` | 성공 | 판정 held (판정기 본체, 반드시 첫 번째) |
| `grasp_failed` | 실패 | 판정 실패: dropped(잡은 위치에서 30 mm), slip(버티기 기준 미달), opened_early(래치로 현재는 안 생김) |
| `drone_knocked` | 실패 | 잡기 전 드론이 hover 목표에서 10 cm 넘게 벗어남 |
| `protective_stop` | 실패 | 팔 관절 추종 오차 > 5° 또는 속도 > 200°/s (기존 레포의 UR 보호 정지). 150 N 접촉력 기준은 아직 없음 |
| `time_out` | 시간 초과 | 12 s |

<br>

## 6-1. 드론 비행 제어기 (`mdp/drone_control.py`)

정책이 아니라 드론 스스로 난다(기존 레포의 PX4 역할). 물리 스텝마다 실행되도록 **행동 차원 0인 action term**으로 만들었다.
그래서 정책의 행동 크기(7)는 그대로다.

| 항목 | 내용 |
|---|---|
| hover 목표 | `p(t) = p0 + Σ (A/2)[sin(2πfⱼt + φⱼ) − sin φⱼ]`, 축마다 사인 2개. 리셋마다 f, φ 새로 뽑음 |
| 프리셋 (기존 `drone_iris.yaml`) | static 0 / small 20·20·10 mm, 0.2~0.5 Hz / **mid 30·30·15 mm**, 0.05~0.3 Hz / big 50·50·20 mm, 0.05~0.3 Hz |
| 학습 / 평가 | 학습: 네 프리셋을 같은 확률로. `play`·평가: mid (기존 레포의 수집·평가 기본) |
| 제어 | 기하 제어기: 위치 PD + 적분(상한 있음) + 목표 가속도 앞먹임 → 원하는 힘. **추력은 몸체 z축 방향만**(쿼드로터처럼 기울어야 이동). 자세 PD로 그 방향을 맞춤 |
| 게인 | Pegasus Iris 기하 제어기 값(Kp 10, Kd 8.5, Ki 1.5, Kr 3.5, Kw 0.5)을 질량·관성 비율로 0.75배 기체에 맞춤 |
| 기타 | 몸체 기준 선형 항력, 무작위 외란 힘 (6-2) |
| 모터 정지 | 판정기의 kill 이후 추력·토크 0. 그 전까지는 잡혀 있어도 목표를 지키려고 그리퍼와 버팀 (PX4와 같음) |
| `mode="float"` | 몸체별 중력 상쇄만(예전 방식). `check_task`가 사용 |

`scripts/check_hover.py` 결과 (외란 없음): 추종 오차 RMS static 2.1 / small 2.7 / mid 2.6 / big 3.6 mm, 0.3 m/s로 밀면 1.5 s에
5 mm 이내로 복귀. 외란 0.1 N이면 축당 8.8~10 mm, 전체 19 mm RMS로 기존 레포의 PX4 hover 측정값(19.5 mm RMS, 축당 8~10 mm)과 같다.

<br>

## 6-2. 도메인 랜덤화

문헌(OpenAI in-hand manipulation, DextrAH-G, 비행 물체 잡기)과 기존 레포 측정값을 기준으로 넣었다.

| 항목 | 값 | 시점 | 근거 |
|---|---|---|---|
| 드론 외란 힘 (Ornstein-Uhlenbeck) | 축당 표준편차 0.1 N, 상관 시간 0.5 s | 매 물리 스텝 | PX4 hover 흔들림 19.5 mm RMS에 맞춤 |
| 팔 명령 지연 | 0~1 정책 스텝 (0~33 ms) | 에피소드마다 | 실물 명령→관절 지연 47 ms (기존 레포), Dactyl 행동 지연 |
| 손가락·드론 마찰 | ×0.7~1.3 (정지 0.42~0.78, 운동 0.35~0.65) | 시작 시 env마다 | Dactyl 범위. 실물 마찰은 아직 미측정 |
| 드론 질량 | ×0.9~1.1 | 시작 시 env마다 | 제어기가 매 스텝 질량을 읽음 |
| 팔 구동 강성·감쇠 | ×0.8~1.25 (로그 균등) | 시작 시 env마다 | Dactyl 게인 ×0.75~1.5 |
| 모션캡처 가림 | 초당 0.2번, 0.1~0.5 s 마지막 값 유지 | 매 스텝 | Dactyl 마커 가림 |
| 센서 잡음 | 관절·그리퍼·TCP 소량 | 매 스텝 | |

`GraspEnvCfg.disable_randomization()`으로 한 번에 끌 수 있다(`check_task`, 스크립트 시연, `eval_policy.py --no_dr`).

<br>

## 6-3. 커리큘럼

| 커리큘럼 | 내용 |
|---|---|
| 시작 상태 | 학습 리셋의 50%는 드론을 시작 TCP 바로 위 0~30 cm에 놓음 (2장) |
| **보상 가중치 (`mdp.ramp_reward_weight`)** | 새 벌점(`approach_speed`, `joint_acc`, `action_rate`, `arm_still`)을 학습 시작부터 24,000 env 스텝(1000 iteration) 동안 직선으로 목표값까지 올림. 학습된 정책에 한꺼번에 넣으면 critic이 모르는 큰 변화가 됨. `min_success`를 주면 학습 중 성공률이 그 이상일 때만 올라가는 성적 기반 방식(unitree_rl_lab의 명령 범위, Isaac Lab ADR과 같은 발상) |

참고한 예: unitree_rl_lab (지형 난이도, 명령 속도 범위를 성적 기반으로 넓힘), Isaac Lab contrib lift (행동 벌점을 10,000 스텝 뒤 강화),
Isaac Lab core lift ADR (관측 잡음·중력을 성적 기반으로), DextrAH-G (성공률 90% 넘으면 제어기 감쇠 증가).
이미 학습된 선생을 이어서 학습할 때는 `env.curriculum.<term>.params.initial_weight`를 도달한 값으로 덮어써서 가중치가 다시 0부터 시작하지 않게 한다.

<br>

## 7. 학습 (PPO, RSL-RL)

| 항목 | 값 | 이유 |
|---|---|---|
| 환경 수 | 2048 (iteration당 약 1 s, RTX 4070 Ti SUPER) | |
| 스텝 / iteration | 24 → 배치 49,152, mini-batch 4개, epoch 5 | Isaac Lab 기본 |
| 네트워크 | actor·critic MLP [256, 128, 64], ELU, 관측 정규화 | |
| **gamma** | **0.995** (약 6.7 s 지평) | 0.98(1.7 s)에서는 닫기 명령 4.5 s 뒤 성공 보상이 6%로 할인돼 안 보였음 |
| lambda | 0.95 | |
| 학습률 | 1e-4, adaptive (KL 0.01) | |
| init_std | 0.5 | 1.0은 손가락을 드론 사이로 넣기엔 너무 흔들림 |
| entropy | 0.003 (마지막 미세 조정은 0.001) | 0.006은 std가 커지고, 처음부터 0.001이면 std가 0.05로 무너짐 |

학습·이어서 학습·평가 명령:
```bash
# 처음부터
uv run --extra isaacsim isaaclab train --rl_library rsl_rl --task DroneGraspRl-Grasp-UR5 --num_envs 2048
# 이어서 (Isaac Lab 3.0은 agent.resume을 무시함, --checkpoint를 써야 함)
uv run --extra isaacsim isaaclab train --rl_library rsl_rl --task DroneGraspRl-Grasp-UR5 --num_envs 2048 \
    --checkpoint logs/rsl_rl/drone_grasp_ur5/<run>/model_<n>.pt
# 평가: 25곳 시작, 판정 결과·접근 기하·그리퍼 기울기·slip 원인
uv run --extra isaacsim python scripts/eval_policy.py --checkpoint logs/rsl_rl/drone_grasp_ur5/<run>/model_<n>.pt
#   --hover static|small|mid|big (기본 mid), --no_dr (랜덤화 끔)
# 보기
uv run --extra isaacsim isaaclab play --rl_library rsl_rl --task DroneGraspRl-Grasp-UR5 --num_envs 16 --viz kit \
    --checkpoint logs/rsl_rl/drone_grasp_ur5/<run>/model_<n>.pt
```

### 학습 지표 (TensorBoard)
- `Judge/grasped`, `Judge/killed`, `Judge/held`, `Judge/fail_*`, `Judge/empty_close`: 판정기 단계별 비율 (커리큘럼 에피소드 포함)
- `Metrics/approach_best_mm`: 에피소드에서 가장 가까이 간 접근 경로 길이
- `Episode_Termination/*`, `Episode_Reward/*`

총 보상보다 이 지표로 판단한다. 실제 성공률은 `eval_policy.py`(25곳만)로 잰다.

<br>

## 8. 학습 기록과 배운 점

| 시도 | 바꾼 것 | 결과 (25곳) | 배운 점 |
|---|---|---|---|
| 1 | 기본 보상 | 허리 38 mm 아래에서 멈춤 | 마지막 몇 cm 정밀 유도가 없음, 탐색 잡음 과다 |
| 2 | `approach_fine` + init_std 0.5 | 허리 8 mm까지, 닫지 않음 | 멀리서 닫기 벌점이 "닫지 않기"를 가르침 |
| 3 | 벌점 완화 | 더 나빠짐 | 벌점은 원인이 아님 |
| 4 | 멀리서 닫기 벌점 → 가까이서 닫기 보상 | 허리까지 왔다 갔다 반복 | 근접 보상을 반복 방문으로 수집 |
| 5 | gamma 0.995 | 같은 반복 | 성공을 한 번도 못 겪음 |
| 6 | 초기 상태 커리큘럼 (0~10 cm) | 잡기 0% | 그리퍼가 실제로 닫힌 적이 없음 (탐색 잡음이 매 스텝 뒤집음) |
| 7 | **그리퍼 래치** | 커리큘럼에서만 성공, 먼 접근 잊음 | 쉬움과 원래 난이도 사이가 비어 있음 |
| 8 | 커리큘럼 0~30 cm | **23% → 89%** (이어서 학습) | |
| 9~10 | 기울기 보상, 허리에서만 닫기, 잡은 뒤 팔 정지 | **93~94%** (정지 드론, 기울기 조건 포함), 잡은 뒤 최대 기울기 39° → 13° | |
| 11 | 비행 제어기 + hover 프리셋 | 제어기 드론에서 10d는 static 75% / mid 69% → 학습 후 95% / 89% (랜덤화 없음) | 닿으면 되밀고 kill 전까지 버티는 드론에 다시 학습 필요 |
| 12 | 도메인 랜덤화 | mid 79% → 85%, 기울기 8° | 랜덤화로 탐색 잡음(std)이 0.84까지 커짐 |
| 12b | entropy 0.001 | 68%, 기울기 13° | std는 줄지 않고 기울기만 커짐 |
| **12c** | entropy 0.001 + 기울기 벌점 −25 | **mid 87~91%** | 채택 |

| 13~15 | 선생 관측 52차원 + 새 벌점 (한꺼번에) | 10 iteration 만에 접근을 멈춤 | 행동 잡음 std를 0.86 → 0.3으로 바꾼 것이 원인 (분리 실험: 보상을 꺼도 무너지고, std를 두면 정상) |
| 16a | std 유지 + 새 벌점 시간 기반 커리큘럼 | 중간 88%, 끝에서 무너짐 (7%) | std가 1 가까이 커지며 벌점이 잡음에만 걸림 |
| 17a | + 팔 행동 저역 필터, 10 cm 감속 | 97.7%, 잡은 뒤 7°/s | 잡은 뒤 팔을 멈추면(진단) 성공 유지·기울기 절반 |
| 18a | + 잡은 뒤 팔 정지, 25 cm 감속 | 95.7%, 137 mm/s, 잡은 뒤 0.1°/s, 기울기 7° | |
| 19a | + 판정 기울기 10°, 속도 벌점 −10 | 94%, 144 mm/s, 0.1°/s, 기울기 3.8° | |
| 20a | + 감속 구간 30 cm, 최소 5%, 속도 벌점 −20 | 82%, 128 mm/s | 느려진 만큼 기울기 실패(10° 초과) 증가 |
| **20b** | + slip 벌점 −1000, `close_near` 4 mm | **93~96%, 138 mm/s, 기울기 3.2~3.5°** | 채택 |
| 20c | 20a + 20b | 81~88%, 122 mm/s | 20a와 같은 문제 |

- 채택 정책 계보: `try5a` → ... → `try12c` → (52차원 변환) → `try16a` 13300 → `try17a` 14100 → `try18a` → `try19a` →
  `try20b_antislip` (`model_17097.pt`, 약 17,100 iteration)
- 잡기 직전 속도: 진단해 보니 마지막 5 cm에서 TCP 자체 속도가 약 140 mm/s로, 감속 규칙이 5 cm에서 허용하는 상한(약 135 mm/s)을
  정책이 그대로 씀. 100 mm/s 이하로 하려면 상한을 더 낮춰야 하는데(20a), 1000 iteration으로는 성공률이 82%로 떨어짐
- 이어서 학습할 때 쓴 오버라이드는 각 run 폴더의 `params/env.yaml`, `params/agent.yaml`에 기록돼 있다
  (예: 커리큘럼 시작 가중치를 도달값으로, `agent.algorithm.entropy_coef=0.0015`)
- `logs/`는 git에 올라가지 않는다. 다른 PC로 옮기려면 run 폴더를 복사한다.

### 현재 선생 평가 (try20b, 원래 25곳, 256회, mid hover, 랜덤화 켬)
| seed | 성공 | slip | 버티기 끝 기울기 (중앙값 / 최대) | 잡기 직전 5 cm 최고 속도 (중앙값) | 잡은 뒤 팔 움직임 |
|---|---|---|---|---|---|
| 1 | 246 (96.1%) | 10 | 3.5° / 9.0° | 138 mm/s | 0.14°/s, TCP 0.6 mm/s |
| 2 | 239 (93.4%) | 17 | 3.2° / 7.7° | 139 mm/s | |

- 시작 정책(try12c) 대비: 잡기 직전 속도 267 → 138 mm/s, 잡은 뒤 팔 움직임 약 7°/s → 0.14°/s, 기울기 8.2° → 3.4°
- 남은 실패는 slip "마지막 0.5 s 변화 ≤ 0.5 mm" (4~7%)

<br>

## 9. 검사 스크립트

| 스크립트 | 확인 내용 |
|---|---|
| `scripts/check_assets.py --backend <engine>` | TCP 위치(URDF FK 비교), 팔 유지, 그리퍼 닫힘·mimic, 드론 hover |
| `scripts/check_task.py physics=<engine>` | 리셋 위치, 시작 시 비접촉, 정지 유지, 모션캡처 모델, 팔·그리퍼 행동 계약, 드론 허리 파지 |
| `scripts/scripted_grasp.py physics=<engine>` | IK 전문가로 전체 에피소드 → 판정 결과, 총 보상 |
| `scripts/eval_policy.py --checkpoint <path>` | 학습된 정책 평가 |

<br>

## 10. 아직 없는 것

- 잡기 직전 속도를 100 mm/s 이하로 (지금 약 144 mm/s)
- 시범 녹화: 카메라(손목·외부), LeRobot v2.1 내보내기, 시각 랜덤화
- Newton에서의 파지 학습 (접촉 조정 필요)
- 보호 정지의 150 N 접촉력 기준, 그리퍼 몸통–드론 접촉 벌점(`bad_contact`)
- 카메라, LeRobot 데이터 내보내기, 스튜어트 플랫폼
