# 드론 파지 RL 설계 (`DroneGraspRl-Grasp-UR5`)

UR5 + ROBOTIS RH-P12-RN(A) 그리퍼가 공중에 떠 있는 Iris 드론의 몸체 허리를 **아래에서 올라가 잡고**, 모터가 꺼진 뒤에도
버티는 태스크다. 이 문서는 정책 구조, 관측, 행동, 보상, 종료, 판정, 학습 방법과 그렇게 정한 이유를 정리한다.
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
                                  드론 회전(TCP 기준) ≤ 10°, 그리퍼 기울기(위 기준) ≤ 15°
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
| 드론 띄우기 | 현재: 몸체별 중력 상쇄 힘 (완전 정지). → hover 제어기로 바꿀 예정 (8장) | |
| 접촉 재질 | PhysX: 손가락·드론 정지 0.6 / 운동 0.5. Newton: 기본값 1.0 유지 | `drive_gains.yaml` |
| 드론 충돌체 | PhysX: 원래 `convexDecomposition`. Newton: 미리 분해한 30조각 | `assets/data/drones/iris/README.md` |

**학습 커리큘럼 (초기 상태):** 학습 리셋의 50%는 드론을 **시작 TCP 바로 위 0~30 cm**(수직선 위, 옆 ±3 mm)에 놓는다.
나머지 50%는 원래 25곳이다. `play`와 평가 스크립트는 커리큘럼 없이 25곳만 쓴다.

<br>

## 3. 관측

비대칭 actor-critic이다. **actor는 실제 로봇에서도 얻는 값만**, critic은 시뮬레이터에만 있는 값까지 본다.
배포할 때는 actor만 쓴다.

### actor (`policy`, 30차원)
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
| `arm_action` (`InterpolatedRelativeJointPositionAction`) | 6 | 행동 [−1, 1] × **0.05 rad**만큼 관절 목표(goal)를 옮기고, 그 스텝의 물리 4번 동안 선형 보간. 관절 한계 − 0.05 rad로 제한 |
| `gripper_action` (`ProfiledBinaryJointPositionAction`) | 1 | 행동 < 0이면 닫기. 목표가 실물 속도 0.516 rad/s로 이동(완전히 닫는 데 2.2 s). **한 번 닫으면 에피소드 끝까지 유지(latch)** |

- 팔: 스텝당 0.05 rad = 최대 1.5 rad/s(기존 IK의 관절 목표 속도 한계). 보간은 기존 레포의 33 ms 보간·실물 UR servoJ와 같다.
  절대 목표가 한 스텝에 0.5 rad씩 바뀌는 방식은 보호 정지(추종 오차 5°)에 계속 걸렸다.
- 데이터셋에 기록할 행동은 `processed_actions` = 절대 관절 목표(기존 LeRobot 스키마와 같음).
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
| `gripper_tilt` | `1 − cos(TCP z와 위쪽 사이 각)` | 항상 | −15 | 잡은 뒤 드론과 같이 기울어 `misalign`이 0이 되는 빈틈 |
| `drone_tilt` | `1 − cos(드론 z와 위쪽 사이 각)` | 항상 | −10 | 잡은 드론을 똑바로 |
| `close_near` | 허리 6 mm 이내·옆 1 cm 이내에서 닫기 명령이면 1 | 잡기 전 | +2 | cabinet `grasp_handle`. 15 mm였을 때 올라가는 중에 닫아 드론을 짜고 기울임 |
| `finger_contact` | 양쪽 손가락 접촉력 > 1 N이면 1 | 모터 정지 전까지 | +2 | core lift 손가락 접촉 보상, 1 N은 기존 판정기 |
| `hold` | 모터 정지 뒤 양쪽 접촉 + 내려앉음 ≤ 15 mm면 1 | 모터 정지 뒤 | +4 | 기존 판정기 기준 |
| `arm_still` | 팔 행동 제곱 합 | 잡기 확인 뒤 | −1.5 | 비행 물체 잡기(Hu et al. 2023)의 "접촉 뒤 손 멈추기". 닫기 명령부터 걸면 기울기를 못 바로잡아 성공률 82→45% |
| `success` | 판정 held | 그 순간 한 번 | **+20** | |
| `failure` | 실패 종료 (놓침·slip, 드론 쳐 냄, 보호 정지) | 그 순간 한 번 | **−5** | core lift `early_termination` |
| `drone_push` | 잡기 전 드론이 리셋 위치에서 5 mm 넘게 밀린 거리 [m] | 잡기 전 | −20 | 근거 없음, 작게 시작 |
| `action_rate` | `‖aₜ − aₜ₋₁‖²` | 항상 | −0.001 | contrib lift |
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
| `drone_knocked` | 실패 | 잡기 전 드론이 리셋 위치에서 10 cm 넘게 밀림 |
| `protective_stop` | 실패 | 팔 관절 추종 오차 > 5° 또는 속도 > 200°/s (기존 레포의 UR 보호 정지). 150 N 접촉력 기준은 아직 없음 |
| `time_out` | 시간 초과 | 12 s |

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
| entropy | 0.003 | 0.006은 std가 커지고, 0.001은 std가 0.05로 무너짐 |

학습·이어서 학습·평가 명령:
```bash
# 처음부터
uv run --extra isaacsim isaaclab train --rl_library rsl_rl --task DroneGraspRl-Grasp-UR5 --num_envs 2048
# 이어서 (Isaac Lab 3.0은 agent.resume을 무시함, --checkpoint를 써야 함)
uv run --extra isaacsim isaaclab train --rl_library rsl_rl --task DroneGraspRl-Grasp-UR5 --num_envs 2048 \
    --checkpoint logs/rsl_rl/drone_grasp_ur5/<run>/model_<n>.pt
# 평가: 25곳 시작, 판정 결과·접근 기하·그리퍼 기울기·slip 원인
uv run --extra isaacsim python scripts/eval_policy.py --checkpoint logs/rsl_rl/drone_grasp_ur5/<run>/model_<n>.pt
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
| 9~10 | 기울기 보상, 허리에서만 닫기, 잡은 뒤 팔 정지 | **93~94%** (기울기 조건 포함), 잡은 뒤 최대 기울기 39° → 13° | |

- 채택 정책 계보: `try5a` → `try6b` → `try7b` → `try8b` → `try8c` → `try9b` → `try10d` (`model_8194.pt`, 약 8200 iteration)
- `logs/`는 git에 올라가지 않는다. 다른 PC로 옮기려면 run 폴더를 복사한다.

### 현재 정책 평가 (try10d, 25곳, 256회)
| seed | 성공 | slip | 드론 쳐 냄 | 시간 초과 | 버티기 끝 기울기 (중앙값) | 잡은 뒤 최대 기울기 (중앙값) |
|---|---|---|---|---|---|---|
| 1 | 237 (93%) | 16 | 3 | 0 | — | 13.7° |
| 2 | 241 (94%) | 14 | 0 | 1 | 8.8° | 13.4° |

남은 slip은 대부분 "마지막 0.5 s 변화 ≤ 0.5 mm" 기준이다. 스크립트 전문가의 그리퍼 기울기는 0.2°로, 정책(약 9°)보다 훨씬 작다.

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

- 드론 hover 제어기와 움직이는 목표 (사인파 프리셋 small/mid/big)
- 도메인 랜덤화 (마찰·질량·게인, 행동 지연, 드론 외란, 모션캡처 가림)
- Newton에서의 파지 학습 (접촉 조정 필요)
- 보호 정지의 150 N 접촉력 기준, 그리퍼 몸통–드론 접촉 벌점(`bad_contact`)
- 카메라, LeRobot 데이터 내보내기, 스튜어트 플랫폼
