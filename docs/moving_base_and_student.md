# 움직이는 베이스와 학생 정책 (설계와 결과)

고정 베이스 선생(`try20b`) 다음 단계: 로봇이 **Stewart 플랫폼** 또는 **Husky** 위에 있을 때 잡기, 그리고 실물에서 얻을 수 있는 입력만 쓰는
**학생 정책(증류)**. 숫자는 코드와 같고, 가정값은 "가정"으로 표시한다(실물 측정 후 교체).

<br>

## 1. Stewart 플랫폼 (폐루프 IK 모델)

기존 레포는 윗판을 정해진 궤적으로 kinematic하게 움직이고 다리는 보이기만 했다. 여기서는 **실물처럼 메커니즘**이다.

| 항목 | 내용 | 코드 |
|---|---|---|
| 치수 | 기존 레포 설계값: 아랫판 반지름 0.8 m, 윗판 0.6 m, 부착 각도 Ingram ik.cpp(MIT), 높이 1.0 m, 다리 0.747~1.395 m | `assets/stewart.py` |
| 구조 | 다리마다 유니버설 조인트(회전 2개) → **구동 프리즘 조인트** → 구면 조인트. 다리 0은 트리 안, 다리 1~5는 루프를 닫는 조인트(PhysX: articulation 밖 구면 조인트, Newton: MuJoCo CONNECT 등식) | `scripts/build_stewart_usd.py` |
| 제어 | 윗판 자세 명령 → **IK로 다리 길이** → 다리 위치 목표 (매 물리 스텝) | `mdp.StewartMotion` |
| 움직임 | 기존 레포의 6축 사인파, 프리셋 static/small/mid/big, 5 s 램프 | `SineMotion` |
| 가정값 | 윗판 20 kg, 다리 강성 2e5 N/m, 힘 한계 3000 N, 속도 0.5 m/s, **다리 등가 질량(armature) 10 kg** | `assets/stewart.py` |

확인 (`scripts/check_stewart.py`, 로봇 대신 20 kg 하중)

| 프리셋 | 위치 오차 RMS (PhysX / Newton) | 회전 오차 | 지연 | 루프 틈 | 다리 힘 최대 |
|---|---|---|---|---|---|
| small | 0.88 / 0.85 mm | 0.02° | 25 ms | < 0.03 mm | 132 N |
| mid | 1.38 / 1.32 mm | 0.05° | 17 ms | < 0.03 mm | 185 N |
| big | 3.22 / 3.06 mm | 0.10° | 17~25 ms | < 0.03 mm | 337 N |

- armature가 없으면 Newton(MuJoCo)에서 가벼운 로드가 진동하며 루프가 22~67 mm 벌어졌다(위치 서보를 명시적으로 적분)

## 2. Husky

기존 레포처럼 정해진 직선 왕복(0.30 m, 0.05 m/s, 가속 0.1 m/s², 양 끝 1 s 정지, 가정값).
- 처음에는 kinematic 물체를 순간이동시켰지만 **joint로 붙인 로봇이 따라오지 않았다**(PhysX) → **구동 레일 위 대차(46 kg)**로 바꿈
  (`scripts/build_husky_rail_usd.py`, `mdp.HuskyMotion`). Husky 모양은 물리 없는 변형(`none`)으로 대차에 얹음
- 로봇 장착: Clearpath 공식 Husky+UR5 위치(윗판 + 0.3058 m, yaw −90°, 가정값). 로봇 base 높이 0.384 m (기존 레포와 같음)

## 3. 공통

| 항목 | 내용 |
|---|---|
| 로봇 고정 | 시뮬레이션 시작 전에 env마다 고정 조인트(윗판/대차 → robot_mount, articulation 밖) + 충돌 필터 (`mdp.attach_robot_to_base`). `replicate_physics=False` 필요 |
| **드론 따라가기** | 로봇이 보낸 **base 자세(x, y, yaw)**를 드론이 지연 0.1 s, 잡음 2 mm로 받아, PX4식 속도 0.5 m/s·가속 1 m/s² 한계 안에서 hover 목표를 옮김. 정책은 드론을 조종하지 않음 (`DroneHoverController.follow_asset`, 가정값) |
| 관측 추가 | **base IMU** 6 (각속도 + 비력, base 좌표, 잡음) → 선생 관측 52 → 58 (`expand_policy_inputs.py`) |
| 커리큘럼 | 움직임 크기 0 → 1을 학습 성공률(EMA)이 0.85 이상일 때만 키움 (`mdp.ramp_base_motion`) |
| 확인 | 스크립트 전문가: Stewart 정지 6/8, 전체 진폭 7/8, Husky 5~6/8 (고정 베이스와 비슷) |

<br>

## 4. 학생 정책 (고정 베이스, 증류)

| 항목 | 내용 |
|---|---|
| 입력 (`student` 그룹) | 팔 관절 위치·속도, 그리퍼 위치, TCP(FK), 직전 행동, 그리퍼 닫기 명령, **ArUco 마커로 본 드론 자세 + 나이** (`docs/marker_study.md`), 최근 3스텝 |
| 빠진 것 | 정확한 드론 자세·속도, TCP 기준 드론 위치, 손가락 접촉력, 잡힘 감지 (시뮬레이터에만 있음) |
| 방법 | RSL-RL Distillation: 학생이 행동하고 선생(`try20b`)이 매 상태에 정답을 줌 (DAgger 방식). Huber 손실, lr 1e-4 (MSE + 5e-4는 3 iteration 만에 발산: 선생의 원시 행동이 최대 12) |
| 태스크 | `DroneGraspRl-Grasp-UR5-Student`, 설정 `agents/rsl_rl_distillation_cfg.py` |

## 5. 결과 (원래 25곳, mid hover, 랜덤화 켬)

### 학생 정책 (고정 베이스)
| 정책 | 입력 | 성공 (seed 1 / 2) | 버티기 끝 기울기 | 잡기 직전 속도 |
|---|---|---|---|---|
| 선생 `try20b` | 특권 정보 52 | 96.1 / 93.4% | 3.2~3.5° | 138 mm/s |
| 학생 증류 `s1_marker` (1500 it) | 마커 + 관절 (3스텝) | 89.1 / 89.5% | 2.9~3.2° | 139~144 mm/s |
| **학생 + RL 미세조정 `s2_rl_finetune`** (1000 it) | 같음 | **93.8 / 93.4%** | 5.0~5.2° | 146~149 mm/s |

- RL 미세조정: 학생 가중치를 actor로, 선생의 critic(특권 입력)을 그대로 (`scripts/student_to_ppo.py`), 행동 잡음 0.05 → 0.1, lr 3e-5
- 실물에서 얻을 수 있는 입력만으로 선생과 비슷한 성공률. 남은 차이는 미끄러짐(slip)

### Husky 선생 `try24a` (20b → 관측 58로 확장, 1500 it, 커리큘럼 1.0 도달)
| 조건 | 성공 | 기울기 |
|---|---|---|
| 전체 움직임, seed 1 | 243/256 (94.9%) | 2.9° |
| 전체 움직임, seed 2 | 241/256 (94.1%) | 3.3° |

- 시범 300개 녹화 (RGB + depth, base_imu 실측): `datasets/lerobot_v21/local/husky24a_vdr_depth_300`, 50,084 프레임, 검수 6/6 PASS

### Stewart 선생 `try23a` (2000 it, 커리큘럼 0.94까지)
| 움직임 프리셋 | 성공 | 드론 밀림 |
|---|---|---|
| small | 116/128 (90.6%) | 0 |
| mid | 104/128 (81.3%) | 12 |
| big | 50/128 (39.1%) | 73 |

- big(x·z ±15 cm, roll·pitch ±6°)에서 팔이 베이스와 같이 흔들리며 드론을 밀어냄. 드론은 x, y만 따라가므로 z 움직임(최대 약 0.14 m/s)은
  팔이 흡수해야 함. 드론은 자기 hover 목표를 30~36 mm 안에서 따라가지만, 따라가기 지연 때문에 베이스 기준으로는 최대 약 10 cm 늦음

### Stewart 선생 `try23b` (23a에서 전체 진폭으로 2000 it 더)
| 움직임 프리셋 | 23a | **23b** |
|---|---|---|
| small | 90.6% | **95.3%** |
| mid | 81.3% | **85.9%** |
| big | 39.1% | **48.4%** (드론 밀림 47%) |

- 시범 300개 녹화 (RGB + depth, base_imu 실측): `datasets/lerobot_v21/local/stewart23b_vdr_depth_300`, 50,754 프레임,
  검수 6/6 PASS. 성공한 것만 저장해서 프리셋 비율은 small 135 / mid 112 / big 53 (녹화 414회 중 301 성공)

### Stewart 선생 `try23c` (23b에서 2000 it 더) — 채택
| 움직임 프리셋 | 23a | 23b | **23c** |
|---|---|---|---|
| small | 90.6% | 95.3% | **95.3%** |
| mid | 81.3% | 85.9% | **90.6%** |
| big | 39.1% | 48.4% | **60.2%** (드론 밀림 34%) |

- 시범 300개 다시 녹화: `datasets/lerobot_v21/local/stewart23c_vdr_depth_300`, 50,386 프레임, 검수 6/6 PASS,
  small 117 / mid 106 / big 77 (녹화 378회 중 300 성공). 23b 데이터셋은 이전 버전
- big을 더 올리려면: 드론이 z도 따라가기(실물 PX4로 가능하지만 지금은 x, y만 따라가기로 정함), 또는 더 긴 학습
