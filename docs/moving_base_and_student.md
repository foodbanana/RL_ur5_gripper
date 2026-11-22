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
| 로봇 고정 | **팔과 베이스를 하나의 articulation으로** (2026-11-23부터, PhysX·Newton 공통): `scripts/build_mounted_robot_usd.py`가 "팔 USD + 베이스 USD + 고정 조인트(articulation 안) + 충돌 필터"로 `assets/data/mounted/ur5_on_{stewart,husky_rail}.usda`를 만듦, 설정은 `assets/mounted.py`. 로봇 기준 좌표계는 articulation root(베이스)가 아니라 팔의 root 몸체 `robot_mount` (`mdp/frames.py`). 이전: 시작 전에 env마다 articulation 밖 고정 조인트(`attach_robot_to_base`, 삭제) |
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

<br>

## 6. 수정: 베이스가 움직이는 중에 잡도록 (2026-11-22)

GUI에서 보니 선생이 **베이스가 움직이기 전에** 잡고 있었다. Stewart는 기존 레포를 따라 처음 5 s 동안 움직임을 키웠는데(램프) 정책은
3~4 s 안에 잡았고, Husky는 매번 정지 상태에서 출발했다. 그래서 위 5장의 Stewart·Husky 결과와 녹화는 "램프 구간" 것이다
(Husky 24a·Stewart 23b 데이터셋은 `~/hdd/drone_grasp_rl/lerobot_v21_ramp_phase/`로 옮김).

| 수정 | 내용 |
|---|---|
| Stewart 램프 5 s → **1 s** | 리셋마다 위상이 무작위라, 1 s 뒤에는 움직임의 무작위 지점에서 전체 크기로 움직임 |
| Husky **무작위 위상에서 출발** | 왕복 주기의 아무 지점에서, 그 지점의 속도까지 1 s 동안 키움 |
| 평가 지표 | `eval_policy.py`가 접근하는 동안 base의 최대 이동·수직 이동·속도·기울기를 출력 |
| 드론 따라가기 | **x, y만** (z 따라가기는 하지 않기로 결정) |
| Stewart 프리셋 | small·mid 위주 + big 조금 (가중치 1 / 1 / 0.25) |

접근 중 base 움직임 (Stewart big, 23c 정책): 램프 5 s → 1 s에서 최대 속도 121 → 306 mm/s, 수직 70 → 149 mm, 기울기 4.4 → 7.3°.
23c는 이 조건에서 big 15%, mid 67%로 떨어짐 → 다시 학습.

### Husky (움직이는 중 출발)
| 정책 | 성공 (seed 1 / 2) | 접근 중 base 이동 (중앙값) |
|---|---|---|
| 선생 `try26a` (24a에서 1000 it) | **97.7 / 96.5%** | 56~58 mm, 50 mm/s |
| 학생 증류 `hs1_husky_marker` | 91.0% (seed 1) | |
| **학생 + RL 미세조정 `hs2_husky_rl`** | **94.1 / 95.7%** | 61~73 mm |

- 시범 300개: `datasets/lerobot_v21/local/husky26a_moving_vdr_depth_300` (50,299 프레임, 검수 6/6 PASS)

### Stewart (움직이는 중 출발, small·mid 위주 + big 조금)
프리셋별 평가 (전체 크기, 256회; 25d는 128회). 접근 중 base 움직임(중앙값): small 78 mm/s·수직 37 mm·기울기 1.8°,
mid 155 mm/s·74 mm·3.8°, big 310 mm/s·149 mm·7.7°.

| 정책 | 바꾼 것 | small | mid | big |
|---|---|---|---|---|
| 23c (이전, 같은 조건) | | - | 67% | 15% |
| 25d | 23c에서 6000 it (커리큘럼 0 → 1) | 93.8% | 80.5% | 30.5% |
| 25e | + 선생에 **base 상태** (속도·이동·기울기 12, 특권) | 96.1% | 90.6% | 46.1% |
| 25f | + 드론 밀기 벌점 −20 → −40 | 96.1% | 94.5% | 49.2% |
| 25h | 25f에서 3000 it 더 | 96.9% | 94.1% | 60.9% |
| **25i** (채택) | 25h에서 3000 it 더 | **99.2%** | **94.1%** | **64.1%** |

- 25d는 학습 성공률 약 81%에서 정체 → base IMU만으로는 base가 어디로 얼마나 빨리 가는지 몰랐음. base 상태를 주자 바로 개선
- 실패의 대부분은 "드론 밀림": 드론은 x, y만 따라가므로 base의 위아래 움직임(big 최대 0.3 m/s)을 팔이 흡수해야 함
- 시범 300개: `datasets/lerobot_v21/local/stewart25i_moving_vdr_depth_300` (선생 25i, 49,428 프레임, 검수 6/6 PASS,
  small 140 / mid 134 / big 26). 이전 25h 데이터셋은 `~/hdd/drone_grasp_rl/lerobot_v21_superseded/`로 옮김

| Stewart 학생 (마커 + base IMU) | small | mid | big |
|---|---|---|---|
| 증류 `ss1_stewart_marker` (1500 it) | 74.2% (세 프리셋 섞음) | | |
| + RL 미세조정 `ss2_stewart_rl` (1500 it) | 93.8% | 87.1% | 34.8% |
| 증류 + **FK 입력** `ss3a_fk_h3` (입력 3스텝) | 88.3% | 78.9% | 38.3% |
| 증류 + FK 입력 `ss3b_fk_h10` (입력 10스텝) | 94.1% | 86.3% | 44.9% |
| ss3b + RL 미세조정 `ss4_fk_h10_rl` (1500 it; 300·600 it도 비슷) | 90.2% | 79.3% | 32.8% |
| **증류(선생 25i) + FK 입력 10스텝 `ss5_fk_h10_t25i`** (채택) | **94.5%** | **87.9%** | **42.6%** |
| 선생 `25i` (비교) | 99.2% | 94.1% | 64.1% |

- 학생은 base 상태(속도·이동)를 못 보고 IMU만 봄 → 선생과의 차이가 mid·big에서 큼. 실패는 대부분 드론 밀림
- **FK 입력**(`stewart_fk_state`): 다리 길이(잡음 0.1 mm) → 순기구학(감쇠 Newton-Raphson, 이전 자세에서 시작)으로
  윗판 자세·속도 12개. 실물 Stewart 제어기가 다리 길이를 알기 때문에 배포 가능. 평가: `--student_history 10`
- FK + 입력 10스텝이면 RL 미세조정 없이 증류만으로 ss2와 비슷하거나 나음(big +8~10%p). 3스텝은 부족
- 이 경우 RL 미세조정은 오히려 나빠짐(학습 성공률 81%에서 정체 후 하락) → 증류 결과를 그대로 채택
- 남은 차이(big 43% vs 64%)는 대부분 드론 밀림. 다음 후보: RNN 학생, DAgger 더 길게, big 비중을 올린 증류

### 팔과 베이스를 하나의 articulation으로 (2026-11-23, PhysX·Newton 공통)
**경과**
- Newton은 Isaac Lab 3.0 cloner가 asset마다 모델을 따로 만들어 합치므로(`isaaclab_newton/cloner/newton_clone_utils.py`)
  asset 사이 조인트(이전 PhysX 방식)가 빠진다. 시도한 것:
  - 떠 있는 root를 매 스텝 마운트 위치로 씀: 한 스텝 동안 로봇이 중력으로 떨어졌다 올라감(0.4 mm, 0.06 m/s) → 그리퍼가 떨려 잡기 판정 안 됨
  - 고정 root(`fix_root_link`)에 자세를 씀(커밋 `075e97d`): **실제로는 로봇이 움직이지 않았다.** Newton은 고정 root의 자세를
    `model.joint_X_p`에만 쓰고 MuJoCo는 그 몸체를 고정물로 둔다. 읽히는 `root_link_pose_w`만 바뀌어서 "0.4 mm 이내로 따라감"으로
    잘못 판단했었다(실제 몸체 `body_link_pose_w`로 재면 마운트 오차 90~180 mm). 정지 베이스 16/16은 맞았고, 움직이는 베이스가 낮았던 원인
- 비교 시험(드론 없는 Stewart + UR5, 같은 동작 5가지, 기준 = 하나의 articulation): 반력 되돌려 주기는 위 문제로 움직이면 실패,
  MuJoCo weld 구속은 마운트 회전관성을 1.0으로 키워야 안정(원래 0.0017이면 yaw 흔들림이 40°까지 커짐) → **하나의 articulation 채택**

**구조** (`scripts/build_mounted_robot_usd.py`, 사용자 결정: 팔 교체는 이 스크립트에 항목을 추가해 새 USD를 만든다)
- default prim이 팔 USD를 참조 → 팔의 prim 경로·센서·`Physics` variant 그대로(`{ENV}/Robot/Geometry/...`)
- `Base`가 베이스 USD를 참조, 베이스 root 몸체가 원점에 오게 배치(`init_state` = 베이스 root 자세)
- 팔 root 몸체(`robot_mount`)를 마운트 자세에 놓고 articulation root를 지움, `MountJoint`(articulation 안 고정 조인트)로 베이스
  마운트 몸체(윗판 / 대차)에 고정, 팔 받침 두 몸체 ↔ 마운트 몸체 충돌 끔
- 참조는 상대 경로(다른 PC에서도 동작). 관절은 이름으로 고르므로 행동·관측·보상은 그대로(다리·레일 관절이 늘어도 영향 없음)

**확인**
- 실제 몸체 위치로 측정: 로봇 base가 베이스와 함께 19~30 cm 움직이고, 마운트 좌표에서 본 위치 변화 0.000 mm (PhysX·Newton, Stewart·Husky)
- 다리 힘 합(정지 Stewart, 팔 정지): 하나의 articulation PhysX 549.4 N, Newton 545.7 N (이전 PhysX 547.8 N, 이전 Newton 282 N = 팔 무게 빠짐)

| 스크립트 전문가 (16회) | PhysX 이전 → 지금 | Newton 이전 → 지금 |
|---|---|---|
| Stewart 정지 | 11 → 12 | 16 → 16 |
| Stewart 움직임 (진폭 1.0) | 11 → 10 | 7 → 7 |
| Husky 정지 | – → 15 | 16 → 16 |
| Husky 움직임 | 11 → 14 | 4 → **16** |

**기존 PhysX 정책을 새 구조에서 평가** (256회씩, mid hover; Stewart는 seed 2 프리셋별, Husky는 seed별)

| 정책 | 이전 구조 | 새 구조 |
|---|---|---|
| Stewart 선생 `25i` small / mid / big | 99.2 / 94.1 / 64.1 % | 98.0 / 93.4 / 62.1 % |
| Stewart 학생 `ss5` small / mid / big | 94.5 / 87.9 / 42.6 % | 94.5 / 87.1 / 38.7 % |
| Husky 선생 `26a` seed 1 / 2 | 97.7 / 96.5 % | 96.1 / 96.5 % |
| Husky 학생 `hs2` seed 1 / 2 / 3 / 4 | 94.1 / 95.7 / – / – % | 94.1 / 90.6 / 88.7 / 94.1 % (평균 91.9) |
| **Husky 학생 `hs3_husky_mounted`** (hs2에서 새 구조로 RL 600 it) | | 90.2 / 94.9 / 93.4 / 94.5 % (평균 93.3) → 채택 |
| 고정 베이스 선생 `20b` seed 2 | 93.4 % | 93.4 % (같음) |

- 관절은 이름으로 고르므로 정책을 그대로 쓸 수 있음. 차이는 대부분 256회 평가의 통계 오차(95% 구간 ±3~6%p) 안
- Husky 학생만 4개 seed 평균 약 3%p 낮아(대부분 slip) 새 구조에서 이어서 학습 → 93.3 %
- Stewart 선생·학생은 그대로 둠(학생 big은 다음 과제 "Stewart 학생 big 개선"에서 다시 학습)

- 같은 하나의 articulation이라도 Stewart가 움직일 때 다리 힘은 PhysX와 Newton이 크게 다름(RMS 500~1000 N): 마운트 방식이 아니라 엔진 차이
- 삭제한 것: `mdp.attach_robot_to_base`, `mdp.SoftMount`, `spawn_mount_joint`, `grasp_env.GraspRLEnv`(Newton 충돌 필터 hook) —
  하나의 articulation에서는 필요 없음. 시험 코드는 커밋하지 않음

<br>

### Stewart big 실패 원인 분석 (2026-11-23, 학생 big 개선 ①)
big, 256회, 같은 seed, 학생 `ss5`(MLP + FK) / 선생 `25i`가 직접 움직이며 매 스텝 기록 (선생은 학생이 겪은 상태마다 정답 행동도 계산)

| | 학생 | 선생 |
|---|---|---|
| 결과 | held 99, **drone_knocked 139**, slip 17 | held 150, **drone_knocked 90**, slip 16 |
| 드론이 밀리기 전 그리퍼가 닫기 시작한 비율 | 96 % | 96 % |
| 닫기 시작 → 드론 밀림(10 cm) | 중앙값 0.72 s | 0.82 s |
| 그 직전 TCP–잡는 점 최소 거리 | 중앙값 1.1 cm | 0.9 cm |

- **실패 대부분은 "잡는 점까지 와서 그리퍼를 닫는 2.2 s 동안"** 일어난다. 학생도 선생도 같은 방식으로 실패(학생이 더 자주)
- 드론은 x, y만 따라가므로 위아래 움직임은 팔이 보상해야 하는데, 닫는 동안 드론–TCP 상하 속도 차이가 최대 0.14~0.15 m/s
- 원인 후보(유력): 행동의 **잡는 점 근처 감속**(`slow_distance` 0.25 m, `min_step_scale` 0.1, 고정 베이스 때 "드론에 돌진 방지"로 넣음).
  2.5 cm 안에서는 관절 한 스텝 최대 0.005 rad(0.15 rad/s) → TCP 약 0.07 m/s 정도까지만 움직일 수 있어 big의 상하 움직임(0.12~0.15 m/s)을
  못 따라간다. 닫는 동안 선생 행동은 80 % 스텝에서 최대값(|a| > 0.95)으로 포화
- 학습 없이 `min_step_scale`만 0.3/0.5로 올리면 오히려 나빠짐(62 → 45 / 9 %): 정책이 0.1에 맞춰져 있어 출력이 3~5배로 커져 과하게 움직임
  → 바꾸려면 다시 학습해야 함
- 학생과 선생의 행동 차이는 TCP 2~5 cm 구간에서 가장 크고, 실패한 에피소드에서 더 큼(0.91 vs 0.78)

<br>

## 7. 앞으로 확인할 것: 실물 Stewart 제어기
- **회전 순서와 각도 단위:** sim의 IK·FK는 Rz·Ry·Rx(yaw·pitch·roll, rad)로, daniel-s-ingram/stewart `ik.cpp`와 같다.
  mlayek21/Stewart-Platform은 Rx·Ry·Rz(도 단위)를 쓴다. 실물 제어기가 쓰는 순서와 단위에 맞춰야 실물에서 같은 동작이 나온다
- **실물 치수:** 부착점 반지름·각도, 다리 행정·속도·힘 (지금은 기존 레포의 sim용 설계값과 가정값)
- 참고 코드: https://github.com/daniel-s-ingram/stewart , https://github.com/mlayek21/Stewart-Platform (둘 다 IK만 있고 FK는 없음)
- IK·FK 구현을 바꿀 때는 먼저 협의한다

## 8. 앞으로 할 것: 보고서용 장면 영상
- 손목 카메라 영상은 녹화 데이터에서 바로 만듦 (`tools/lerobot_export/make_report_videos.py`, `outputs/report_videos/`)
- **밖에서 본 장면 영상**: `scripts/render_scene_video.py`(녹화 태스크에 고정 시점 카메라를 더해 PNG로 저장) →
  `tools/lerobot_export/frames_to_mp4.py`(H.264). `isaaclab play --video`는 MoviePy가 필요해 쓰지 않음.
  `outputs/report_videos/scene_{fixed,husky,stewart_small,stewart_mid,stewart_big}.mp4` (1280×720, 30 fps, 11 s)
