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

### Newton에서 움직이는 베이스 (2026-11-23 원인 확인, 고정 root로 수정)
- 구조: 로봇팔은 베이스와 **별도 asset**으로 둔다(여러 팔로 바꿔 끼우기 위해, 사용자 결정). PhysX는 asset 사이 고정 조인트
  (`mdp.attach_robot_to_base`). Newton은 Isaac Lab 3.0 cloner가 asset마다 builder를 따로 만들어 합치므로
  (`isaaclab_newton/cloner/newton_clone_utils.py`, `add_builder(asset, xform=...)`) asset 사이 조인트가 빠진다.
  Newton 자체에는 `parent_body`(다른 몸체에 FIXED로 붙이기)가 있지만 Isaac Lab은 쓰지 않는다.
  Isaac Lab의 이동 매니퓰레이터 예(`RIDGEBACK_FRANKA_PANDA_CFG`)는 베이스+팔을 articulation 하나로 만든다(팔 교체에 불리해서 채택 안 함)
- 이전 방식: 로봇 root를 떠 있게(FREE 조인트, Newton USD import 기본값) 두고 `mdp.SoftMount`가 매 물리 스텝 자세·속도를 씀
- **원인**: 한 물리 스텝 동안 root가 자유라 로봇 전체가 중력으로 떨어졌다가 다시 올려짐 → 스텝 직후 root가 마운트보다 0.4 mm 낮고
  0.063 m/s(그리퍼 0.087 m/s)로 떨어지는 중. 그리퍼가 드론에 대해 120 Hz로 떨려서 판정(그리퍼 정지 + 양쪽 힘)이 안 됨
- **수정**: Newton에서만 로봇을 `fix_root_link=True`로 spawn(`env_cfg.py` preset). 고정 root에 쓴 자세는 Newton이
  `model.joint_X_p`에 쓰고 MuJoCo가 root를 mocap 몸체처럼 옮긴다 → 떨어짐 없음, 로봇이 마운트를 0.4 mm 이내로 따라감. PhysX는 그대로

| 스크립트 전문가 (16회) | PhysX | Newton 이전 (floating) | **Newton 고정 root** |
|---|---|---|---|
| Husky 정지 | – | 0/16 | **16/16** |
| Stewart 정지 | – | 4/16 | **16/16** |
| Husky 움직임 | 11/16 | 0/16 | 4/16 |
| Stewart 움직임 | 11/16 | 1/16 | 7/16 |

- 남은 차이(움직일 때, 대부분 드론 밀림): mocap 몸체는 접촉 계산에서 속도가 0이라서라고 추정(**미확인**)
- 한계: 한 방향 연결이라 로봇 무게·반력이 베이스에 실리지 않는다
- 진단 주의: 학습용 태스크는 커리큘럼이 진폭 0에서 시작 → 움직이는 베이스를 보려면 `env.curriculum.base_motion.params.initial=1.0`
- 측정(정지 Stewart, 다리 6개 힘 합): PhysX 547.8 N, Newton 282.2 N → 차이 266 N ≈ 로봇 무게(22.6 kg). **PhysX는 팔 무게·반력이
  베이스에 실리고(팔을 흔들면 ±150 N), Newton은 안 실림.** Newton에서 팔을 흔들 때 변하는 힘은 팔-윗판 접촉으로 추정(PhysX의
  FilteredPairs가 Newton에는 없음, 미확인)

#### 앞으로 (Newton 양방향 연결, 팔은 별도 asset 유지가 조건)
| 후보 | 내용 | 비고 |
|---|---|---|
| ① 반력 되돌려 주기 | 지금처럼 고정 root로 옮기고, root 조인트가 받는 힘(joint wrench)을 읽어 반대로 윗판에 가함 | 변경 가장 작음. 1스텝 늦음, 안정성 확인 필요 |
| ② MuJoCo weld 등식 | Isaac Lab cloner의 world builder hook에서 로봇 root와 윗판 사이에 weld equality 추가(Stewart 고리처럼) | 양방향, 약간 soft. hook이 공개 API인지 확인 필요 |
| ③ 시작할 때 자동 합치기 | spawner가 "베이스 USD + 팔 USD + 고정 조인트"를 articulation 하나로 합침 (손으로 만든 통합 USD 아님) | 가장 정확, 작업 큼. 관절 순서가 바뀜 |
| ④ Isaac Lab 업데이트 | cloner가 asset 사이 조인트를 지원하면(Newton에는 `parent_body`가 있음) 지금 구조 그대로 | 대기 |
- 함께: Newton에서 로봇-베이스 충돌 끄기(PhysX FilteredPairs와 같게) — 작은 수정, 먼저 할 것

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
