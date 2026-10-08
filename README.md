# Hangeul Decision Lab

### Budgeted Search and a Research Agenda for a Hangul Block Puzzle

**Implemented:** DFS assistant, synthetic simulator/benchmark, offline CPU CNN imitation pilot · **Application policy:** DFS · **RL:** FUTURE

## Abstract

Hangeul Decision Lab은 메이플스토리 한글날 이벤트의 16행 × 10열 블록 퍼즐을 상태 공간 문제로 모델링하고, 제한된 계산 예산 안에서 현재 조각과 특수 능력의 사용 순서를 추천하는 웹 도구다. 현재 구현은 합법 배치 열거, 공통 게임 전이, 사전식 상태 평가, 완료된 부분 탐색의 재사용을 결합한 deterministic depth-first search(DFS)를 사용한다. 화면 입력은 로컬 캡처·판별·사용자 검토를 거쳐 명시적으로 확정하며 실제 게임 조작은 사용자가 수행한다.

장기 연구 질문은 **같은 의사결정 시간 안에서 탐색과 학습이 장기 생존 및 줄 삭제를 얼마나 개선할 수 있는가**이다. 현재 seeded simulator/DFS 비교와 오프라인 CPU CNN 교사 모방을 구현했다. 작은 합성 시험에서 CNN의 선택 일치율은 53.3%로 첫 합법 행동 기준선 64.4%보다 낮았다. 학습 경로는 검증했으나 앱 정책 채택이나 생존 향상을 뒷받침하지 않는다. 이후 탐색 개선·학습 정책/가치 모델을 비교하며 강화학습은 향후 연구다.

**Keywords:** combinatorial planning, bounded DFS, legal-action masking, stochastic planning, reinforcement learning

## 1. Motivation and Research Questions

줄 삭제를 많이 만드는 선택이 다음 조각을 놓기 좋은 상태를 보장하지는 않는다. 조각 순서·회전·반전·아이템·능력과 미래 입력의 불확실성을 함께 고려해야 한다.

현재 구현의 기여는 UI와 분리된 전이/솔버, 예산과 후보 제한을 드러내는 탐색, 추천의 단계별 재생, 인식 결과의 명시 검토 및 공유 세션의 턴 연결이다. 아래 질문들은 **향후 비교 연구**다.

- **RQ1 — Budget:** 예산 확대로 놓친 유효 경로를 찾을 수 있는가?
- **RQ2 — Evaluation:** 후보 순서·가지 제거·평가 편향 중 무엇이 선택을 제한하는가?
- **RQ3 — Uncertainty:** 미래 조각/재뽑기 샘플링이 같은 시간의 결정적 탐색보다 유리한가?
- **RQ4 — Learning:** 앞선 개선 후에도 남는 한계를 학습한 policy/value가 추가 비용을 감수할 만큼 해결하는가?

## 2. Problem Formulation

$$s=(B,P,I,A,r)$$

| 기호 | 현재 의사결정 상태                          |
| ---- | ------------------------------------------- |
| $B$  | 16 × 10 Boolean 점유 보드                   |
| $P$  | 남은 최대 3조각의 instance·화면 slot·형상   |
| $I$  | 미획득 아이템의 좌표/종류, 보드 위 최대 3개 |
| $A$  | 점 찍기/바꿔 뽑기 횟수, 보유 합계 최대 7개  |
| $r$  | 재뽑기 결과를 기다리는 대상 또는 null       |

action은 조각 배치, 빈칸 한 곳에 점 찍기, 남은 조각 재뽑기다. 조각 사용 순서는 자유롭고 slot은 화면 위치 식별자다. 회전/반전의 중복 variant를 제거하고 실제 true 셀만 경계/충돌 검사한다. 가로줄은 동시에 삭제되며 중력은 없다. 능력을 소모한 뒤 삭제 행의 아이템을 행·열 순으로 획득하고, 상한 때문에 미획득한 아이콘은 같은 좌표에 남긴다.

19종 형상/한글 이름은 제공된 자료를 기반으로 한다. ㄹ/ㅌ은 각각 8칸, 점·ㅡ·ㅣ는 각각 1·3·5칸이다. 전체 형상/규칙의 독립 실전 대조, 새 아이콘 생성, 단계별 출현 확률은 미완료다. 이벤트 배경은 [공식 안내][1]를 참고한다.

![Synthetic line-clear example](assets/figures/line-clear.svg)

_Figure 1. 합성 규칙 예제. 마지막 행의 빈 3칸에 ㅡ를 배치하면 그 행만 비워지고 위의 셀은 제자리에 남는다. 게임 캡처·솔버 추천·성능 실험 결과가 아니다._

## 3. Implemented Method

### 3.1 Bounded DFS

기본값은 `maxNodes=512`, `maxAlternatives=3`(최선 포함), `useMemoization=true`다. 일반 probe는 전체 예산의 최대 1/4을 사용하고 실제 방문량을 능력 DFS 예산에서 차감한다. 루트를 포함한 합산 방문 수를 제한한다. 일반 조각 소진 깊이는 최대 남은 3개지만 능력 사용/재획득을 포함한 action 깊이는 고정 3이 아니다.

```text
probe := ordinary-only DFS using part of the node budget
seed candidates with suitable terminal paths from the probe
DFS(state):
    stop and evaluate when awaiting next pieces or reroll input
    reuse only completed subtrees with the same search key
    enumerate configured ordinary and ability candidates
    apply actions through the shared domain transition
    stop at the node limit and mark the search incomplete
    rank evaluated candidates; retain up to three results
```

search key는 점유·남은 instance/형상·아이템·능력·pending·누적 삭제/획득을 포함한다. 예산 때문에 중단된 subtree는 memo하지 않는다. 동등 경로를 재사용하므로 대안과 경로 count가 모든 고유 경로의 열거는 아니다.

### 3.2 Evaluation and Ability Candidates

현재 gameover 회피 → 현재 조각 소진 → catalog 배치 가능 종류/배치 수 → 삭제 행 → 아이템 획득 → 능력 잔여 → 적은 고립/coverage gap → 많은 빈칸 순으로 **사전식 비교**한다. 학습한 weighted reward가 아니다.

점 찍기는 모든 합법 빈칸을 후보로 두고 즉시 행 삭제부터 탐색한다. 재뽑기는 막힌 조각, 초기 probe에서 소진 경로 미발견, 또는 catalog 배치 가능 종류 0인 경우에 제한적으로 추가한다. 미완료 probe의 미발견은 불가능 증명이 아니다. 전체 domain 합법 mask와 solver의 제한된 후보를 구별한다.

mobility는 catalog 배치 종류/개수이며 미래 생존 또는 출현 확률이 아니다. coverage gap도 셀 덮임 지표다. 점 조각이 있으면 gap이 0이어도 큰 조각 소진을 보장하지 않는다. 재뽑기 결과는 실제 다른 조각을 관측해 입력받고 현재 탐색은 pending에서 멈춘다.

![Implemented decision pipeline](assets/figures/decision-pipeline.svg)

_Figure 2. 구현된 입력–검토–탐색–재생 경계. 판별기는 초안을 만들고 사용자가 확인한 뒤 GameState에 반영한다. Apply Step은 앱 상태 재생이며 실제 게임 배치는 사용자 조작이다._

### 3.3 Observation and Review

선택한 화면 공유의 한 프레임에서 보드 점유·보유 조각 도형·능력 숫자를 로컬 판별한다. 미확정 칸은 캡처 위에서 바로 수정하고 아이템 위치/종류는 점유와 별도로 검토한다. 초안을 자동 승인하지 않는다. 같은 공유에서는 영역/직전 확인 상태를 이어 쓰고 마지막 정상 Apply 뒤 제한된 자동 재읽기로 새 조각을 기다린다. 차이와 새 아이콘은 명시 확인한다.

캡처/임시 픽셀은 제한된 탭 메모리 수명 뒤 폐기한다. 저장·업로드·계정 연동·게임 자동 조작 기능은 없다.

## 4. Validation Scope and Limitations

검사는 합성 보드, 독립 좌표 oracle, 전이 replay, 예산/memo, 입력 수명 및 브라우저 여정을 다룬다. 작은 fixture의 완전 비교는 해당 범위의 correctness 근거이고 전체 게임 oracle가 아니다. `optimalWithinScope`는 설정 후보/평가 범위에 한정된다.

실제 연속 턴·전체 형상/숫자/배율의 인식 정확도·신규 아이콘/확률·사람 사용감/별도 제품 수락은 미완료다. 짧은 합성 episode의 관측은 다음 절에 제시한다. 실제 게임 장기 생존·알고리즘 승률·학습 curve의 증거로 확대하지 않는다.

## 5. Synthetic Benchmark and Future Research

### 5.1 Benchmark First

실제 게임 플레이 없이 기존 domain 전이를 재사용하는 headless simulator를 실행할 수 있다. 합성 환경은 19종 조각에 동등 가중치를 주고 재뽑기에서 현재 종류를 제외하며, 초기 아이템만 사용한다. 실제 출현 확률·새 아이콘 생성·7수 카운터는 모델링하지 않는다. 실제 게임 일치 검증은 별도 과제로 유지한다. 오프라인 모방 학습은 아래 5.4의 이 환경만 사용한다.

초기 상태, catalog/rules version, 다음 조각/재뽑기/아이콘 과정, seed/PRNG, 종료/horizon을 고정한다. 환경/정책/rollout/학습 난수를 분리하고 action별 난수 소비 차이를 통제한다. 미래 tape를 정책에 노출하지 않는다. 다음 입력 대기·실제 gameover·시간/episode truncation을 구별하고 임의 균등분포를 실제 출현 확률로 가정하지 않는다.

| Metric                      | 비교 정의                                                    |
| --------------------------- | ------------------------------------------------------------ |
| Survival / terminal step    | 소진 세트와 일반/능력 action 수, gameover/truncation 분리    |
| Clears / acquired items     | 누적 삭제 행/실제 획득, 잔류 아이템 제외                     |
| Ability usage               | 종류별 소모·재획득·잔여/구제 결과                            |
| Decision latency            | 합법 후보/평가/탐색/추론의 mean/p95, load/warmup 분리        |
| Search cost / memory        | 알고리즘별 nodes/simulations/memo, 정의한 heap/RSS/피크 한계 |
| Reference / reproducibility | 유효 경로 도달률, action/state hash, 오류/timeout/분산       |

같은 초기 상태·paired seeds·종료/시간 상한으로 비교한다. 알고리즘 간 node count를 동일 비용으로 취급하지 않는다. train/dev/test는 seed·episode 단위로 분리하며 계열 밖 일반화는 별도 실험이 필요하다. 아래 작은 CNN pilot은 sparse/pressure 계열을 모든 split에서 공유한다. test benchmark로 학습/tuning하지 않고 생존/삭제·지연/메모리·안정성을 구분한다.

**실행된 pilot — 2026-10-07.** seed 17·42·2026 × sparse/pressure 보드의 6개 dev fixture를 예산 128/512, memo 사용, 대안 3개, 최대 12행동으로 각각 2번 실행했다. 총 24episode이며 반복을 독립 표본으로 세지 않는다. 실행 순서는 회전하고 별도 seed 0의 warmup은 제외했다. 모든 episode 재생과 반복 trace 일치가 통과했다.

| 관측                           |   DFS 128 |    DFS 512 |
| ------------------------------ | --------: | ---------: |
| 평균 수행 행동 / 최대 12       |      7.67 |       8.00 |
| 평균 삭제 행                   |      0.83 |       1.00 |
| 평균 판단 시간                 |  395.79ms | 1,400.62ms |
| 판단 시간 p95                  |  596.17ms | 2,254.72ms |
| 불완전 탐색 / 판단 횟수        |   86 / 94 |    72 / 96 |
| gameover / horizon / 정책 포기 | 4 / 6 / 2 |  6 / 6 / 0 |

측정 환경은 Windows 10, i5-9400F, Node 24.15.0, 단일 worker다. 판단 시간은 `solveTurn` 호출부터 반환까지이며 simulator 합법 action 생성·환경 공급·replay·모듈 로드는 제외한다. p95는 유효 episode의 판단 표본에서 nearest-rank로 계산한다. 보드마다 행동 경로가 달라 두 설정의 판단 표본 수는 다르다. 이 비교는 **동일 시간 예산 실험이 아니다**. 전후 Node heap은 약 43.87→108.68MiB, RSS는 226.77→358.22MiB였으며 compiler·누적 기록을 포함한 전체 프로세스 관측이다. 순간 피크나 브라우저 메모리 기준이 아니다.

6개 paired fixture 중 pressure-42에서만 DFS 512가 2행동 더 진행하고 1행 더 지웠다. 그 외 합산 행동/삭제는 같았다. sparse 3개는 모두 길이 제한에 도달했고 아이템 획득은 전부 0이었다. 따라서 장기 생존·아이템 전략·학습 효과를 판정할 표본은 부족하다. 정책 포기는 합법 action이 남아도 정책이 action을 반환하지 않은 경우이며 gameover와 구분한다. 관측·protocol·source/trace hash의 기계 판독 기록은 [pilot summary](assets/research/pilot-budget-20261007-v1.json)에 있다. CLI는 전체 fixture/tape/action/state 기록을 로컬 JSON으로 생성한다.

### 5.2 Search Before Learning

![Future research progression](assets/figures/research-roadmap.svg)

그림의 policy/value hybrid·RL은 조건부 후속이다. 별도로 실행한 CPU CNN 모방 feasibility는 5.4에 기록하며, 이 그림은 측정 chart가 아니다.

_Figure 3. DFS와 합성 benchmark 기반은 구현되어 있다. 점선 방법은 향후 비교안이며, 반복 가능한 한계와 비용 대비 개선 근거에 따라 선택한다. 도식 자체는 측정 결과가 아니다._

1. **Baseline / budget:** revision/config·probe·memo·평가를 고정하고 예산 곡선을 만든다.
2. **Pruning / evaluation:** 작은 완전 reference로 잘못된 제거를 검출하고 후보 순서·평가를 ablation한다.
3. **Beam Search:** width/depth·중복 제거·미래 sampling·cutoff 평가를 같은 시간의 DFS와 비교한다.
4. **MCTS:** 검증된 확률 모델에서 selection → expansion → rollout → evaluation → backpropagation을 검토한다. rollout 정책/깊이/수·chance 전이·재사용·변동성을 기록한다. UCT[2]는 개념 참고이며 본 게임 효용의 증거가 아니다.
5. **Conditional ML/RL:** 앞선 개선 후 유의미한 잔여 한계가 있고 학습/추론 예산과 별도 승인이 있을 때만 검토한다.

### 5.3 Conditional Policy / Value Learning

점유/아이템 plane과 조각/능력 입력의 소형 CNN candidate scorer를 오프라인 모방 pilot에 구현했다. 앱에 채택된 학습 정책·가치 모델은 없으며 RNN/GNN·policy/value hybrid는 측정 가능한 이점이 있을 때 검토한다.

DQN[3], Actor-Critic/PPO[4], policy-guided search, learned value + MCTS는 후보 방법이다. 합법성은 기존 domain mask와 최종 전이 검증이 맡는다. 학습은 승인된 simulator episode 생성이며 상대방과 대전하는 무조건적 self-play로 설명하지 않는다.

reward 후보는 생존/삭제/획득과 mobility·능력 잔여·고립/불필요한 소모의 shaping이다. 가중치/terminal penalty는 미정이며 proxy 편향을 ablation한다. 모델은 action ordering/leaf value를 보완하는 hybrid로도 비교한다. 개선이 작거나 지연/메모리/복잡도/불안정성이 크면 deterministic search를 유지한다.

기록은 문제 → 가설 → fixture/seed/config → 관찰 → 해석 → 결정으로 연결하고 실패/기각도 보존한다. chart는 CSV/JSON 원본·생성 방법을 필요로 하며 개념 그림을 측정 결과처럼 사용하지 않는다.

### 5.4 Technology Briefing

| 역할                      | 현재 채택                                           | 다음 제안                                            |
| ------------------------- | --------------------------------------------------- | ---------------------------------------------------- |
| 게임 규칙·탐색·시뮬레이션 | TypeScript / Node.js, pure domain, 예산 제한 DFS    | 현재 전이를 학습 자료 생성·합법 행동 검증에도 재사용 |
| 화면 입력·검토            | React / Next.js, 브라우저 Canvas의 로컬 판별        | 이번 CNN 실험에서 변경 없음                          |
| 신경망 학습               | 연구 전용 Python3.13.13 / PyTorch2.14.1+cpu         | 소형 CNN 오프라인 모방 pilot 완료                    |
| 학습 목표                 | 합성 상태에서 DFS512 첫 행동의 supervised imitation | 장기 생존·RL은 별도 후속 연구                        |

PyTorch의 [공식 학습 경로](https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)를 참고해 CPU 실행을 연구 전용으로 채택했다. 보드3plane의 3×3 convolution8/8채널과 4×5 pooling, 조각/능력 metadata140값, 합법 candidate52값을 32차원에서 함께 점수화한다. 총12,189parameter이며 전체 domain 합법 후보에만 softmax를 적용한다. 미래 조각/seed/instance명/교사 결과는 forward 입력에서 제외한다. bounded DFS512 교사 선택은 최적 정답이 아니며 search scope/미완료 진단을 보존한다.

**실행된 CNN pilot — 2026-10-08.** 고정 train1000–1015/dev2000–2003/test3000–3003 × sparse/pressure, horizon6의48episode에서168/45/45선택을 얻었다. 교사202/258선택은 탐색 미완료, skip/error0이며 모든 episode/encoding/key를 domain에 대조했다. Adam(lr0.001, weight decay0.0001), batch8,12epoch, seed20261008, CPU2thread로 학습했다. dev top1이 가장 높은 첫 checkpoint(epoch11,40.0%)를 선택한 뒤 test를 평가했다. 같은 train/dev 재실행은 전체 loss/선택과 가중치가 bit-equal이었고 재현 검사에서 test 점수는 사용하지 않았다.

| 시험 관측                             | 결과                            |
| ------------------------------------- | ------------------------------- |
| CNN top1 / 첫 합법 행동 기준선        | 24/45(53.3%) / 29/45(64.4%)     |
| fixture macro top1                    | 54.2%, 독립 test fixture8개     |
| 교사 complete / incomplete의 CNN 일치 | 8/13(61.5%) / 16/32(50.0%)      |
| 최종 domain 적용                      | 45/45합법, NaN0·반복 예측 일치  |
| CPU 추론 mean / p95                   | 0.56 / 0.92ms,45선택×고정3반복  |
| tensor 생성→선택 key mean / p95       | 1.99 / 4.68ms,45선택            |
| 학습+dev / 연구 runner 처리 시간      | 3.67 / 7.28초                   |
| process working set 전후              | 183.9 / 349.7MiB, 순간피크 제외 |

Windows10/i5-9400F 개발 PC 관측이다. 추론은 load/warmup을 제외한 finite 검사/argmax 포함이며 domain 후보 생성·JSON·프로세스 시작·IPC·UI 시간은 포함하지 않는다. 처리 시간도 Python/PyTorch import를 제외한다. 45개 연속 상태를 독립45게임으로 세지 않으며 계열 밖 일반화·통계 우월성·실전 생존/RL은 미검증이다. **단순 기준선보다 낮아 현재 모델의 제품 적용을 추천하지 않는다.** 동일한 가치의 다른 합법 선택도 top1에서는 오답이 될 수 있어 생존 성과를 뜻하지 않는다. 비민감 수치·protocol/source/data hash는 [CNN pilot summary](assets/research/pilot-cnn-imitation-20261008-v1.json)에 있다. raw data/checkpoint는 Git 밖 로컬에 보관하며 설치 다운로드 외 자료 전송은 없다.

### 5.5 Teacher Order and Coverage Audit

**진단 — 2026-10-08.** 재학습에 앞서 기존 자료의 분포와 교사 선택을 검사했다. label이나 모델 성적을 보지 않고 train1000–1003/dev2000–2003 × sparse/pressure의 최초 상태16개를 고정했다. 각 상태에 원본, 조각 배열 뒤집기 대조, slot 번호 순환(0→1→2→0)을 적용해 DFS512를 총48회 실행했다. slot 변경은 보드·조각 instance/형상·능력·아이템과 전체 물리적 합법 행동 집합을 보존한다. 게임에서 조각 선택 순서는 자유롭다.

| 진단 관측                        | 결과                           |
| -------------------------------- | ------------------------------ |
| 배열만 뒤집은 대조               | 16/16 경로·평가·탐색 정보 동일 |
| slot 순환 뒤 물리적 첫 행동 변경 | 14/16 상태                     |
| 선택 경로의 사전식 평가          | 향상7 / 저하7 / 동일2          |
| 변경된 첫 행동 중 경로 평가 동점 | 0/14                           |
| 원본 탐색 미완료                 | 16/16                          |
| 반환 경로의 domain 재생          | 144/144 통과                   |
| 모델 입력의 정확 중복            | split 내부·split 간 모두0      |

교사는 후보를 slot 순서로 방문하고 제한된 탐색에서 찾은 경로를 반환한다. 위 관측은 **이 예산과 표본에서의 순서 민감성**을 보여준다. 다른 첫 행동을 같은 정답으로 합쳐도 된다는 증거나 CNN 성적 저하의 단독 원인은 아니다. 반환 대안3에서는 서로 다른 첫 행동의 최상 동점이 관측되지 않았지만, memo와 잘린 탐색 때문에 전체 동점 후보의 부재를 증명하지 않는다. 경로 평가 동점도 즉시 행동 가치·미래 생존의 동일성을 뜻하지 않는다.

학습 자료에서 first-legal label은 전체84/168(50.0%), 탐색 완료5/37(13.5%), 미완료79/131(60.3%)다. 배치 label의124/131(94.7%)는 남은 조각 중 가장 작은 slot이고, sparse는89/89다. 아이템 획득은 train/dev/test에서2/1/0회에 불과하다. seed 분리와 정확한 관측 중복0은 계열 밖 일반화나 충분한 전략 표본을 보장하지 않는다. 기존 test 통계는 설명용으로만 집계했으며 새 test 탐색·튜닝·점수 측정을 하지 않았다. 현재 CNN 결과와 앱 DFS는 유지한다. [진단 요약과 고정 protocol](assets/research/teacher-order-audit-20261008-v1.json)에 분모·계열별 수치·hash를 보관한다.

### 5.6 Complete Reference and Experiment Preflight

순서 진단 뒤 다음 비교 실험의 [제안 protocol](research/experiments/teacher-quality-v1.json)과 실행 전 검사 도구를 추가했다. train4000–4003/dev5000–5001, reserved test6000–6003을 분리하고 노드예산128/512·slot 대조·반복2를 설계했다. **이 preflight 당시 본 실험은 PROPOSED / NOT RUN**이었다. 제안 원본은 보존하며 이후 train/dev 진단은 5.7에 별도 기록한다. reserved test를 생성하거나 새 모델을 학습하지 않았다. 동일 노드 수는 동일 실행 시간이 아니므로 현재 동기 DFS에 실제 deadline이 생기기 전까지 시간은 관측값으로만 비교하도록 명시했다.

실행한 검사는 7개의 의도적으로 만든 합성 상태를 사용한다. 연구 reference는 최대2조각의 전체 일반 배치 경로를 memo/pruning 없이 열거한다. domain 전이와 평가를 공유하므로 게임 규칙의 독립 oracle가 아니다. 노드 상한1024에서 미완료이면 최적 첫 행동 집합을 반환하지 않는다. 능력 사용은 별도로 지정한 전이를 검사하며 일반 reference는 능력을 탐색하지 않는다.

| 실행 전 검사 관측 — 2026-10-08                      | 결과                         |
| --------------------------------------------------- | ---------------------------- |
| 완전 일반 reference                                 | 7/7, 총903노드               |
| production ordinary DFS의 memo off/on 비교          | 14/14 최상 평가·첫 행동 일치 |
| 반환 경로의 domain 재생                             | 34/34                        |
| 직접 지정한 획득·상한 잔류·점 소모/획득·reroll 대기 | 4/4                          |
| 가장 큰 최상 첫 행동 집합                           | 22개                         |
| 새 reserved test 생성 / 학습 실행                   | 0 / 0                        |

완전한 작은 트리에서는 대안3보다 많은 최상 첫 행동을 확인할 수 있었다. 이는 **해당 ordinary-only 현재 턴의 경로 평가**에 한정되며 3조각/능력 전체 탐색·미래 생존의 동등 label을 증명하지 않는다. deliberate coverage 표본은 규칙을 통과하는 예이며 실제 빈도·모델 효용의 증거가 아니다. 기존 CNN top1과 앱의 DFS는 유지한다. 비민감 결과·hash는 [preflight summary](assets/research/teacher-quality-preflight-20261008-v1.json)에 있다.

### 5.7 Train/Dev Order and Budget Diagnostics

위 제안의 새 train/dev seed × sparse/pressure **12개 초기 상태**를 기존 `solveTurn`으로 진단했다. 노드128/512 × 순서3가지 × 반복2의 **144호출**, 반환경로 **432개**를 domain으로 재생했다. 현재3조각과 능력만 전달하며 future tape를 전달하지 않는다. memo=true/대안3은 유지하고, 상태 인덱스와 반복에 따라 6조건의 실행 순서를 회전했다. 반복72조건이 동일했고 단순 배열 뒤집기 대조24쌍도 동일했다. 반복은 독립 표본으로 세지 않는다.

| 현재 턴 관측, 독립 초기 상태12개               | DFS128    | DFS512    |
| ---------------------------------------------- | --------- | --------- |
| slot 순환 후 첫 추천 변화                      | 10/12     | 10/12     |
| slot 순환 후 경로 평가 향상 / 저하 / 동일      | 7 / 3 / 2 | 5 / 6 / 1 |
| 기본 순서 탐색 미완료                          | 12/12     | 12/12     |
| 기본 순서 선택 경로 줄 삭제 합계               | 4         | 5         |
| 기본 순서 점 능력 사용 경로 / reroll 대기 경로 | 4 / 2     | 3 / 2     |
| 기본 순서 아이템 획득 / 상한 잔류              | 0 / 0     | 0 / 0     |

기본 순서에서128→512로 늘리면 공유 사전식 경로 평가가 **8/12 향상·4/12 동일**했고 첫 추천은2/12에서 바뀌었다. slot 순환 조건에서는5/12 향상·7/12 동일이었다. 모든72개 독립 상태·조건의 탐색이 미완료이고 획득/잔류 표본은0이었다. 따라서 더 큰 예산이 순서 민감성을 해소하거나 아이템을 충분히 대표한다고 볼 수 없다. 같은 경로 평가의 첫 행동 변경1건도 장기 생존의 동등 정답이 아니다.

시간은 워밍업 없이 cold/JIT·개발 PC의 동시 테스트/정적 검사 영향을 포함한 관측값이다. 작은 집단의 p95를 속도 수락이나 동일 시간 비교로 사용하지 않는다. 보드12개는 현재 합성 profile의 초기 상태이며 장기 episode 분포나 실제 게임 확률이 아니다. 새 test 평가·추가 학습·앱 정책 변경은 실행하지 않았다. 집단별 coverage/시간·고정 실행 설정·hash는 [matrix summary](assets/research/teacher-matrix-20261008-v1.json)에 있다.

### 5.8 Scripted Item and Capacity Coverage

초기 상태 진단에서 획득/상한 잔류가0이었던 한계를 점검하기 위해 **의도적으로 만든8개 상태**를 별도로 실행했다. 일반 획득, 상한7의 미획득 잔류, 점 소모 후 획득, reroll 대기와 함께 용량6의 획득순서, 상한에서 점소모 후 획득/초과잔류,5행 동시삭제의 row/col 획득, 상한에서 reroll소모 후 관측 대기를 검사한다. 이들은 실제 이벤트 빈도나 학습자료 분포를 나타내지 않는다.

| 검사                                               | 결과                     |
| -------------------------------------------------- | ------------------------ |
| 명시 정답 전이 / ordinary-only 완전 reference 대조 | 8/8 / 8/8                |
| 기존 full solver 호출 / 동일 반복조건              | 32 / 16/16               |
| 일반·능력 반환경로 domain 재생                     | 106/106                  |
| 선택 경로 획득 / 상한 잔류 발생, 각 예산8상태      | 5/8 / 2/8 (128·512 모두) |
| 선택 경로 점사용 / reroll 대기, 각 예산8상태       | 6/8 / 1/8 (128·512 모두) |
| full 탐색 미완료,128 / 512                         | 6/8 / 5/8                |

강제 전이의 기대값과 솔버 선택 결과를 분리했다. **규칙을 통과하는 상황을 만들었다는 사실은 솔버가 같은 첫 행동을 선택한다는 뜻이 아니다.** scripted 첫 행동과 추천이 일치한 상태는 각예산2/8이며 목표는 그 일치율을 최대화하는 것이 아니다. 상한에서 점을 먼저 쓰면 용량을 확보해 아이콘 하나를 획득하고 초과 아이콘은 남는다. 역순으로 넣은 아이콘도 여러행 동시삭제에서 row/col 순으로 획득했다. tiny reference의 완전성은 ordinary-only 공유평가 범위에 한정되고 능력 전체/미래 최적성은 보장하지 않는다. [coverage summary](assets/research/item-coverage-20261008-v1.json)에 상황별 강제/선택 결과와 hash를 보관한다. 새 학습/test평가·앱 의미변경은 없다.

## 6. Reproduction and Usage

환경: **Node.js 24.x**, **pnpm 11.16.0**. package/lock에 고정된 TypeScript·Next.js·React·Vitest·Playwright를 사용한다.

```sh
git clone https://github.com/jhpark-jarvis/hangeul-decision-lab.git
cd hangeul-decision-lab
pnpm install --frozen-lockfile
pnpm dev
```

`http://127.0.0.1:45000`에서 수동 입력 후 **Analyze → Apply Step**을 사용한다. 실제 게임 배치는 직접 수행한다. 화면 입력은 **화면 캡처·검토 열기 → Start Screen Capture → 창의 게임 화면 공유 → Capture Frame → 미확정/아이템 수정 → 전체 확인하고 분석**으로 진행한다. 같은 공유에서 조각을 소진하면 자동 다음 턴 읽기 뒤 다시 확인한다. 영역이 어긋나면 **영역 설정**에서 보드/조각/능력을 각각 지정한다. Stop/공유 교체·새로고침은 관련 상태를 초기화한다.

```sh
pnpm test
pnpm lint
pnpm build
pnpm start                  # production: 127.0.0.1:45001
pnpm test:e2e               # 별도 터미널, 설치된 Chrome/Edge 필요
pnpm research:benchmark     # 게임 실행 없이 고정 합성 pilot; 결과는 로컬 .research-output/runs/
node scripts/figures/render.mjs
```

benchmark는 기존 TypeScript compiler로 순수 모듈을 `.research-output/runtime`에 생성하고, 새 결과 폴더에 `protocol.json`, 전체 `report.json`, `summary.csv`를 남긴다. Node/기존 개발 dependency 외에 Python·GPU·새 서비스는 필요하지 않다. 다른 dev 설정은 `pnpm research:benchmark --protocol <json파일> --output <새폴더>`로 지정하며 기존 폴더를 덮어쓰지 않는다. 최대 2,000행동의 실험 행렬과 최대 8,192노드/판단을 허용하며 강제 wall-time timeout은 없다. 저장되는 것은 합성 상태이며 앱의 화면 캡처 저장 기능은 추가하지 않는다.

별도 CPU CNN 연구의 Windows 재현 절차다. `research/cnn/protocol.json`으로 data와 학습을 함께 고정하며 각 실행은 **새 출력 폴더**를 사용한다. NumPy/torchvision/GPU는 필요하지 않고 앱 실행에는 Python을 사용하지 않는다.

```powershell
python -m venv .research-output/venv-cnn
.research-output/venv-cnn/Scripts/python.exe -m pip install --only-binary=:all: -r research/cnn/requirements-lock.txt
pnpm research:dataset --output .research-output/datasets/my-pilot
.research-output/venv-cnn/Scripts/python.exe research/cnn/train.py --dataset .research-output/datasets/my-pilot --output .research-output/models/my-pilot
node scripts/research/verify-imitation.mjs .research-output/datasets/my-pilot .research-output/models/my-pilot/predictions.json .research-output/models/my-pilot/domain-validation.json
.research-output/venv-cnn/Scripts/python.exe research/cnn/reproduce.py --dataset .research-output/datasets/my-pilot --reference .research-output/models/my-pilot --output .research-output/models/my-pilot-repeat
.research-output/venv-cnn/Scripts/python.exe -m unittest discover -s research/cnn -p test_cnn.py
```

자료/모델/schema·hash가 맞지 않거나 NaN이 나오면 중단한다. 실패한 출력도 보존하고 같은 경로를 덮어쓰지 않는다. checkpoint는 자체 생성 state dict만 `weights_only=True`로 읽는다. `summarize-cnn.mjs <model폴더> <repeat폴더> <새JSON>`으로 domain/repro 판정이 모두 통과한 비민감 요약을 재생성할 수 있다.

그림 생성 명령은 SVG 세 개와 합성 입력을 재생성하며 게임 이미지·학습/benchmark를 사용하지 않는다. 입력/metadata는 [figure-data.json](assets/figures/figure-data.json)에 있다. `src/domain`은 규칙/지표/탐색, `src/research`는 simulator/비교, `src/features`는 캡처/인식/검토/세션, `src/components`/`src/app`은 UI, `tests/fixtures`는 비민감 재현 데이터다.

기존 자료만 읽는 순서 진단은 아래 명령으로 실행한다. 기본 `audit-protocol.json`은 위 pilot의 train/dev 초기16상태만 지정하며 test 탐색이나 학습을 수행하지 않는다. 원본 자료의 hash와 전체 encoding/replay를 확인한 뒤 새 폴더에 `report.json`, 완료 상태별 `probes.jsonl`, 공개용 `summary.json`을 남긴다. 중단된 raw도 보존하며 같은 출력 폴더를 다시 사용하지 않는다.

```powershell
pnpm research:teacher-audit --dataset .research-output/datasets/my-pilot --output .research-output/audits/my-audit
```

다음 실험의 실행 전 검사는 아래 명령으로 실행한다. 제안한 seed 행렬과 reserved test를 사용하지 않고 7개 scripted case만 검사한다. 전체 상태/경로·환경은 새 로컬 폴더의 `report.json`, 비민감 집계는 `summary.json`에 보관하며 기존 폴더는 덮어쓰지 않는다.

```powershell
pnpm research:teacher-quality --output .research-output/preflight/my-preflight
```

새 train/dev 초기 상태 진단은 아래 명령이다. 기존 제안 JSON을 검증하고 실행 snapshot을 상태 생성 전에 저장한다. reserved test6000대와 기존 test3000대를 생성/평가하지 않고 학습하지 않는다. 전체 상태·경로/환경은 `report.json`과 `probes.jsonl`, 시작 시 hash/환경은 `started.json`, 비민감 집계는 `summary.json`이다. 잘못된 설정과 기존 output은 거절한다. 실행 중 검증 실패 시 중간 raw와 `failure.json`을 보존한다.

```powershell
pnpm research:teacher-matrix --output .research-output/matrices/my-matrix
```

아이템·상한 scripted 진단은 아래 명령이다. seed 없이 고정8상태를 사용하고 시작 전에 protocol/hash를 저장한다. `cases.jsonl`에는 완료한 상황별 전이·reference·선택경로, `report.json`에는 전체 실행, `summary.json`에는 강제/선택을 분리한 비민감 집계를 남긴다. 기존 출력은 덮어쓰지 않으며 학습/test를 실행하지 않는다.

```powershell
pnpm research:item-coverage --output .research-output/coverage/my-coverage
```

## References

[1]: https://maplestory.nexon.com/News/Event/Ongoing/1393
[2]: https://sites.ualberta.ca/~szepesva/papers/ecml06.pdf
[3]: https://arxiv.org/abs/1312.5602
[4]: https://arxiv.org/abs/1707.06347

1. Nexon. [한글 모아모아 이벤트 안내][1]. 게임 배경 자료이며 독립 형상/확률 검증을 대체하지 않는다.
2. Kocsis, L., & Szepesvári, C. (2006). [Bandit Based Monte-Carlo Planning][2]. ECML, 282–293.
3. Mnih, V., et al. (2013). [Playing Atari with Deep Reinforcement Learning][3]. arXiv:1312.5602.
4. Schulman, J., et al. (2017). [Proximal Policy Optimization Algorithms][4]. arXiv:1707.06347.

논문은 방법 참고이며 본 프로젝트의 성능 증거가 아니다. 모든 그림은 자체 생성 설명용 도식이다. 메이플스토리 상표/게임 저작물의 권리는 각 권리자에게 있으며 공식/제휴 프로젝트가 아니다.
