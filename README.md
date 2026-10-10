# Hangeul Decision Lab

### Budgeted Search and a Research Agenda for a Hangul Block Puzzle

**Implemented:** DFS assistant, synthetic simulator/benchmark, offline CPU CNN imitation pilot · **Application policy:** DFS · **RL:** FUTURE

## Abstract

Hangeul Decision Lab은 메이플스토리 한글날 이벤트의 16행 × 10열 블록 퍼즐을 상태 공간 문제로 모델링하는 웹 도구다. 제한된 계산 예산 안에서 현재 조각과 특수 능력의 사용 순서를 추천한다.

현재 구현은 합법 배치 열거, 공통 게임 전이, lexicographic evaluation, 완료된 subtree의 재사용을 결합한 deterministic depth-first search (DFS)를 사용한다. 화면 입력은 로컬 캡처와 판별을 거쳐 사용자가 명시적으로 확인한다. 실제 게임 조작은 사용자가 수행한다.

장기 연구 질문은 **같은 의사결정 시간 안에서 탐색과 학습이 장기 생존 및 줄 삭제를 얼마나 개선할 수 있는가**이다. 현재 seeded simulator를 통한 DFS 비교와 오프라인 CPU CNN의 supervised imitation pilot을 구현했다.

작은 합성 실험에서 CNN의 teacher 선택 일치율은 53.3%로, first-legal baseline의 64.4%보다 낮았다. 학습 절차는 검증했지만 앱의 policy 채택이나 생존 향상을 뒷받침하지 않는다. 이후 탐색 개선과 learned policy/value model을 비교하며, reinforcement learning (RL)은 향후 연구로 남는다.

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

| 기호 | 현재 의사결정 상태                           |
| ---- | -------------------------------------------- |
| $B$  | 16 × 10 Boolean 점유 보드                    |
| $P$  | 남은 최대 3개 조각의 instance·화면 slot·형상 |
| $I$  | 미획득 아이템의 좌표/종류, 보드 위 최대 3개  |
| $A$  | 점 찍기/바꿔 뽑기 횟수, 보유 합계 최대 7개   |
| $r$  | 재뽑기 결과를 기다리는 대상 또는 null        |

action은 조각 배치, 빈칸 한 곳에 점 찍기, 남은 조각 재뽑기다. 조각 사용 순서는 자유롭고, slot은 화면 위치 식별자다. 회전과 반전의 중복 variant를 제거하고 실제 `true`인 셀만 경계·충돌 검사에 사용한다.

가로줄은 동시에 삭제되며 중력은 없다. 능력을 소모한 뒤 삭제 행의 아이템을 행·열 순으로 획득한다. 보유 상한 때문에 획득하지 못한 아이콘은 같은 좌표에 남긴다.

19종 조각의 형상과 한글 이름은 제공된 자료를 기반으로 한다. ㄹ과 ㅌ은 각각 8칸이며, 점·ㅡ·ㅣ는 각각 1·3·5칸이다.

전체 형상과 규칙의 독립적인 실전 대조, 새 아이콘 생성 규칙, 단계별 출현 확률은 아직 확인을 마치지 않았다. 이벤트 배경은 [공식 안내][1]를 참고한다.

![Synthetic line-clear example](assets/figures/line-clear.svg)

_Figure 1. 합성 규칙 예제. 마지막 행의 빈 3칸에 ㅡ를 배치하면 그 행만 비워지고 위의 셀은 제자리에 남는다. 게임 캡처·솔버 추천·성능 실험 결과가 아니다._

## 3. Implemented Method

### 3.1 Bounded DFS

기본값은 `maxNodes=512`, `maxAlternatives=3` (최선 포함), `useMemoization=true`다.

ordinary-only probe는 전체 예산의 최대 1/4을 사용한다. 실제 방문한 노드 수를 능력 DFS의 예산에서 차감하며, root를 포함한 합산 방문 수를 제한한다. 일반 조각 소진까지의 깊이는 최대 남은 3개 조각에 해당한다. 능력 사용과 재획득을 포함한 action 깊이는 3으로 고정되지 않는다.

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

search key에는 점유 상태, 남은 조각의 instance와 형상, 아이템, 능력, pending 상태, 누적 삭제·획득 수가 포함된다.

예산 때문에 중단된 subtree는 memoization에 저장하지 않는다. 완료된 subtree의 결과를 재사용하므로, 반환한 alternatives와 path count를 모든 고유 경로의 열거로 해석할 수는 없다.

### 3.2 Evaluation and Ability Candidates

평가는 다음 우선순위를 따른다. 앞선 항목이 같을 때만 다음 항목을 비교하는 **lexicographic comparison**이다.

gameover 회피 → 현재 조각 소진 → catalog의 배치 가능 종류·배치 수 → 삭제 행 → 아이템 획득 → 능력 잔여 → 적은 고립·coverage gap → 많은 빈칸.

이 평가는 현재 규칙에 따른 순서 비교이며, 학습한 weighted reward는 아니다.

점 찍기는 모든 합법 빈칸을 후보로 두고, 즉시 행을 삭제하는 action부터 탐색한다. 재뽑기는 막힌 조각이 있거나, 초기 probe에서 소진 경로를 찾지 못했거나, catalog에서 배치 가능한 종류가 0개일 때 제한적으로 추가한다.

미완료 probe에서 경로를 찾지 못한 것만으로 배치가 불가능하다고 결론 내리지 않는다. 전체 domain의 legal-action mask와 solver가 탐색하는 제한된 후보를 구별한다.

mobility는 catalog에서 배치 가능한 조각 종류와 배치 수다. 미래 생존이나 출현 확률을 뜻하지 않는다. coverage gap은 빈 셀을 덮을 수 있는지를 나타내는 지표이며, 점 조각 때문에 gap이 0이어도 큰 조각의 소진을 보장하지 않는다.

재뽑기 결과는 실제로 다른 조각을 관측한 뒤 입력받는다. 현재 탐색은 그 입력을 기다리는 pending 상태에서 멈춘다.

![Implemented decision pipeline](assets/figures/decision-pipeline.svg)

_Figure 2. 구현된 입력–검토–탐색–재생 경계. 판별기는 초안을 만들고 사용자가 확인한 뒤 GameState에 반영한다. Apply Step은 앱 상태 재생이며 실제 게임 배치는 사용자 조작이다._

### 3.3 Observation and Review

선택한 화면 공유의 한 frame에서 보드 점유, 보유 조각 도형, 능력 숫자를 로컬에서 판별한다. 미확정 칸은 캡처 위에서 바로 수정할 수 있다. 아이템의 위치와 종류는 점유 상태와 별도로 검토하며, 초안을 자동 승인하지 않는다.

같은 공유 세션에서는 영역과 직전 확인 상태를 이어서 사용한다. 마지막 정상 Apply 뒤에는 제한된 자동 재읽기로 새 조각을 기다린다. 이전 상태와의 차이와 새 아이콘은 사용자가 명시적으로 확인한다.

캡처/임시 픽셀은 제한된 탭 메모리 수명 뒤 폐기한다. 저장·업로드·계정 연동·게임 자동 조작 기능은 없다.

## 4. Validation Scope and Limitations

검사는 합성 보드, 독립 좌표 oracle, 전이 replay, 예산과 memoization, 입력 수명, 브라우저 여정을 다룬다. 작은 fixture의 완전 비교는 해당 범위의 correctness 근거이며 전체 게임의 oracle는 아니다. `optimalWithinScope`는 설정한 후보와 평가 범위에 한정된다.

실제 게임의 연속 턴, 전체 형상·숫자·배율의 인식 정확도, 신규 아이콘과 출현 확률, 사용감과 별도 제품 수락은 미완료다.

짧은 합성 episode의 관측은 다음 절에 제시한다. 이를 실제 게임의 장기 생존, 알고리즘 승률, learning curve의 증거로 확대하지 않는다.

## 5. Synthetic Benchmark and Future Research

### 5.1 Benchmark First

실제 게임 플레이 없이 기존 domain 전이를 재사용하는 headless simulator를 실행할 수 있다. 합성 환경은 19종 조각에 동등 가중치를 주고, 재뽑기에서 현재 종류를 제외하며, 초기 아이템만 사용한다.

실제 출현 확률, 새 아이콘 생성, 7수 카운터는 모델링하지 않는다. 실제 게임과의 일치 여부는 별도 검증 과제로 유지한다. 5.4의 오프라인 imitation learning도 이 합성 환경만 사용한다.

초기 상태, catalog와 rules version, 다음 조각·재뽑기·아이콘 과정, seed와 PRNG, 종료 조건과 horizon을 고정한다. 환경, policy, rollout, 학습의 난수를 분리하고 action별 난수 소비 차이를 통제한다. 미래 tape는 policy에 노출하지 않는다.

다음 입력 대기, 실제 gameover, 시간 또는 episode truncation을 구별한다. 임의의 균등분포를 실제 출현 확률로 가정하지 않는다.

| Metric                      | 비교 정의                                                    |
| --------------------------- | ------------------------------------------------------------ |
| Survival / terminal step    | 소진 세트와 일반/능력 action 수, gameover/truncation 분리    |
| Clears / acquired items     | 누적 삭제 행/실제 획득, 잔류 아이템 제외                     |
| Ability usage               | 종류별 소모·재획득·잔여/구제 결과                            |
| Decision latency            | 합법 후보/평가/탐색/추론의 mean/p95, load/warmup 분리        |
| Search cost / memory        | 알고리즘별 nodes/simulations/memo, 정의한 heap/RSS/피크 한계 |
| Reference / reproducibility | 유효 경로 도달률, action/state hash, 오류/timeout/분산       |

같은 초기 상태, paired seeds, 종료 조건과 시간 상한으로 비교한다. 알고리즘 간 node count를 동일 비용으로 취급하지 않는다. train/dev/test는 seed와 episode 단위로 분리하며, 계열 밖 generalization은 별도 실험이 필요하다.

아래 작은 CNN pilot은 sparse/pressure 계열을 모든 split에서 공유한다. test benchmark로 학습하거나 tuning하지 않는다. 생존·삭제 성과, 지연·메모리, 안정성을 구분해 보고한다.

**실행된 pilot — 2026-10-07.** seed 17·42·2026 × sparse/pressure 보드의 6개 dev fixture를 사용했다. 예산 128/512, memoization 사용, alternatives 3개, 최대 12개 action으로 각각 2번 실행했다.

총 24개 episode이며, 반복을 독립 표본으로 세지 않는다. 실행 순서는 회전했고 별도 seed 0의 warmup은 집계에서 제외했다. 모든 episode의 replay와 반복 trace 일치 검사를 통과했다.

| 관측                           |   DFS 128 |    DFS 512 |
| ------------------------------ | --------: | ---------: |
| 평균 수행 행동 / 최대 12       |      7.67 |       8.00 |
| 평균 삭제 행                   |      0.83 |       1.00 |
| 평균 판단 시간                 |  395.79ms | 1,400.62ms |
| 판단 시간 p95                  |  596.17ms | 2,254.72ms |
| 불완전 탐색 / 판단 횟수        |   86 / 94 |    72 / 96 |
| gameover / horizon / 정책 포기 | 4 / 6 / 2 |  6 / 6 / 0 |

측정 환경은 Windows 10, i5-9400F, Node 24.15.0, 단일 worker다. 판단 시간은 `solveTurn` 호출부터 반환까지다. simulator의 legal-action 생성, 환경 공급, replay, 모듈 로드는 제외한다.

p95는 유효 episode의 판단 표본에서 nearest-rank로 계산한다. 보드마다 action 경로가 달라 두 설정의 판단 표본 수는 다르다. 이 비교는 **동일 시간 예산 실험이 아니다**.

전후 Node heap은 약 43.87 → 108.68 MiB, RSS는 226.77 → 358.22 MiB였다. compiler와 누적 기록을 포함한 전체 프로세스 관측이며, 순간 peak나 브라우저 메모리 기준을 뜻하지 않는다.

6개 paired fixture 중 pressure-42에서만 DFS 512가 action 2개를 더 수행하고 1행을 더 지웠다. 그 외 합산 action 수와 삭제 행 수는 같았다. sparse 3개는 모두 horizon에 도달했고, 아이템 획득은 전부 0이었다. 장기 생존, 아이템 전략, 학습 효과를 판정하기에는 표본이 부족하다.

policy abstention은 legal action이 남아 있는데도 policy가 action을 반환하지 않은 경우이며, gameover와 구분한다. 관측값, protocol, source·trace hash는 [pilot summary](assets/research/pilot-budget-20261007-v1.json)에 있다. CLI는 전체 fixture, tape, action, state 기록을 로컬 JSON으로 생성한다.

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

점유·아이템 plane과 조각·능력 입력을 사용하는 소형 CNN candidate scorer를 오프라인 imitation pilot에 구현했다. 앱에 채택된 learned policy나 value model은 없다. RNN, GNN, policy/value hybrid는 측정 가능한 이점이 있을 때 검토한다.

DQN[3], Actor-Critic/PPO[4], policy-guided search, learned value + MCTS는 후보 방법이다. 합법성은 기존 domain mask와 최종 전이 검증이 맡는다.

학습 자료는 승인된 simulator에서 생성한 episode다. 상대방과 대전하는 self-play로 설명하지 않는다.

reward 후보는 생존·삭제·획득과 mobility, 능력 잔여, 고립, 불필요한 소모에 대한 reward shaping이다. 가중치와 terminal penalty는 미정이며, proxy bias는 ablation으로 검사한다.

모델은 action ordering과 leaf value를 보완하는 hybrid로도 비교한다. 개선이 작거나 지연·메모리·복잡도·불안정성이 크면 deterministic search를 유지한다.

기록은 문제 → 가설 → fixture/seed/config → 관찰 → 해석 → 결정으로 연결하고 실패/기각도 보존한다. chart는 CSV/JSON 원본·생성 방법을 필요로 하며 개념 그림을 측정 결과처럼 사용하지 않는다.

### 5.4 Technology Briefing

| 역할                      | 현재 채택                                              | 다음 제안                                            |
| ------------------------- | ------------------------------------------------------ | ---------------------------------------------------- |
| 게임 규칙·탐색·시뮬레이션 | TypeScript / Node.js, pure domain, 예산 제한 DFS       | 현재 전이를 학습 자료 생성·합법 행동 검증에도 재사용 |
| 화면 입력·검토            | React / Next.js, 브라우저 Canvas의 로컬 판별           | 이번 CNN 실험에서 변경 없음                          |
| 신경망 학습               | 연구 전용 Python 3.13.13 / PyTorch 2.14.1+cpu          | 소형 CNN의 오프라인 imitation pilot 완료             |
| 학습 목표                 | 합성 상태에서 DFS 512 첫 action의 supervised imitation | 장기 생존·RL은 별도 후속 연구                        |

PyTorch의 [공식 학습 경로](https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)를 참고해 연구 전용 CPU 실행을 채택했다.

모델은 보드의 3개 input plane에 3 × 3 convolution (채널 8/8)과 4 × 5 pooling을 적용한다. 조각·능력 metadata 140개 값과 legal candidate feature 52개 값을 32차원 embedding에서 함께 점수화한다. 총 12,189개 parameter를 사용하며, 전체 domain의 legal candidate에만 softmax를 적용한다.

미래 조각, seed, instance 이름, teacher 결과는 forward 입력에서 제외한다. bounded DFS 512의 teacher 선택은 최적 정답이 아니다. 각 선택의 search scope와 incomplete 여부를 함께 보관한다.

**실행된 CNN pilot — 2026-10-08.** seed 범위는 train 1000–1015, dev 2000–2003, test 3000–3003이다. 각 seed에 sparse/pressure를 적용하고 horizon을 6으로 고정했다. 총 48개 episode에서 train/dev/test 각각 168/45/45개 선택을 얻었다.

teacher의 202/258개 선택은 incomplete search에서 나왔다. skip과 error는 0건이었으며, 모든 episode, encoding, key를 domain에 대조했다.

학습 설정은 Adam (learning rate 0.001, weight decay 0.0001), batch size 8, 12 epochs, seed 20261008, CPU 2 threads다. dev top1이 가장 높은 첫 checkpoint (epoch 11, 40.0%)를 선택한 뒤 test를 평가했다.

같은 train/dev 재실행에서 전체 loss, checkpoint 선택, weights가 bit-equal이었다. 이 재현 검사에는 test 점수를 사용하지 않았다.

| 시험 관측                                | 결과                                      |
| ---------------------------------------- | ----------------------------------------- |
| CNN top1 / first-legal baseline          | 24/45 (53.3%) / 29/45 (64.4%)             |
| fixture macro top1                       | 54.2%, 독립 test fixture 8개              |
| teacher complete / incomplete의 CNN 일치 | 8/13 (61.5%) / 16/32 (50.0%)              |
| 최종 domain 적용                         | 45/45 합법, NaN 0건·반복 예측 일치        |
| CPU 추론 mean / p95                      | 0.56 / 0.92 ms, 45개 선택 × 고정 3회 반복 |
| tensor 생성 → 선택 key mean / p95        | 1.99 / 4.68 ms, 45개 선택                 |
| 학습+dev / 연구 runner 처리 시간         | 3.67 / 7.28초                             |
| process working set 전후                 | 183.9 / 349.7 MiB, 순간 peak 제외         |

Windows 10 / i5-9400F 개발 PC에서 관측한 결과다. 추론 시간에는 finite 검사와 argmax를 포함하고 load와 warmup은 제외한다. domain 후보 생성, JSON 처리, 프로세스 시작, IPC, UI 시간은 포함하지 않는다. 연구 runner의 처리 시간도 Python/PyTorch import를 제외한다.

45개 연속 상태를 독립적인 45개 게임으로 세지 않는다. 계열 밖 generalization, 통계적 우월성, 실전 생존, RL은 검증하지 않았다. **first-legal baseline보다 낮아 현재 모델의 제품 적용을 추천하지 않는다.** 같은 가치의 다른 legal action도 hard-label top1에서는 불일치로 집계될 수 있으므로, 이 점수는 생존 성과를 뜻하지 않는다.

비민감 수치와 protocol·source·data hash는 [CNN pilot summary](assets/research/pilot-cnn-imitation-20261008-v1.json)에 있다. raw data와 checkpoint는 Git 밖 로컬에 보관한다. 설치 다운로드 외에 자료 전송은 없다.

### 5.5 Teacher Order and Coverage Audit

**진단 — 2026-10-08.** 재학습에 앞서 기존 자료의 분포와 teacher 선택을 검사했다. label이나 모델 성적을 보지 않고, train 1000–1003 / dev 2000–2003 × sparse/pressure의 최초 상태 16개를 고정했다.

각 상태에 원본, 조각 배열 뒤집기, slot 번호 순환 (0 → 1 → 2 → 0)을 적용해 DFS 512를 총 48회 실행했다. slot 변경은 보드, 조각 instance와 형상, 능력, 아이템, 전체 물리적 legal-action 집합을 보존한다. 게임에서 조각 선택 순서는 자유롭다.

| 진단 관측                            | 결과                           |
| ------------------------------------ | ------------------------------ |
| 배열만 뒤집은 대조                   | 16/16 경로·평가·탐색 정보 동일 |
| slot 순환 뒤 물리적 첫 행동 변경     | 14/16 상태                     |
| 선택 경로의 lexicographic evaluation | 향상 7 / 저하 7 / 동일 2       |
| 변경된 첫 행동 중 경로 평가 동점     | 0/14                           |
| 원본 탐색 미완료                     | 16/16                          |
| 반환 경로의 domain 재생              | 144/144 통과                   |
| 모델 입력의 정확 중복                | split 내부·split 간 모두 0건   |

teacher는 후보를 slot 순서로 방문하고 제한된 탐색에서 찾은 경로를 반환한다. 위 관측은 **이 예산과 표본에서의 순서 민감성**을 보여준다. 서로 다른 첫 action을 같은 정답으로 합칠 수 있다는 증거나, CNN 성적 저하의 단독 원인은 아니다.

반환 결과 3개에서는 서로 다른 첫 action의 최상 동점이 관측되지 않았다. 그러나 memoization과 truncated search 때문에 전체 동점 후보가 없다고 증명할 수는 없다. 경로 평가의 동점도 즉시 action value나 미래 생존의 동일성을 뜻하지 않는다.

학습 자료에서 first-legal label의 비율은 전체 84/168 (50.0%), complete search 5/37 (13.5%), incomplete search 79/131 (60.3%)다. 배치 label의 124/131 (94.7%)는 남은 조각 중 가장 작은 slot을 선택했고, sparse에서는 89/89였다. 아이템 획득은 train/dev/test에서 각각 2/1/0회에 불과하다.

seed 분리와 정확한 관측 중복 0건만으로 계열 밖 generalization이나 충분한 전략 표본을 보장할 수 없다. 기존 test 통계는 설명용으로만 집계했으며, 새 test 탐색·tuning·점수 측정은 하지 않았다. 현재 CNN 결과와 앱 DFS는 유지한다.

분모, 계열별 수치, hash는 [진단 요약과 고정 protocol](assets/research/teacher-order-audit-20261008-v1.json)에 보관한다.

### 5.6 Complete Reference and Experiment Preflight

순서 진단 뒤 다음 비교 실험의 [제안 protocol](research/experiments/teacher-quality-v1.json)과 preflight 도구를 추가했다. train 4000–4003, dev 5000–5001, reserved test 6000–6003을 분리하고, 노드 예산 128/512, slot 대조, 반복 2회를 설계했다.

**이 preflight 당시 본 실험은 PROPOSED / NOT RUN이었다.** 제안 원본은 보존하며, 이후 train/dev 진단은 5.7에 별도로 기록한다. reserved test 생성과 새 모델 학습은 하지 않았다.

동일한 노드 수는 동일한 실행 시간을 뜻하지 않는다. 현재 동기 DFS에 실제 deadline이 생기기 전까지 시간은 관측값으로만 비교하도록 명시했다.

실행한 검사는 의도적으로 만든 합성 상태 7개를 사용한다. 연구 reference는 최대 2개 조각의 전체 ordinary placement 경로를 memoization과 pruning 없이 열거한다. domain 전이와 평가를 공유하므로 게임 규칙의 독립 oracle는 아니다.

노드 상한 1024에서 미완료이면 최적 첫 action 집합을 반환하지 않는다. 능력 사용은 별도로 지정한 전이를 검사하며, ordinary-only reference는 능력을 탐색하지 않는다.

| 실행 전 검사 관측 — 2026-10-08                      | 결과                         |
| --------------------------------------------------- | ---------------------------- |
| 완전 일반 reference                                 | 7/7, 총 903개 노드           |
| production ordinary DFS의 memo off/on 비교          | 14/14 최상 평가·첫 행동 일치 |
| 반환 경로의 domain 재생                             | 34/34                        |
| 직접 지정한 획득·상한 잔류·점 소모/획득·reroll 대기 | 4/4                          |
| 가장 큰 최상 첫 행동 집합                           | 22개                         |
| 새 reserved test 생성 / 학습 실행                   | 0 / 0                        |

완전한 작은 트리에서는 반환 결과 3개보다 많은 최상 첫 action을 확인할 수 있었다. 이는 **해당 ordinary-only 현재 턴의 경로 평가**에 한정된다. 3개 조각과 능력을 모두 탐색했을 때나 미래 생존에서도 같은 label로 취급할 수 있다는 증거는 아니다.

deliberate coverage 표본은 규칙을 검사하기 위한 예이며, 실제 빈도나 모델 효용을 나타내지 않는다. 기존 CNN top1과 앱의 DFS는 유지한다. 비민감 결과와 hash는 [preflight summary](assets/research/teacher-quality-preflight-20261008-v1.json)에 있다.

### 5.7 Train/Dev Order and Budget Diagnostics

위 제안의 새 train/dev seed × sparse/pressure에서 생성한 **12개 초기 상태**를 기존 `solveTurn`으로 진단했다. 노드 예산 128/512 × 순서 3가지 × 반복 2회로 **총 144회 호출**했으며, 반환 경로 **432개**를 domain으로 replay했다.

현재 조각 3개와 능력만 전달하고 future tape는 전달하지 않는다. memoization과 alternatives 3개를 유지했으며, 상태 index와 반복에 따라 6개 조건의 실행 순서를 회전했다.

반복한 72개 조건의 결과가 동일했고, 단순 배열 뒤집기 대조 24쌍도 동일했다. 반복은 독립 표본으로 세지 않는다.

| 현재 턴 관측, 독립 초기 상태 12개              | DFS 128   | DFS 512   |
| ---------------------------------------------- | --------- | --------- |
| slot 순환 후 첫 추천 변화                      | 10/12     | 10/12     |
| slot 순환 후 경로 평가 향상 / 저하 / 동일      | 7 / 3 / 2 | 5 / 6 / 1 |
| 기본 순서 탐색 미완료                          | 12/12     | 12/12     |
| 기본 순서 선택 경로 줄 삭제 합계               | 4         | 5         |
| 기본 순서 점 능력 사용 경로 / reroll 대기 경로 | 4 / 2     | 3 / 2     |
| 기본 순서 아이템 획득 / 상한 잔류              | 0 / 0     | 0 / 0     |

기본 순서에서 노드 예산을 128 → 512로 늘리면 공통 lexicographic evaluation이 **8/12 향상·4/12 동일**했고, 첫 추천은 2/12에서 바뀌었다. slot 순환 조건에서는 5/12 향상·7/12 동일이었다.

반복을 제외한 72개 상태·조건 조합의 탐색이 모두 incomplete였고, 획득·잔류 표본은 0이었다. 더 큰 예산이 순서 민감성을 해소하거나 아이템 전략을 충분히 대표한다고 볼 수 없다. 같은 경로 평가에서 첫 action이 달라진 1건도 장기 생존의 동등 정답을 뜻하지 않는다.

시간은 warmup 없이 측정했다. cold start, JIT, 개발 PC에서 동시에 실행한 테스트·정적 검사의 영향을 포함하는 관측값이다. 작은 집단의 p95를 속도 수락이나 동일 시간 비교로 사용하지 않는다.

보드 12개는 현재 합성 profile의 초기 상태이며, 장기 episode 분포나 실제 게임 확률을 대표하지 않는다. 새 test 평가, 추가 학습, 앱 policy 변경은 실행하지 않았다. 집단별 coverage와 시간, 고정 실행 설정, hash는 [matrix summary](assets/research/teacher-matrix-20261008-v1.json)에 있다.

### 5.8 Scripted Item and Capacity Coverage

초기 상태 진단에서 획득·상한 잔류가 0건이었던 한계를 점검하기 위해 **의도적으로 만든 8개 상태**를 별도로 실행했다. 검사한 상황은 다음과 같다.

- 일반 획득, 보유 상한 7에서의 미획득 잔류, 점 소모 후 획득, reroll 대기.
- 보유량 6에서의 획득 순서, 상한에서 점 소모 후 획득·초과 잔류.
- 5개 행의 동시 삭제에서 row/col 순으로 획득, 상한에서 reroll 소모 후 관측 대기.

이 상태들은 실제 이벤트 빈도나 학습 자료 분포를 나타내지 않는다.

| 검사                                               | 결과                     |
| -------------------------------------------------- | ------------------------ |
| 명시 정답 전이 / ordinary-only 완전 reference 대조 | 8/8 / 8/8                |
| 기존 full solver 호출 / 동일 반복조건              | 32 / 16/16               |
| 일반·능력 반환경로 domain 재생                     | 106/106                  |
| 선택 경로 획득 / 상한 잔류 발생, 각 예산 8개 상태  | 5/8 / 2/8 (128·512 모두) |
| 선택 경로 점 사용 / reroll 대기, 각 예산 8개 상태  | 6/8 / 1/8 (128·512 모두) |
| full 탐색 미완료, 128 / 512                        | 6/8 / 5/8                |

지정한 action의 기대 전이와 solver의 선택 결과를 분리했다. **규칙을 검사하는 상황을 만들었다는 사실만으로 solver가 그 action부터 선택해야 하는 것은 아니다.** scripted 첫 action과 추천이 일치한 상태는 각 예산에서 2/8이었으며, 목표는 그 일치율을 최대화하는 것이 아니다.

상한에서 점을 먼저 쓰면 보유 공간을 확보해 아이콘 하나를 획득하고, 초과 아이콘은 보드에 남는다. 아이콘 배열을 역순으로 넣어도 여러 행을 동시에 삭제할 때 row/col 순으로 획득했다.

tiny reference의 완전성은 ordinary-only의 공통 평가 범위에 한정된다. 능력 전체나 미래 최적성은 보장하지 않는다. 상황별 forced transition과 선택 결과, hash는 [coverage summary](assets/research/item-coverage-20261008-v1.json)에 보관한다. 새 학습, test 평가, 앱 의미 변경은 없다.

### 5.9 Permutation Union and Equal Node-Cap Comparison

기존 **generated 초기 상태 12개와 scripted 상태 8개**를 재사용해 slot 순서의 영향과 탐색량 증가의 영향을 분리했다. 현재 teacher와 모델을 변경하지 않는 diagnostic이다.

- **A:** 기존 slot 순서의 단일 DFS 512.
- **B:** 같은 순서의 단일 DFS 3072.
- **C:** 여섯 slot 순서에서 각각 DFS 512를 실행하고, 반환 경로를 원래 slot으로 복원한 합집합에서 선택.

B와 C의 총 노드 상한은 상태당 **3,072**로 같다. memo는 호출마다 새로 시작하며, 같은 상한이 같은 실행 시간이나 실제 탐색 비용을 뜻하지는 않는다.

| 현재 턴 경로 평가: 향상 / 저하 / 동일 | C vs A512 | C vs B3072 |
| ------------------------------------- | --------- | ---------- |
| generated 초기 상태 12개              | 8 / 0 / 4 | 5 / 6 / 1  |
| scripted 상태 8개                     | 0 / 0 / 8 | 0 / 0 / 8  |

**C는 A보다 나은 경로를 찾았지만, 같은 노드 상한의 B보다 우월하다는 근거는 얻지 못했다.** generated에서 C의 기본 순서 호출이 A를 포함하므로 C vs A의 저하 0은 구조적으로 기대한 결과다. C vs B의 저하 0·향상 1 이상이라는 pilot gate는 미충족이며, 기존 teacher label과 CNN은 유지한다.

반복·slot 순환·배열 대조를 포함해 measured **600호출**, 반환 경로 **1,740개**를 검사했다. 반복 100조건, 배열 대조 20쌍, C의 slot 순환 대조 20쌍이 일치했다. 과거 A512의 best/alternatives와 SearchInfo도 20/20 일치했다. 별도 warmup 8호출은 집계에서 제외했다.

generated의 A/B/C 선택 경로는 획득·상한 잔류가 모두 0이었다. scripted에서는 세 조건 모두 획득 발생 5/8·잔류 2/8·점 사용 6/8·reroll 대기 1/8이었다. 의도적 coverage 상태는 실제 빈도나 학습 분포를 대표하지 않는다.

generated B/C의 실제 visited nodes는 반복을 제외하면 각각 36,864였다. condition latency 평균은 B 약 7.02초, C 약 7.71초로 관측했다. 이 시간에는 변환·합법성 대조·solve·선택·replay가 포함되며 per-call 파일 기록 비용은 분리했다. warmup, GC/JIT, 표본 크기의 한계 때문에 앱 응답 시간 수락이나 동일 시간 비교로 해석하지 않는다.

measured 호출 중 **510/600은 incomplete**였다. 합집합은 반환된 경로만 사용하며 제한 reroll과 alternatives 누락 가능성을 유지한다. 현재 턴 평가의 동점은 미래 가치의 동등 label을 뜻하지 않는다. 추가 학습, 새 test 평가, 앱 policy 변경은 수행하지 않았다. 고정 설정·집단별 coverage/시간·hash는 [permutation summary](assets/research/permutation-teacher-20261008-v1.json)에 있다.

### 5.10 Canonical Piece Order Diagnostic

**도형 기준의 canonical 순서에서 단일 DFS 512를 실행하는 연구 진단**을 구현했다. 화면 slot·배열·instance ID 변화가 결과에 미치는 영향을 같은 노드 예산에서 검사한다. 실제 고정 표본 실행 결과는 검증 후 별도로 기록한다.

도형의 unique variants에서 정렬 key를 고정하고, 중복 인스턴스와 pending reroll을 보존한 채 slot을 변환·복원한다. 기존 slot 순서와 같은 노드 예산으로 비교하며, 여섯 slot 변경·배열 뒤집기·instance rename·반복에 대한 semantic 결과를 검사하도록 설계했다. solver의 best 선택 규칙은 유지한다.

기존 20개 초기 상태만 재사용하고 새 학습 자료나 held-out test를 만들지 않는다. 결과가 좋아도 teacher label 채택과 CNN 재학습은 별도 결정이다. 현재 앱 DFS와 기존 모델은 유지한다.

## 6. Reproduction and Usage

환경: **Node.js 24.x**, **pnpm 11.16.0**. package/lock에 고정된 TypeScript·Next.js·React·Vitest·Playwright를 사용한다.

```sh
git clone https://github.com/jhpark-jarvis/hangeul-decision-lab.git
cd hangeul-decision-lab
pnpm install --frozen-lockfile
pnpm dev
```

`http://127.0.0.1:45000`에서 수동 입력 후 **Analyze → Apply Step**을 사용한다. 실제 게임 배치는 직접 수행한다.

화면에서 입력을 가져올 때는 다음 순서로 진행한다.

- **화면 캡처·검토 열기 → Start Screen Capture**를 누른다.
- 공유 선택창에서 **창**을 선택하고 게임 화면을 공유한다.
- **Capture Frame**을 누르고 미확정 칸과 아이템을 수정한다.
- **전체 확인하고 분석**을 누른다.

같은 공유에서 조각을 소진하면 자동으로 다음 턴을 읽은 뒤 다시 확인한다. 영역이 어긋나면 **영역 설정**에서 보드·조각·능력 영역을 각각 지정한다. Stop, 공유 교체, 새로고침은 관련 상태를 초기화한다.

```sh
pnpm test
pnpm lint
pnpm build
pnpm start                  # production: 127.0.0.1:45001
pnpm test:e2e               # 별도 터미널, 설치된 Chrome/Edge 필요
pnpm research:benchmark     # 게임 실행 없이 고정 합성 pilot; 결과는 로컬 .research-output/runs/
node scripts/figures/render.mjs
```

benchmark는 기존 TypeScript compiler로 순수 모듈을 `.research-output/runtime`에 생성한다. 새 결과 폴더에는 `protocol.json`, 전체 `report.json`, `summary.csv`를 남긴다. Node와 기존 개발 dependency 외에 Python, GPU, 새 서비스는 필요하지 않다.

다른 dev 설정은 `pnpm research:benchmark --protocol <json파일> --output <새폴더>`로 지정하며 기존 폴더를 덮어쓰지 않는다. 실험 행렬의 총 action은 최대 2,000개, 판단당 노드는 최대 8,192개로 제한한다. 강제 wall-time timeout은 없다.

저장되는 것은 합성 상태다. 앱의 화면 캡처 저장 기능을 추가하지 않는다.

다음은 별도 CPU CNN 연구의 Windows 재현 절차다. `research/cnn/protocol.json`으로 data와 학습 설정을 함께 고정하며, 각 실행은 **새 출력 폴더**를 사용한다.

NumPy, torchvision, GPU는 필요하지 않다. 앱 실행에는 Python을 사용하지 않는다.

```powershell
python -m venv .research-output/venv-cnn
.research-output/venv-cnn/Scripts/python.exe -m pip install --only-binary=:all: -r research/cnn/requirements-lock.txt
pnpm research:dataset --output .research-output/datasets/my-pilot
.research-output/venv-cnn/Scripts/python.exe research/cnn/train.py --dataset .research-output/datasets/my-pilot --output .research-output/models/my-pilot
node scripts/research/verify-imitation.mjs .research-output/datasets/my-pilot .research-output/models/my-pilot/predictions.json .research-output/models/my-pilot/domain-validation.json
.research-output/venv-cnn/Scripts/python.exe research/cnn/reproduce.py --dataset .research-output/datasets/my-pilot --reference .research-output/models/my-pilot --output .research-output/models/my-pilot-repeat
.research-output/venv-cnn/Scripts/python.exe -m unittest discover -s research/cnn -p test_cnn.py
```

data, model, schema, hash가 맞지 않거나 NaN이 나오면 중단한다. 실패한 출력도 보존하고 같은 경로를 덮어쓰지 않는다. checkpoint는 자체 생성 state dict만 `weights_only=True`로 읽는다.

`summarize-cnn.mjs <model폴더> <repeat폴더> <새JSON>`으로 domain 검증과 reproducibility 검사를 모두 통과한 비민감 요약을 재생성할 수 있다.

그림 생성 명령은 SVG 세 개와 합성 입력을 재생성한다. 게임 이미지나 학습·benchmark를 사용하지 않는다. 입력과 metadata는 [figure-data.json](assets/figures/figure-data.json)에 있다.

코드의 역할은 다음과 같다.

- `src/domain`: 규칙, 지표, 탐색.
- `src/research`: simulator와 비교 실험.
- `src/features`: 캡처, 인식, 검토, 세션.
- `src/components`, `src/app`: UI.
- `tests/fixtures`: 비민감 재현 데이터.

기존 자료만 읽는 순서 진단은 아래 명령으로 실행한다. 기본 `audit-protocol.json`은 위 pilot의 train/dev 초기 상태 16개만 지정하며, test 탐색이나 학습을 수행하지 않는다.

원본 자료의 hash와 전체 encoding·replay를 확인한 뒤, 새 폴더에 `report.json`, 완료 상태별 `probes.jsonl`, 공개용 `summary.json`을 남긴다. 중단된 raw도 보존하며 같은 출력 폴더를 다시 사용하지 않는다.

```powershell
pnpm research:teacher-audit --dataset .research-output/datasets/my-pilot --output .research-output/audits/my-audit
```

다음 실험의 preflight는 아래 명령으로 실행한다. 제안한 seed 행렬과 reserved test를 사용하지 않고, 7개 scripted case만 검사한다.

전체 상태·경로·환경은 새 로컬 폴더의 `report.json`, 비민감 집계는 `summary.json`에 보관한다. 기존 폴더는 덮어쓰지 않는다.

```powershell
pnpm research:teacher-quality --output .research-output/preflight/my-preflight
```

새 train/dev 초기 상태 진단은 아래 명령으로 실행한다. 기존 제안 JSON을 검증하고, 상태 생성 전에 실행 snapshot을 저장한다. reserved test 6000대와 기존 test 3000대를 생성하거나 평가하지 않으며 학습도 수행하지 않는다.

전체 상태·경로·환경은 `report.json`과 `probes.jsonl`, 시작 시 hash와 환경은 `started.json`, 비민감 집계는 `summary.json`에 기록한다. 잘못된 설정과 기존 output은 거절한다. 실행 중 검증에 실패하면 중간 raw와 `failure.json`을 보존한다.

```powershell
pnpm research:teacher-matrix --output .research-output/matrices/my-matrix
```

아이템·상한 scripted 진단은 아래 명령으로 실행한다. seed 없이 고정 상태 8개를 사용하고, 시작 전에 protocol과 hash를 저장한다.

`cases.jsonl`에는 완료한 상황별 전이·reference·선택 경로, `report.json`에는 전체 실행, `summary.json`에는 forced transition과 선택 결과를 분리한 비민감 집계를 남긴다. 기존 출력은 덮어쓰지 않으며 학습이나 test를 실행하지 않는다.

```powershell
pnpm research:item-coverage --output .research-output/coverage/my-coverage
```

slot 순서와 탐색 예산을 분리하는 diagnostic은 아래 명령으로 실행한다. 기존 generated 초기 상태 12개와 scripted 상태 8개를 재사용한다. A (DFS 512), B (단일 DFS 3072), C (여섯 순서의 DFS 512)를 비교하고 slot 순환·배열 대조·반복을 검사한다.

고정 protocol의 원본 상태 hash를 실행 전에 대조한다. warmup 8회는 집계에서 제외하고, 본 진단은 총 600회 호출한다. 같은 총 노드 한도가 같은 실행 시간을 뜻하지는 않는다. 반환 경로를 원래 slot으로 복원해 domain에서 replay하며, 현재 teacher label과 학습 모델은 바꾸지 않는다.

```powershell
pnpm research:permutation-teacher --output .research-output/permutation/my-diagnostic
```

protocol과 hash·환경은 `protocol.json`과 `started.json`, 진행 기록은 `calls.jsonl`과 `conditions.jsonl`, 전체 결과와 비민감 집계는 `report.json`과 `summary.json`에 남긴다. 기존 output은 거절하고 실행 중 실패 기록은 보존한다. reserved test 생성·평가와 학습은 수행하지 않는다.

canonical 진단은 아래 명령으로 실행한다. 원본 20개 상태의 hash를 확인한 뒤 A (DFS 512)와 canonical (DFS 512)을 모든 slot 변경, 배열 반전, instance rename에서 2회 비교한다. warmup 2회와 본 진단 360회를 분리하고, full path·evaluation·final state의 invariance와 물리적 target 변화를 구분한다.

```powershell
pnpm research:canonical-teacher --output .research-output/canonical/my-diagnostic
```

protocol, hash, 실행 환경, 진행 JSONL, report와 summary, 실패 기록은 새 output에 남긴다. 기존 출력은 덮어쓰지 않는다. 원래 shape와 solveTurn best를 유지하며 teacher label이나 학습 모델을 변경하지 않는다.

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
