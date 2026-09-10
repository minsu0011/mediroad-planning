# 점수·노출·coverage·최적화의 역할

## 데이터 단위

기본 지역 단위는 153개 읍면동입니다. 시군 건강통계는 11개 시군 단위이고, 장소·격자·도로 자료는 다른 공간 단위를 갖습니다. 분석 단위를 맞추지 않은 상관이나 fitting은 과도한 표본 수를 만들 수 있습니다.

특징 registry는 원천·방향·단위·proxy·활성 여부를 구분합니다. 과거 점수와 순위를 새 점수의 원천 특징처럼 넣지 않습니다.

## Stage1: Need Score

Percentile 계열 변환과 다기준 가중합을 사용합니다. 여러 특징을 semantic component로 묶고 component를 축으로 결합합니다. 실제 환자 수를 맞힌 예측 모델이 아니라 정책 우선순위용 구성 지표입니다.

[stage1_features.yaml](../../configs/model_v1/stage1_features.yaml)과 [scoring](../../src/mediroad/scoring)에서 변환·가중치를 볼 수 있습니다.

## Stage2: specialty와 exposure

Specialty는 어떤 진료 서비스가 상대적으로 더 부족한지 다룹니다. 접근성 excess는 개념적으로 다음 형태입니다.

```text
log1p(진료과별 이동시간) − log1p(일반 의료기관 이동시간)
```

이는 일반 접근성을 두 번 가중하지 않기 위한 표현입니다. 건강·연령 differential, 공급과 public support의 역할도 구분합니다. Exposure와 temporal 단계는 필요도와 서비스 노출 규모·시점을 분리합니다.

[gap.py](../../src/mediroad/specialty/gap.py), [specialty 설정](../../configs/model_v1/specialty_gap.yaml), [temporal](../../src/mediroad/temporal)에 구현돼 있습니다.

## Stage3: 공간 coverage

장소와 지역의 관계를 구성해 어느 후보가 누구를 커버하는지 표현합니다. 여러 장소가 같은 인구를 포함할 수 있으므로 개별 점수 합과 계획 전체 coverage는 다릅니다.

## Stage4: 제약을 가진 조합

방문 수, 지역 cap, 중복 coverage, equity 목표가 함께 들어갑니다. 단순 Top-K 점수 합이 아니라 제약을 만족하는 계획을 구합니다. 후보군을 고정했을 때의 최적성 범위와 후보군 자체의 적절성도 별개의 문제입니다.

Equity 후속 설정은 최소 시군 coverage와 high-need coverage의 보존 조건을 분리합니다. [equity 설정](../../configs/model_v1/stage4_2d_equity_certification.yaml)과 [threshold oracle](../../src/mediroad/stage4_2d_equity_certification/threshold_oracles.py)에서 선택 기준을 읽을 수 있습니다.
