# MEDIROAD — 이동진료 방문계획

충북에서 의료 접근성이 부족한 지역을 찾고, 어떤 진료과를 어디에 배치할지 검토하는 의사결정 지원 프로젝트입니다. 지역의 필요도를 정하는 단계와 진료과별 부족, 방문 후보지의 coverage, 일정 최적화를 나눠 구현했습니다.

## 프로젝트를 시작한 이유

의료기관이 적다는 사실만으로 실제 방문진료 수요를 알 수는 없습니다. 지역 인구·건강·접근성을 함께 보되, 자료의 공간 단위와 proxy 의미를 구분해야 합니다. 그래서 환자 수를 예측한다고 가정하기보다 설명 가능한 필요도 점수와 제약을 가진 방문계획으로 접근했습니다.

## 사용 기술

Python, pandas, NumPy, SciPy, scikit-learn, GeoPandas, Shapely, Matplotlib, Plotly를 사용합니다. 최적화에는 HiGHS 기반 경로가 있고, 설정으로 정책 가중치와 방문 제약을 분리합니다.

## 데이터

153개 읍면동의 인구·고령화·건강·의료 공급·접근성 자료와 후보 장소, 공간 coverage를 연결합니다. 읍면동 자료와 11개 시군 단위 건강통계는 분석 단위가 다릅니다. 시군 값을 반복한 행을 독립 관측처럼 취급하지 않습니다.

실제 환자 수 label이 없어 Need Score는 감독학습 예측이 아니라 여러 기준을 결합한 점수입니다. 도로 기반 추정 시간과 `*_drive_min_proxy` 같은 proxy도 구분합니다. 원천 데이터의 이용 조건은 [데이터 안내](data/README.md)를 따릅니다.

## 알고리즘 구조

```text
인구·건강·접근성 → Stage1 Need Score
                         ↓
                 Stage2 진료과별 부족·exposure
                         ↓
                 Stage3 장소·공간 coverage
                         ↓
                 Stage4 방문계획·형평성 제약
```

최적화 결과는 어떤 자료와 목적함수 아래에서 선택된 계획인지 함께 해석합니다. 계산상 제약 충족이 의료 효과나 현장 운영 가능성을 대신하지는 않습니다.

## 개발 과정

### 전체 열을 넣는 대신 의미 단위로 점수 구성

초기 자료에는 인구 규모, 반복된 시군 값, 유사한 고령화 지표와 이전 점수가 섞여 있었습니다. 그대로 합치면 같은 개념에 여러 번 가중치를 주게 됩니다. 특징의 역할과 단위를 먼저 구분하고, 보완 지표를 semantic component로 묶은 뒤 축별 점수를 계산했습니다.

### 필요도와 진료과별 부족 분리

여러 진료과 점수가 모두 일반적인 농촌·의료 접근성 패턴을 반복하는 문제가 있었습니다. 일반 접근성은 Stage1에 남기고, 진료과 접근성은 일반 의료기관 대비 추가로 필요한 이동 정도로 표현했습니다. 건강·연령 특징도 공통 효과와 진료과별 차이를 나눴습니다.

숫자상 상관만 낮추는 대안은 채택하지 않았습니다. 예를 들어 유효한 진료과 공급 정보를 제거하면 분리력 수치가 좋아져도 설명하려는 의미가 사라집니다.

### Exposure와 공간 coverage 연결

필요도와 실제 서비스에 노출될 수 있는 규모·시간 조건을 분리하고, 후보 장소가 어느 지역을 커버하는지 연결했습니다. 동일 지역 중복 coverage를 단순 합산하면 효과가 부풀 수 있어 계획 전체의 coverage로 계산합니다.

### 좋은 후보에서 제약을 만족하는 계획으로

방문 후보를 탐색한 뒤 방문 수·지역별 제한·형평성을 만족하는 조합을 최적화합니다. 후속 단계에서는 최소 시군 coverage를 먼저 확보하고 high-need 인구 coverage를 다루는 순차 구조로 확장했습니다.

단순히 점수가 높은 계획을 찾는 것과 더 좋은 계획이 남아 있는지 확인하는 것은 다릅니다. 상한·하한과 제약을 따로 확인하고, 증명 가능한 상한을 이용해 불가능한 후보만 제외하도록 구성했습니다.

## 결과와 한계

당시 Stage1 기록에서 대표 구성은 30개 특징을 19개 component로 묶었고, 기본 정책의 안정성 조건을 만족했습니다. 일부 대안 정책은 더 민감해 탐색적 설정으로 남겼습니다. 이는 실제 의료수요 예측 정확도가 아니라 점수 구성의 일관성 결과입니다.

진료과별 분리력, coverage, 최적화 gap은 서로 다른 평가 대상입니다. 이동시간·장소 수용성의 현장 확인과 실제 수요·의료 효과 검증은 남아 있습니다.

## 실행

[src/mediroad](src/mediroad)에서 scoring, specialty, temporal, spatial과 Stage4 모듈을 읽을 수 있습니다. 단계별 입력과 설정을 먼저 맞춘 뒤 실행합니다.

```powershell
pip install -r requirements-model-v1.txt
python run_model_v1.py --help
python run_model_v1.py --package-root data/local/mediroad-v6 --bootstrap 1000 --n-jobs 8
```

`run_model_v1.py`는 기존 Stage1 필요도 파이프라인을 실행합니다. `--package-root` 아래에 원본 master·자료 감사표·도로 보고서와 `configs/model_v1/stage1_features.yaml`을 준비합니다. 결과는 해당 패키지의 `outputs/model_v1`과 `reports/model_v1`에 저장하며 Stage2–4를 자동 실행하지 않습니다.

단계별 실행 명령과 필요한 모듈 경로는 [실행 안내](https://github.com/minsu0011/mediroad-planning/wiki/How-to-Run)에 구분했습니다.

[개발 과정](https://github.com/minsu0011/mediroad-planning/wiki/Development-Journey) · [알고리즘별 역할](https://github.com/minsu0011/mediroad-planning/wiki/Model-Evolution) · [병목과 해결](https://github.com/minsu0011/mediroad-planning/wiki/Bottlenecks-and-Solutions) · [결과 해석](https://github.com/minsu0011/mediroad-planning/wiki/Validation-and-Results)

[Wiki 전체 보기](https://github.com/minsu0011/mediroad-planning/wiki) · [저장소 내 문서 사본](docs/wiki/Home.md)
