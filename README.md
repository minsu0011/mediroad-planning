# MEDIROAD

충북 지역의 이동진료 필요도를 평가하고, 진료과·방문 장소·일정을 계획하는 의사결정 지원 프로젝트입니다. 인구, 건강, 의료 공급과 접근성 자료를 연결합니다.

## 파이프라인

1. **Need Score** — 의미가 겹치는 지표를 묶어 지역별 필요도 구성
2. **진료과별 부족** — 일반 접근성과 진료과별 추가 부족분 분리
3. **공간 coverage** — 후보 장소가 커버하는 지역과 인구 계산
4. **방문계획** — 방문 수·지역 제한·형평성 제약 아래 조합 최적화

Python, pandas, NumPy, SciPy, scikit-learn, GeoPandas와 Shapely를 사용합니다. 최적화에는 HiGHS 기반 경로가 있습니다.

## 실행

```powershell
pip install -r requirements-model-v1.txt
python run_model_v1.py --help
python run_model_v1.py --package-root data/local/mediroad-v6 --bootstrap 1000 --n-jobs 8
```

위 진입점은 Stage1을 실행합니다. 원본 master·자료 감사표·도로 보고서와 `configs/model_v1/stage1_features.yaml`을 준비해야 합니다. Stage2–4의 실행은 [단계별 안내](docs/wiki/How-to-Run.md)에 정리했습니다.

## 해석 범위

필요도는 실제 환자 수를 학습한 예측값이 아니라 지표를 결합한 점수입니다. 153개 읍면동 자료와 11개 시군 건강통계의 단위를 구분하며, 시군 값을 반복한 행을 독립 표본으로 취급하지 않습니다. 방문계획의 현장 적용에는 이동시간·장소 수용성과 실제 수요 확인이 필요합니다.

[코드](src/mediroad) · [모델 구조](docs/wiki/Model-Evolution.md) · [검증 결과](docs/wiki/Validation-and-Results.md) · [데이터](data/README.md)

[출처 및 이용 조건](ATTRIBUTION.md)
