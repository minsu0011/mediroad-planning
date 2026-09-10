# Stage1 실행과 다음 단계

저장소 루트에서 의존성을 설치합니다.

```bash
pip install -r requirements-model-v1.txt
python run_model_v1.py --help
python run_model_v1.py --package-root data/local/mediroad-v6 --bootstrap 1000 --n-jobs 8
```

`--package-root`는 별도로 준비한 입력 패키지입니다. 아래 상대 경로가 필요합니다.

- `derived/mediroad_admin_dong_master_v6_road_integrated.csv`
- `reports/V6_FEATURE_COLUMN_AUDIT.csv`
- `configs/model_v1/stage1_features.yaml`
- `14_v6_external_data/road_network/derived/osm_road_network_integration_report_v6.json`

Stage1은 특징 감사·상관·ablation·필요도 계산과 조건 검사를 수행합니다. 결과는 패키지 내부의 `outputs/model_v1`과 `reports/model_v1`에 저장합니다. 필수 조건을 만족하지 못하면 종료 코드 2를 반환합니다. 원본을 보존하려면 별도 작업 패키지에서 실행합니다.

후속 실행 파일은 `run_model_v1_stage2.py`, `run_model_v1_stage3.py`, `run_model_v1_stage4.py`입니다. 각 `--help`로 필요한 이전 결과와 인자를 먼저 확인합니다. Stage1 실행이 전체 방문계획을 자동 완성하지는 않습니다.

연산 테스트는 `PYTHONPATH=src python -m pytest tests/test_stage1_scoring.py -q`로 구분합니다. [설정](../../configs/model_v1) · [알고리즘](Model-Evolution.md)
