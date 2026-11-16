# 테스트와 실행 확인

의존성을 설치한 뒤 저장소 루트에서 실행합니다.

```bash
PYTHONPATH=src python -m pytest tests/test_stage1_scoring.py -q
```

Stage1의 변환·가중치·순위 경계를 검사합니다. 위 환경변수 표기는 POSIX shell 기준입니다.

Stage2–4 통합에는 master·공간 자료와 이전 단계 결과가 필요합니다. [단계별 실행](wiki/How-to-Run.md)을 참고하세요.
