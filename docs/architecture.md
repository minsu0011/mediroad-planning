# 필요도에서 방문계획까지

`mediroad.data`는 자료를 읽는 코드입니다. 특징 registry와 scoring에서 원천 특징과 이전 점수를 구분하고, specialty·temporal·spatial 계층 뒤에 Stage4 최적화를 연결합니다.

필요도는 우선순위 점수, exposure는 서비스 노출, coverage는 공간 포함 범위입니다. 제약과 상한·하한 검사는 고정된 계획 문제의 계산적 범위를 다룹니다. 이를 실제 의료수요 정확도나 현장 적합성으로 해석하지 않습니다.
