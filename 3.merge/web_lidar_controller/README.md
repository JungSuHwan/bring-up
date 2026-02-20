# Web LiDAR Controller

별도 웹앱 UI로 `merge_viewer.py`의 실시간 제어 API를 사용합니다.

## 실행

1. 기존 프로그램 실행

```powershell
python merge_viewer.py
```

2. 브라우저에서 컨트롤러 열기

```text
http://localhost:8080/controller
```

## 기능

- LiDAR 선택
- `x/y/z/yaw` 직접 설정(Set)
- `x/y/z/yaw` 증감(+/-)
- `XYZ step`, `Yaw step` 적용
- `Reset XYZ`, `Reset Yaw`

모든 값은 `/control`로 전송되어 즉시 반영됩니다.
