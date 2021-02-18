# mark-yolo5
[특허정보넷 키프리스](http://www.kipris.or.kr/khome/main.jsp)의 상표이미지 데이터셋에 대한 객체 인식 모델을 제공합니다. [YoLOv5 모델](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)을 활용하여 커스터마이징 하였습니다.

## 특징
- 다음 객체가 있는 상표 이미지에 대한 모델 제공 (repository에 가중치 포함)
    - 동물의 머리
    - 요리사
    - 심장 모양
    - 남성
    - 머리와 상반신

- docker 기반의 환경 제공
- API 서버 형태로 모델 이용 가능

## 사용법
**개발시**
```bash
# docker image 생성
$ ./scripts/build_dev_env.sh
# docker container 접속
$ ./scripts/conn_dev_env.sh

# (권장) 컨테이너 내 가상환경 생성 및 접속
$ python -m venv venv
$ source ./venv/bin/activate

# (의존성 설치)
$ pip install -r requirements.txt

# 서버 구동 (default PORT BINDING: 8001:8000)
$ ./scripts/run_dev_server.sh
```

**배포시**
> VERSION 파일에서 버전 확인 후 진행

```bash
$ ./scripts/build_prod.sh
$ ./scripts/run_prod_server.sh
```
