# 배경·객체 합성과 Harmonizer 전후 비교

학습과 별도로 배경 한 장·객체 한 장·객체 마스크를 지정해 붙여넣고, Harmonizer 색·조명 보정 전후를 비교한다. COCO JSON·검증 데이터·클래스 매핑은 필요하지 않다. 학습 설정이나 학습 코드를 바꾸지 않는다.

## 1. 모델·추가 패키지 준비

모델 없이 붙이기만 보려면 이 단계 없이 아래 명령에 `--plain-only`를 추가하면 된다. Harmonizer 전후 비교에는 다음 준비가 필요하다.

**인터넷이 되는 준비 머신에서** 프로젝트 폴더에 들어가 실행한다.

```bash
python scripts/prepare_harmonizer.py
pip download --only-binary=:all: --no-deps -r requirements-harmonizer.txt \
  --dest harmonizer-wheelhouse
```

첫 명령은 공식 저장소의 고정 revision에서 필요한 소스 9개와 약 19MB의 `harmonizer.pth`를 다운로드하고 SHA256을 검사한다. 소스·가중치와 `SOURCE.json`이 `models/Harmonizer/`에 저장된다. 임시 폴더에서 모두 검증한 뒤 완성 폴더로 이동하며 기존 폴더는 덮어쓰지 않는다. Torch가 없어도 준비 스크립트는 실행할 수 있다. 두 번째 명령은 **기존 YOLOX 환경에 Torch·torchvision·packaging이 설치되어 있다는 전제**에서 추가 패키지 kornia 0.8.2·kornia-rs 0.2.0만 수집한다. 대상과 같은 Python 3.12·OS·CPU 아키텍처에서 준비한다.

`models/Harmonizer/` 전체와 `harmonizer-wheelhouse/`를 폐쇄망 서버로 복사한다. 서버의 기존 학습용 Python 환경에서 설치한다.

```bash
python -m pip install --no-index --find-links=harmonizer-wheelhouse \
  -r requirements-harmonizer.txt
```

내부 pip 저장소에 패키지가 있으면 `python -m pip install -r requirements-harmonizer.txt`로 설치할 수 있다. 외부 소스·가중치·wheel 폴더는 Git 및 소스 압축에 포함하지 않으며 별도로 반입한다. 기본 requirements와 Qwen 패키지는 변경하지 않았다. Diffusers·FlashAttention·bitsandbytes는 이 명령에서 사용하지 않는다.

Kornia는 다른 기능의 초기화 과정에서 FlashAttention import를 시도할 수 있다. Harmonizer는 색 보정만 사용하므로 해당 선택 모듈을 import 단계에서 제외하고 원래 상태로 복원한다. 기존 FlashAttention 바이너리 불일치가 이 색 보정 명령을 막지 않도록 처리했으며 시스템 패키지는 수정하지 않는다.

공식 저장소: https://github.com/ZHKKKe/Harmonizer

고정 revision: `48ecd70becbff50ccaf576db0e64212dbc494e26`. 파일 목록과 SHA256은 `configs/harmonizer-source.json`에 있다. 로컬 실행 시 소스 hash를 확인하고, checkpoint는 `weights_only=True`로 로드한다. EfficientNet은 `.from_name()`으로 생성하므로 런타임에 별도 가중치를 다운로드하지 않는다. `run.py`는 IPv4/IPv6 네트워크 차단 아래 실행한다.

## 2. 이미지 지정해 합성

기존 데이터 구조를 그대로 사용할 수 있다. 파일명은 실제 서버에 있는 것으로 바꾼다.

```bash
python run.py paste-check \
  --background data/assets/background/1.png \
  --object data/assets/object/object_1/001.png \
  --mask data/assets/mask/object_1/001.png \
  --object-size 64 \
  --output outputs/paste_check_v1
```

객체 이미지와 마스크는 같은 원본 해상도여야 한다. 마스크는 단일 채널 uint8 이진 이미지(0/255 또는 0/1)로 준비한다. PNG alpha 대신 지정한 마스크를 사용한다. 배경·객체는 읽을 수 있는 uint8 BGR/BGRA 이미지여야 한다. 절대 경로도 사용할 수 있으며 상대 경로는 프로젝트 루트 기준이다.

기본적으로 배경의 중앙에서 640×640 패치를 자르고, 회전 없이 긴 변 64픽셀로 바꾼 객체를 중앙에 붙인다. 객체 크기는 tight foreground 기준이다. 원본이 패치보다 작으면 114로 패딩한다. 객체와 배경을 임의로 골라주는 학습 합성 명령과 달리 여기서는 지정한 파일을 그대로 사용한다.

모델 없이 기본 합성만 확인하려면:

```bash
python run.py paste-check \
  --background data/assets/background/1.png \
  --object data/assets/object/object_1/001.png \
  --mask data/assets/mask/object_1/001.png \
  --object-size 64 --plain-only --output outputs/paste_plain_v1
```

`--plain-only`에서는 모델 폴더와 kornia가 필요 없고 `after.png`·`comparison.png`도 생성하지 않는다.

## 3. 결과와 조절 옵션

```text
outputs/paste_check_v1/
  background.png   # 자른 배경
  before.png       # 기본 합성
  after.png        # Harmonizer 색·조명 보정
  comparison.png   # 왼쪽 before / 오른쪽 after
  mask.png         # 패치 좌표의 이진 객체 마스크
  alpha.png        # 안쪽 feathering을 적용한 알파
  boxes.png        # 보정 이미지 위 정답 박스
  result.json      # 경로·배치 좌표·박스·모델 hash·보정값·시간
```

먼저 `comparison.png`를 연다. `boxes.png`는 배치한 객체의 정답 영역이며 탐지 모델 예측 결과가 아니다. 기존 파일은 덮어쓰지 않으므로 재실행 시 `--output`을 새 폴더로 지정한다.

| 옵션 | 의미 |
| --- | --- |
| `--position 200 280` | 최종 캔버스에서 변환한 객체 이미지의 왼쪽 위 X/Y. 생략하면 중앙 |
| `--object-size 64` | 회전 전 foreground 긴 변 픽셀 크기 |
| `--angle 0` | 객체 회전 각도. 기본 0 |
| `--patch-size 640 640` | 배경에서 자를 높이·너비. 기본 640×640 |
| `--crop-xy 100 200` | 원본 배경에서 자르기 시작할 X/Y. 생략하면 중앙 |
| `--full-background` | 자르지 않고 배경 원본 크기 사용 |
| `--feather 0.6` | 마스크 안쪽 경계 Gaussian sigma. 0이면 hard paste |
| `--strength 0.5` | 기본 합성에서 보정 결과로 섞는 강도 0~1. 기본 1 |
| `--allow-partial` | 캔버스 밖으로 나가는 객체 배치를 명시적으로 허용 |
| `--device auto` | GPU가 있으면 cuda, 없으면 cpu. cpu/cuda 직접 지정도 가능 |
| `--harmonizer-repo /path/Harmonizer` | 반입한 소스·가중치 폴더 위치 |
| `--checkpoint /path/harmonizer.pth` | 같은 구조의 별도 로컬 checkpoint |

위치·크기를 고정하고 `--strength 0.5`와 `1.0`을 각각 새 폴더에 저장해 비교할 수 있다. 같은 파일과 좌표를 사용하면 합성은 결정적이다. 보정은 기존 합성의 마스크 내부에만 적용하고 경계에서는 알파만큼 약하게 섞는다. 마스크 밖 배경 픽셀은 원본 배경과 동일하게 유지한다. 객체의 마스크 좌표·정답 박스는 보정 전후 동일하다.

Harmonizer는 색온도·밝기·대비·채도·하이라이트·그림자를 보정하는 모델이다. 별도의 항공 crop을 입력받아 촬영 질감·시점을 옮기는 기능은 포함하지 않는다. 어울리는지 먼저 확인하는 별도 도구이며, 이 결과가 자동으로 YOLOX 학습에 적용되지는 않는다.

## 검증 범위

Python 3.12 / Torch 2.8 CPU / torchvision 0.23 / kornia 0.8.2에서 공식 실제 가중치를 strict load하고, 네트워크 차단 아래 실제 명령의 전후 이미지 저장과 배경 픽셀 보존을 검증했다. 한 개발용 640×640 합성 샘플에서 CPU 보정 연산은 약 0.22초, 모델 로딩은 약 0.29초였다. 특정 환경·샘플의 1회 측정이며 A100 성능이나 모든 이미지의 처리 시간 기준은 아니다. 실제 사용자 항공영상의 시각적 품질·탐지 AP 개선과 GPU 실행은 미검증이다.
