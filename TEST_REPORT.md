# 실제 실행 테스트 결과

## 2026-10-21 합성 이미지·패치의 클래스 중복 방지

`copy_paste.unique_classes`와 공용 CLI `--unique-classes`/`--no-unique-classes`를 추가했다. 활성화하면 객체 자산이 있는 클래스를 중복 없이 균등 선택한 뒤 각 클래스 안에서 소스 이미지·마스크를 랜덤 선택한다. 따라서 서로 다른 파일이어도 같은 COCO category ID의 객체는 한 합성 이미지·패치에 최대 하나만 들어간다. 요청한 최대 객체 수가 실제 자산이 있는 클래스 수를 초과하면 시작 시 오류를 낸다. 기본값은 false이며 기존 파일 단위 `unique_objects`의 의미와 옵션 생략 시 난수 흐름은 유지했다. 실제 검증 데이터의 객체 수는 변경하지 않는다.

Python 3.12 / torch 2.8.0+cu129 / NumPy 1.26.4 / OpenCV 4.11.0.86 / TensorBoard 2.16.2 환경에서 네트워크 차단 상태로 새 테스트 **19 passed, 0 failed, 4 warnings**, 15.65초를 확인했다. [`reports/pytest-unique-classes.xml`](reports/pytest-unique-classes.xml)에 결과가 있다.

동일 환경의 전체 회귀 검사는 **201 passed, 4 GPU skipped, 0 failed, 35 warnings**, 80.22초다. [`reports/pytest-unique-classes-regression.xml`](reports/pytest-unique-classes-regression.xml)에 기록했다. 공식 YOLOX 원본 170개 파일의 hash 검증도 통과했다.

비연속 category ID 1·3·7·29·51 및 같은 클래스의 여러 파일을 사용해 Manifest/폴더 입력, online/online_patch 모드와 unique_objects 동시 설정을 검증했다. 클래스별 하나씩 고르면서 같은 클래스 안에서는 소스가 바뀌는지, 같은 seed에서 픽셀·박스·metadata 재생, 최종 visible mask와 박스·클래스 대응, 0~5개 랜덤 개수와 빈 패치를 확인했다. 기존 파일 중복 방지 모드의 클래스 반복 허용 및 생략/false 동일성, 자산 없는 클래스의 count 한도, boolean 검증과 CLI 우선순위·설정 복사를 확인했다.

0/2 Worker 학습 Loader에서 클래스 중복 방지와 재생을 확인했다. 별도 작업 디렉터리에서 실제 공개 preview·CPU train CLI를 두 합성 모드 모두 실행하여 클래스별 하나씩 저장·학습하고, 유한한 양의 loss와 checkpoint의 유효 설정 저장을 확인했다. 기존 online COCO export에서도 이미지별 클래스 중복 없이 저장했다. GPU 학습과 실제 데이터 탐지 성능 개선은 이 환경에서 검증하지 못했다.

## 2026-10-20 일반 이미지·폴더 추론과 박스 저장

`python run.py predict`를 추가했다. 이미지 한 장 또는 폴더를 재귀적으로 읽고 저장한 YOLOX checkpoint로 추론하여 원본 크기 이미지에 예측 박스·클래스·점수를 그린다. 전체 이미지 추론과 패치 배치 추론을 지원한다. 패치 예측은 letterbox 비율·원본 crop 위치를 역변환하고 실제 이미지 영역으로 잘라 클래스별 NMS로 합친다. 파일별 원본 좌표 JSON과 실행 summary를 저장하며, 정답 박스·학습 데이터·합성 자산을 초기화하지 않는다. 클래스 이름을 얻을 때만 존재하는 검증 JSON의 categories를 읽을 수 있다. EMA 기본 선택과 일반 가중치 선택을 지원한다.

Python 3.12 / torch 2.8.0+cu129 / NumPy 1.26.4 / OpenCV 4.11.0.86 / TensorBoard 2.16.2 환경에서 네트워크 차단 상태로 전체 **182 passed, 4 GPU skipped, 0 failed, 31 warnings**, 67.92초를 확인했다. [`reports/pytest-predict-regression.xml`](reports/pytest-predict-regression.xml)에 결과를 기록했다. 추론 전용 테스트는 **12 passed**, 6.25초이며 [`reports/pytest-predict.xml`](reports/pytest-predict.xml)에 결과가 있다.

단일 후보·빈 예측, 전체 이미지 좌표 복원, 배치 크기별 패치 offset 및 겹친 예측 제거, 다른 클래스 유지, 작은 원본의 패딩 영역 예측 제거를 확인했다. 실제 공식 nano 모델 checkpoint를 사용한 공개 CLI를 별도 작업 디렉터리에서 실행하여 Unicode·공백 경로의 폴더 입력, 하위 폴더 유지, 같은 stem의 다른 확장자 충돌 방지, 박스가 그려진 이미지와 JSON 저장을 확인했다. 없는 학습·합성 폴더에서도 추론했고 EMA/일반 가중치 선택, EXIF 방향 정책, category ID 및 클래스 이름 fallback, 입력·가중치 불변과 출력 덮어쓰기 거부를 검증했다.

공식 YOLOX 원본 170개 파일 hash 검증과 기존 전체 학습·평가 회귀 검사를 통과했다. 정확도 검증용으로 학습한 실제 사용자 모델은 이 환경에 없으며, 테스트용 checkpoint의 예측은 좌표·입출력 검사용이다. 실제 데이터 탐지 성능과 GPU 추론은 이 머신에서 검증하지 못했다. 패치 예측을 원본으로 합치는 추론을 제공하며, 기존 검증 AP의 계산 방식은 바꾸지 않았다.

## 2026-10-20 AP 0 원인 확인용 체크포인트 진단

`python run.py diagnose`를 추가했다. 실제 검증 샘플과 새로 생성한 합성 샘플에 대해 저장된 checkpoint로 추론하고, 정답·예측 이미지를 따로 저장한다. 체크포인트의 학습 설정·완료 EPOCH·학습률, 현재 합성 설정과의 차이, 필터 전 최고 score, 예측 개수, 클래스 포함/제외 IoU 0.5 매칭을 기록한다. 학습이나 원본·checkpoint 수정은 수행하지 않는다. 소수의 양성 위주 표본에 대한 진단으로 전체 AP 평가를 대체하지 않는다. 진단 NMS의 후보 수 제한과 적용 여부도 기록한다.

같은 Python 3.12 / torch 2.8.0+cu129 / NumPy 1.26.4 / OpenCV 4.11.0.86 / TensorBoard 2.16.2 환경에서 네트워크 차단 상태로 전체 **170 passed, 4 GPU skipped, 0 failed, 31 warnings**, 62.10초를 확인했다. [`reports/pytest-diagnose-regression.xml`](reports/pytest-diagnose-regression.xml)에 결과가 있다. 새 진단 테스트는 **5 passed**, 5.12초이며 [`reports/pytest-diagnose.xml`](reports/pytest-diagnose.xml)에 기록했다.

클래스가 틀린 박스와 중복 예측의 일대일 매칭, 공식 letterbox·NMS 후 정답 좌표 복원, 단일 NMS 후보와 후보 수 제한, confidence 미달 시 빈 예측 및 필터 전 최고 score 보존을 확인했다. 양성 샘플 분산 선택과 빈 샘플 fallback도 확인했다. 별도 작업 디렉터리에서 공개 CLI로 실제 모델 checkpoint를 로드하여 실제·합성 이미지와 summary를 저장하고, EMA 선택·checkpoint 학습 설정과 현재 설정 차이·학습률을 확인했다. checkpoint 파일 bytes가 변하지 않고 기존 출력 덮어쓰기를 거부하는 것도 확인했다.

기존 전체 학습·평가 회귀 테스트와 공식 YOLOX 원본 170개 파일 hash 검증을 통과했다. 실제 사용자 서버의 AP 0 원인은 아직 확인하지 못했으며, 이 도구의 실제 checkpoint·데이터 진단 결과가 필요하다. GPU 진단 실행도 이 머신에서는 검증하지 못했다.

## 2026-10-20 객체 전용 블러와 흙색 표면 효과

`object_blur_sigma_range`, `object_dirt_strength_range`, `object_dirt_color_bgr` 설정과 CLI 옵션을 추가했다. 크기·회전 및 기존 색 보정 후 객체 Mask 내부에 절차적으로 만든 갈색 얼룩을 적용하고, Mask로 정규화한 Gaussian blur를 적용한다. 표면 색상 변경이며 별도 물체로 가리지 않으므로 Mask·Box·클래스는 유지한다. 두 강도 범위의 기본값 `[0, 0]`은 기존 결과를 유지한다. 표면 효과의 별도 난수와 seed를 사용하고 적용 수치·흙색·seed를 객체 metadata에 기록한다.

같은 Python 3.12 / torch 2.8.0+cu129 / NumPy 1.26.4 / OpenCV 4.11.0.86 / TensorBoard 2.16.2 환경에서 네트워크 차단 상태로 전체 **165 passed, 4 GPU skipped, 0 failed, 31 warnings**, 58.82초를 확인했다. [`reports/pytest-object-surface-regression.xml`](reports/pytest-object-surface-regression.xml)에 결과를 기록했다. 관련 테스트는 **27 passed**, 8.33초이며 [`reports/pytest-object-surface.xml`](reports/pytest-object-surface.xml)에 결과가 있다.

균일 전경의 경계에 검은 halo가 생기지 않는지, 고주파 무늬가 블러로 완화되는지, 갈색 얼룩의 공간 변화·불투명도 한도·seed 재현성, 입력 이미지·Mask·Mask 밖 픽셀 보존을 확인했다. `online`, `online_patch`에서 표면 효과를 켜도 기존 배경·객체 선택·회전·위치·Mask·bbox와 기존 채도·대비 추출값이 동일했다. 표면 효과를 끈 명시적 기본값과 옵션 생략 결과도 같았다. 범위·색상 검증, CLI 설정 복사·우선순위·Original 모드 거부를 확인했다.

기존 0/2 Worker 재현성과 공개 CLI 테스트에 표면 효과를 포함하여 미리보기 metadata, 실제 CPU 학습·검증·checkpoint 및 설정 저장을 확인했다. 기존 전체 회귀 검사와 공식 YOLOX 원본 170개 파일 hash 검증을 통과했다. 실제 항공영상의 흙 외형과 일치하는지, 탐지 성능이 개선되는지는 검증하지 않았으며 GPU 검증도 수행하지 못했다.

## 2026-10-20 합성 객체 채도·대비 보정

`copy_paste.object_saturation_range`, `object_contrast_range` 및 같은 이름의 CLI 옵션을 추가했다. 객체의 Mask 내부 픽셀에만 보정하며, 마스크·박스·기하학적 변환과 클래스는 유지한다. 기본값 `[1, 1]`은 기존 결과를 유지한다. 채도는 BT.601 밝기와 원래 색 사이의 보간이고, 대비는 전경 평균 밝기를 기준으로 조절한다. 흙 질감 생성이나 실제 촬영 데이터와의 자동 색 정합은 제공하지 않는다.

Python 3.12 / torch 2.8.0+cu129 / NumPy 1.26.4 / OpenCV 4.11.0.86 / TensorBoard 2.16.2 환경에서 네트워크 차단 상태로 전체 **155 passed, 4 GPU skipped, 0 failed, 31 warnings**, 58.11초를 확인했다. [`reports/pytest-object-appearance-regression.xml`](reports/pytest-object-appearance-regression.xml)에 전체 결과를 기록했다. 새 관련 테스트는 별도 **17 passed**, 9.38초이며 [`reports/pytest-object-appearance.xml`](reports/pytest-object-appearance.xml)에 결과가 있다.

채도 0의 회색 변환과 대비 감소, 입력 이미지·Mask 불변, Mask 밖 픽셀 보존을 확인했다. `online`, `online_patch` 모두 0개/5개 객체 사례에서 기본·항등 보정 결과가 같고, 색 보정의 전용 RNG를 사용할 때 기존과 같은 배경·객체 선택·크기·회전·위치·visible mask·bbox를 유지했다. 전체 이미지 증강을 끈 비교에서 Mask 밖 배경 픽셀도 동일했다. 0/2 Worker 새 Loader에서 재현성을 확인했고 잘못된 수치·범위와 Original 모드 CLI 사용을 거부했다.

공개 Python CLI에서 보정 옵션을 지정한 미리보기 파일과 객체별 적용 metadata를 확인하고, 실제 CPU 1 epoch·유한 loss·검증·checkpoint 저장 및 유효 설정 기록을 확인했다. 기존 학습·평가·Preview·export 회귀 테스트와 공식 YOLOX 원본 170개 파일 hash 검증도 통과했다. GPU 검증과 실제 항공영상에서의 외형 정합·정확도 개선은 수행하지 않았다.

## 2026-10-20 합성 시작 전 전체 디코딩 지연 개선

`copy_paste.validate_assets_on_start: false`를 추가했다. 파일 경로·객체/마스크 대응·클래스 검사는 시작할 때 유지하고, 픽셀 디코딩·자료형·크기·마스크 이진값·전경 검사는 실제 선택된 파일을 합성할 때 수행한다. 옵션 생략 시 기존 전체 초기 검사를 유지하며, 새 `online_patch.json`에는 `false`를 명시했다. 초기 처리 단계와 소요 시간, 실제 설정 경로, 파일 개수와 검증 패치 수를 `[startup]` 로그로 표시한다.

Python 3.12 / torch 2.8.0+cu129 / NumPy 1.26.4 / OpenCV 4.11.0.86 / TensorBoard 2.16.2 환경에서 네트워크 차단 상태로 전체 **138 passed, 4 GPU skipped, 0 failed, 27 warnings**, 50.68초를 확인했다. [`reports/pytest-startup-regression.xml`](reports/pytest-startup-regression.xml)에 결과가 있다. 새 관련 테스트는 별도 **10 passed**, 6.16초이며 [`reports/pytest-startup.xml`](reports/pytest-startup.xml)에 기록했다.

6개 객체/마스크 세트와 배경 4개를 가진 샘플에서 기존 초기화는 디코딩 16회, 새 옵션은 0회였다. 단일 객체를 선택한 합성 호출에서는 필요한 배경·객체·마스크 3개만 읽었다. 같은 입력과 seed에서 기존 검사 방식과 합성 픽셀·박스·metadata가 완전히 같음을 `online`, `online_patch` 모두 확인했다. 이 수치는 디코딩 호출 수이며 실제 내부 서버의 시작 시간 개선 비율을 측정한 결과는 아니다.

비이진 마스크와 손상 배경은 선택 시 오류를 내고, 없는 마스크·미해결 클래스는 초기화 단계에서 계속 거부한다. 설정 값의 boolean 검증, 템플릿 재사용 시 불필요한 디코딩 없음, 실제 공개 CLI의 CPU 학습·checkpoint·held-out 평가 및 시작 로그도 검사했다. 기존 학습·Preview 테스트를 포함한 전체 회귀 검사와 공식 YOLOX 원본 170개 파일의 hash 검증을 통과했다. GPU 속도는 실제 서버에서 확인해야 한다. 폴더 검색·경로 검사·COCO JSON 읽기·검증 patch metadata 구성 비용과 전체 검증 분량은 이 옵션으로 제거되지 않는다.

## 2026-10-19 패치 모드 추가 검증

기존 `original`, `online` 설정은 유지하고 `original_patch`, `online_patch`를 별도 모드로 추가했다. Python 3.12 / torch 2.8.0+cu129 / NumPy 1.26.4 / OpenCV 4.11.0.86 / TensorBoard 2.16.2 환경에서 네트워크 차단 상태로 전체 **128 passed, 4 GPU skipped, 0 failed, 27 warnings**, 47.70초를 확인했다. [`reports/pytest-patch-regression.xml`](reports/pytest-patch-regression.xml)에 전체 결과가 있다.

```bash
.venv-server-check/bin/python scripts/no_network.py \
  .venv-server-check/bin/python -m pytest -q --junitxml=reports/pytest-patch-regression.xml
```

새 패치 테스트는 별도 **15 passed**, 9.30초다([`reports/pytest-patch-modes.xml`](reports/pytest-patch-modes.xml)). 실제 공개 CLI를 별도 작업 디렉터리에서 실행하여 원본 패치 8개/4 iterations, 합성 패치 4개/2 iterations의 CPU 1 epoch를 각각 완료했다. 유한한 양의 탐지 loss, checkpoint 저장, held-out 패치 4개 COCO 평가와 입력 파일 SHA256 불변을 확인했다. 기존 두 모드의 학습·재개·평가 테스트도 전체 실행에 포함했다.

좌표 패턴 이미지에서 실제 crop 픽셀, 오른쪽·아래쪽 마지막 패치, 경계 박스·부분 가시성·최소 크기, 음수 원본 좌표 보정, 빈 패치 정책, 작은 원본의 패딩, annotation과 실제 원본 크기 불일치 오류를 검사했다. category ID 7·29를 사용하는 정답 박스를 다른 모델 입력 크기의 공식 COCO 변환기로 역변환한 뒤 AP=1을 확인하여 패치 평가 좌표와 클래스 대응을 검증했다. 최대 정답 개수 초과 시 조용히 자르지 않고 실패하는 것도 확인했다.

합성은 배경을 원본 픽셀에서 먼저 랜덤 crop한 뒤 객체 5개를 배치한다. 반복 호출의 crop 위치 변화, 동일 seed 재생, 최종 visible mask와 bbox 일치, 증강을 끈 경우 원본 배경 crop과 픽셀 일치, 2 Worker finite loader 재현성, native infinite sampler 배치·no-Mosaic 전환, 두 모드 Preview 저장을 검증했다. 공식 YOLOX 원본 170개 파일은 hash가 동일하다. 새 의존성이나 설치 변경은 없다.

GPU가 없는 환경이므로 실제 CUDA 학습은 검증하지 못했다. 원본 단위 예측 병합 평가도 제공하지 않으며 두 새 모드의 지표는 패치 단위 bbox AP다. 아래는 이전 기능들의 검증 기록이다.

2026-10-14, Debian 13 x86_64 / CPython 3.12.14 / torch 2.8.0+cu129 / torchvision 0.23.0+cu129에서 검증했습니다. GPU가 없는 준비 머신입니다. **CPU 개발·오프라인 설치·두 학습 모드의 진단 실행을 검증했으며, 전체 GPU 학습 호환성은 아직 검증하지 못했습니다.**

## 전체 결과

```bash
source scripts/env.sh
python scripts/no_network.py python -m pytest -q --junitxml=reports/pytest-cpu.xml
```

**51 passed, 4 skipped, 0 failed, 12 warnings**, 11.38초, exit 0. 실행 전에 차단 도구가 IPv4/IPv6 socket 거부와 AF_UNIX 사용 가능 여부를 확인했습니다. 이 규칙은 자식 프로세스에도 상속되므로 multi-worker와 spawn 실행도 같은 차단 아래 검증했습니다. 자세한 case별 결과는 [`reports/pytest-cpu.xml`](reports/pytest-cpu.xml)에 있습니다.

| 범위 | 실제 결과 | 검증한 동작 |
|---|---|---|
| Original | 13 passed | COCO/Mosaic batch, positive detection loss/backward, SGD/EMA, 실제 유한 epoch, 체크포인트·재개, category 검증, 로컬 80-class→1-class 모델 초기화 |
| Online Copy-Paste | 26 passed | 객체/mask 로드, affine/cardinal 회전, 가림·클리핑 후 visible bbox, 매 호출 변화, 비연속 category ID, 저장 없는 합성, worker 재현성, 학습·재개, Preview 파일 정합성 |
| 의존성/평가 | 5 passed | NumPy/OpenCV/torch 배열, pycoco RLE/area/bbox, TensorBoard 기록 재읽기, CPU NMS, 실제 C++/표준 COCO 평가 일치 |
| 오프라인 검증기 | 7 passed | 정확한 hash lock, URL/index/unpinned/unhashed 거부, 누락·변조 wheel 거부 |
| GPU | 4 skipped | GPU 없음을 이유로 CUDA/NMS, FP32/AMP backward, GPU 검증 context의 extension 검사를 skip |

GPU context에서 건너뛴 C++ extension은 별도 dependency 테스트와 실제 빌드에서 **CPU C++ extension으로 빌드·로드·연산을 확인**했습니다. YOLOX에는 이번 workflow에서 별도 custom CUDA extension이 없으며 CUDA NMS는 torchvision의 binary op입니다.

## 실제 CLI 실행

두 모드 모두 CLI wrapper로 커널 네트워크 차단 상태에서 실행했습니다. 단순히 import만 하거나 PID만 확인한 결과가 아닙니다.

| 모드 | epoch 1 | epoch 2 | validation |
|---|---|---|---|
| Original | 8 images / 4 iterations, 평균 loss 11.84237 | 로컬 epoch 1 checkpoint에서 resume, 8 images / 4 iterations, 평균 loss 14.10818 | 샘플 4 images, standard 및 fast 평가, AP=0 |
| Online Copy-Paste | 새 합성 8 images / 4 iterations, 평균 loss 11.27483 | 새 합성 8 images / 4 iterations, 평균 loss 13.42255 | 같은 샘플 4 images, standard 평가, AP=0 |

두 epoch의 실제 backward와 optimizer 업데이트를 완료했습니다. 모델은 사전학습 없이 초기화했으며 작은 샘플로 2 epoch만 실행했습니다. 두 모드의 실제 validation detection 수가 0이고 AP=0인 결과를 그대로 기록했습니다. **정확도나 학습 수렴을 확인한 시험은 아닙니다.** Empty result도 COCO evaluation을 수행하도록 처리했고, 별도의 정답 박스 fixture에서 official postprocess/category conversion을 거친 AP=1 및 standard/fast precision·recall 일치를 확인했습니다.

- [`reports/original-validation.json`](reports/original-validation.json): 두 실제 epoch와 원본 모드 테스트
- [`reports/original-fast-evaluation.json`](reports/original-fast-evaluation.json): 공식 fast extension 평가
- [`reports/copy-paste-validation.json`](reports/copy-paste-validation.json): 합성 모드 실행·테스트·Preview
- [`reports/dependency-tests.json`](reports/dependency-tests.json): C++/표준 평가 parity

실제 출력 checkpoint와 학습 로그는 `outputs/original/sample_nano/`, `outputs/online/sample_nano/`에 있습니다. 이 출력은 로컬 산출물이며 Git에 포함하지 않습니다.

## 체크포인트, worker와 Preview

원본 모드는 Mosaic·HSV·flip을 활성화한 전체 2 epoch와 1 epoch 중단 후 재개의 raw model, optimizer, EMA, scaler, RNG를 비교해 정확히 일치했습니다. Online 모드도 동일한 비교를 통과했습니다. 이 **bit-exact checkpoint 시험은 CPU 진단, 0 worker, 유한 epoch loader** 범위입니다. Native GPU infinite sampler/prefetch cursor의 완전한 복원은 제공하지 않습니다.

Online raw Dataset은 0 worker, 2 worker fork, 2 worker spawn의 새 Loader에서 각각 동일 결과를 재현했습니다. 0/2 worker의 official HSV·flip 증강 loader도 재현됐고 worker별 stream seed는 서로 달랐습니다. worker 수를 변경했을 때 같은 이미지가 생성된다고 보장하지 않습니다. GPU에서의 worker 동작은 GPU 시험에서 다시 확인해야 합니다.

Preview를 명시적으로 호출하여 4개의 이미지, Box overlay, instance ownership mask와 Metadata JSON을 `outputs/preview/`에 생성했습니다. 실제 이미지도 확인했으며, 별도 테스트에서 저장된 mask 픽셀·bbox·area의 일치를 검사했습니다. Dataset의 합성 실행에서는 image write를 금지하고 입력 파일의 SHA256을 전후 비교하여 학습용 합성 이미지가 디스크에 저장되지 않음을 확인했습니다.

## 오프라인 설치와 빌드

51개 wheel, 총 **4,572,267,583 bytes (4.26 GiB)**를 공식 HTTPS 출처의 게시 hash로 확인했습니다. 첫 `.venv` 설치와 별도의 깨끗한 가상환경 설치를 모두 IPv4/IPv6 socket 차단 상태에서 실행했습니다. 두 설치에서 전체 hash/크기/고정 버전, `pip check`, 주요 import, CPU NMS가 통과했습니다. 재검증 venv는 디스크를 확보하기 위해 삭제했습니다.

원본 `setup.py build_ext --inplace`를 실제 실행해 FastCOCOEval을 빌드·로드했습니다. Python sysconfig의 없는 clang++를 GCC/G++ 14.2로 명시한 뒤 **원본 C++14 flag 그대로** 성공했습니다. 같은 명령을 다시 실행해 반복 가능성도 확인했습니다. `scripts/check_upstream.py`는 원본 파일 170개의 hash가 모두 그대로임을 확인했습니다. 원본 application/requirements/loss 코드를 수정하지 않았습니다.

[`reports/offline-install-verification.json`](reports/offline-install-verification.json)이 두 실제 설치의 결과를 기록합니다. 이 결과는 현재 준비 호스트에서 네트워크가 차단된 설치·실행의 증거이며, 실제 폐쇄망 서버의 OS·GPU·드라이버 검증을 대신하지 않습니다.

## 진단된 경고와 남은 검증

12개 경고는 Python 3.12의 multi-threaded fork deprecation, GPU 없는 테스트에서의 pin_memory 경고, 원본 YOLOX legacy AMP FutureWarning, NumPy 2.x에서 pycocotools의 `__array__(copy=...)` deprecation입니다. 해당 실행에서는 모든 기능 검사가 통과했고 hang이나 비정상 수치가 없었습니다. 경고를 숨기기 위해 원본을 일괄 변경하지 않았습니다.

[`reports/gpu_validation.json`](reports/gpu_validation.json)은 **blocked**, 도구 exit 2입니다. GPU 없음이 확인되어 다음은 실행하지 못했습니다.

- NVIDIA GPU 인식·compute capability·driver, CUDA Tensor와 torchvision CUDA NMS
- 단일 GPU FP32/AMP YOLOX 학습 1 iteration·1 epoch 및 native resume/평가
- 실제 GPU·드라이버·최종 OS에서의 CUDA 12.9 호환성
- 실제 SAM3 객체·마스크와 실제 항공영상 크기 분포 적용
- 실제 사용자 COCO validation의 정확도 평가와 제공된 사전학습 가중치

대상 서버에서 [OFFLINE_INSTALL.md](OFFLINE_INSTALL.md)에 따라 다시 설치하고 `scripts/environment_probe.py`, `scripts/validate_gpu.py`를 네트워크 차단 상태로 실행해야 합니다. 최종 서버 정보와 실제 데이터가 제공되기 전에는 **전체 GPU 호환 완료·실제 폐쇄망 적용 완료**라고 주장하지 않습니다.

## 기존 라이브러리 재사용 추가 검증

`requirements.txt`와 기존 Python 선택 경로를 추가한 뒤, 네트워크 차단 상태에서 `tests/test_existing_env.py` **11개 테스트가 통과**했습니다. 기존 패키지 교체 거부, OpenCV 재사용/중복 검사, wheel hash·출처 검사, wheelhouse 없는 실제 설치 불필요 경로를 검증했습니다. 결과는 [`reports/pytest-existing-env.xml`](reports/pytest-existing-env.xml)에 있습니다. 이 수치는 앞의 원래 전체 51개 테스트와 별도의 추가 실행입니다.

프로젝트에 `.venv`가 없는 임시 소스 디렉터리에서도 `YOLOX_PYTHON`으로 기존 인터프리터를 선택해 두 모드의 실제 CPU 1 epoch를 각각 실행했습니다. 각 모드 8 images / 4 iterations 및 COCO 평가가 완료됐습니다. [`reports/existing-python-validation.json`](reports/existing-python-validation.json)에 결과를 저장했습니다.

별도의 가상환경에서 loguru만 없는 상태를 만들고, **loguru wheel 한 개(61,595 bytes)**만 제공했습니다. 점검·설치 계획이 loguru 한 개만 선택했고, 네트워크 차단 상태의 설치 뒤 기존 모든 패키지 버전이 그대로임을 확인했습니다. 전체 wheelhouse 또는 CUDA 패키지 재설치가 필요하지 않았습니다. [`reports/existing-env-missing-only-validation.json`](reports/existing-env-missing-only-validation.json)에 기록했습니다. 실제 내부 서버의 버전 조합과 GPU는 여전히 해당 장비에서 검증해야 합니다.

## NumPy 1.26.4 고정 서버 호환성 추가 검증

서버가 `numpy==1.26.4`를 constraint로 고정한 상황에서 초기 requirements의 NumPy>=2.2가 충돌했다. NumPy 하한을 조정하고 `constraints-numpy126.txt`에 NumPy 1.26.4 / OpenCV headless 4.11.0.86을 고정해 해결했다. 기존 준비 환경의 NumPy 2.5.4나 Python/PyTorch 버전은 변경하지 않았다.

추가 환경은 Python 3.12.14 / torch 2.8.0+cu129 / torchvision 0.23.0+cu129 / NumPy 1.26.4 / OpenCV 4.11.0.86 / pycocotools 2.0.11이다. 공식 SHA256을 확인한 wheel을 네트워크 차단 상태에서 설치했고, `-r requirement.txt -c constraints-numpy126.txt`의 실제 pip dependency 해결을 확인했다. 이어서 **62 passed, 4 GPU skipped, 0 failed, 11 warnings**, 14.06초, exit 0으로 전체 테스트를 완료했다. Original/Online 학습·backward·checkpoint 재개, fork/spawn worker, 표준/C++ COCO 평가까지 포함한다. GPU는 계속 미검증이다.

[`reports/numpy126-compatibility.json`](reports/numpy126-compatibility.json), [`reports/pytest-numpy126.xml`](reports/pytest-numpy126.xml)에 결과와 wheel 출처·SHA256을 저장했다. 내부 저장소에 OpenCV 4.11.0.86이 없으면 해당 로컬 wheel을 별도로 제공해야 한다. 원래 전체 wheelhouse의 NumPy 2.x lock을 이 NumPy 1.x constraint와 함께 사용하지 않는다.

## TensorBoard 2.16.2 고정 서버 추가 검증

NumPy 수정 후에도 서버의 TensorBoard 2.16.2 constraint가 높은 기존 하한(2.20)과 충돌했다. TensorBoard 하한을 2.16.2로 낮추고 보조 패키지의 불필요한 높은 하한도 제거했다. 이 해결은 서버 constraint 삭제나 NumPy/TensorBoard 업그레이드를 요구하지 않는다.

Python 3.12.14 / torch 2.8.0+cu129 / torchvision 0.23.0+cu129 / NumPy 1.26.4 / OpenCV 4.11.0.86 / TensorBoard 2.16.2로 별도 환경을 구성했다. psutil 5.9.8, tqdm 4.66.4, loguru 0.7.2, tabulate 0.9.0, six 1.17.0도 공식 SHA256으로 확인한 로컬 wheel을 설치했다. 네트워크 차단 상태에서 `-r requirement.txt -c constraints-existing-server.txt`의 실제 pip dependency 해결과 전체 **62 passed, 4 GPU skipped, 0 failed**, 14.67초, exit 0을 확인했다. THOP profiling·psutil memory·loguru file logging도 별도로 통과했다.

[`reports/existing-server-compatibility.json`](reports/existing-server-compatibility.json), [`reports/pytest-existing-server.xml`](reports/pytest-existing-server.xml)에 증거를 저장했다. GPU 및 실제 내부 서버의 추가 고정 버전 조합은 해당 서버에서 검증해야 한다. 전체 51-package baseline lock은 별도로 유지했다.

## Python 직접 실행 추가 검증

2026-10-15, 위 TensorBoard 2.16.2 / NumPy 1.26.4 환경에서 `run.py` 진입점을 추가한 뒤 전체 테스트를 실행했다.

```bash
.venv-server-check/bin/python scripts/no_network.py \
  .venv-server-check/bin/python -m pytest -q --junitxml=reports/pytest-python-cli.xml
```

**66 passed, 4 GPU skipped, 0 failed, 11 warnings**, 38.12초, exit 0이다. [`reports/pytest-python-cli.xml`](reports/pytest-python-cli.xml)에 결과를 저장했다.

추가한 4개 CLI 테스트는 환경 스크립트를 source하지 않고, `PYTHONPATH`를 제거하고, `YOLOX_PYTHON`에 존재하지 않는 경로를 넣은 상태에서 실행했다. 공백을 포함한 별도 작업 폴더에서 현재 인터프리터로 샘플 생성 → Original/Online 각각 CPU 1 epoch → 저장된 checkpoint로 COCO 평가 → 합성 미리보기를 완료했다. 각 모드는 학습 이미지 2개와 검증 이미지 1개를 사용했고 유한한 양의 loss, checkpoint, 평가 JSON 및 미리보기 파일을 확인했다. 현재 Python을 그대로 쓰는 환경 점검, 의존성 없이 읽는 최상위 도움말, 잘못된 모드 및 모드와 설정의 불일치 거부도 확인했다.

`python run.py build`도 같은 인터프리터로 공식 `setup.py build_ext --inplace`와 FastCOCOEval 로드를 완료했다. 실제 명령에는 기존 seccomp 네트워크 차단이 적용된다. 공식 YOLOX 원본 170개 파일의 hash는 변하지 않았다. GPU 학습 검증은 여전히 대상 GPU 서버에서 수행해야 한다.

## 객체 5개·크기 조절·COCO 저장 추가 검증

2026-10-15, 같은 NumPy 1.26.4 / TensorBoard 2.16.2 환경에서 전체 **87 passed, 4 GPU skipped, 0 failed, 12 warnings**, 43.11초를 확인했다. [`reports/pytest-synthetic-export.xml`](reports/pytest-synthetic-export.xml)에 결과를 저장했다.

새 테스트 21개는 객체 5개 보장과 bounded retry 실패, 중복 없는 객체 선택, 고정·범위 크기와 RNG 재현성, 잘못된 설정 거부를 확인했다. COCO JSON은 비연속 category ID 7·29를 유지하고, 저장한 PNG 픽셀과 RLE mask를 원래 합성 결과와 비교했다. 각 mask의 실제 픽셀에서 bbox·area를 다시 계산해 annotation과 일치함을 확인했다. 생성한 `train_config.json`으로 실제 CPU 1 epoch 및 기존 held-out validation 평가도 통과했다.

자동 학습 설정이 Original 기본 Mosaic/MixUp으로 객체 수를 늘리지 않도록, 생성 설정에서 Mosaic·MixUp·multiscale을 끄도록 마무리했다. 이 변경 뒤 관련 **21 passed, 0 failed**, 7.81초를 다시 확인했다. [`reports/pytest-synthetic-export-controls.xml`](reports/pytest-synthetic-export-controls.xml)에 최종 관련 테스트 결과가 있다. 일반 Original 설정과 Online 메모리 합성의 동작은 이 자동 생성 설정과 별개다.

실제 Python CLI로 샘플 4장을 16~24픽셀 객체 5개씩 합성하여 표준 COCO annotation 20개와 학습 설정을 생성했다. 이어 생성된 설정으로 Original CPU 2 epochs 학습과 COCO 평가를 완료했다. [`reports/synthetic-export-validation.json`](reports/synthetic-export-validation.json)에 결과를 기록했다. 출력 파일은 로컬 `outputs/synthetic-five-objects/`에 있고 Git에는 포함하지 않는다. export는 기존 검증 이미지·annotation의 경로를 참조하며 검증 데이터를 복제하거나 학습 이미지로 대체하지 않는다. GPU 학습 검증은 수행하지 못했다.

## 객체·마스크 폴더 자동 탐색 추가 검증

2026-10-15, 같은 서버 호환 환경에서 전체 **106 passed, 4 GPU skipped, 0 failed, 13 warnings**, 45.44초를 확인했다. 결과는 [`reports/pytest-folder-inputs.xml`](reports/pytest-folder-inputs.xml)에 있다.

새 테스트 19개는 PNG 객체·마스크를 같은 상대 하위 폴더와 파일명으로 연결하고, 여러 폴더에 같은 `001.png`가 있어도 올바르게 구분함을 확인했다. 확장자·선택적 마스크 접미사, 단일/다중 클래스 매핑, 누락·중복 마스크와 잘못된 경로도 검사했다. Manifest가 없는 폴더만으로 strict 5개 합성, 실제 CPU 1 epoch와 기존 COCO 검증, 다른 작업 폴더에서의 Python CLI 실행을 통과했다. 입력 파일의 SHA256은 변하지 않았고 파일 목록 JSON이나 합성 이미지 저장도 없었다.

별도 실제 CLI 실행에서도 객체 폴더 5개와 대응 PNG 마스크 5개, 배경 4개를 사용해 CPU 1 epoch(8 images / 4 iterations)를 완료했다. 객체 수 5개와 16~24픽셀 크기 설정이 checkpoint에 반영됐고 입력 폴더에 JSON이 생성되지 않았음을 확인했다. [`reports/folder-input-validation.json`](reports/folder-input-validation.json)에 경로와 실행 결과를 기록했다. `configs/online_folders.json`이 폴더 방식 예제이며 기존 Manifest 입력도 유지한다. 공식 YOLOX 원본 170개 파일의 hash는 그대로다. GPU 학습은 이번 준비 머신에서 검증하지 못했다.

## 재사용 클라우드 설정

설치 지침 `install_script`와 시작 지침 `start_skill`을 환경 설정 초안에 저장했습니다. 전체 저장 설치 script도 현재 머신에서 실행하여 exit 0을 확인했습니다. 현재 venv·wheel·소스 준비 상태와 초안 저장은 확인했으며, **환경 게시 및 새 작업에서의 복원은 아직 수행하지 않았습니다.** 환경 설정에서 변경 내용을 검토·저장한 뒤 환경을 게시해야 활성화됩니다. 운영 프로세스의 유지에 의존하는 설정은 없습니다.
