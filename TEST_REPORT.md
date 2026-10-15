# 실제 실행 테스트 결과

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
