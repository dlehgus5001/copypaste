# YOLOX 환경 호환성 보고서

## 선택한 환경과 검증 범위

선택한 목표는 **CPython 3.12 / PyTorch 2.8.0+cu129 / torchvision 0.23.0+cu129 / Linux x86_64**이다. Python 또는 PyTorch를 낮추지 않았다. GPU 실행 환경은 실제 폐쇄망 서버의 GPU·드라이버를 확인한 뒤 확정해야 한다.

이번 준비 호스트는 Debian 13 x86_64, Python 3.12.14, GCC/G++ 14.2이며 GPU, `nvidia-smi`, `nvcc`가 없다. 따라서 CPU에서 수행한 설치·모델·데이터셋·평가 검증과 **GPU 미검증 항목**을 구분한다. 자세한 환경 및 설치 전후 패키지 목록은 `environment_report.md`와 `reports/`의 환경 조사 자료를 기준으로 한다.

다른 후보에 언급된 PyTorch 2.1/2.2/2.4, CUDA 12.1/12.3/12.5 및 PyTorch 2.6/Python 3.12/CUDA 12.8 조합은 이번 구현·검증 대상이 아니다. 후보의 Python 표기 `3.1`은 임의로 `3.10` 등으로 해석하지 않았다. 해당 후보의 조합이 공식 wheel로 제공되거나 YOLOX에서 정상 동작한다고 주장하지 않는다.

## 공식 소스 및 배포 증거

원본은 [Megvii 공식 YOLOX](https://github.com/Megvii-BaseDetection/YOLOX/tree/6ddff4824372906469a7fae2dc3206c7aa4bbaee)의 commit `6ddff4824372906469a7fae2dc3206c7aa4bbaee`이다. 소스의 버전은 `0.3.0`, 저장소 라이선스는 Apache-2.0이다. `third_party/YOLOX`에 원본 소스를 포함하고 LICENSE 및 저작권 고지를 보존했다. 실행 중 clone을 요구하지 않는다.

원본 파일은 변경하지 않았다. `third_party/UPSTREAM.json`에 출처를, `third_party/SOURCE_SHA256.json`에 원본 파일 해시를 기록했다. `scripts/check_upstream.py`로 원본 무결성을 확인한다. 원본 Python 파일 91개는 Python 3.12 `ast.parse` 문법 검사에서 오류가 없었다. 문법 통과만으로 실행 호환성을 보장하지 않는다.

다음 공식 HTTPS index를 직접 조회했다. 조회 정보와 pycocotools 비교 증거는 [`reports/package-index-evidence.json`](reports/package-index-evidence.json)에 저장했다.

| 패키지 | 확인한 wheel | 공식 index의 SHA256 |
|---|---|---|
| torch | `2.8.0+cu129`, `cp312-cp312-manylinux_2_28_x86_64` | `02c7258e917f3043c978b53acf6f02b818db0d0d85db0e58ae578af333b9b4e2` |
| torchvision | `0.23.0+cu129`, `cp312-cp312-manylinux_2_28_x86_64` | `6226be1b8399ef655a11965ea4975250f7823fc9b200b35deb9eeac350c667a9` |

공식 조회 URL:

- https://download.pytorch.org/whl/cu129/torch/
- https://download.pytorch.org/whl/cu129/torchvision/
- https://pypi.org/pypi/onnx-simplifier/0.4.10/json
- https://pypi.org/pypi/pycocotools/json

Wheel이 존재한다는 사실은 Python ABI에 맞는 배포물이 있다는 증거다. CUDA NMS, 단일 GPU 학습, AMP, GPU resume까지 정상 동작한다는 증거는 아니다.

## 의존성 및 설치 과정

기존 서버 라이브러리 재사용에는 `requirements.txt` / `requirements-build.txt`와 `scripts/check_existing_env.py`를 추가했다. 이 점검은 기존 버전·OpenCV 제공 패키지·framework CUDA build 및 기본 runtime 동작을 확인한다. 모두 있으면 설치하지 않고, 누락 시에는 pip의 실제 오프라인 전이 의존성 계획을 검토하여 기존 패키지를 교체하는 계획을 거부한다. 설치 후보는 기존 manifest의 wheel·SHA256과 일치해야 한다. 기존 환경의 CUDA 라이브러리가 Conda/OS로 제공될 수 있으므로 pip NVIDIA metadata 부재만으로 runtime이 부족하다고 판단하지 않는다. **NumPy 1.26.4 / OpenCV headless 4.11.0.86** 조합은 추가 실행 검증했으며 `constraints-numpy126.txt`로 선택할 수 있다. 그보다 낮은 버전이나 확인하지 않은 조합은 별도 실행 검증이 필요하다.

[`requirements-core.txt`](requirements-core.txt)는 수집 입력이고, 실제 폐쇄망 설치에는 SHA256을 포함해 모든 전이 의존성을 고정한 [`requirements-lock.txt`](requirements-lock.txt)를 사용한다. 고정 버전의 실제 실행 결과는 테스트 결과와 함께 판단해야 한다. 준비 환경과 다른 OS/아키텍처의 wheelhouse로 재사용하지 않는다.

### 필수 학습 패키지와 선택적 내보내기

원본 `requirements.txt`는 `torch>=1.7`, torchvision, NumPy 등의 넓은 범위와 `onnx-simplifier==0.4.10`을 함께 선언한다. ONNX, ONNX Runtime, TensorRT, torch2trt, OpenVINO, MegEngine는 이번 두 학습 모드에 필요하지 않으므로 기본 설치에 포함하지 않았다. SAM3는 미리 생성한 이미지·마스크를 읽는 단계에 필요하지 않다.

PyPI의 `onnx-simplifier==0.4.10` 배포 파일에는 cp36~cp310 wheel만 있고 **cp312 wheel이 없다**. sdist는 있으나 이번 환경에서 빌드를 검증하지 않았다. 원본 requirements를 그대로 설치하면 학습과 무관한 소스 빌드가 발생할 수 있다. 이를 해결하려고 원본 pin을 바꾸는 대신, 외부 패키지 목록과 `PYTHONPATH`를 사용한다. 따라서 원본 YOLOX distribution의 선택적 ONNX metadata까지 설치할 필요가 없다.

서버에서 GUI 창을 열지 않는 학습·전처리·파일 preview를 위해 `opencv-python-headless`를 선택했다. `cv2.imshow`를 사용하는 원본 GUI demo는 별도 디스플레이 환경이 필요하며 이번 기본 환경에 포함하지 않는다. `opencv-python`과 headless 패키지를 함께 설치하지 않는다.

### 원본 setup.py와 빌드 격리

원본 `setup.py`는 Linux에서 PyTorch를 import하고 `FastCOCOEvalOp` extension을 빌드한다. 격리된 빌드 환경에 torch가 없으면 원본 assert에 의해 실패한다. 이 때문에 **로컬 wheel을 먼저 설치한 후 extension을 빌드**한다. 기본 실행은 vendored source를 사용하므로 원본 전체 package의 의존성 설치를 다시 실행하지 않는다.

`scripts/build_yolox.sh`는 원본 `setup.py build_ext --inplace`로 extension만 빌드한다. pip resolver나 build isolation을 사용하지 않고, `MAX_JOBS=2`로 병렬도를 제한한다. 이번 Python 배포판의 sysconfig가 설치되지 않은 clang을 가리켜 초기 빌드가 실패했고, 설치된 GCC/G++를 `CC=gcc`, `CXX=g++`로 선택하여 해결했다. 빌드 스크립트는 이를 기본값으로 사용하며 호출자가 지정한 compiler는 보존한다. 원본을 별도로 pip 설치할 필요가 생기면 torch 설치 후 `--no-deps --no-build-isolation`가 필요한지 검토한다. 이 경우 원본 metadata의 선택적 ONNX 요구도 별도로 처리해야 한다.

### C++ / CUDA extension

FastCOCOEval은 COCO 평가를 빠르게 하는 **CPU C++ extension**이다. 원본 `cocoeval.h`는 pybind11/STL을 포함하고 torch/ATen 헤더 또는 CUDA kernel을 포함하지 않는다. 이 extension에는 nvcc가 필요하지 않지만 C++ 컴파일러, Python/PyTorch 제공 헤더, ninja가 필요하다.

원본 Linux flag는 `-std=c++14`이다. **이번 환경에서 원본 flag와 C++ 소스를 그대로 사용해 GCC/G++ 14.2로 빌드·로드에 성공했다.** 근거는 [`artifacts/yolox-build.log`](artifacts/yolox-build.log)이며, IPv4/IPv6 socket이 차단된 seccomp 프로세스 안에서 수행했다. PyTorch 2.x의 일반 ATen extension에서 C++17을 요구한다는 사실만으로 이 CPU extension의 실패를 단정할 수 없다. 이번 워크플로에 별도 CUDA extension은 없다. CUDA NMS는 설치한 torchvision의 사전 빌드 연산이며 GPU에서의 실행은 미검증이다.

원본 COCOEvaluator는 `COCOeval_opt` import의 ImportError에만 표준 COCOeval로 fallback한다. COCOeval_opt 생성 중 JIT 빌드 실패까지 처리하지는 않는다. 외부 평가 어댑터에서 `standard` 또는 `fast` backend를 명시적으로 선택해 불명확한 fallback을 피한다. CPU 표준 평가가 성공하더라도 C++ extension 로드 성공은 별도로 확인해야 한다.

## API 및 데이터 호환성 분석

| 항목 | 원본 상태와 진단 | 적용한 대응 |
|---|---|---|
| NumPy | 현재 commit의 `yolox/`, `tools/`에는 제거된 `np.float`, `np.int`, `np.bool` 등의 alias가 없다. | 소스 일괄 치환 없음. 실제 증강·데이터셋·Tensor 변환으로 검증한다. |
| OpenCV | 전처리/합성은 cv2의 배열·리사이즈·변환 API를 사용한다. | headless wheel 고정, GUI 기능은 기본 범위 밖. |
| pycocotools | 원본 COCODataset은 메모리 절약을 위해 `info`를 제거한다. 2.0.10의 `loadRes`는 이 key를 무조건 읽어 `KeyError`를 낸다. | 이를 수정한 **2.0.11**을 고정한다. 외부 어댑터의 `setdefault('info', {})`는 이전 버전과의 방어적 호환 조치다. |
| Torchvision NMS | `yolox.utils.boxes`가 torchvision `nms`/`batched_nms`를 사용한다. | torch와 torchvision의 cu129 wheel을 함께 고정한다. CPU NMS와 CUDA NMS는 별도로 시험한다. |
| meshgrid | 원본 compat helper가 torch>=1.10에서 `indexing='ij'`를 지정한다. | 추가 패치 없음. |
| Loss / SimOTA | IOU loss, BCE, positive matching을 유지할 수 있는 구조다. | 원본 모델/head/loss 재사용. non-empty targets의 forward/backward 및 finite gradient를 시험한다. |
| AMP | 원본 Trainer의 `torch.cuda.amp` API는 legacy이며 deprecation warning 대상이다. 제거되었다고 확인한 것은 아니다. | CUDA Trainer의 원본 lifecycle을 유지한다. 경고를 숨기려고 전역 monkeypatch하지 않는다. CPU diagnostic은 AMP를 사용하지 않는다. |
| DataLoader | 원본 Trainer/DataPrefetcher는 CUDA stream을 요구한다. | 실제 CUDA 학습과 명시적 CPU diagnostic을 구분한다. worker seed와 dataset 재현성을 별도 시험한다. |
| Checkpoint | PyTorch 2.6 이후 `torch.load` 기본값은 weights_only=True다. NumPy scalar가 들어간 legacy checkpoint는 거부될 수 있다. | 외부 loader가 `weights_only=True`를 명시하고 로컬 경로만 허용한다. 새 checkpoint는 NumPy RNG를 Tensor·기본 타입으로 저장한다. |

### pycocotools 2.0.10 / 2.0.11의 확인된 차이

공식 cp312 Linux wheel을 각각 다운로드하고 PyPI의 SHA256과 일치함을 확인한 뒤 `pycocotools/coco.py`를 직접 비교했다.

```python
# 2.0.10: YOLOX가 info를 제거하면 KeyError
res.dataset['info'] = copy.deepcopy(self.dataset['info'])

# 2.0.11: info가 없는 COCO dataset도 허용
res.dataset['info'] = copy.deepcopy(self.dataset.get('info', {}))
```

2.0.10의 확인한 wheel SHA256은 `4539d8b29230de042f574012edd0b5227528da083c4f12bbd6488567aabd3920`이고, 선택한 2.0.11 wheel SHA256은 `a82d1c9ed83f75da0b3f244f2a3cf559351a283307bd9b79a4ee2b93ab3231dd`이다. 이 문제는 원본 코드를 수정하지 않고 패키지 선택으로 해결했다.

실제 **NumPy 2.5.4 / OpenCV headless 4.14.0.94 / pycocotools 2.0.11**에서 PNG mask 저장·로드, RLE encode/decode, bbox/area, NumPy↔Torch roundtrip이 통과했다. `mask.decode`에서 NumPy 2의 `__array__(copy=...)` 규약과 관련한 DeprecationWarning 1개를 관찰했다. 마스크, bbox, area의 값 비교는 모두 통과했으므로 실행 실패로 분류하지 않는다. 향후 NumPy 변경에서는 재검증이 필요하다.

서버의 `numpy==1.26.4` constraint와 초기 `numpy>=2.2` 선언이 충돌한 실제 설치 오류를 확인했다. 일반 학습 requirements를 `numpy>=1.26.4,<3`, `opencv-python-headless>=4.11,<5`로 수정하고, NumPy 1.26.4 / OpenCV 4.11.0.86의 선택적 constraint를 제공한다. OpenCV 4.11 공식 metadata는 Python 3.12에서 NumPy>=1.26.0을 요구하므로 이 조합은 dependency 범위에 들어간다. Python 3.12·torch 2.8.0+cu129·torchvision 0.23.0+cu129를 유지한 별도 환경에서 네트워크 차단 설치·pip 해결 및 전체 **62 passed / 4 GPU skipped**를 확인했다. 공식 wheel SHA256과 결과는 `reports/numpy126-compatibility.json`에 기록했다. 전체 51-package lock과 wheelhouse는 기존 NumPy 2.5.4 재현 프로필로 유지한다.

### Checkpoint와 Resume의 범위

기본 tensor state_dict는 weights-only loader로 읽을 수 있다. 그러나 원본 GPU evaluator의 AP는 NumPy scalar일 수 있어 외부 Trainer subclass는 새 checkpoint의 `best_ap`, `curr_ap`를 저장 시 Python float로 정규화한다. 원본은 latest를 평가 전에 저장하므로, 외부 subclass에서 평가 완료 후 latest를 다시 저장해 갱신된 best AP를 보존한다. FP16/FP32 설정도 저장하고 resume 시 동일한 설정을 요구한다. 모델, optimizer, scaler, EMA, RNG 상태를 포함한 새 형식의 epoch 경계 resume를 시험한다. 모델 fine tuning을 위한 초기 checkpoint 로딩과 optimizer/RNG까지 복구하는 resume는 다른 기능이다.

GPU의 원본 infinite sampler 및 prefetch queue 상태를 모두 저장하는 것은 아니므로, GPU resume가 중단 없는 실행과 동일한 샘플 순서나 비트 단위 결과를 재현한다고 보장하지 않는다. CPU finite diagnostic의 정확한 resume 검증과 구분한다.

임의의 legacy checkpoint를 자동으로 `weights_only=False`로 다시 로딩하지 않는다. 출처가 확인된 legacy 파일을 반드시 사용해야 한다면 파일별로 필요한 safe_globals 또는 별도 변환 절차를 검토해야 한다. 새 checkpoint 검증 통과를 모든 외부 checkpoint 호환성으로 확대하지 않는다.

## 변경 내역: 원본 수정 없이 외부 어댑터

| 위치 | 원본과 다른 부분 | 이유 |
|---|---|---|
| `src/offline_yolox/bootstrap.py` | vendored source 경로 활성화, URL이 아닌 로컬 파일 경로 확인 | 실행 중 clone/download 없이 고정 소스를 사용한다. |
| `src/offline_yolox/experiment.py` | JSON 설정, COCO 경로·category 수 검증, seed/worker 설정 | 원본 모델·optimizer·증강을 재사용하면서 폐쇄망 데이터 경로를 명시한다. |
| `src/offline_yolox/train.py` | 원본 CUDA Trainer subclass의 로컬 checkpoint 처리, 별도 CPU diagnostic | 원본 CUDA 훈련 lifecycle을 유지하고 CPU 시험을 GPU 훈련으로 오인하지 않도록 한다. |
| `src/offline_yolox/evaluate.py` | device와 표준/빠른 evaluator 선택, empty detections 처리 | CPU 평가와 선택적 extension을 명시하고 빈 결과도 실제 평가한다. |
| `requirements-*.txt`, `scripts/` | 학습 의존성 분리, wheel/해시 검증, offline install, extension 별도 빌드 | 원본 dependency 선언을 변경하지 않고 재현 가능한 설치를 제공한다. |

Online Copy-Paste는 외부 Dataset이 사전 생성 객체·마스크와 배경을 읽어 합성하고 원본 YOLOX 학습 API로 전달하는 방식이다. 모델 구조, loss, optimizer를 별도 구현하지 않는다. 합성 Dataset의 mask/bbox 정합성, iteration별 변화, worker 재현성은 호환성 분석과 별개로 실제 테스트 결과를 확인해야 한다.

## GPU / CUDA / 서버 적용 조건

- `nvidia-smi`의 CUDA 표시는 **드라이버가 지원하는 CUDA 범위**, `nvcc --version`은 **설치한 Toolkit**, `torch.version.cuda`는 **PyTorch wheel의 CUDA 빌드 버전**이다. 서로 같은 의미가 아니다.
- cu129 wheel과 전이 NVIDIA library wheel을 함께 반입한다. NVIDIA 커널 드라이버와 GPU는 wheel에 포함되지 않는다. 일반 학습 및 이 CPU extension에는 별도 nvcc가 필수는 아니다.
- 선택한 torch/torchvision wheel 자체의 조건은 glibc>=2.28이지만, **전체 설치 묶음의 최소 조건은 glibc>=2.31**이다. `tensorboard_data_server-0.7.2-py3-none-manylinux_2_31_x86_64.whl`을 함께 포함하기 때문이다. 근거는 [`wheelhouse-manifest.json`](wheelhouse-manifest.json)과 [`reports/offline-install-verification.json`](reports/offline-install-verification.json)이다. 전체 묶음은 Linux x86_64, CPython 3.12를 기준으로 한다. Debian 13 호스트는 ABI 범위에 들어가지만 실제 폐쇄망 서버의 OS/CPU가 같다고 가정하지 않는다. Windows/ARM용 설치 묶음으로 재사용할 수 없다.
- GPU의 compute capability가 이 wheel의 포함 아키텍처와 맞는지 실제 서버에서 확인한다. GPU 없는 호스트의 `torch.cuda.get_arch_list()` 빈 결과는 wheel의 GPU 지원 범위를 의미하지 않는다.
- driver 최소 버전 및 GPU 지원은 대상 장비와 공식 NVIDIA 문서로 확인해야 한다. CUDA 12.x minor compatibility 범위와 CUDA 12.9의 모든 기능/PTX 실행 조건을 혼동하지 않는다.
- 이번 준비 환경에서 `pytorch.org`, `docs.pytorch.org`, `docs.nvidia.com` 공식 문서 요청은 HTTPS proxy의 403으로 차단되었다. 아래 문서 URL은 참고 위치이며 본문을 확인했다고 주장하지 않는다. driver 최소 숫자를 임의로 확정하지 않았다.

공식 문서 위치:

- https://pytorch.org/get-started/previous-versions/
- https://docs.pytorch.org/docs/2.8/generated/torch.load.html
- https://docs.pytorch.org/docs/2.8/amp.html
- https://docs.nvidia.com/cuda/archive/12.9.0/cuda-toolkit-release-notes/index.html
- https://docs.nvidia.com/cuda/archive/12.9.0/cuda-installation-guide-linux/index.html
- https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html

## 실행 검증 상태를 기록할 기준

설치·CPU smoke 결과는 별도 테스트 보고서/로그를 기준으로 추가한다. 이 문서의 소스 분석과 wheel 배포 확인을 실행 테스트 통과로 세지 않는다.

`tests/test_dependencies.py`는 seccomp로 IPv4/IPv6 socket을 차단한 상태에서 **5 passed, 1 warning**으로 통과했다. TensorBoard 2.21.0 SummaryWriter의 파일 기록과 EventAccumulator 재읽기, torchvision CPU NMS, pycocotools의 missing-info 처리, 완벽한 detection의 COCO AP를 실제 확인했다. CPU C++ FastCOCOEval의 stats 12개와 precision/recall 배열은 표준 COCOeval과 `atol=1e-12`로 일치했고 AP는 1.0이었다. GPU 연산을 포함한 결과는 아니다.

| 검사 | 이 준비 호스트에서 가능한 범위 |
|---|---|
| offline wheel 설치 / import / CPU NMS | 실제 실행 가능, 설치 및 테스트 로그로 판정 |
| 원본 C++ extension 빌드 및 로드 | **통과**: 원본 C++14 + GCC/G++14.2, seccomp 네트워크 차단 |
| 표준/빠른 COCO evaluator AP 및 precision/recall parity | **통과**: 합성 정답 detection, AP=1.0, 오차 허용 1e-12 |
| COCO/Mosaic Dataset, positive loss/backward | 실제 실행 가능, 테스트 결과로 판정 |
| Original/Copy-Paste CPU diagnostic epoch 및 checkpoint/resume | 실제 실행 가능, GPU 학습과 구분 |
| 합성 mask/bbox, 변화, multi-worker 재현성 | 실제 실행 가능, 테스트 결과로 판정 |
| 실제 사용자 COCO/객체/마스크 데이터 평가 | 데이터가 제공되지 않아 미검증; 합성 fixture 검증과 구분 |
| CUDA Tensor, CUDA NMS, GPU forward/backward | **미검증: GPU 없음** |
| 두 모드의 GPU 1 iteration / 1 epoch / 평가 / resume | **미검증: GPU 없음** |
| GPU FP32 및 AMP, cuDNN kernel, VRAM 사용량 | **미검증: GPU 없음** |
| 실제 폐쇄망 서버 OS/GPU/driver/아키텍처 | **미검증: 대상 서버 미제공** |

폐쇄망 적용 절차는 [`OFFLINE_INSTALL.md`](OFFLINE_INSTALL.md)를 따른다. 네트워크 접근 방지 장치가 있는 테스트와 실제로 네트워크가 차단된 서버에서의 재검증도 구분해서 기록해야 한다.
