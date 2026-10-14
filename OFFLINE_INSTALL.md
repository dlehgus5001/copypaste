# 폐쇄망 설치·반입·문제 해결

## 준비 환경과 적용 범위

이번 wheelhouse는 Debian 13 x86_64 / CPython 3.12에서 수집·검증했습니다. torch 2.8.0+cu129, torchvision 0.23.0+cu129와 모든 전이 의존성을 포함합니다. **묶음 전체는 glibc 2.31 이상**을 요구합니다. PyTorch wheel만의 최소값 2.28과 혼동하지 마세요. GPU·NVIDIA 커널 드라이버·Python 인터프리터 자체는 wheelhouse에 포함되지 않습니다.

최종 서버가 Windows, ARM, Python 3.10/3.11 또는 다른 Linux ABI이면 이 묶음을 재사용하지 않습니다. 대상 OS·아키텍처·Python과 일치하는 온라인 준비 머신에서 별도의 lock/wheelhouse를 수집하고 모든 테스트를 다시 실행해야 합니다. 특정 CUDA 버전을 드라이버나 Toolkit 표기만으로 선택하지 않습니다.

설치 파일과 가상환경의 합계보다 여유 있는 디스크를 확보하세요. wheelhouse 크기는 `wheelhouse-manifest.json`의 `total_bytes`에 기록되며 약 4 GiB입니다. 가상환경까지 포함한 작업 공간에는 최소 15 GiB 여유를 권장합니다. C++ extension 빌드용 GCC/G++, Python development headers, libstdc++가 필요합니다. 준비 호스트에는 GCC/G++ 14.2가 있었습니다. CMake/ninja/setuptools/wheel은 잠금 목록에 포함합니다. Python 또는 컴파일러가 없는 서버의 OS 패키지는 해당 배포판의 서명된 설치 매체/저장소에서 별도 준비해야 합니다.

## 인터넷 가능한 준비 머신에서 수집

현재 준비 환경에는 이미 수집한 wheelhouse가 있습니다. 다시 수집하려면:

```bash
cd /path/to/copypaste
python3.12 scripts/collect_wheels.py --reuse-resolution --jobs 4
python3.12 scripts/verify_bundle.py --wheels-only
python3.12 scripts/check_upstream.py
```

수집만 온라인에서 실행합니다. PyTorch의 checksum이 있는 공식 cu129 wheel을 먼저 선택하고, 나머지는 PyPI에서 torch metadata의 요구 버전에 맞춰 수집합니다. 필요한 HTTPS 목적지는 `download.pytorch.org`, `download-r2.pytorch.org`, `pypi.org`, `files.pythonhosted.org`입니다. 런타임에는 이 목적지가 필요하지 않습니다. 프록시를 우회하거나 TLS·패키지 해시 검증을 끄지 않습니다.

`requirements-core.txt`는 수집 시 선택 범위이고, `requirements-lock.txt`는 실제 선택된 51개 패키지의 버전·SHA256을 고정합니다. `--reuse-resolution`은 `artifacts/resolution.json`이 있으면 이를 사용하고, fresh clone처럼 해당 파일이 없으면 Git에 포함된 `wheelhouse-manifest.json`의 정확한 버전·출처·SHA256을 사용합니다. 현재 검증한 환경을 재현할 때는 위 명령을 사용하세요. 옵션 없이 새로 resolve하면 새로운 라이브러리가 선택될 수 있으므로 새 lock은 다시 검증해야 합니다. wheelhouse에 이전 버전 wheel이 남아 있으면 수집 도구가 중단하므로, 별도 디렉터리로 옮긴 후 원인을 확인합니다.

ONNX/ONNX Runtime/TensorRT/torch2trt/OpenVINO와 SAM3 추론 의존성은 기본 학습 묶음에서 제외합니다. 이 도구들은 이번 두 학습 모드의 필수 요소가 아니며, 별도 export 검증 없이 설치하지 않습니다. 고정한 모든 Python 의존성은 wheel이 있으므로 이 묶음에서는 sdist 빌드가 필요하지 않습니다.

## 반입할 파일

소스 전체와 다음 파일·폴더를 함께 전달합니다.

- `wheelhouse/`: 모든 51개 wheel
- `wheelhouse-manifest.json`: 게시된 SHA256, 크기, 출처와 대상 조건
- `requirements-lock.txt`, `requirements-core.txt`
- `third_party/YOLOX/`, `third_party/UPSTREAM.json`, `third_party/SOURCE_SHA256.json`
- `src/`, `scripts/`, `configs/`, `tests/`, `pyproject.toml`, 보고서와 문서
- 사용할 실제 COCO 데이터·객체/마스크·배경 및 사전학습 가중치: 별도 로컬 파일

wheelhouse와 가상환경은 Git ignore 대상입니다. `git archive`만 반입하면 wheelhouse가 빠질 수 있습니다. `.venv`를 다른 OS/경로로 복사하지 말고 대상에서 다시 설치하세요. `third_party/YOLOX`의 로컬 `*.so`, `build/`, `*.egg-info/`도 대상 컴파일러로 다시 생성합니다.

소스와 wheel을 함께 압축하려면 `python3.12 scripts/package_bundle.py --include-wheels --output artifacts/yolox-offline.tar.gz`를 사용합니다. 압축 파일 옆에 SHA256을 저장하며 venv·데이터·가중치·컴파일 출력을 제외합니다. 검증 보고서를 모두 작성한 후 실행하세요. 소스만 필요하면 `--include-wheels`를 생략합니다.

## 대상 서버에서 네트워크 없는 설치

```bash
cd /path/to/copypaste
python3.12 scripts/verify_bundle.py --wheels-only
PYTHON_BIN=python3.12 bash scripts/install_offline.sh
source scripts/env.sh
bash scripts/build_yolox.sh
python scripts/check_upstream.py
python scripts/environment_probe.py --output reports/environment-target.json
```

설치 도구는 다음 방식으로 PyTorch를 포함한 의존성을 먼저 설치합니다.

```bash
.venv/bin/python -m pip --isolated install --no-index --find-links=./wheelhouse \
  --require-hashes --only-binary=:all: --no-cache-dir -r requirements-lock.txt
```

`--no-index`는 온라인 index를 사용하지 않습니다. 잠금 파일에 direct URL이 있으면 검증 도구가 거부합니다. wheelhouse의 파일 집합·크기·SHA256을 설치 전에 검사하고, 설치 후 버전·`pip check`·import·CPU NMS를 검사합니다. 기존 가상환경이 있으면 명확한 경로의 Python 3.12 venv만 재사용합니다.

YOLOX는 vendored source를 `.pth`로 연결하여 실행합니다. 원본 `requirements.txt`의 ONNX metadata를 설치하지 않습니다. 이후 `setup.py build_ext --inplace`로 FastCOCOEval만 별도 빌드합니다. 이미 설치한 PyTorch와 빌드 도구를 사용하므로 격리 빌드나 추가 다운로드가 없습니다. 원본 YOLOX를 별도로 pip 빌드하는 경우에는 PyTorch 설치 순서와 `--no-deps --no-build-isolation` 필요성을 확인하세요. 이 프로젝트의 기본 절차에는 별도 pip YOLOX 설치가 없습니다.

## 차단 상태 검증과 학습

Linux에 `libseccomp.so.2`가 있으면 다음 명령은 부모·자식 프로세스에서 IPv4/IPv6 socket 생성을 kernel 수준에서 차단합니다. Python의 HTTP 함수만 바꾸는 방식이 아닙니다. DataLoader IPC용 AF_UNIX는 허용합니다. 이 테스트 장치는 `libseccomp`가 없거나 규칙을 적용할 수 없으면 실패하며, 차단 없이 진행하지 않습니다. 다른 서버에서는 OS 방화벽/네트워크 격리를 사용하세요.

제공한 `train_original.sh`, `train_online.sh`, `evaluate.sh`도 같은 차단 도구를 사용하므로 `libseccomp.so.2`가 필요합니다. 이를 제공하지 않는 서버는 OS의 네트워크 격리를 적용하고 `python -m offline_yolox.train` / `python -m offline_yolox.evaluate` 진입점을 직접 실행하세요. 파이썬 wheel은 libseccomp·NVIDIA driver·OS build tool 설치를 대신하지 않습니다.

```bash
python3.12 scripts/no_network.py bash scripts/install_offline.sh .venv-offline-check
source scripts/env.sh
python scripts/no_network.py bash scripts/build_yolox.sh
python scripts/no_network.py python -m pytest -q
python scripts/no_network.py python scripts/validate_gpu.py
```

동일 venv에 설치 명령을 다시 실행할 수 있습니다. 별도 venv 설치는 추가 디스크 공간이 필요합니다. `validate_gpu.py`는 CUDA tensor/NMS, FP32/AMP forward/backward, 두 학습 모드의 epoch·resume·평가를 검사합니다. GPU가 없으면 exit 2 및 `blocked` 보고서를 남깁니다. 샘플 CPU 진단 성공만으로 GPU 적용을 승인하지 마세요.

사전학습 가중치를 다운로드하는 코드 경로는 사용하지 않습니다. `--checkpoint /local/model.pth`로 직접 지정하며, 실제 제공되지 않은 가중치는 미검증입니다. Runtime은 로컬 이미지·마스크·annotation·가중치만 읽고 외부 API·온라인 인증이 필요하지 않습니다.

## 문제 해결

| 증상 | 확인과 조치 |
|---|---|
| `No matching distribution` / wheel tag 불일치 | OS·Python 3.12·x86_64·glibc와 manifest를 비교합니다. 버전을 무작정 바꾸지 말고 대상에 맞춰 새 wheelhouse를 준비합니다. |
| 누락 wheel / SHA256 불일치 | 검증 도구가 지정한 파일을 원래 출처·원래 hash로 다시 수집합니다. hash나 TLS 검증을 끄지 않습니다. |
| `CUDA is unavailable` | NVIDIA GPU·커널 드라이버·컨테이너 GPU 노출을 확인합니다. cu129 wheel 설치는 GPU 제공을 대신하지 않습니다. |
| `nvidia-smi`와 `torch.version.cuda` 값이 다름 | 전자는 드라이버 지원 범위, 후자는 wheel build입니다. `nvcc`는 별도 Toolkit입니다. |
| `no kernel image` / CUDA NMS 실패 | GPU compute capability, wheel 아키텍처, torch/vision 같은 cu129 조합 및 driver를 확인하고 GPU 테스트 로그를 기록합니다. |
| compiler/header/ninja 오류 | 대상 OS의 GCC/G++, Python headers, 빌드 도구를 확인합니다. `MAX_JOBS=2`로 재빌드하고 이전 서버의 `.so`를 재사용하지 않습니다. |
| COCO `info` KeyError | 잠금된 pycocotools 2.0.11인지 확인합니다. 2.0.10에서 확인된 문제입니다. |
| category 수·ID 불일치 | train/val taxonomy와 객체 manifest의 category_id를 같은 COCO ID로 맞춥니다. |
| `weights_only` checkpoint 오류 | 로컬 파일의 실제 형식과 출처를 확인합니다. 자동 unrestricted 로딩은 하지 않으며 legacy 변환은 별도 작업입니다. |
| Resume 설정 불일치 | 동일 모델·데이터·학습 schedule로 재개합니다. 다른 설정에는 `--checkpoint`로 모델만 초기화합니다. |
| DataLoader 메모리/worker 문제 | `num_workers`·batch·객체 개수를 조정합니다. 같은 seed·worker 수로 재현성을 검사합니다. |

최종 폐쇄망 서버에서 `environment_probe.py` 결과, `validate_gpu.py` 로그 및 실제 검증 데이터 평가 결과를 보관하세요. 현재 클라우드의 차단 테스트와 실제 서버의 드라이버·데이터 검증은 서로 별개의 증거입니다.
