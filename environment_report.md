# 환경 조사 및 설치 결과

조사 대상은 현재 Codex 클라우드 준비 머신입니다. 실제 폐쇄망 서버의 OS·GPU·드라이버 정보는 제공되지 않았습니다. 따라서 아래 결과를 대상 서버의 결과로 확대하지 않습니다. 사용자 후보 중 5번 Python 3.12 / PyTorch 2.8 / CUDA 12.9를 유지했습니다.

## 현재 머신

| 항목 | 설치 전 / 확인 결과 | 준비 후 |
|---|---|---|
| OS | Debian GNU/Linux 13.6 (trixie) | 동일 |
| Kernel / libc | Linux 6.18.44, glibc 2.41 | 동일 |
| CPU 아키텍처 | x86_64, 보이는 CPU 5개, cgroup quota 4 CPU | 학습 진단 thread 2, 빌드 MAX_JOBS 2 |
| 메모리 / 디스크 | cgroup 32 GiB, workspace 약 31.45 GiB | venv 약 7.5 GiB, wheelhouse 4.26 GiB |
| Python | CPython 3.12.14 | 별도 `.venv`, Python 3.12.14 |
| pip / setuptools / wheel | 26.2.1 / 84.0.0 / 0.48.0 | 같은 고정 버전 |
| NVIDIA GPU / compute capability | GPU 없음, 조사 불가 | GPU 학습 미검증 |
| NVIDIA Driver / nvidia-smi | 명령 및 driver 없음 | 동일 |
| CUDA Toolkit / nvcc | 없음 | 설치하지 않음: 기본 학습 wheel 및 YOLOX CPU C++ extension에는 불필요 |
| PyTorch | 미설치 | 2.8.0+cu129 |
| PyTorch CUDA build | 해당 없음 | `torch.version.cuda == '12.9'` |
| torchvision | 미설치 | 0.23.0+cu129 |
| cuDNN | 해당 없음 | runtime version 91002 (9.10.2), GPU kernel 실행은 미검증 |
| GCC / G++ | 14.2.0 | 동일, C++ extension 실제 빌드 성공 |
| CMake / ninja | 미설치 | 4.4.4 / 1.13.2 |
| YOLOX | 미설치, 프로젝트는 빈 저장소 | 공식 0.3.0 소스 commit `6ddff4824372906469a7fae2dc3206c7aa4bbaee` 포함 |
| 환경변수 / 인증 | 실행에 필요한 별도 secret 없음 | 외부 API·온라인 인증 없이 실행 |

`nvidia-smi`의 CUDA 표시는 드라이버가 지원하는 범위, `nvcc --version`은 Toolkit, `torch.version.cuda`는 PyTorch wheel build입니다. 이번 머신에서는 앞의 두 명령이 없으며, 마지막 값만 12.9입니다. CUDA runtime wheel 설치가 GPU·커널 driver·nvcc를 제공한다고 해석하지 않습니다.

PyTorch binary가 기록하는 CUDA architecture flags는 `sm_70 sm_75 sm_80 sm_86 sm_90 sm_100 sm_120 compute_120`입니다. 이는 binary metadata이며 실제 장착 GPU에서 연산을 확인한 결과가 아닙니다. 실제 장비의 compute capability·드라이버·torchvision CUDA NMS는 GPU 검증 도구로 확인해야 합니다.

## 패키지 선택과 재현 조건

| 주요 패키지 | 고정 버전 |
|---|---|
| torch / torchvision | 2.8.0+cu129 / 0.23.0+cu129 |
| numpy | 2.5.4 |
| opencv-python-headless | 4.14.0.94 |
| pycocotools | 2.0.11 |
| loguru / tqdm | 0.7.3 / 4.70.1 |
| tensorboard | 2.21.0 |
| thop | 0.1.1-2209072238 |
| ninja / cmake | 1.13.2 / 4.4.4 |
| setuptools / wheel / pip | 84.0.0 / 0.48.0 / 26.2.1 |

全51개 패키지의 정확한 배포 버전·SHA256은 [`requirements-lock.txt`](requirements-lock.txt), 출처·wheel 이름·크기는 [`wheelhouse-manifest.json`](wheelhouse-manifest.json)에 기록했습니다. 위 표는 사람이 읽는 주요 버전 요약이며 잠금 파일이 설치 기준입니다. 설치 전 전체 패키지 목록은 [`reports/initial-packages.json`](reports/initial-packages.json), 설치 전후 시스템·패키지 조사는 [`reports/environment-before.json`](reports/environment-before.json), [`reports/environment-after.json`](reports/environment-after.json)에 있습니다.

torch/torchvision은 공식 cu129 cp312 wheel, 나머지는 PyPI의 호환 wheel을 선택했습니다. 최종 선택을 실제 import·NumPy/OpenCV/pycoco 배열 처리·DataLoader·YOLOX loss/backward·COCO 평가로 검증했습니다. NVIDIA library는 torch metadata가 요구한 정확한 전이 버전이며 **설치 및 hash는 확인했지만 GPU 실행 호환성은 미검증**입니다. ONNX Export와 TensorRT 등은 제외했습니다.

이 wheelhouse 전체의 최소 glibc는 2.31입니다. torch만의 최소값 2.28과 달리 TensorBoard data server wheel이 2.31을 요구합니다. 원본 C++ extension은 대상 OS에서 다시 빌드해야 하며, 현재 생성된 Debian 13 `.so`를 다른 서버에 그대로 전달하지 않습니다.

## 실제 발견하고 해결한 설정 문제

Python 배포의 `sysconfig`는 `CC=clang`, `CXX=clang++`를 기록하지만 이 머신에는 clang++ 실행 파일이 없습니다. 첫 공식 `setup.py build_ext --inplace`는 `No such file or directory: clang++`로 실패했습니다. `scripts/build_yolox.sh`에서 기존 사용자 선택이 없을 때 `CC=gcc`, `CXX=g++`를 명시하고 venv의 ninja 경로를 활성화하여 해결했습니다. 원본 YOLOX 또는 C++14 compile flag는 수정하지 않았습니다. 수정 후 실제 빌드·로드와 재실행에 성공했습니다.

패키지 수집 초기에는 PyTorch index가 광고하는 `pypi.nvidia.com` 목적지가 제한되어 실패했습니다. 공식 framework wheel URL을 checksum으로 검증하고 동일한 NVIDIA package versions를 PyPI에서 수집하는 경로로 해결했습니다. 네트워크 allowlist를 넓히거나 TLS·서명·체크섬 검증을 끄지 않았습니다.

현재의 검증 결과 및 GPU·실데이터 미검증 목록은 [`TEST_REPORT.md`](TEST_REPORT.md)에 있습니다. 설치·검증 완료 상태는 현재 클라우드에 대한 것이며, 환경 설정 초안 저장·게시·새 작업 복원 여부는 별도입니다.
