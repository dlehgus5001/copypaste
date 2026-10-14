# 폐쇄망 YOLOX: Original / Online Copy-Paste

공식 Megvii YOLOX를 포함한 **Python 3.12 / PyTorch 2.8.0+cu129 / torchvision 0.23.0+cu129** 학습 프로젝트입니다. 두 모드는 모델, Loss, Optimizer 및 COCO 평가를 공유합니다. Version 2는 이미 추출된 객체 이미지·마스크를 읽으며 SAM3 실행이나 인증이 필요하지 않습니다.

현재 검증 환경에는 GPU가 없습니다. CPU 진단 학습의 성공은 GPU 학습 성공을 의미하지 않습니다. 실제 결과는 [TEST_REPORT.md](TEST_REPORT.md), 시스템 정보는 [environment_report.md](environment_report.md), 분석은 [compatibility_report.md](compatibility_report.md)를 확인하세요.

## 설치와 첫 실행

프로젝트 폴더에서 사용할 Python 3.12 환경을 활성화하고, 다음 명령부터 시작합니다. 폐쇄망에서는 pip가 내부 저장소 또는 로컬 wheel만 사용하도록 설정한 상태에서 실행하세요.

```bash
pip install -r requirement.txt
```

`requirement.txt`는 `requirements.txt`를 불러오는 진입 파일이므로 두 이름 모두 같은 학습 의존성을 설치합니다. 조건을 만족하는 기존 패키지는 재사용합니다. 기존 버전을 변경하지 않거나 설치된 OpenCV 제공 패키지를 구분해야 한다면 아래 **기존 환경 사용**의 점검 절차를 사용하세요.

외부 저장소에 접근하지 않고 로컬 wheel에서만 설치하려면 다음과 같이 실행합니다. 이미 필요한 패키지가 모두 있으면 추가 wheel은 필요하지 않습니다.

```bash
pip install --no-index --find-links=./wheelhouse -r requirement.txt
```

설치 후 같은 Python으로 환경을 연결하고, 필요한 C++ 평가 확장을 빌드한 뒤 샘플을 실행합니다.

```bash
export YOLOX_PYTHON="$(python -c 'import sys; print(sys.executable)')"
source scripts/env.sh
python scripts/check_existing_env.py --include-build --output artifacts/installed-env
# 빌드 도구가 부족할 때만 로컬 wheel로 설치
# pip install --no-index --find-links=./wheelhouse -r requirements-build.txt
bash scripts/build_yolox.sh
python scripts/make_sample_data.py  # 기존 샘플이 있으면 생략
bash scripts/train_original.sh --device cuda
bash scripts/train_online.sh --device cuda
```

GPU가 없는 머신은 두 학습 명령의 `cuda`를 `cpu`로 바꾸어 진단합니다. 별도 venv와 전체 wheelhouse는 아래처럼 새 환경을 완전히 재현할 때 사용하는 경로입니다.

## 새 환경의 오프라인 설치 묶음 준비

GitHub에서 소스를 받는 온라인 준비 머신에서는 아래 순서로 **이미 검증한 버전의 wheel**을 수집합니다. Git에는 대용량 wheelhouse·가상환경·학습 결과가 포함되지 않습니다.

```bash
git clone https://github.com/dlehgus5001/copypaste.git
cd copypaste
python3.12 scripts/collect_wheels.py --reuse-resolution --jobs 4
```

폐쇄망 서버에서는 clone이나 위 수집 명령을 실행하지 않습니다. 소스와 수집한 wheelhouse를 함께 반입하거나, 미리 준비한 `yolox-offline.tar.gz`를 풀어서 사용합니다. 압축 파일은 Git 저장소와 별도로 제공합니다.

`wheelhouse/`, `wheelhouse-manifest.json`, `requirements-lock.txt`, `third_party/`를 함께 반입합니다. Linux x86_64, Python 3.12, glibc 2.31 이상에 맞춘 묶음입니다. 최종 서버의 OS·GPU·드라이버는 별도로 확인해야 합니다. 자세한 준비·설치·문제 해결은 [OFFLINE_INSTALL.md](OFFLINE_INSTALL.md)에 있습니다.

```bash
cd /path/to/copypaste  # 반입하거나 clone한 프로젝트 경로
bash scripts/install_offline.sh
source scripts/env.sh
bash scripts/build_yolox.sh
python scripts/check_upstream.py
python scripts/make_sample_data.py
```

샘플 생성 도구는 기존 데이터를 덮어쓰지 않습니다. 이미 `data/sample/sample_manifest.json`이 있으면 생성 명령을 생략하세요. `--force`는 직접 생성한 샘플을 다시 만들 때만 사용합니다. 샘플은 단순한 차량 모양과 배경이며, 실제 SAM3 결과나 실제 항공영상이 아닙니다.

## 기존 환경 사용

이미 사용하는 Python 3.12의 Conda/venv를 활성화한 뒤 실행합니다. `requirements.txt`는 학습에 필요한 직접 의존성 11개, `requirements-build.txt`는 선택적 C++ 빌드 도구입니다. 전체 전이 의존성 51개를 고정한 `requirements-lock.txt`와 용도가 다릅니다.

```bash
cd /path/to/copypaste
export YOLOX_PYTHON="$(python -c 'import sys; print(sys.executable)')"
"$YOLOX_PYTHON" scripts/check_existing_env.py --plan --include-build \
  --output artifacts/server-env
```

이 명령은 패키지를 설치하지 않습니다. 현재 버전, torch/torchvision의 CUDA 12.9 build, CPU NMS와 주요 배열·mask·TensorBoard 동작을 확인합니다. 조건을 모두 만족하면 `no installation needed`를 기록하며 **wheelhouse와 pip 실행이 필요하지 않습니다**. 기존 OpenCV는 python/headless/contrib 중 하나를 재사용합니다. 다른 버전의 패키지로 교체해야 하는 경우에는 충돌을 보고하고 설치 계획을 만들지 않습니다.

부족한 패키지가 있으면 로컬 wheelhouse만 사용해 전이 의존성까지 계획하고 `artifacts/server-env/requirements-missing.txt`에 정확한 버전·SHA256을 기록합니다. 필요한 wheel만 있어도 되며 전체 4.3 GiB를 요구하지 않습니다. 누락된 wheel, 기존 의존성 교체, 중복 OpenCV, 확인되지 않은 wheel은 설치 계획을 차단합니다. 생성된 파일에 실제 패키지 행이 있을 때만 같은 Python으로 설치합니다.

```bash
PIP_CONFIG_FILE=/dev/null "$YOLOX_PYTHON" scripts/no_network.py "$YOLOX_PYTHON" -m pip \
  --isolated install --no-index --find-links=./wheelhouse \
  --no-deps --require-hashes -r artifacts/server-env/requirements-missing.txt
```

그런 다음 동일 환경을 연결합니다. 프로젝트의 `.venv`를 새로 만들거나 강제로 선택하지 않습니다.

```bash
source scripts/env.sh
bash scripts/build_yolox.sh
bash scripts/train_original.sh --device cuda
bash scripts/train_online.sh --device cuda
```

이 모드는 기존 버전을 자동으로 높이거나 낮추지 않습니다. 현재 requirements 범위를 벗어난 기존 NumPy/OpenCV/TensorBoard 등은 충돌로 표시되며, 해당 버전을 유지할 수 있는지는 대상 서버에서 별도 호환성 검증 후 범위를 조정해야 합니다. 버전 범위 안의 모든 조합을 학습 검증했다고 주장하지 않습니다. `--output`은 빈 폴더를 지정하고, 설치 후 재점검할 때는 다른 폴더를 사용하세요. GPU 학습 검증은 `scripts/validate_gpu.py`로 별도 수행합니다.

`pip install -r requirements.txt`도 조건을 만족하는 기존 패키지는 재설치하지 않습니다. 다만 조건이 맞지 않으면 pip가 교체할 수 있고 OpenCV 제공 패키지를 자동 구분하지 않으므로, 기존 환경을 유지하려면 위 점검·누락 목록 절차를 권장합니다.

GPU가 있는 서버에서 두 학습 모드를 실행합니다.

```bash
bash scripts/train_original.sh --config configs/original.json --device cuda
bash scripts/train_online.sh --config configs/online.json --device cuda
python -m offline_yolox.preview --config configs/online.json --output outputs/preview --count 4
```

GPU가 없을 때는 CPU 진단을 명시합니다. 이 경로는 공식 모델·Loss·SGD·증강을 사용하지만 CUDA 전용 공식 Trainer를 CPU에서 실행하는 경로는 아닙니다.

```bash
bash scripts/train_original.sh --config configs/original.json --device cpu --stop-after-epoch 1
bash scripts/train_original.sh --config configs/original.json --device cpu \
  --resume outputs/original/sample_nano/latest_ckpt.pth
bash scripts/train_online.sh --config configs/online.json --device cpu
```

샘플 설정은 작은 `nano`, 128×128, batch 2, 2 epochs입니다. 실제 학습에는 별도 설정 파일에서 `preset`, `input_size`, `test_size`, `batch_size`, `max_epoch`, `basic_lr_per_img`, 증강과 schedule을 지정하세요. 기본 학습률은 **이미지당 학습률**이며, YOLOX가 batch size를 곱합니다. `warmup_epochs + no_aug_epochs < max_epoch`를 유지해야 합니다. `preset`은 nano/tiny/s/m/l/x를 지원합니다. 멀티 GPU/DDP는 이 프로젝트의 검증 범위에 포함하지 않습니다.

## 기존 COCO 데이터와 로컬 가중치

`configs/original.json`을 복사하여 데이터 경로를 수정합니다. 상대 경로는 프로젝트 루트를 기준으로 하며 절대 경로도 허용합니다.

```text
DATA_ROOT/
  annotations/instances_train2017.json
  annotations/instances_val2017.json
  train2017/...
  val2017/...
```

이미지 폴더·annotation 파일명은 `train_name`, `val_name`, `train_ann`, `val_ann`으로 변경할 수 있습니다. `num_classes`는 COCO categories 수와 같아야 하며 train/val category ID와 이름은 같은 매핑이어야 합니다. 실제 validation 데이터는 두 모드 모두 같은 COCO 형식으로 제공합니다.

사전학습 가중치는 로컬 파일을 직접 지정합니다. 지정하지 않으면 초기화된 모델로 시작하며 자동 다운로드하지 않습니다.

```bash
bash scripts/train_original.sh --config configs/original.json --device cuda --checkpoint /data/weights/model.pth
bash scripts/train_original.sh --config configs/original.json --device cuda --fp16
bash scripts/evaluate.sh --config configs/original.json --device cuda \
  --checkpoint outputs/original/sample_nano/latest_ckpt.pth --evaluator fast \
  --output outputs/original/evaluation.json
```

`--checkpoint`는 공식 YOLOX의 shape matching으로 호환되는 텐서를 복원합니다. 예를 들어 80-class checkpoint에서 1-class 모델을 초기화할 때 분류 predictor는 초기값을 유지하고 해당 layer를 로그에 기록합니다. `--resume`는 동일한 설정으로 model/optimizer/scaler/EMA/RNG를 엄격하게 복구하는 epoch 경계 재개입니다. CPU 진단과 CUDA Trainer의 checkpoint를 서로 resume하지 않습니다. 평가에서는 저장된 EMA를 우선 사용하며 `--raw-model`로 학습 가중치를 선택합니다. 로딩에는 `weights_only=True`를 사용합니다. 외부 legacy checkpoint가 거부되면 자동으로 unrestricted pickle 로딩을 재시도하지 않으며, 별도 변환이 필요할 수 있습니다. 실제 사전학습 가중치는 이번 작업에서 제공되지 않았습니다.

## 온라인 합성 데이터

데이터 계약과 모든 합성 옵션은 [configs/ONLINE_DATA.md](configs/ONLINE_DATA.md)를 따릅니다. 핵심 입력은 다음과 같습니다.

```json
[
  {"image": "objects/car.png", "mask": "objects/car_mask.png", "category_id": 1}
]
```

객체 manifest와 배경 manifest의 상대 경로 기준은 `asset_root`입니다. 배경 manifest는 이미지 경로 문자열의 배열입니다. COCO category ID를 사용하고 내부 YOLOX label index로 변환합니다. 학습 Dataset은 결과 이미지를 메모리에서 생성하여 바로 DataLoader에 전달합니다. Preview 도구를 명시적으로 호출했을 때만 결과 이미지·bbox·mask를 저장합니다.

실제 항공영상의 객체 크기 관측값으로 크기 분포를 보정해야 합니다. 샘플 설정의 픽셀 크기와 증강값은 기능 검증용이며 실제 크기 분포를 추정한 값이 아닙니다.

## 검증

```bash
# IPv4/IPv6 소켓을 kernel seccomp로 차단한 CPU 테스트. AF_UNIX IPC는 허용합니다.
python scripts/no_network.py python -m pytest -q --junitxml=reports/pytest-cpu.xml

# 실제 GPU: CUDA/NMS/FP32/AMP, 두 모드의 native epoch/resume/COCO 평가.
python scripts/no_network.py python scripts/validate_gpu.py

# 대상 서버 조사
python scripts/environment_probe.py --output reports/environment-target.json
```

GPU 검증 도구는 GPU가 없으면 `blocked` 결과를 저장하고 exit 2로 종료합니다. CPU 테스트의 GPU 항목은 skipped로 표시합니다. GPU 검증은 샘플 설정이 기본이며, 실제 데이터 설정을 `--original-config`, `--online-config`로 전달해야 실데이터를 검사합니다. Native CUDA prefetcher의 대기 batch와 sampler cursor는 checkpoint에 저장하지 않으므로 GPU resume의 합성·증강 순서가 bit-exact하게 같다고 보장하지 않습니다.

원본 출처·라이선스는 `third_party/UPSTREAM.json`, 원본 파일 SHA256은 `third_party/SOURCE_SHA256.json`에 있습니다. 원본 YOLOX 파일은 수정하지 않고 `src/offline_yolox/`에 실행 어댑터를 구현했습니다. wheel과 로컬 가상환경·출력은 Git에 포함하지 않습니다. 폐쇄망 반입 시에는 wheelhouse를 소스와 함께 별도로 전달해야 합니다.
