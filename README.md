# 폐쇄망 YOLOX: Original / Online Copy-Paste / Patch

공식 Megvii YOLOX를 포함한 **Python 3.12 / PyTorch 2.8.0+cu129 / torchvision 0.23.0+cu129** 학습 프로젝트입니다. `original`, `online`, `original_patch`, `online_patch`는 모델, Loss, Optimizer 및 COCO 평가를 공유합니다. 합성 모드는 이미 추출된 객체 이미지·마스크를 읽으며 SAM3 실행이나 인증이 필요하지 않습니다.

현재 검증 환경에는 GPU가 없습니다. CPU 진단 학습의 성공은 GPU 학습 성공을 의미하지 않습니다. 실제 결과는 [TEST_REPORT.md](TEST_REPORT.md), 시스템 정보는 [environment_report.md](environment_report.md), 분석은 [compatibility_report.md](compatibility_report.md)를 확인하세요.

## 설치와 첫 실행

프로젝트 폴더에서 사용할 Python 3.12 환경을 활성화하고, 다음 명령부터 시작합니다. 폐쇄망에서는 pip가 내부 저장소 또는 로컬 wheel만 사용하도록 설정한 상태에서 실행하세요.

```bash
pip install -r requirement.txt
```

`requirement.txt`는 `requirements.txt`를 불러오는 진입 파일이므로 두 이름 모두 같은 학습 의존성을 설치합니다. 조건을 만족하는 기존 패키지는 재사용합니다. 기존 버전을 변경하지 않거나 설치된 OpenCV 제공 패키지를 구분해야 한다면 아래 **기존 환경 사용**의 점검 절차를 사용하세요.

서버에서 `numpy==1.26.4`를 고정한 경우에는 다음 호환 프로필을 사용합니다. NumPy는 그대로 유지하고 NumPy 1.x를 지원하는 OpenCV headless 4.11.0.86을 선택합니다. OpenCV 4.12 이상은 NumPy 2.x를 요구할 수 있으므로 버전을 함께 맞춰야 합니다.

```bash
pip install -r requirement.txt -c constraints-numpy126.txt
```

폐쇄망의 내부 저장소에 이 OpenCV 버전이 없으면 해당 wheel을 반입한 뒤 같은 명령에 `--no-index --find-links=./wheelhouse`를 추가하세요. Python 3.12·PyTorch 2.8·CUDA build 12.9 기준은 유지합니다.

서버에서 **NumPy 1.26.4와 TensorBoard 2.16.2를 함께 고정**한 경우에는 두 조건을 포함한 프로필을 사용합니다. 서버의 constraint를 삭제하거나 TensorBoard를 올릴 필요는 없습니다.

```bash
pip install -r requirement.txt -c constraints-existing-server.txt
```

학습용 requirements에서 TensorBoard의 하한은 2.16.2로 조정했고, loguru/tqdm/tabulate/psutil 등의 불필요한 높은 하한은 제거했습니다. 기존 고정 버전을 재사용하되 실제 학습과 로그 기능은 해당 환경에서 점검해야 합니다. 정확한 51-package 재현용 lock은 별도이며, 기존 서버 고정 버전과 혼합해서 적용하지 않습니다.

외부 저장소에 접근하지 않고 로컬 wheel에서만 설치하려면 다음과 같이 실행합니다. 이미 필요한 패키지가 모두 있으면 추가 wheel은 필요하지 않습니다.

```bash
pip install --no-index --find-links=./wheelhouse -r requirement.txt
```

설치 후 같은 Python으로 환경 점검, C++ 평가 확장 빌드, 샘플 생성과 학습을 실행합니다. `run.py`는 실행한 Python을 그대로 사용하며 프로젝트 경로와 실행 환경을 자동으로 설정하므로 `source scripts/env.sh`나 `YOLOX_PYTHON` 설정이 필요하지 않습니다.

```bash
python run.py check-env --include-build --output artifacts/installed-env
# 빌드 도구가 부족할 때만 로컬 wheel로 설치
# pip install --no-index --find-links=./wheelhouse -r requirements-build.txt
python run.py build
python run.py sample-data  # 기존 샘플이 있으면 생략
python run.py train --mode online --config configs/online.json --device cuda \
  --objects-per-image 5 --object-size-range 16 24 --require-object-count
```

위 학습 명령은 샘플 이미지마다 객체 **5개**를 새로 합성하고 크기를 **16~24픽셀**에서 선택합니다. 합성 이미지와 정답 박스·클래스를 메모리에서 바로 모델에 전달하므로 **합성 이미지나 학습용 COCO JSON을 미리 만들거나 저장할 필요가 없습니다.** 모델 checkpoint와 학습 로그는 `outputs/online/<exp_name>/`에 저장합니다. 기존 검증용 COCO annotation은 입력으로 필요합니다.

학습 횟수는 `configs/online.json`의 `max_epoch`, EPOCH당 합성할 이미지 분량은 `copy_paste.epoch_length`, 배치 크기는 `batch_size`에서 지정합니다. 객체 크기는 회전 전 긴 변 기준이며 `--object-size 24`를 쓰면 24픽셀로 고정할 수 있습니다.

합성 객체의 색이 너무 선명하면 `preview` 또는 `train` 명령에 `--object-saturation-range 0.4 0.7`을 추가해 객체 채도만 낮출 수 있습니다. 객체 대비도 줄이려면 `--object-contrast-range 0.85 0.95`를 함께 사용합니다. 계속 적용하려면 실제 설정의 `copy_paste`에 `object_saturation_range`, `object_contrast_range`를 저장하세요. 두 옵션의 기본값은 원래 색을 유지하는 `[1, 1]`입니다.

기본적으로 같은 객체·마스크 세트를 여러 번 선택할 수 있습니다. 이미지 안에서 서로 다른 세트만 선택하려면 `--unique-objects`를 추가하고 객체 세트를 5개 이상 준비하세요. 객체를 5개 배치할 수 없으면 재시도 후 오류로 알립니다. `preview`와 `export`는 합성 결과를 파일로 확인하거나 반출할 때만 직접 실행하는 선택 도구이며 학습에서 자동 호출하지 않습니다.

각 명령의 옵션은 `python run.py --help`, `python run.py train --help` 등으로 확인합니다. `train --mode original`은 `configs/original.json`, `train --mode online`은 `configs/online.json`을 기본으로 사용합니다. `--config`로 다른 설정을 지정할 수 있으며, 명시한 `--mode`와 설정의 모드가 다르면 오류로 종료합니다. C++ 빌드에는 시스템의 `gcc`·`g++`가 필요하고, 학습·평가 등의 네트워크 차단 실행에는 Linux의 `libseccomp`가 필요합니다.

GPU가 없는 머신은 두 학습·평가 명령의 `cuda`를 `cpu`로 바꾸어 진단합니다. 별도 venv와 전체 wheelhouse는 아래처럼 새 환경을 완전히 재현할 때 사용하는 경로입니다.

## 내 객체·마스크 폴더로 합성 학습

`manifest.json`을 직접 작성할 필요 없이 아래처럼 파일을 넣으세요. 객체와 마스크는 **같은 하위 폴더 경로와 파일명**으로 연결합니다. 각 폴더에 여러 PNG를 넣어도 됩니다.

```text
data/assets/
  object/
    type1/001.png
    type2/001.png
    type3/001.png
    type4/001.png
    type5/001.png
  mask/
    type1/001.png
    type2/001.png
    type3/001.png
    type4/001.png
    type5/001.png
  background/
    bg_001.png
    bg_002.png
```

```bash
python run.py train --mode online --config configs/online_folders.json --device cuda \
  --objects-per-image 5 --object-size-range 16 24 --require-object-count
```

이 설정은 `data/assets/object`, `mask`, `background`를 하위 폴더까지 자동으로 읽습니다. `object/type1/001.png`는 `mask/type1/001.png`와 한 세트입니다. 마스크는 객체 이미지와 같은 크기의 단일 채널 이진 이미지로, 객체는 흰색(255), 나머지는 검정(0)으로 준비하세요. 대응하는 마스크가 없거나 후보가 여러 개이면 학습 전에 오류로 알립니다.

`configs/online_folders.json`에서 `max_epoch`, `batch_size`, `copy_paste.epoch_length`를 바꾸면 학습 횟수와 배치 크기, EPOCH당 이미지 분량을 조절합니다. 이 파일은 작은 샘플 모델·입력 크기와 `data/sample`의 검증 데이터를 사용하므로, 실제 학습에서는 `data_dir`·`num_classes`·모델 설정도 맞추세요. 검증 클래스가 하나라면 위 5개 폴더의 객체는 모두 같은 클래스입니다. 5개 객체는 전체 파일 목록에서 랜덤으로 선택하며, 폴더마다 하나씩 선택하는 규칙은 아닙니다. 폴더별 클래스 지정과 경로 변경은 [configs/ONLINE_DATA.md](configs/ONLINE_DATA.md)를 참고하세요.

## 큰 이미지를 패치로 학습하기

기존 `original`, `online`은 그대로 사용합니다. 새 모드는 다음 두 가지입니다.

| 모드 | 학습 입력 |
| --- | --- |
| `original_patch` | 큰 원본 이미지와 COCO 박스를 격자 패치로 잘라 학습 |
| `online_patch` | 큰 배경에서 매번 랜덤 패치를 자른 뒤 그 위에 객체·마스크 합성 |

먼저 사용할 새 설정 파일의 `data_dir`, `num_classes`, `train_ann`·`train_name`(원본 학습), `val_ann`·`val_name`(검증)을 실제 데이터에 맞춥니다. `online_patch`는 `copy_paste.asset_root`를 현재 사용하는 객체·마스크·배경 폴더의 루트로 지정합니다. 기본값은 `data/assets/object`, `mask`, `background`입니다. 기존 JSON의 `file_name`은 그대로 두면 됩니다.

```bash
# 원본 이미지의 패치 학습
python run.py train --mode original_patch --config configs/original_patch.json --device cuda

# 배경 패치 한 장마다 객체 5개를 합성해 학습
python run.py train --mode online_patch --config configs/online_patch.json --device cuda \
  --objects-per-image 5 --object-size-range 16 24 --require-object-count
```

새 설정의 기본 패치·모델 입력 크기는 **640×640**입니다. `patch.size`는 자를 원본 픽셀 크기 `[높이, 너비]`, `patch.stride`는 원본 격자를 이동할 간격입니다. 겹침을 주려면 예를 들어 `size: [640, 640]`, `stride: [512, 512]`로 지정합니다. `online_patch`의 배경 위치는 매번 랜덤이며 `stride`는 검증 패치에만 적용됩니다. `input_size`, `test_size`도 `[640, 640]`으로 맞추면 패치를 축소하지 않습니다. 크기를 다르게 지정하면 자른 후 모델 입력 크기에 맞춰 리사이즈합니다. 합성 객체의 `--object-size-range`는 최종 합성 캔버스에서 회전 전 긴 변 픽셀 크기입니다.

원본 모드는 한 EPOCH에 모든 학습 패치를 한 번씩 순회합니다. 합성 모드는 `copy_paste.epoch_length`가 EPOCH당 패치 수입니다. `max_epoch`, `batch_size`는 기존과 같습니다. 패치 모드에서는 Mosaic·MixUp·multiscale을 사용하지 않으며, 색상·좌우 반전은 `hsv_prob`·`flip_prob`로 설정합니다.

결과 확인은 선택적으로 실행합니다. 새 폴더에 원본 패치와 박스 표시 이미지를 저장합니다.

```bash
python run.py preview --config configs/original_patch.json --output outputs/preview_original_patch --count 4
python run.py preview --config configs/online_patch.json --output outputs/preview_online_patch --count 4
```

학습 중 패치 이미지나 새 COCO JSON을 디스크에 저장하지 않습니다. 검증도 같은 크기의 격자 패치에서 수행하므로 **AP는 패치 기준**이며 전체 이미지로 예측을 합치는 평가는 아닙니다. 경계 박스·빈 패치·경로 설정은 [configs/PATCH_DATA.md](configs/PATCH_DATA.md)를 확인하세요.

합성 시작 전에 오래 걸리면 실제 사용하는 설정의 `copy_paste`에 `"validate_assets_on_start": false`를 지정하세요. 전체 이미지의 초기 디코딩 대신 선택된 파일을 사용할 때 검사합니다. 새 `online_patch.json`에는 이 값이 들어 있습니다. `[startup]` 로그에서 파일 목록 검사와 검증 패치 구성 중 어느 단계인지 확인할 수 있습니다. `epoch_length`는 학습 분량을 정하며 전체 폴더 검색이나 검증 패치 수를 제한하지 않습니다.

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
bash scripts/install_offline.sh  # 선택 사항: 전체 lock으로 별도 .venv를 만드는 기존 설치 도구
.venv/bin/python run.py build
.venv/bin/python scripts/check_upstream.py
.venv/bin/python run.py sample-data
```

위 설치 도구는 `.venv`를 만들므로 이후 명령에서는 `.venv/bin/python`을 사용합니다. 기존 환경을 그대로 쓸 때는 이 설치 도구를 실행하지 않습니다.

샘플 생성 도구는 기존 데이터를 덮어쓰지 않습니다. 이미 `data/sample/sample_manifest.json`이 있으면 생성 명령을 생략하세요. `--force`는 직접 생성한 샘플을 다시 만들 때만 사용합니다. 샘플은 단순한 차량 모양과 배경이며, 실제 SAM3 결과나 실제 항공영상이 아닙니다.

## 기존 환경 사용

이미 사용하는 Python 3.12의 Conda/venv를 활성화한 뒤 실행합니다. `requirements.txt`는 학습에 필요한 직접 의존성 11개, `requirements-build.txt`는 선택적 C++ 빌드 도구입니다. 전체 전이 의존성 51개를 고정한 `requirements-lock.txt`와 용도가 다릅니다.

```bash
cd /path/to/copypaste
python run.py check-env --plan --include-build \
  --output artifacts/server-env
```

이 명령은 패키지를 설치하지 않습니다. 현재 버전, torch/torchvision의 CUDA 12.9 build, CPU NMS와 주요 배열·mask·TensorBoard 동작을 확인합니다. 조건을 모두 만족하면 `no installation needed`를 기록하며 **wheelhouse와 pip 실행이 필요하지 않습니다**. 기존 OpenCV는 python/headless/contrib 중 하나를 재사용합니다. 다른 버전의 패키지로 교체해야 하는 경우에는 충돌을 보고하고 설치 계획을 만들지 않습니다.

부족한 패키지가 있으면 로컬 wheelhouse만 사용해 전이 의존성까지 계획하고 `artifacts/server-env/requirements-missing.txt`에 정확한 버전·SHA256을 기록합니다. 필요한 wheel만 있어도 되며 전체 4.3 GiB를 요구하지 않습니다. 누락된 wheel, 기존 의존성 교체, 중복 OpenCV, 확인되지 않은 wheel은 설치 계획을 차단합니다. 생성된 파일에 실제 패키지 행이 있을 때만 같은 Python으로 설치합니다.

```bash
PIP_CONFIG_FILE=/dev/null python scripts/no_network.py python -m pip \
  --isolated install --no-index --find-links=./wheelhouse \
  --no-deps --require-hashes -r artifacts/server-env/requirements-missing.txt
```

그런 다음 활성화한 환경의 Python으로 실행합니다. `run.py`는 프로젝트의 `.venv`를 새로 만들거나 강제로 선택하지 않습니다.

```bash
python run.py build
python run.py train --mode original --device cuda
python run.py train --mode online --device cuda
```

이 모드는 기존 버전을 자동으로 높이거나 낮추지 않습니다. 현재 requirements 범위를 벗어난 기존 NumPy/OpenCV/TensorBoard 등은 충돌로 표시되며, 해당 버전을 유지할 수 있는지는 대상 서버에서 별도 호환성 검증 후 범위를 조정해야 합니다. 버전 범위 안의 모든 조합을 학습 검증했다고 주장하지 않습니다. `--output`은 빈 폴더를 지정하고, 설치 후 재점검할 때는 다른 폴더를 사용하세요. GPU 학습 검증은 `python run.py validate-gpu`로 별도 수행합니다.

`pip install -r requirements.txt`도 조건을 만족하는 기존 패키지는 재설치하지 않습니다. 다만 조건이 맞지 않으면 pip가 교체할 수 있고 OpenCV 제공 패키지를 자동 구분하지 않으므로, 기존 환경을 유지하려면 위 점검·누락 목록 절차를 권장합니다.

GPU가 있는 서버에서 두 학습 모드를 실행합니다.

```bash
python run.py train --mode original --config configs/original.json --device cuda
python run.py train --mode online --config configs/online.json --device cuda
python run.py preview --config configs/online.json --output outputs/preview --count 4
```

GPU가 없을 때는 CPU 진단을 명시합니다. 이 경로는 공식 모델·Loss·SGD·증강을 사용하지만 CUDA 전용 공식 Trainer를 CPU에서 실행하는 경로는 아닙니다.

```bash
python run.py train --mode original --config configs/original.json --device cpu --stop-after-epoch 1
python run.py train --mode original --config configs/original.json --device cpu \
  --resume outputs/original/sample_nano/latest_ckpt.pth
python run.py train --mode online --config configs/online.json --device cpu
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
python run.py train --mode original --config configs/original.json --device cuda --checkpoint /data/weights/model.pth
python run.py train --mode original --config configs/original.json --device cuda --fp16
python run.py evaluate --config configs/original.json --device cuda \
  --checkpoint outputs/original/sample_nano/latest_ckpt.pth --evaluator fast \
  --output outputs/original/evaluation.json
```

`--checkpoint`는 공식 YOLOX의 shape matching으로 호환되는 텐서를 복원합니다. 예를 들어 80-class checkpoint에서 1-class 모델을 초기화할 때 분류 predictor는 초기값을 유지하고 해당 layer를 로그에 기록합니다. `--resume`는 동일한 설정으로 model/optimizer/scaler/EMA/RNG를 엄격하게 복구하는 epoch 경계 재개입니다. CPU 진단과 CUDA Trainer의 checkpoint를 서로 resume하지 않습니다. 평가에서는 저장된 EMA를 우선 사용하며 `--raw-model`로 학습 가중치를 선택합니다. 로딩에는 `weights_only=True`를 사용합니다. 외부 legacy checkpoint가 거부되면 자동으로 unrestricted pickle 로딩을 재시도하지 않으며, 별도 변환이 필요할 수 있습니다. 실제 사전학습 가중치는 이번 작업에서 제공되지 않았습니다.

## 온라인 합성 데이터

데이터 계약과 모든 합성 옵션은 [configs/ONLINE_DATA.md](configs/ONLINE_DATA.md)를 따릅니다. 폴더 자동 수집은 `configs/online_folders.json`의 `copy_paste.objects_dir`, `masks_dir`, `backgrounds_dir`를 사용합니다. 상대 경로는 `copy_paste.asset_root` 기준입니다. 기존 샘플 설정 `configs/online.json`의 manifest 입력도 계속 지원합니다.

Online 학습은 결과를 메모리에서 생성하여 바로 DataLoader에 전달하며 합성 이미지와 annotation JSON을 저장하지 않습니다. 클래스는 기존 검증용 COCO category ID를 사용하고 내부 YOLOX label index로 변환합니다. 선택 도구인 `preview`는 눈으로 확인할 이미지·bbox·mask·Metadata JSON을, `export`는 데이터셋 반출용 이미지와 표준 COCO annotation JSON을 저장합니다. 저장 도구의 사용법은 [configs/ONLINE_DATA.md](configs/ONLINE_DATA.md)를 참고하세요.

객체 수는 `copy_paste.object_count: [5, 5]`, 크기는 `object_long_side_range: [16, 24]`로 설정할 수 있습니다. 범위 방식과 기존 `object_long_side_pixels`/`object_size_weights` 방식은 함께 지정하지 않습니다. 명령의 `--objects-per-image`, `--object-size`, `--object-size-range`로 해당 실행에만 값을 바꿀 수도 있습니다. 기본 Online 설정은 객체 5개를 요구하며 `require_object_count: true`로 부족한 개수의 합성 결과를 거부합니다. `unique_objects: false`이면 같은 객체·마스크 세트의 중복 선택을 허용합니다.

실제 항공영상의 객체 크기 관측값으로 크기 분포를 보정해야 합니다. 샘플 설정의 픽셀 크기와 증강값은 기능 검증용이며 실제 크기 분포를 추정한 값이 아닙니다.

## 검증

```bash
# IPv4/IPv6 소켓을 kernel seccomp로 차단한 CPU 테스트. AF_UNIX IPC는 허용합니다.
python scripts/no_network.py python -m pytest -q --junitxml=reports/pytest-cpu.xml

# 실제 GPU: CUDA/NMS/FP32/AMP, 두 모드의 native epoch/resume/COCO 평가.
python run.py validate-gpu

# 대상 서버 조사
python scripts/environment_probe.py --output reports/environment-target.json
```

`run.py`는 학습·평가·미리보기·GPU 검증 실행 시 기존 네트워크 차단 도구를 적용합니다. GPU 검증 도구는 GPU가 없으면 `blocked` 결과를 저장하고 exit 2로 종료합니다. CPU 테스트의 GPU 항목은 skipped로 표시합니다. GPU 검증은 샘플 설정이 기본이며, 실제 데이터 설정을 `--original-config`, `--online-config`로 전달해야 실데이터를 검사합니다. Native CUDA prefetcher의 대기 batch와 sampler cursor는 checkpoint에 저장하지 않으므로 GPU resume의 합성·증강 순서가 bit-exact하게 같다고 보장하지 않습니다.

원본 출처·라이선스는 `third_party/UPSTREAM.json`, 원본 파일 SHA256은 `third_party/SOURCE_SHA256.json`에 있습니다. 원본 YOLOX 파일은 수정하지 않고 `src/offline_yolox/`에 실행 어댑터를 구현했습니다. wheel과 로컬 가상환경·출력은 Git에 포함하지 않습니다. 폐쇄망 반입 시에는 wheelhouse를 소스와 함께 별도로 전달해야 합니다.
