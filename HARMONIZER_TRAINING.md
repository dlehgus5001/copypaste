# Harmonizer를 켜고 끄며 YOLOX 합성 학습하기

합성한 이미지와 객체 마스크를 Harmonizer로 보정한 뒤 YOLOX에 입력하는 학습 모드다. 기존 `online`·`online_patch`는 그대로 사용할 수 있다. 비교용 설정 두 개는 합성·모델·학습 조건이 같고 Harmonizer 적용 여부와 결과 저장 위치만 다르다.

## 1. 데이터와 모델 준비

기존 객체·마스크·배경 폴더를 사용한다.

```text
data/assets/
├─ object/
│  ├─ object_1/001.png
│  ├─ object_2/001.png
│  ├─ object_3/001.png
│  ├─ object_4/001.png
│  └─ object_5/001.png
├─ mask/
│  ├─ object_1/001.png
│  ├─ object_2/001.png
│  ├─ object_3/001.png
│  ├─ object_4/001.png
│  └─ object_5/001.png
└─ background/1.png
```

객체와 대응 마스크는 같은 크기·같은 하위 경로 및 파일 이름이어야 한다. 마스크는 객체가 흰색(255), 배경이 검정(0)인 단일 채널 이진 이미지다. 폴더마다 여러 후보를 넣을 수 있으며 기본 설정은 클래스마다 하나씩 총 5개를 무작위로 선택한다.

**두 비교 config 모두 실제 검증 데이터에 맞춰** `data_dir`, `val_ann`, `val_name`, `num_classes`를 설정한다. `val_ann`은 `data_dir/annotations/`의 COCO JSON이다. 객체 클래스는 이 JSON의 `categories`에서 읽는다. `copy_paste.folder_categories`의 1~5도 실제 category ID와 맞아야 한다. 기본 `data/sample`이 한 클래스짜리 샘플이면 새 5클래스 설정으로 그대로 학습할 수 없으므로 실제 검증 데이터 경로로 바꾼다. 합성 학습용 COCO annotation은 따로 만들 필요 없다.

Harmonizer를 사용하는 경우 기존 학습 Python 환경에 추가 패키지를 설치한다.

```bash
python -m pip install -r requirements-harmonizer.txt
```

모델이 없다면 **인터넷이 되는 준비 머신에서** 다음 명령을 실행한다.

```bash
python scripts/prepare_harmonizer.py
```

생성된 `models/Harmonizer/` 전체를 폐쇄망 프로젝트의 같은 위치로 복사한다. 이 폴더에는 필요한 소스와 `pretrained/harmonizer.pth`가 함께 들어 있다. 추가 패키지도 오프라인으로 반입해야 한다면 [HARMONIZER_PREVIEW.md](HARMONIZER_PREVIEW.md)의 준비 절차를 사용한다. Kornia Rust는 내부 저장소의 0.1.14 등을 허용하는 `>=0.1.9,<0.2` 범위를 사용한다.

## 2. 먼저 학습 입력 확인

```bash
python run.py preview \
  --config configs/online_patch_harmonized.json \
  --object-resolution-scale-range 0.25 0.5 \
  --object-blur-sigma-range 0.8 1.6 \
  --harmonizer-device cpu \
  --count 5 \
  --output outputs/preview_train_degraded_v1
```

`sample_0000.png`는 Harmonizer까지 적용된 학습 입력이고, `sample_0000_boxes.png`는 정답 박스를 그린 이미지다. 개별 마스크와 JSON도 함께 저장한다. 출력 폴더는 새 이름을 사용한다.

보정 없는 버전과 비교하려면 config를 `configs/online_patch_baseline.json`으로 바꾸고 다른 출력 폴더에 저장한다. 같은 seed·객체 및 배경·크기 설정을 사용하면 두 버전의 객체 선택·배치·박스·마스크를 맞춰 비교할 수 있다.

## 3. 학습 실행

**Harmonizer 없이 패치 합성 학습:**

```bash
python run.py train \
  --mode online_patch \
  --config configs/online_patch_baseline.json \
  --device cuda
```

**Harmonizer를 적용한 패치 합성 학습:**

```bash
python run.py train \
  --mode online_patch_harmonized \
  --config configs/online_patch_harmonized.json \
  --device cuda
```

`max_epoch`, `batch_size`, `preset`, `copy_paste.epoch_length`는 기존과 같은 위치에서 조절한다. 비교 실험에서는 두 파일의 이 값들을 동일하게 맞춘다. `--mode`는 config의 `mode`와 일치해야 한다.

새 비교 config는 `copy_paste.epoch_length: 10000`으로 EPOCH당 합성 이미지 10,000장을 사용한다. 먼저 짧게 동작을 확인하려면 두 파일 모두 이 값을 8 등으로 줄인다. 기존 config의 값은 변경하지 않았다.

| 입력 방식 | 보정 없는 config / mode | 보정하는 config / mode |
| --- | --- | --- |
| 배경 전체를 입력 크기로 맞춘 뒤 합성 | `online_baseline.json` / `online` | `online_harmonized.json` / `online_harmonized` |
| 배경에서 패치를 자른 뒤 합성 | `online_patch_baseline.json` / `online_patch` | `online_patch_harmonized.json` / `online_patch_harmonized` |

각 config의 `output_dir/exp_name/` 아래에 checkpoint와 로그를 저장한다. 두 버전의 저장 폴더가 다르므로 결과가 섞이지 않는다. 기존 `configs/online.json`·`configs/online_patch.json`도 계속 사용할 수 있다.

## 4. 색·해상도·블러·Harmonizer 강도 조절

두 비교 config의 `copy_paste` 기본값은 다음과 같다.

```json
"object_saturation_range": [0.2, 0.5],
"object_contrast_range": [0.9, 1.0],
"object_resolution_scale_range": [0.25, 0.5],
"object_blur_sigma_range": [0.8, 1.6]
```

처리 순서는 **배경 선택/패치 추출 → 객체 크기 변환 → 채도·대비 → 흙색 얼룩(선택) → 객체 해상도 저하 → 객체 블러 → 합성 → Harmonizer → YOLOX 학습 전처리**다. 작은 채도 값은 빨간색 등 쨍한 색을 줄인다. Harmonizer는 이후 색·조명을 추가로 바꾸므로 최종 preview에서도 색을 확인한다. 클래스별 크기는 `copy_paste.object_long_side_range_by_category`를 사용하며, 기존 크기·객체 수 CLI 옵션도 지원한다.

`object_resolution_scale_range`는 객체마다 무작위로 선택하는 내부 축소 비율이다. 예를 들어 최종 긴 변이 80픽셀인 객체에 `[0.25, 0.5]`를 사용하면 약 20~40픽셀로 줄였다가 80픽셀로 복원해 디테일을 없앤다. 객체의 최종 크기·마스크·정답 박스와 배경 픽셀은 유지한다. 마스크로 정규화한 축소·확대로 외부의 검은 픽셀이 섞이는 것을 막는다. 값은 `0 < MIN <= MAX <= 1`이며 작을수록 더 강하게 저하된다. 약하게는 `[0.5, 0.75]`, 더 강하게는 `[0.15, 0.3]`부터 비교할 수 있다. 생략 또는 `[1, 1]`은 해상도 저하를 끄며 기존 처리 결과를 유지한다. 추가 패키지는 필요 없다.

위 `--object-resolution-scale-range`·`--object-blur-sigma-range` 옵션은 `train`과 `paste-multi`에도 동일하게 사용할 수 있다. 계속 적용하려면 실제 학습 JSON의 `copy_paste`를 바꾼다. 네 비교 config와 `paste_multi.json`에는 위 기본값을 적용했고, 기존 `online.json`·`online_patch.json` 등은 변경하지 않았다. 실행 중인 학습은 설정을 자동으로 다시 읽지 않으므로 새 실행부터 적용된다. 설정을 바꿔 `--resume`이 거부되면 새 config와 `--checkpoint`로 fine-tuning한다.

보정하는 config의 최상위 `harmonizer` 설정은 다음과 같다.

```json
"harmonizer": {
  "repository": "models/Harmonizer",
  "checkpoint": null,
  "device": "auto",
  "strength": 1.0
}
```

`strength`는 기본 합성과 보정 결과를 섞는 강도다. 1은 전체 보정, 0.5는 절반, 0은 보정을 생략한다. `checkpoint: null`이면 repository의 `pretrained/harmonizer.pth`를 읽는다. 외부 파일은 실행 중 다운로드하지 않는다.

학습·preview 명령에 다음 옵션을 추가해 config를 덮어쓸 수 있다.

```text
--harmonizer-repo /path/Harmonizer
--harmonizer-checkpoint /path/harmonizer.pth
--harmonizer-device cuda
--harmonizer-strength 0.5
```

**CUDA 또는 auto Harmonizer를 사용하면 `num_workers: 0`으로 설정한다.** 모델은 메인 프로세스에서 처음 필요할 때 로드하고 EPOCH가 바뀌어도 재사용한다. `device: "cpu"`를 명시하면 여러 데이터 Worker를 사용할 수 있지만 Worker마다 모델을 별도로 로드하므로 메모리·로딩 비용이 늘어난다. Harmonizer 연산이 합성 이미지마다 추가되므로 보정 없는 학습보다 느려질 수 있다.

Harmonizer는 고정 가중치·FP32·평가 모드로 실행한다. YOLOX optimizer와 EMA에 포함하지 않으며 이미지만 바꾼다. 전체 객체 마스크의 합집합으로 보정하고, 마스크 밖 배경과 정답 박스·클래스·마스크 좌표를 유지한다.

## 5. checkpoint와 검증

YOLOX 가중치를 불러와 학습하려면 기존 `--checkpoint /path/yolox_ckpt.pth`를 사용한다. Harmonizer 가중치는 `--harmonizer-checkpoint`로 지정한다. 두 옵션의 용도가 다르다.

`--resume`은 같은 config의 EPOCH 경계 재개에 사용한다. 보정 없는 학습에서 보정하는 학습으로 전환할 때는 새 config와 `--checkpoint`를 사용해 fine-tuning한다. YOLOX checkpoint에는 적용 config를 기록하며 Harmonizer 가중치 자체를 함께 넣지 않으므로 재학습·재개 시 로컬 모델 폴더를 유지한다.

재개할 때는 Harmonizer 가중치 파일의 내용도 동일하게 유지한다. 현재 resume 설정 검사는 파일 경로를 비교하며, 같은 경로에 다른 가중치를 덮어쓴 경우까지 감지하지는 않는다.

검증과 일반 추론에는 실제 이미지를 그대로 입력한다. Harmonizer는 합성 학습 입력에만 적용하며 검증·추론에서 모델 가중치를 로드하지 않는다. 효과는 동일한 검증 데이터의 AP로 비교한다. 이 기능이 실제 항공영상의 질감 차이나 탐지 성능을 개선한다고 보장하지 않는다.
