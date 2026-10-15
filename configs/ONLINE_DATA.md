# Online Copy-Paste 로컬 데이터 설정

`configs/online.json`은 기능 확인용 작은 샘플 설정이다. 샘플 객체는 직접 생성한 도형이며 SAM3 출력이나 실제 항공 영상이 아니다. SAM3 추론은 학습 중 실행하지 않는다. 실제 서버에서는 사전에 추출한 객체 이미지·마스크와 객체가 없는 항공뷰 배경을 로컬로 제공한다.

`data_dir`에는 평가용 COCO 데이터의 `annotations/<val_ann>`과 `<val_name>` 이미지가 필요하다. Online 모드는 Original 모드의 COCO 학습 Annotation이나 학습 이미지를 요구하지 않는다. `num_classes`는 검증 COCO의 클래스 개수와 일치해야 한다. 객체 Manifest의 `category_id`는 그 COCO taxonomy의 ID다. 비연속 ID도 오름차순으로 정렬해 모델의 0부터 시작하는 클래스 인덱스로 변환한다.

`copy_paste.asset_root`는 객체와 배경의 공통 로컬 루트다. 설정 파일의 루트 경로는 프로젝트 루트 기준이고, Manifest 및 Manifest 내부 이미지 경로는 `asset_root` 기준이다. 루트 밖으로 나가는 경로나 URL은 거부한다. 예:

```json
[
  {"image": "objects/vehicle_001.png", "mask": "objects/vehicle_001_mask.png", "category_id": 1},
  {"image": "objects/vehicle_002.png", "mask": "objects/vehicle_002_mask.png", "category_id": 1}
]
```

객체 이미지는 uint8 3채널 컬러 이미지 또는 4채널 PNG다. 4채널 이미지의 alpha 대신 별도 Mask를 사용한다. Mask는 이미지와 같은 높이·폭의 uint8 단일 채널 이진 이미지다. 0은 배경, 1 또는 255는 전경이며 비어 있거나 중간 회색 값이 있으면 거부한다. 설정 초기화 때 모든 파일의 존재 여부, 디코딩, Mask 정합성, category를 검사한다. SAM3의 연속 확률 Mask는 신뢰할 수 있는 임곗값으로 별도 이진화한 뒤 제공한다.

배경 Manifest는 다음과 같은 로컬 경로 배열이다. 배경은 `input_size`의 높이·폭으로 크기를 변경하므로 실제 배경은 같은 종횡비로 준비하는 것을 권장한다. 이미 있는 객체를 배경에 포함하면 그 객체는 합성 label에 포함되지 않는다.

```json
["backgrounds/aerial_001.png", "backgrounds/aerial_002.png"]
```

`epoch_length`는 한 Epoch에서 생성할 합성 샘플 수다. 각 Dataset 호출은 같은 index여도 새 배경·객체·크기·각도·위치를 선택한다. Native Trainer는 공식 `InfiniteSampler`를 사용하고 한 Epoch의 iteration 수는 `ceil(epoch_length / batch_size)`다. 마지막 배치 크기까지 일정하므로 필요하면 `epoch_length`를 `batch_size`의 배수로 둔다. CPU 진단은 유한 Sampler로 정확히 `epoch_length`개를 소비한다. Online 모드에는 Mosaic/MixUp을 추가하지 않는다. 모델, Loss, SGD, Scheduler, EMA와 실제 COCO 검증은 Original 모드와 공유한다.

객체의 크기는 tight foreground Mask의 회전 전 긴 변을 기준으로 정한다. 다음 중 한 가지 방식만 사용한다.

- `object_long_side_pixels: [24]`와 `object_size_weights: [1]`: 객체의 긴 변을 24픽셀로 고정한다.
- `object_long_side_range: [16, 48]`: 객체마다 16~48픽셀의 정수를 균등하게 선택한다. 양 끝값을 포함하고 최소 크기는 2픽셀이다. 이 방식을 사용할 때는 `object_long_side_pixels`와 `object_size_weights`를 제거한다.
- `object_long_side_pixels: [12, 18, 24, 32]`와 `object_size_weights: [1, 3, 3, 1]`: 목록의 크기를 가중치에 따라 선택한다. 예시에서는 18·24픽셀을 더 자주 선택한다. weight를 생략하면 같은 확률로 선택한다.

실측 항공영상 크기 분포를 사용할 때는 학습 입력 해상도에서 측정한 긴 변 길이와 발생 빈도를 목록·가중치에 지정하고 `distribution_source`에 출처를 기록한다. 현재 샘플 크기는 기능 확인용이며 실제 항공 크기 분포라고 주장하지 않는다. Native multiscale을 활성화하면 합성 후 학습 입력 변경에 따라 Box와 객체 크기도 비례해 변한다. 지정한 pixels 기준을 유지하려면 `multiscale_range: 0`을 사용한다.

`object_count`는 이미지 한 장에 넣을 최소·최대 객체 수다. `[5, 5]`이면 5개를 요청한다. `unique_objects: false`이면 같은 객체 이미지·마스크 세트를 여러 번 선택할 수 있고, `true`이면 한 이미지에서 서로 다른 세트를 선택한다. 서로 다른 클래스만 선택한다는 뜻은 아니다. `true`일 때는 요청한 수 이상의 세트를 Manifest에 준비해야 한다.

기본 설정의 `require_object_count: true`는 요청한 개수를 만족하는 이미지만 반환한다. 각 객체는 `placement_attempts`만큼 배치를 시도하고, 개수를 채우지 못하면 전체 이미지를 다시 합성한다. 전체 시도 한도인 `composition_attempts`는 기본 20이다. 한도를 모두 사용하면 부족한 객체 수로 학습하는 대신 오류로 종료한다. 객체 크기·개수·가림 제한을 조정해서 해결한다. `require_object_count: false`에서는 배치 실패한 객체를 생략할 수 있고, 요청 수와 실제 수를 Metadata에 기록한다.

`rotation_degrees`는 최소·최대 각도다. Mask는 nearest-neighbor 보간으로 확대·회전하며 확장된 캔버스에서 처리한다. `allow_partial_objects`는 영상 경계에 걸친 객체를 허용한다. `min_visible_fraction`, `min_visible_pixels`, `min_box_side_pixels` 미만이 되는 배치는 재시도한다. 이미 놓은 객체가 심하게 가려질 배치도 거부한다.

새 객체는 기존 객체의 ownership Mask에서 겹친 전경을 제거한다. 마지막 visible Mask로 Box를 다시 계산하기 때문에 회전·잘림·가림 뒤에도 정합성을 유지한다. Box 형식은 `[xmin, ymin, xmax, ymax]`이며 xmax/ymax는 마지막 전경 pixel의 좌표 + 1이다. 모든 label은 서로 겹치지 않는 최종 visible Mask와 대응한다. `gaussian_blend_sigma`는 Mask 내부 경계를 부드럽게 하는 inward feathering이다. Blend의 낮은 alpha에서도 전경 소유권은 나중 객체에 있다. `blur_sigma`, `noise_std`, `brightness_range`는 선택적 전체 이미지 증강이며 기하학적 label은 바꾸지 않는다.

학습 Dataset은 합성 이미지·Mask를 메모리에서 바로 YOLOX로 전달하고 파일을 쓰지 않는다. `train`, `preview`, `export`에서 객체 수·크기를 명령 옵션으로 바꾸면 해당 실행에만 적용한다. 원래 JSON 파일은 수정하지 않는다. `--object-size`와 `--object-size-range`는 함께 지정하지 않는다.

```bash
# 매 이미지에 5개, 객체 긴 변 16~24픽셀로 랜덤 합성하면서 학습
python run.py train --mode online --device cuda \
  --objects-per-image 5 --object-size-range 16 24 --require-object-count

# 크기 24픽셀 고정, 서로 다른 객체·마스크 세트 5개로 미리보기
python run.py preview --config configs/online.json --output outputs/preview --count 4 \
  --objects-per-image 5 --object-size 24 --unique-objects --require-object-count
```

Preview는 PNG, Box overlay, 각 instance의 최종 Mask와 Metadata JSON을 내보낸다. `--unique-objects`를 사용하려면 객체 세트를 요청한 개수 이상 준비한다. 기존 파일을 보호하기 위해 Preview 출력 폴더는 비어 있어야 한다. `--no-unique-objects`와 `--no-require-object-count`로 JSON의 해당 옵션을 명시적으로 끌 수 있다. CLI로 바꾼 설정도 checkpoint에 기록되므로 Online 학습을 `--resume`으로 재개할 때는 같은 합성 옵션을 함께 지정한다.

Online 학습은 위의 `train --mode online` 명령만 실행하면 된다. 합성 이미지와 annotation JSON은 저장하지 않으며, 정답 박스·클래스는 메모리에서 자동 생성한다. `max_epoch`는 총 EPOCH, `copy_paste.epoch_length`는 EPOCH당 합성 이미지 분량, `batch_size`는 한 번에 처리하는 이미지 수다. 기존 검증용 COCO JSON과 객체·배경 manifest는 입력으로 유지한다.

합성 데이터를 다른 학습 작업에 반출할 때만 선택 도구인 `export`를 사용한다. Online 학습에 필요한 사전 단계는 아니다. Preview의 Metadata JSON과 달리 `annotations/instances_train2017.json`은 표준 COCO 형식이며 `images`, `annotations`, `categories`를 포함한다. 각 annotation의 `bbox`는 `[x, y, width, height]`, `area`는 최종 visible Mask의 전경 픽셀 수, `segmentation`은 같은 Mask의 COCO RLE다. 클래스 ID와 이름은 원래 검증 COCO의 category 매핑을 유지한다.

```bash
python run.py export --config configs/online.json --output outputs/synthetic \
  --count 100 --objects-per-image 5 --object-size-range 16 24 --require-object-count

python run.py train --mode original --config outputs/synthetic/train_config.json --device cuda
```

기본 `--split train2017`로 내보낸 결과는 다음과 같다.

```text
outputs/synthetic/
  train2017/*.png
  annotations/instances_train2017.json
  metadata/*.json
  train_config.json
```

출력 폴더는 새 경로나 빈 폴더여야 한다. `train_config.json`은 저장된 합성 이미지를 학습 데이터로 사용하고, 기존 Online 설정의 검증 이미지·annotation을 그대로 참조한다. 합성한 객체 수와 픽셀 크기를 유지하기 위해 이 학습 설정은 Mosaic·MixUp·multiscale을 끈다. EPOCH·배치 크기 등은 원래 Online 설정을 이어받는다. 검증 데이터는 합성하거나 복사하지 않으므로 기존 경로를 유지해야 한다. 이동해서 사용할 때는 설정의 데이터 경로도 확인한다. `--split val2017`은 `val2017/`과 `annotations/instances_val2017.json`으로 저장하며 학습 설정은 만들지 않는다. 실제 성능 검증에는 학습용 합성과 별도로 확보한 검증 데이터를 사용한다.

재현성은 같은 파일·의존 버전·seed·epoch·batch size·worker 수·Sampler 순서의 새 Loader에서 검증했다. 각 Worker는 PyTorch가 설정한 서로 다른 seed를 사용한다. 0 Worker와 2 Worker는 각각 반복 실행 결과를 재현하지만 서로 동일한 이미지를 생성한다고 보장하지 않는다. 유한 CPU 진단은 Epoch마다 seed+epoch로 Loader를 새로 만들어 Epoch 경계 Resume를 정확히 재현한다. Native CUDA Trainer의 체크포인트는 모델/Optimizer/EMA/Scaler/RNG를 복원하지만 무한 Loader의 worker/prefetch 진행 상태는 복원하지 않으므로 합성 스트림의 bit-exact 재개를 보장하지 않는다. 중간 iteration 재개도 제공하지 않는다.
