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

객체의 크기는 tight foreground Mask의 회전 전 긴 변을 기준으로 정한다. `object_long_side_pixels`와 `object_size_weights`는 실측 항공영상 크기 분포를 이산 경험 분포로 표현한다. 실측 객체를 학습 입력 해상도로 변환한 뒤 긴 변 길이들의 목록을 넣고, 각각의 발생 빈도를 weight로 지정한다. 표본 목록 자체를 넣고 모든 weight를 1로 지정해도 된다. `distribution_source`에는 측정 출처를 기록한다. 현재 12/18/24/32 pixels 설정은 미보정 진단용이며 실제 항공 크기 분포라고 주장하지 않는다. Native multiscale을 활성화하면 합성 후 학습 입력 변경에 따라 Box와 객체 크기도 비례해 변한다. 물리적 크기 분포를 입력 pixels로 유지하려면 `multiscale_range: 0`을 사용한다.

`object_count`는 최소·최대 객체 수, `rotation_degrees`는 최소·최대 각도다. Mask는 nearest-neighbor 보간으로 확대·회전하며 확장된 캔버스에서 처리한다. `allow_partial_objects`는 영상 경계에 걸친 객체를 허용한다. `min_visible_fraction`, `min_visible_pixels`, `min_box_side_pixels` 미만이 되는 배치는 재시도한다. 이미 놓은 객체가 심하게 가려질 배치도 거부한다. 정해진 `placement_attempts` 안에 성공하지 못하면 그 객체를 넣지 않으므로 실제 객체 수는 요청 수보다 적을 수 있으며 Metadata에 기록한다.

새 객체는 기존 객체의 ownership Mask에서 겹친 전경을 제거한다. 마지막 visible Mask로 Box를 다시 계산하기 때문에 회전·잘림·가림 뒤에도 정합성을 유지한다. Box 형식은 `[xmin, ymin, xmax, ymax]`이며 xmax/ymax는 마지막 전경 pixel의 좌표 + 1이다. 모든 label은 서로 겹치지 않는 최종 visible Mask와 대응한다. `gaussian_blend_sigma`는 Mask 내부 경계를 부드럽게 하는 inward feathering이다. Blend의 낮은 alpha에서도 전경 소유권은 나중 객체에 있다. `blur_sigma`, `noise_std`, `brightness_range`는 선택적 전체 이미지 증강이며 기하학적 label은 바꾸지 않는다.

학습 Dataset은 합성 이미지·Mask를 메모리에서 바로 YOLOX로 전달하고 파일을 쓰지 않는다. 필요할 때만 다음 Preview 명령으로 PNG, Box overlay, 각 instance의 최종 Mask와 Metadata JSON을 내보낸다. 기존 파일을 보호하기 위해 Preview 출력 폴더는 비어 있어야 한다.

```bash
python scripts/no_network.py python -m offline_yolox.preview --config configs/online.json --output outputs/preview --count 4
bash scripts/train_online.sh --device cpu
bash scripts/train_online.sh --device cuda
```

재현성은 같은 파일·의존 버전·seed·epoch·batch size·worker 수·Sampler 순서의 새 Loader에서 검증했다. 각 Worker는 PyTorch가 설정한 서로 다른 seed를 사용한다. 0 Worker와 2 Worker는 각각 반복 실행 결과를 재현하지만 서로 동일한 이미지를 생성한다고 보장하지 않는다. 유한 CPU 진단은 Epoch마다 seed+epoch로 Loader를 새로 만들어 Epoch 경계 Resume를 정확히 재현한다. Native CUDA Trainer의 체크포인트는 모델/Optimizer/EMA/Scaler/RNG를 복원하지만 무한 Loader의 worker/prefetch 진행 상태는 복원하지 않으므로 합성 스트림의 bit-exact 재개를 보장하지 않는다. 중간 iteration 재개도 제공하지 않는다.
