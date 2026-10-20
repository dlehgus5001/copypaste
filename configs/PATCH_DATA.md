# 큰 이미지의 패치 학습

`original_patch.json`, `online_patch.json`은 별도 설정이다. 기존 `original.json`, `online.json`, `online_folders.json`은 변경하지 않는다. 모델·Loss·Optimizer·Checkpoint는 기존 YOLOX 흐름을 공유한다. 기존 모드의 가중치는 새 설정에서 `--checkpoint /로컬/가중치.pth`로 미세조정할 수 있다. 설정이 달라지므로 기존 학습의 `--resume`으로 전환하지 않는다.

## 원본 데이터와 경로

큰 원본 이미지에 대한 기존 COCO JSON을 그대로 입력한다. 별도의 패치 JSON이나 이미지 사전 생성은 필요하지 않다.

```json
{
  "data_dir": "/data/coco",
  "train_ann": "train_clean.json",
  "val_ann": "val_clean.json",
  "train_name": "/data/큰원본/학습",
  "val_name": "/data/큰원본/검증"
}
```

이 예시의 annotation은 `/data/coco/annotations/train_clean.json`, `val_clean.json`이다. annotation 경로 자체를 절대 경로로 지정해도 된다. 이미지는 `train_name` 또는 `val_name` 뒤에 JSON의 `file_name`을 붙인다. `file_name`이 절대 경로면 그 경로를 사용한다. COCO의 `width`, `height`는 실제 원본 크기와 일치해야 한다. DRM 이미지는 먼저 `scripts/filter_coco.py`로 제외한다.

`online_patch`는 학습용 COCO annotation이 필요하지 않다. 검증 경로와 클래스 목록은 필요하다. 객체·마스크·배경 경로는 기존 폴더 설정을 그대로 복사한다.

```json
"copy_paste": {
  "asset_root": "/data/assets",
  "objects_dir": "object",
  "masks_dir": "mask",
  "backgrounds_dir": "background",
  "epoch_length": 10000,
  "object_count": [5, 5],
  "object_long_side_range": [16, 24]
}
```

위 항목은 전체 설정이 아니라 경로·개수·크기 변경 예시다. 나머지는 제공된 `online_patch.json`을 사용한다. 이미지·마스크는 `object/type1/001.png` ↔ `mask/type1/001.png`로 연결한다. 배경에 이미 표시된 객체는 새로운 정답으로 자동 인식하지 않으므로 합성용 배경은 별도로 준비한다.

## 크기와 경계 처리

```json
"input_size": [640, 640],
"test_size": [640, 640],
"patch": {
  "size": [640, 640],
  "stride": [512, 512],
  "min_visible_fraction": 0.0,
  "min_box_side": 2,
  "keep_empty": true,
  "max_labels": 500
}
```

- 크기·간격은 `[높이, 너비]`다. `input_size`, `test_size`는 32의 배수이며 최소 64다. 패치 크기도 입력 크기와 같게 하는 것을 권장한다.
- 원본 학습과 검증은 좌상단부터 격자로 순회한다. 오른쪽·아래쪽 마지막 패치는 가장자리에 맞춰 이동하므로 간격이 크기와 같아도 마지막 부분은 겹칠 수 있다. 원본이 패치보다 작으면 오른쪽·아래쪽을 114로 채우며 확대하지 않는다.
- 박스는 패치 안에 남은 부분으로 잘라 좌표를 바꾼다. 남은 박스의 가로·세로가 `min_box_side` 미만이면 제외한다. 원본 이미지 안으로 보정한 박스 면적 대비 남은 비율이 `min_visible_fraction` 미만이어도 제외한다. 기본값 0.0은 일부만 남은 박스도 크기 조건을 만족하면 사용한다. 작은 조각을 정답에서 제외하면 그 조각은 이미지에 남으므로 임계값을 올릴 때는 preview를 확인한다.
- 겹친 패치에는 같은 원본 객체가 각각 포함될 수 있다. 패치마다 COCO 박스와 ID를 메모리에서 따로 만들고 원본 ID는 metadata로 보관한다.
- 학습의 crowd 박스는 기존 YOLOX처럼 정답에서 제외한다. 검증 COCO에는 crowd 속성을 유지한다. 이 구현은 bbox 탐지용이며 세그멘테이션 평가는 제공하지 않는다.
- `keep_empty: true`는 객체가 없는 학습 패치도 사용한다. `false`는 원본 학습에서만 빈 패치를 제외한다. 검증은 설정과 관계없이 전체 패치를 포함한다. 합성 학습의 개수는 `copy_paste.object_count`가 결정한다.
- 원본 학습 패치의 박스 수가 `max_labels`를 넘으면 정답을 잘라내지 않고 설정을 늘리라는 오류로 알린다. 합성 학습의 최대 정답 수는 기존 `copy_paste.max_labels`를 사용한다.

## 합성 순서와 반복 수

`online_patch`: 배경 파일 랜덤 선택 → 원본 해상도에서 `patch.size` 영역의 좌상단 위치 랜덤 선택 → 부족한 부분 패딩 → `input_size`로 맞춤 → 객체·마스크 선택 및 크기·회전·배치 → 보이는 마스크에서 박스 생성. 따라서 `object_count: [5, 5]`는 큰 배경 전체가 아니라 **패치 한 장에 5개**다. 원본 배경이 패치보다 큰 경우 호출마다 새로운 위치를 선택한다. 배경이 같거나 위치가 다시 선택되는 것도 가능하다. 합성 가시성 조건은 `copy_paste.min_visible_fraction` 등 기존 설정을 따른다.

`original_patch`: 같은 격자 패치를 EPOCH마다 섞어 순회한다. 위치를 매번 랜덤으로 다시 자르는 방식은 아니다. EPOCH당 이미지 수는 빈 패치 정책을 적용한 격자 패치 수다. `online_patch`의 EPOCH당 이미지 수는 `copy_paste.epoch_length`다. iteration 수는 `ceil(패치 수 / batch_size)`이고, 총 횟수는 `max_epoch`으로 설정한다. 실제 CUDA Trainer의 무한 Sampler에서는 이 크기를 기준으로 EPOCH 길이를 계산한다.

원본은 필요할 때 디코딩하고 패치 픽셀을 메모리에서 자른다. 이미지를 미리 타일 파일로 만들거나 모든 픽셀을 캐싱하지 않으므로 같은 큰 원본을 여러 번 디코딩할 수 있다. 디코딩 비용과 Worker별 메모리를 고려해 `num_workers`를 조절한다.

## 검증과 결과 확인

### 합성 시작 전 오래 걸리는 경우

실제 사용하는 설정 파일의 `copy_paste` 안에 아래 항목을 넣는다. `online_patch_mapped.json`처럼 별도 설정을 사용하는 경우에도 그 파일에 추가해야 한다.

```json
"validate_assets_on_start": false
```

폴더·파일 대응·클래스는 먼저 검사하지만 전체 객체·마스크·배경의 픽셀을 미리 디코딩하지 않는다. 선택된 파일은 합성에 사용할 때 기존과 동일하게 디코딩·자료형·크기·마스크 값을 검사한다. 따라서 아직 선택하지 않은 손상 파일은 시작 단계에서 발견되지 않을 수 있다. 파일을 조용히 건너뛰거나 정답 검사를 생략하지는 않는다. 기존 `online` 설정에서 옵션을 생략하면 전체 초기 검사를 유지하고, 새 `online_patch.json`에는 `false`를 명시한다. 학습과 preview에 모두 적용된다.

`[startup]` 로그에 실제 설정 경로, 객체·마스크 파일 목록 작성, 배경 파일 목록 작성, 개수·픽셀 검사 방식·소요 시간을 출력한다. 전체 초기 검사를 유지하면 처리 중인 파일을 주기적으로 표시한다. 검증 패치 목록 생성 단계와 완성된 패치 수도 따로 출력한다. 학습을 시작하기 전 GPU 모델 준비와 검증 패치 목록 구성은 여전히 필요하다. 폴더 검색·파일 존재 확인과 COCO JSON 읽기·패치 metadata 생성은 `epoch_length`에 의해 줄어들지 않는다.

### 패치 이미지 확인

두 새 모드 모두 기존 held-out 검증 원본을 격자 패치로 나누고 같은 박스 변환을 적용하여 COCO bbox AP를 계산한다. **패치 AP**이므로 이전 전체 이미지 AP와 직접 같은 지표로 비교하지 않는다. 겹치는 패치를 원본 좌표로 합치거나 중복 제거하는 전체 이미지 추론은 이 학습 모드에 포함되지 않는다. 학습·검증 원본을 먼저 분리한 뒤 각 split을 패치로 나누어 같은 원본의 패치가 양쪽에 섞이지 않게 준비한다.

```bash
python run.py preview --config configs/original_patch.json --output outputs/preview_original_patch --count 4
python run.py preview --config configs/online_patch.json --output outputs/preview_online_patch --count 4
```

`sample_0000.png`는 실제 패치, `sample_0000_boxes.png`는 박스 표시 이미지다. 원본 모드는 앞에서부터 최대 `count`개를 보여준다. JSON에 원본 경로·ID·`crop_xy`를 기록한다. 합성 모드는 JSON의 `background_patch.xy`, `size`, `source_size`로 잘라온 위치를 확인한다. 이 파일들은 preview를 직접 실행할 때만 저장한다. 출력 폴더는 새 폴더나 빈 폴더를 사용한다. `export` 명령은 기존 `online` 전용으로 유지한다.

체크포인트와 TensorBoard 로그는 각각 `outputs/original_patch/<exp_name>/`, `outputs/online_patch/<exp_name>/`에 저장한다. `python run.py evaluate --config configs/original_patch.json --checkpoint /로컬/가중치.pth --device cuda`도 동일한 패치 검증을 사용한다.
