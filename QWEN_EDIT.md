# Qwen-Image-Edit-2511로 객체 외형 변환

별도 촬영한 객체 사진과 원본 마스크, 실제 항공영상의 객체 crop을 입력한다. 같은 클래스의 crop을 참조해서 객체 외형을 편집한 뒤 기존 Copy-Paste 학습에서 사용하는 파일로 저장한다. 항공 crop의 segmentation이나 일대일 대응 사진은 필요하지 않다. 이 도구는 공개 편집 모델을 실행하며 LoRA를 학습하지 않는다.

## 설치

YOLOX를 실행하는 Python 3.12 / Torch 2.8 / torchvision 0.23 환경에서 선택 패키지를 설치한다. `requirements.txt`에는 Qwen 의존성을 추가하지 않았다.

```bash
pip install -r requirements-qwen.txt
```

폐쇄망에서는 반입한 wheel만 사용한다. 조건을 만족하는 기존 Torch·torchvision은 재사용한다.

```bash
pip install --no-index --find-links=/반입경로/qwen-wheelhouse -r requirements-qwen.txt
```

`diffusers==0.36.0`, `transformers==4.57.3`, `accelerate==1.12.0`, `peft==0.17.1`, `safetensors==0.7.0`, `huggingface-hub==0.36.0`을 사용한다. 이 조합의 실제 설치·의존성 검사와 QwenImageEditPlusPipeline import·호출 인자 확인은 수행했다. 실제 Qwen 가중치 로딩·GPU 생성은 이 개발 환경에서 수행하지 못했다. 기존 환경에 별도 패키지 제약이 있으면 분리된 Python 환경을 사용할 수 있으며, 생성된 PNG는 기존 YOLOX 환경에서 읽을 수 있다.

이미 requirements를 설치했는데 `peft>=0.17.0 ... found peft==0.15.2`가 발생하면, 초기 Qwen requirements에 PEFT 조건이 빠져 있어 기존 구버전이 남은 경우다. LoRA를 직접 사용하지 않아도 Diffusers가 설치된 PEFT 버전을 검사한다. 갱신된 requirements를 같은 Python으로 다시 설치하거나 아래 한 패키지를 설치한다.

```bash
python -m pip install "peft==0.17.1"
# 반입 wheel만 사용할 때
python -m pip install --no-index --find-links=/반입경로/qwen-wheelhouse "peft==0.17.1"
```

wheel 방식이면 온라인 준비 머신에서 갱신된 requirements로 wheel을 다시 수집해 PEFT와 전이 의존성도 반입한다. 서버 pip constraint가 PEFT 0.15.2를 고정한 경우 해당 환경에서는 두 요구사항이 충돌하므로 Qwen 편집용 환경을 분리한다.

`flash_attn_2_cuda...so: undefined symbol: _ZN3c104cuda9SetDeviceEa`는 설치된 외부 FlashAttention CUDA 확장이 현재 Torch 바이너리와 호환되지 않는 오류다. Qwen 편집 명령은 PyTorch 기본 SDPA를 사용한다. 갱신된 `src/offline_yolox/qwen_edit.py`는 HF 클래스 import 동안 외부 FlashAttention·xformers·SageAttention·aiter 탐지를 제외하고, 텍스트/비전 인코더는 `attn_implementation="sdpa"`, 이미지 transformer는 `native` backend로 명시한다. 설치된 시스템 패키지를 삭제하거나 수정하지 않는다. 탐지 함수는 클래스 import 후 원래대로 복원한다. `DIFFUSERS_ATTN_BACKEND=native` 환경 변수만 설정하면 Diffusers의 초기 optional extension import를 피하지 못할 수 있으므로 수정된 코드를 사용해야 한다. 새 Python 프로세스로 기존 `run.py adapt-objects` 명령을 다시 실행한다. SDPA는 Torch 자체 CUDA 구현을 사용할 수 있으며 CPU offload 설정과는 별개다.

## 모델 반입

Qwen 가중치는 Git에 포함하지 않는다. 온라인 준비 머신에서 **전체 Diffusers snapshot**을 준비해서 반입한다. 단일 `.safetensors`나 YOLOX `.pth` 파일만으로 실행할 수 없다.

```text
models/Qwen-Image-Edit-2511/
  model_index.json
  transformer/      # config + 모든 가중치 shard
  text_encoder/    # config + 모든 가중치 shard
  tokenizer/
  processor/
  vae/
  scheduler/
```

온라인 준비 머신에서만 다음 명령을 사용할 수 있다. `qwen-wheelhouse`에는 전이 의존성과 큰 Torch wheel도 수집될 수 있다. 대상 서버와 같은 Python·OS·아키텍처로 준비한다. 새 환경에서 Torch를 설치해야 한다면 CUDA build도 기존 프로젝트의 12.9 환경에 맞춰 별도로 준비한다.

```bash
pip install huggingface-hub==0.36.0
hf download Qwen/Qwen-Image-Edit-2511 --local-dir models/Qwen-Image-Edit-2511
pip download --only-binary=:all: -r requirements-qwen.txt --dest qwen-wheelhouse
```

이 모델 폴더와 wheel 폴더를 서버에 복사한다. 기존 YOLOX wheelhouse·lock·manifest에는 Qwen 의존성이 포함되지 않으며, 소스 압축 도구도 Qwen 가중치와 별도 wheel 폴더를 포함하지 않는다. 모델 경로가 다르면 `configs/qwen_edit.json`의 `model_dir` 또는 명령의 `--model-dir`을 바꾼다.

실행 시 Hugging Face offline 환경 변수와 `local_files_only=True`, `use_safetensors=True`를 사용하고, `run.py`의 커널 네트워크 차단도 적용한다. 누락된 가중치를 인터넷에서 자동으로 받지 않는다.

## 데이터 배치

```text
data/assets/
  object/
    object_1/001.png
    object_2/001.png
    object_3/001.png
    object_4/001.png
    object_5/001.png
  mask/
    object_1/001.png
    object_2/001.png
    object_3/001.png
    object_4/001.png
    object_5/001.png
  aerial_crop/
    object_1/ref_001.png
    object_1/ref_002.png
    object_2/ref_001.png
    object_3/ref_001.png
    object_4/ref_001.png
    object_5/ref_001.png
  background/
    bg_001.png
```

객체·마스크는 같은 상대 하위 경로와 파일명으로 연결하며 확장자는 달라도 된다. 마스크는 이미지와 같은 해상도의 단일 채널 이진 이미지(0/255 또는 0/1)여야 한다. 객체 PNG alpha 대신 이 마스크를 사용한다. 객체와 crop 이미지는 읽을 수 있는 uint8 BGR/BGRA 이미지여야 한다.

각 클래스 폴더에 여러 이미지와 하위 폴더를 넣을 수 있다. 항공 crop은 해당 차량 주변을 네모로 잘라 저장하면 된다. 클래스별 crop의 파일명은 원본 객체와 같을 필요가 없다. 같은 클래스의 참조를 하나씩 랜덤 선택하고 파일 경로·seed를 기록한다.

`configs/qwen_edit.json`에는 앞서 지정한 5개 클래스와 `object_1`~`object_5` 매핑이 들어 있다. 다른 데이터라면 `class_ids`, `class_names`, `folder_categories`를 함께 수정한다. 폴더 이름을 COCO category ID 또는 클래스 이름으로 사용해도 된다. 변환할 객체 클래스에 참조 crop이 없으면 시작 시 오류가 발생한다. 최종 독립 테스트용 사진은 참조 crop 제작에서 제외한다.

## 실행

프로젝트 폴더에서 실행한다. 상대 경로는 프로젝트 루트 기준이다. 처음에는 객체 하나만 변환한다.

```bash
python run.py adapt-objects --config configs/qwen_edit.json --limit 1
```

입력 파일만 먼저 검사하려면 다음 명령을 사용한다. 모델 폴더의 기본 구조와 입력 픽셀·마스크를 확인하며 Qwen 패키지를 import하거나 모델을 로드하지 않는다. 따라서 실제 가중치 완전성·GPU 메모리·생성 성공을 검증하는 명령은 아니다. 출력 폴더도 만들지 않는다.

```bash
python run.py adapt-objects --config configs/qwen_edit.json --check-inputs
```

결과가 적절하면 새 출력 폴더에 전체를 실행한다.

```bash
python run.py adapt-objects --config configs/qwen_edit.json \
  --output data/assets/qwen_adapted_all --variants-per-object 3
```

입력 객체가 100개이고 variants가 3이면 300번 편집한다. 참조 선택과 모델 seed는 원본 경로·기본 seed·variant 번호에서 결정한다. 같은 조건의 job은 limit이 달라도 같은 참조와 seed를 사용한다. 동일 seed라도 GPU·라이브러리·offload 조건에 따라 생성 픽셀의 완전한 재현을 보장하지 않는다.

| 설정 | 의미 |
| --- | --- |
| `model_dir` | 반입한 Qwen 전체 모델 폴더 |
| `asset_root` | 기존 객체·마스크·배경을 포함한 루트 |
| `objects_dir`, `masks_dir`, `references_dir` | 루트 아래 입력 폴더 |
| `output` | 루트 안의 새/빈 결과 폴더. 입력 폴더와 겹칠 수 없음 |
| `work_size: [512, 512]` | Qwen 편집 출력 크기 `[높이, 너비]`. 최종 객체 픽셀 크기 설정과 다름 |
| `crop_padding: 0.15` | 마스크 bounding box 긴 변의 15%만큼 주변 여백 추가 |
| `variants_per_object: 1` | 원본 하나당 생성할 후보 수 |
| `num_inference_steps: 40` | 편집 샘플링 step 수 |
| `prompt`, `negative_prompt` | 편집 지시. `{class_name}`·`{category_id}`는 객체별로 치환 |
| `true_cfg_scale: 4.0`, `guidance_scale: 1.0` | 공식 예제에 맞춘 guidance 설정. 원본 보존 정도를 직접 보장하는 값은 아님 |
| `offload: sequential` | GPU 메모리를 줄이기 위해 모듈을 순차적으로 CPU↔GPU 이동. 느릴 수 있음 |
| `offload: model` | 모델 단위 offload. 더 큰 GPU 메모리가 필요할 수 있음 |
| `offload: none` | 전체 모델을 GPU에 올림. 메모리가 충분한 경우 사용 |

Qwen은 YOLOX보다 훨씬 큰 모델이다. A100 일부 메모리를 할당받은 환경에서는 예제의 bfloat16·sequential offload로 소량부터 확인한다. CPU RAM도 필요하고 실제 실행 가능 메모리·속도는 이 환경에서 측정하지 못했다. offload는 CPU 연산으로 YOLOX를 학습하는 옵션이 아니다.

`--limit`, `--variants-per-object`, `--num-inference-steps`, `--seed`, `--offload`, `--model-dir`, `--asset-root`, `--references-dir`, `--output`은 JSON을 덮어쓴다. 도움말은 `python run.py adapt-objects --help`로 확인한다.

## 결과 확인과 합성 학습 연결

```text
data/assets/qwen_adapted_v1/
  object/object_1/001__qwen_000.png   # 원본 해상도로 복원한 객체 이미지
  mask/object_1/001__qwen_000.png     # 원본과 동일한 foreground 영역
  raw/object_1/001__qwen_000.png      # Qwen의 전체 편집 결과
  review/object_1/001__qwen_000.png   # 원본·참조·raw+마스크 윤곽·내보낸 객체 비교
  metadata/object_1/001__qwen_000.json
  copy_paste_settings.json
  manifest.json
```

원본 마스크의 bounding box에 여백을 더해 crop하고, 비율을 유지해 중립 회색 캔버스에 배치한다. 같은 클래스의 실제 항공 crop을 두 번째 이미지로 넣는다. 출력은 원본 좌표로 되돌리고 **마스크 내부 픽셀만 교체**한다. 마스크 밖 픽셀은 원본을 유지하며, 최종 마스크·이미지 크기도 원본과 동일하다. 출력 크기가 요청과 다르면 조용히 변형하지 않고 오류로 알린다. 원본 자산·참조·모델 파일과 기존 출력은 덮어쓰지 않는다.

먼저 `review/`를 확인한다. 초록색 선은 원본 마스크 윤곽이며, 생성된 차량 윤곽을 새로 추정한 선이 아니다. 생성 모델이 차체·부품·위치를 바꿨거나 차량을 지운 경우, 마스크로 잘라도 올바른 객체가 복구되지는 않는다. **마스크 좌표 보존과 실제 차량 형태 보존은 다르다.** 이 도구는 형태·클래스 보존을 자동 인증하거나 성능 개선을 보장하지 않으며 metadata에 `semantic_shape_preservation_verified: false`, `review_required: true`를 기록한다. 좋지 않은 후보는 학습용 폴더에서 이미지·마스크를 한 세트로 제외한다.

사용할 후보를 확인한 뒤 기존 `configs/online_patch.json`의 `copy_paste` 안에서 경로를 수정한다. 위 전체 실행 예시의 출력 폴더를 사용하면 다음과 같다.

```json
"objects_dir": "qwen_adapted_all/object",
"masks_dir": "qwen_adapted_all/mask",
"mask_suffix": ""
```

`asset_root`는 기존 `data/assets`로 유지하고 배경 경로·객체 개수·크기 등 나머지 설정도 유지한다. 실제 경로와 매핑은 결과의 `copy_paste_settings.json`에도 기록한다. 이 파일은 **기존 copy_paste에 병합할 경로 조각**으로, 단독 학습 config가 아니다. `output`을 바꿨다면 해당 결과 파일에 적힌 경로를 사용한다.

`--limit 1`로 생성한 폴더에는 원본 객체 하나의 후보만 있다. 이 폴더로 서로 다른 클래스 5개를 요청하면 클래스 부족 오류가 발생한다. 다중 클래스 합성 학습에는 필요한 모든 클래스가 포함된 전체 실행 결과를 사용한다.

```bash
python run.py preview --config configs/online_patch.json \
  --output outputs/preview_qwen_v1 --count 20 --unique-classes
python run.py train --mode online_patch --config configs/online_patch.json \
  --device cuda --unique-classes
```

실제 합성 크기는 계속 `copy_paste.object_long_side_range` 또는 CLI `--object-size-range`로 결정한다. Qwen의 512×512 작업 해상도는 객체가 패치 안에서 512픽셀이라는 뜻이 아니다. 기존 saturation·blur·dirt 효과도 함께 적용되므로 중복 열화를 피하려면 먼저 항등/꺼짐 설정으로 비교한다. YOLOX iteration마다 Qwen을 호출하지 않는다.

## 확인한 범위

네트워크 차단 아래 실제 선택 의존성 설치·import와 호출 API를 확인했다. 테스트 대역을 사용해 같은 클래스 참조 선택, crop/letterbox 역변환, 마스크 밖 픽셀·원본 파일 보존, 원래 해상도의 객체·마스크 저장, 기존 Copy-Paste에서 다시 읽기, 기존 출력 보호와 로컬 로딩 인자를 확인했다. **테스트 대역의 색상 출력은 Qwen 생성 결과가 아니다.** GPU와 실제 Qwen 가중치가 없는 환경이므로 실제 모델 로딩·생성 품질·형태 보존·실제 항공영상 AP 개선은 미검증이다.

참고: [Qwen 공식 구현](https://github.com/QwenLM/Qwen-Image), [Diffusers 0.36.0 Qwen 편집 파이프라인](https://github.com/huggingface/diffusers/blob/v0.36.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit_plus.py).
