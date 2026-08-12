# Face Separation

사람 사진에서 얼굴의 눈, 코, 입 영역만 잘라내 각각 이미지 파일로 저장하는 CLI 도구입니다.
[MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)로 얼굴 랜드마크를 검출합니다.

## 설치

```bash
pip install -r requirements.txt
```

처음 실행할 때 얼굴 랜드마크 모델 파일(약 4MB)을 자동으로 다운로드합니다(인터넷 연결 필요, 이후에는 재사용).

## 사용법

```bash
python separate.py 사진.jpg
```

기본적으로 `output/` 폴더에 다음 파일들이 생성됩니다:

- `사진_left_eye.png`
- `사진_right_eye.png`
- `사진_nose.png`
- `사진_mouth.png`

### 여러 장 한번에 처리하기

파일을 여러 개 나열하거나, 사진들이 들어있는 폴더를 통째로 넘길 수 있습니다.

```bash
# 파일 여러 개
python separate.py 사진1.jpg 사진2.jpg 사진3.jpg

# 폴더 전체 (jpg, jpeg, png, bmp, webp 파일을 모두 찾아 처리)
python separate.py 사진폴더/
```

얼굴이 없는 사진은 건너뛰고 나머지는 계속 처리합니다.

### 옵션

```bash
python separate.py 사진.jpg -o 결과폴더 --padding 0.3 --max-faces 3
```

- `-o, --output`: 출력 폴더 (기본값: `output`)
- `--padding`: 잘라낼 영역 주변에 추가할 여백 비율 (기본값: `0.3`)
- `--max-faces`: 한 이미지에서 처리할 최대 얼굴 수 (기본값: `1`). 2 이상이면 파일명에 `_face0`, `_face1` 등이 붙습니다.
