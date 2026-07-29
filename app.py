import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
from tensorflow.keras.layers import DepthwiseConv2D

# -------------------------------------------------------------
# 1. 페이지 기본 설정
# -------------------------------------------------------------
st.set_page_config(
    page_title="AI 외떡잎 vs 쌍떡잎 식물 분류기",
    page_icon="🌿",
    layout="centered",
)

st.title("🌿 AI 외떡잎 vs 쌍떡잎 식물 탐구 분류기")
st.markdown("""
식물의 **잎(잎맥)** 또는 **줄기/뿌리 사진**을 카메라로 찍거나 업로드해 보세요!  
AI가 **외떡잎식물**인지 **쌍떡잎식물**인지 분석하고 생물학적 특징을 알려줍니다. 🌱
""")

# -------------------------------------------------------------
# 2. 생물학적 관찰 포인트 안내
# -------------------------------------------------------------
with st.expander("📖 **[수업 참고] 외떡잎식물 vs 쌍떡잎식물 핵심 비교**"):
  col1, col2 = st.columns(2)
  with col1:
    st.markdown("""
        #### 🌾 외떡잎식물 (Monocot)
        * **잎맥**: 나란히맥 (줄무늬 형태)
        * **관다발**: 흩어져 있음 (산재)
        * **뿌리**: 수염뿌리
        * **대표 식물**: 벼, 옥수수, 강아지풀, 대나무, 튤립
        """)
  with col2:
    st.markdown("""
        #### 🌸 쌍떡잎식물 (Dicot)
        * **잎맥**: 그물맥 (그물망 형태)
        * **관다발**: 고리 모양으로 정렬
        * **뿌리**: 원뿌리와 옆뿌리
        * **대표 식물**: 봉선화, 장미, 단풍나무, 해바라기, 나팔꽃
        """)

st.divider()


# -------------------------------------------------------------
# 3. 모델 및 라벨 파일 로딩 (DepthwiseConv2D 호환성 패치 적용)
# -------------------------------------------------------------
# 티처블 머신 h5 모델의 groups 파라미터 오류 방지용 클래스
class FixedDepthwiseConv2D(DepthwiseConv2D):

  def __init__(self, *args, **kwargs):
    kwargs.pop('groups', None)  # 에러 원인인 groups 파라미터 제거
    super().__init__(*args, **kwargs)


@st.cache_resource
def load_tm_model():
  # custom_objects 옵션으로 호환성 클래스 전달
  model = tf.keras.models.load_model(
      'keras_model.h5',
      compile=False,
      custom_objects={'DepthwiseConv2D': FixedDepthwiseConv2D},
  )
  with open('labels.txt', 'r', encoding='utf-8') as f:
    class_names = [line.strip() for line in f.readlines()]
  return model, class_names


try:
  model, class_names = load_tm_model()
except Exception as e:
  st.error(
      f'❌ 모델 파일을 불러올 수 없습니다: {e}\n\n`keras_model.h5`와'
      ' `labels.txt` 파일이 `app.py`와 같은 위치에 있는지 확인해 주세요!'
  )
  st.stop()


# -------------------------------------------------------------
# 4. 이미지 전처리 및 예측 함수
# -------------------------------------------------------------
def predict_plant(image_data, model, class_names):
  # 티처블 머신 표준 입력 크기 (224x224)
  size = (224, 224)
  image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
  image_array = np.asarray(image.convert('RGB'))

  # 정규화: (pixel / 127.5) - 1
  normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

  data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
  data[0] = normalized_image_array

  prediction = model.predict(data)
  index = np.argmax(prediction)

  raw_label = class_names[index]
  clean_label = raw_label.split(' ', 1)[-1] if ' ' in raw_label else raw_label
  confidence = prediction[0][index]

  return clean_label, confidence, prediction[0]


# -------------------------------------------------------------
# 5. 카메라 촬영 / 이미지 업로드 탭
# -------------------------------------------------------------
tab1, tab2 = st.tabs(['📸 실시간 카메라 촬영', '🖼️ 이미지 파일 업로드'])

input_image = None

with tab1:
  camera_file = st.camera_input('식물 잎맥이나 모습을 카메라에 비춰주세요!')
  if camera_file:
    input_image = Image.open(camera_file)

with tab2:
  uploaded_file = st.file_uploader(
      '식물 사진 파일(JPG, PNG)을 업로드하세요.', type=['jpg', 'jpeg', 'png']
  )
  if uploaded_file:
    input_image = Image.open(uploaded_file)

# -------------------------------------------------------------
# 6. 분석 및 탐구 결과 출력
# -------------------------------------------------------------
if input_image is not None:
  st.divider()
  st.subheader('🔍 AI 식물 분석 결과')

  col_img, col_res = st.columns([1, 1.2])

  with col_img:
    st.image(input_image, caption='관찰 대상 이미지', use_column_width=True)

  with col_res:
    with st.spinner('AI가 잎맥과 식물 형태를 분석하는 중... 💭'):
      label, confidence, all_predictions = predict_plant(
          input_image, model, class_names
      )

      if '외떡잎' in label or 'monocot' in label.lower():
        display_title = '🌾 외떡잎식물'
        description = (
            '이 식물은 **나란히맥** 잎맥 구조나 산재된 관다발 특징을 가질'
            ' 확률이 높습니다!'
        )
      elif '쌍떡잎' in label or 'dicot' in label.lower():
        display_title = '🌸 쌍떡잎식물'
        description = (
            '이 식물은 **그물맥** 잎맥 구조나 규칙적인 관다발 특징을 가질'
            ' 확률이 높습니다!'
        )
      else:
        display_title = f'🌱 {label}'
        description = '분류 결과를 확인해 보세요.'

      st.success(f'### 예측 결과: **{display_title}**')
      st.metric(label='분류 신뢰도(정확도)', value=f'{confidence * 100:.1f}%')
      st.write(f'💡 **생물학적 특징**: {description}')

  st.write('---')
  st.subheader('📊 클래스별 예측 확률')

  for i, class_raw in enumerate(class_names):
    c_name = class_raw.split(' ', 1)[-1] if ' ' in class_raw else class_raw
    score = float(all_predictions[i])

    col_name, col_bar = st.columns([1, 3])
    with col_name:
      st.write(f'**{c_name}**')
    with col_bar:
      st.progress(int(score * 100))
      st.caption(f'{score * 100:.1f}%')

  st.info("""
    🔬 **[탐구 질문] AI의 분류 결과를 비판적으로 생각해 봅시다!**
    * AI가 잎의 진짜 **'잎맥 패턴'**을 보고 판단했을까요, 아니면 **'배경(화분, 손가락, 흙)'**을 보고 판단했을까요?
    * 만약 오진이 발생했다면, 잎을 확대하여 잎맥만 또렷하게 촬영한 후 다시 시도해 보세요!
    """)
