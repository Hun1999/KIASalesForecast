import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 엑셀 파일 경로 설정
file_path = 'static/sales_data_original.xlsx'
model_dir = 'models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# 데이터 로드 및 전처리
data = pd.read_excel(file_path)

# 날짜 형식 변환
data['Year-Month'] = pd.to_datetime(data['Year-Month'], format='%Y-%m')

# 추가적인 시계열 특성 생성
data['Year'] = data['Year-Month'].dt.year
data['Month'] = data['Year-Month'].dt.month
data['Lag1_Sales'] = data.groupby('Car')['Sales'].shift(1)
data['Lag2_Sales'] = data.groupby('Car')['Sales'].shift(2)
data['Lag3_Sales'] = data.groupby('Car')['Sales'].shift(3)
data['Rolling_Mean_3'] = data.groupby('Car')['Sales'].transform(lambda x: x.rolling(window=3).mean())
data['Rolling_Std_3'] = data.groupby('Car')['Sales'].transform(lambda x: x.rolling(window=3).std())

# 결측값 처리
data.fillna(0, inplace=True)

# Attention Layer 정의
class TimeSeriesAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(TimeSeriesAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight', shape=(input_shape[-1], input_shape[-1])
                                 , initializer='glorot_uniform', trainable=True)
        self.U = self.add_weight(name='att_U_weight', shape=(input_shape[-1], input_shape[-1])
                                 , initializer='glorot_uniform', trainable=True)
        self.v = self.add_weight(name='att_v_weight', shape=(input_shape[-1],)
                                 , initializer='zeros', trainable=True)
        super(TimeSeriesAttention, self).build(input_shape)

    def call(self, x):
        x_prev = tf.concat([tf.zeros_like(x[:, :1, :]), x[:, :-1, :]], axis=1)
        e = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + tf.tensordot(x_prev, self.U, axes=1))
        e = tf.tensordot(e, self.v, axes=1)

        a = tf.nn.softmax(e, axis=1)

        context = x * tf.expand_dims(a, axis=-1)
        context = tf.reduce_sum(context, axis=1)

        return context

# 커스텀 GRU 모델 정의
def build_custom_gru(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.GRU(100, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.GRU(100, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        TimeSeriesAttention(),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model

# Early Stopping 콜백 정의
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

# 차량별로 데이터 분리 및 학습
car_models = data['Car'].unique()
for car_model in car_models:
    car_data = data[data['Car'] == car_model]

    # 데이터 준비
    features = car_data.drop(columns=['Year-Month', 'Car', 'Sales'])
    feature_names = features.columns  # 특성 이름 저장
    target = car_data['Sales']

    # 스케일링
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    scaled_features = feature_scaler.fit_transform(features)
    scaled_target = target_scaler.fit_transform(target.values.reshape(-1, 1))

    # 학습 및 테스트 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(scaled_features, scaled_target, test_size=0.2, random_state=42)

    # 데이터 형태 변환
    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)

    # 모델 파일 경로 설정 (슬래시를 언더스코어로 대체)
    model_filename = car_model.replace("/", "_") + "_model.h5"
    model_path = os.path.join(model_dir, model_filename)

    # 모델 학습 또는 로드
    if os.path.exists(model_path):
        # 저장된 모델 로드
        with tf.keras.utils.custom_object_scope({'TimeSeriesAttention': TimeSeriesAttention}):
            custom_gru_model = tf.keras.models.load_model(model_path)
        print(f"Saved model for {car_model} loaded.")
    else:
        # 커스텀 GRU 모델 정의 및 학습
        custom_gru_model = build_custom_gru((X_train.shape[1], X_train.shape[2]))
        history = custom_gru_model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test), verbose=1, callbacks=[early_stopping])
        custom_gru_model.save(model_path)
        print(f"Model for {car_model} trained and saved at {model_path}")

    # 모델 평가
    custom_gru_eval = custom_gru_model.evaluate(X_test, y_test, verbose=0)
    print(f"Evaluation for {car_model}: {custom_gru_eval}")

    # 예측
    y_pred_train = custom_gru_model.predict(X_train)
    y_pred_test = custom_gru_model.predict(X_test)

    # 스케일링 되돌리기
    y_pred_train = target_scaler.inverse_transform(y_pred_train)
    y_pred_test = target_scaler.inverse_transform(y_pred_test)
    y_train = target_scaler.inverse_transform(y_train)
    y_test = target_scaler.inverse_transform(y_test)

    # 결과 시각화
    plt.figure(figsize=(14, 5))
    plt.plot(y_train, label='Actual Sales (Train)')
    plt.plot(y_pred_train, label='Predicted Sales (Train)')
    plt.legend()
    plt.title(f'Actual vs Predicted Sales (Train) for {car_model}')
    plt.xlabel('Time')
    plt.ylabel('Sales')
    plt.show()

    plt.figure(figsize=(14, 5))
    plt.plot(y_test, label='Actual Sales (Test)')
    plt.plot(y_pred_test, label='Predicted Sales (Test)')
    plt.legend()
    plt.title(f'Actual vs Predicted Sales (Test) for {car_model}')
    plt.xlabel('Time')
    plt.ylabel('Sales')
    plt.show()
