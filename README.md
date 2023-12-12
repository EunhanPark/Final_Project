***모델을 학습시키고 저장한 파일('model.pkl')을 eclass에 추가로 업로드하였습니다. (github에는 용량이 커서 업로드되지 않음)***

# Scikit Learn을 활용한 종양 분류

뇌 종양 이미지를 "glioma tumor," "meningioma tumor," "no tumor," "pituitary tumor" 네 가지 범주로 분류하는 프로젝트입니다. 머신 러닝 모델을 구축하고 평가하기 위해 scikit-learn 라이브러리를 사용합니다.

## 데이터셋

데이터셋은 'tumor_dataset/Training' 디렉토리에 위치하며 각 종양 범주에 대한 이미지를 포함합니다. 이미지는 전처리되어 크기가 조정되고 회색조로 변환됩니다.

## 정규화 및 데이터 증강

StandardScaler를 사용하여 특성 벡터를 정규화하고, 훈련 데이터셋을 향상시키기 위해 이미지를 수직으로 뒤집는 데이터 증강을 수행합니다.

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 데이터 증강
def augment_data(images, labels):
    # 함수 코드 작성...

X_train_augmented, y_train_augmented = augment_data(X_train_scaled, y_train)
X_train_combined = np.vstack([X_train, X_train_augmented])
y_train_combined = np.concatenate([y_train, y_train_augmented])

X_train_combined, y_train_combined = shuffle(X_train_combined, y_train_combined, random_state=46)
```

## 서포트 벡터 머신 (SVM) 모델

특정 hyperparameter로 조정된 RBF(Radial Basis Function) 커널을 사용하는 SVM 모델을 훈련합니다.

```python
model = SVC(kernel='rbf', C=175, gamma=0.0002, random_state=46)
model.fit(X_train_combined, y_train_combined)
```

## Hyperparameter 튜닝

Hyperparameter 튜닝은 최적의 조합을 찾기 위해 수행할 수 있습니다. 값을 바꾸며 여러번 실행한 결과 C=175, gamma=0.0002를 얻었고 이를 적용하였습니다.

```python
# param_grid = {'C': [170], 'gamma': [0.0002], 'kernel': ['rbf']}
# grid = GridSearchCV(SVC(random_state=46), param_grid, refit=True, verbose=3, cv=3, n_jobs=-1)
# grid.fit(X_train_combined, y_train_combined)
# print(grid.best_params_)
# print(grid.best_estimator_)
```

## 모델 저장 및 불러오기

모델을 파일로 저장하고, 평가를 위해 불러옵니다.

```python
# 모델 저장
joblib.dump(model, 'model.pkl')

# 모델 로드
model_load = joblib.load('model.pkl')
```

## 모델 평가

훈련된 모델을 로드하고 테스트 데이터셋에서의 성능을 평가합니다.

```python
# 예측 수행
y_pred = model_load.predict(X_test_scaled)

# 정확도 출력
print('정확도: %.2f' % sklearn.metrics.accuracy_score(y_test, y_pred))
```
