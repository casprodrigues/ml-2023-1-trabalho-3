import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Carregar o conjunto de dados
base_dados = pd.read_csv("./ds_salaries.csv")

# Remover as linhas que são diferentes de USD
data_clean = base_dados.drop(base_dados[base_dados['salary_currency'] != 'USD'].index)

# Imprimir informações iniciais sobre o dataset
print("Dataset: {} linhas | {} colunas".format(data_clean.shape[0], data_clean.shape[1]))

# Identificar as colunas que são do tipo 'object'
label_encoder = LabelEncoder()
cols_object = data_clean.select_dtypes(include=['object']).columns

# Aplicar a transformação LabelEncoder nas colunas identificadas
data_clean[cols_object] = data_clean[cols_object].apply(lambda col: label_encoder.fit_transform(col.astype(str)))

# Separar os dados em features (X) e rótulo (y)
X = data_clean.drop(['job_title'], axis=1)
y = data_clean['job_title']

# Dividir os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pré-processamento dos dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Criação do modelo MLP
model = Sequential()
model.add(Dense(10, activation='relu', input_dim=X_train_scaled.shape[1]))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compilação do modelo
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Treinamento do modelo
history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, verbose=1)

# Avaliação do modelo
y_pred_prob = model.predict(X_test_scaled)
y_pred = (y_pred_prob > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)
print("Acurácia:", accuracy)

# Plotar gráfico da acurácia ao longo das épocas
plt.plot(history.history['accuracy'])
plt.title('Acurácia do Modelo')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.show()


print()
print()

# Plotar gráfico da perda ao longo das épocas
plt.plot(history.history['loss'])
plt.title('Curva de Aprendizado')
plt.xlabel('Época')
plt.ylabel('Perda')
plt.show()

print()
print()

# Acessar os pesos da primeira camada oculta
weights = model.layers[0].get_weights()[0]

# Plotar histograma dos pesos
plt.hist(weights.flatten(), bins=30)
plt.title('Distribuição dos Pesos')
plt.xlabel('Pesos')
plt.ylabel('Frequencia')
plt.show()

print()
print()

# Plotar a superfície de decisão (apenas para conjuntos de dados com duas features)
if X_train_scaled.shape[1] == 2:
    x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
    y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    Z = model.predict_classes(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap=plt.cm.Paired)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Superfície de Decisão')
    plt.show()
