import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Carregar dados do arquivo CSV
filename = input("Digite o nome do arquivo CSV: ")
data = pd.read_csv(filename, header=None)

# Pré-processamento dos dados
data = data.fillna('') 
X = data.iloc[:, 0]  # Texto
y = data.iloc[:, 1]  # Sentimento

# Dividir conjunto de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Extrair características do texto
vectorizer = TfidfVectorizer(max_features=1000)  # Usando TF-IDF com no máximo 1000 características
X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)

# Treinar modelo SVM
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train_features, y_train)

# Avaliar modelo SVM
y_pred = svm_classifier.predict(X_test_features)
accuracy = accuracy_score(y_test, y_pred)

print("Precisão:", accuracy)