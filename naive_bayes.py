import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Carregar o arquivo CSV
data = pd.read_csv("tweets_ekman.csv")

# Remover linhas com valores NaN
data.dropna(subset=['Texto', 'Sentimento'], inplace=True)

# Dividir os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(data['Texto'], data['Sentimento'], test_size=0.2, random_state=42)

# Vetorizar os textos
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Treinar o classificador Naive Bayes
modelo = MultinomialNB()
modelo.fit(X_train_counts, y_train)

# Fazer previsões no conjunto de teste
predictions = modelo.predict(X_test_counts)

# Avaliar a precisão do modelo
accuracy = accuracy_score(y_test, predictions)
print("Precisão:", accuracy)

# Exibir o relatório de classificação
print("\nRelatório de Classificação:")
print(classification_report(y_test, predictions))

# Avaliar o modelo com validação cruzada
cv_scores = cross_val_score(modelo, X_train_counts, y_train, cv=5)
print("\nValidação Cruzada - Pontuações:")
print(cv_scores)
print("Precisão Média:", cv_scores.mean())

def detector():
    # Texto de entrada
    input_text = input("\nInsira um texto para classificação: ")

    # Pré-processamento do texto de entrada
    input_text_counts = vectorizer.transform([input_text])

    # Fazer previsões com o modelo
    predicted_sentiment = modelo.predict(input_text_counts)

    # Interpretar os resultados
    print("Sentimento Previsto:", predicted_sentiment)

continuar = True
while(continuar):
    detector()
    escolha = input("\nDigite 1 para sair: o")
    if escolha == '1': continuar = False