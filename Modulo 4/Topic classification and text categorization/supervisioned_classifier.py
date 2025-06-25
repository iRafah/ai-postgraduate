from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn import metrics

# Dados de exemplo mais robustos
texts = [
    "Eu amo programar em Python", "Messi é melhor do que o Cristiano Ronaldo", "A máquina de lavar está quebrada", "Eu gosto de pizza", 
    "Python é uma linguagem de programação", "Eu preciso consertar minha máquina de lavar",
    "Pizza é minha comida favorita", "Estou aprendendo a programar", "O forno está quebrado",
    "Eu amo pizza de pepperoni", "A geladeira parou de funcionar", "O curso de Python é ótimo",
    "Preciso de um técnico para consertar minha geladeira", "A pizza de marguerita é deliciosa",
    "Eu gosto de aprender novas linguagens de programação", "O conserto do micro-ondas foi caro",
    "O Milan tem 7 ligas dos campeões", "A seleção brasileira é a seleção mais vitoriosa com cinco copas do mundo"
]

labels = [
    "tecnologia", "futebol", "doméstico", "comida", "tecnologia", "doméstico", "comida", 
    "tecnologia", "doméstico", "comida", "doméstico", "tecnologia", "doméstico",
    "comida", "tecnologia", "doméstico", "futebol", "futebol"
]

X_train, X_test, y_train, y_test = train_test_split(texts, labels, random_state=42)

model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train, y_train)
labels_predicted = model.predict(X_test)

print(metrics.classification_report(y_test, labels_predicted, zero_division=0))

new_texts = ["Eu preciso aprender PHP", "A pizza está deliciosa", "Hoje terá o jogo em Real Madrid e Barcelona pelo campeonato espanhol"]
new_labels_predicted = model.predict(new_texts)

print(new_labels_predicted)