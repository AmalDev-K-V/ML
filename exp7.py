from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, f1_score

# Load dataset
train = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
test = fetch_20newsgroups(subset="test", remove=("headers", "footers", "quotes"))

# Convert text into numerical features
vectorizer = CountVectorizer(stop_words="english", max_features=5000)
X_train = vectorizer.fit_transform(train.data)
X_test = vectorizer.transform(test.data)

y_train = train.target
y_test = test.target

# Train and evaluate models
models = {
    "Multinomial NB": MultinomialNB(),
    "Bernoulli NB": BernoulliNB()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print(name)
    print("Accuracy:", accuracy_score(y_test, pred))
    print("F1 Score:", f1_score(y_test, pred, average="weighted"))
    print()