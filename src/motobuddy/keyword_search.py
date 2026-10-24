from sklearn.feature_extraction.text import TfidfVectorizer

docs = [
    "Office equipment policy",
    "Office furniture guidelines",
    "Office travel policy"
]

analyzer = TfidfVectorizer()

word_scores = analyzer.fit_transform(docs)
print(word_scores.toarray())