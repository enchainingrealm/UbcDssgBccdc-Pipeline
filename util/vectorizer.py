def vectorize(vectorizer, documents):
    """
    Fits the given vectorizer on the given documents. bi/tri/n-grams that span
    across a pipe character (e.g.: the trigram "not performed | Accession") will
    not be included in the vectorizer's vocabulary. Then, transforms the given
    documents into a sparse matrix feature representation based on the fitted
    vocabulary.
    :param vectorizer: the vectorizer to fit and transform the documents
    :param documents: an Iterable of result_full_description strings
    :return: the sparse matrix feature representation of the documents;
             a List whose jth element is the feature represented by the jth
             column of the sparse matrix;
             a Dict mapping feature names to column indices
    """
    phrases = []
    for document in documents:
        phrases.extend(document.split("|"))

    vectorizer.fit(phrases)

    X = vectorizer.transform(documents)
    feature_names = vectorizer.get_feature_names()
    vocabulary = vectorizer.vocabulary_

    return X, feature_names, vocabulary
