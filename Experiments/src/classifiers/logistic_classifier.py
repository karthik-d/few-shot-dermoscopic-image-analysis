from sklearn.linear_model import LogisticRegression


def fit_predict(support_X, support_y, query):

    clf = LogisticRegression(
        penalty='l2',
        solver='lbfgs',
        max_iter=1000,
        multi_class='multinomial'
    )

    clf.fit(support_X, support_y)
    query_pred = clf.predict(query)

    return query_pred