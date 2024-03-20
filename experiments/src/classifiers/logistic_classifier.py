from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def fit_predict(support_X, support_y, query):

    clf = make_pipeline(
            # StandardScaler(), 
            LogisticRegression(
                penalty='l2',
                solver='lbfgs',
                max_iter=1000,
                multi_class='multinomial'
            )
    )

    clf.fit(support_X, support_y)
    query_pred = clf.predict(query)
    query_prob = clf.predict_proba(query)

    return query_pred, query_prob