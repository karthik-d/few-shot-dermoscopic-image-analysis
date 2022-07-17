from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def fit_predict(support_X, support_y, query):
    
    clf = make_pipeline(
        StandardScaler(),
        DecisionTreeClassifier(
            criterion='gini', 
            splitter='best'
        )
    )

    clf.fit(support_X, support_y)
    query_pred = clf.predict(query)
    query_prob = clf.predict_proba(query)

    return query_pred, query_prob