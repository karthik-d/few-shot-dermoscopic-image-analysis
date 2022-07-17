from sklearn import svm 
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def fit_predict(support_X, support_y, query):
    
    clf = make_pipeline(
        StandardScaler(), 
        SVC(
            gamma='auto',
            kernel='poly',
            decision_function_shape='ovr'
        )
    )
    
    clf.fit(support_X, support_y)
    query_pred = clf.predict(query)
    query_prob = clf.predict_proba(query)

    return query_pred, query_prob