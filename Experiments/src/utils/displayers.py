import pandas as pd 


def get_printable_confusion_matrix(
    all_labels,
    all_predictions,
    classes
):
    """
    Renders a `printable` confusion matrix
    Uses the pandas display utility
    """

    map_classes = {
        x: class_ 
        for x, class_ in enumerate(classes)
    }

    pd.options.display.float_format = "{:.4f}".format
    pd.options.display.width = 0

    truth = pd.Series(
        pd.Categorical(
            pd.Series(all_labels).replace(map_classes), 
            categories=classes
        ),
        name="Truth"
    )

    prediction = pd.Series(
        pd.Categorical(
            pd.Series(all_predictions).replace(map_classes), 
            categories=classes
        ),
        name="Prediction"
    )

    confusion_matrix = pd.crosstab(
        index=truth, 
        columns=prediction, 
        normalize="index", 
        dropna=False
    )
    confusion_matrix.style.hide(axis='index')
    return f'\n{confusion_matrix}\n'