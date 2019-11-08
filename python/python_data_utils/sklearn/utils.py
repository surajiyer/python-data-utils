# coding: utf-8

"""
    description: Scikit-learn utility functions and classes
    author: Suraj Iyer
"""

__all__ = [
    'get_estimator_name',
    'display_topics']


def get_estimator_name(clf):
    """
    Extract the estimator name from the the estimator object {clf}
    :param clf: estimator object
    :return: string name
    """
    return str(type(clf)).split('.')[-1].replace("'>", "")


def display_topics(model, feature_names, n_top_words):
    """
    Display keywords associated with topics detected with topic
    modeling models, e.g., LDA, TruncatedSVD (LSA) etc.
    """
    assert hasattr(model, 'components_')
    for topic_idx, topic in enumerate(model.components_):
        print("Topic {}:".format(topic_idx))
        print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
