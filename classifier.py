from enum import Enum, auto
from ai import CNN, CellNN


class ClassifierType(Enum):
    CNN = auto()
    CellNN = auto()


def init_classifier(classifier_type, input_shape, classes, **kwargs):
    if classifier_type is ClassifierType.CNN:
        classifier = CNN(CNN, input_shape, classes, **kwargs)
    elif classifier_type is ClassifierType.CellNN:
        classifier = CellNN()
    else:
        raise NotImplementedError('Classifier {} is not implemented.'.format(classifier_type.name))
    return classifier
