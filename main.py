import numpy as np
import tensorflow
import torch.optim as optim
import torch

from feature_extraction import FeatureType
from ioprocess import OfflineDataPreprocessor, init_base_config, Databases
from classifier import init_classifier, ClassifierType
from ai import train
torch.cuda.is_available()

class BCISystem(object):

    def __init__(self):
        self._base_dir = init_base_config()
        self._proc = OfflineDataPreprocessor(self._base_dir)

    def _train_classifier(self, trainx, trainy, testx, testy, classifier_type):
        if classifier_type == ClassifierType.CNN:
            classifier = init_classifier(classifier_type, (59,100), 2)
            print('Learning:')
            classifier.fit(x=trainx, y=trainy, batch_size=30, epochs=5)
            print('Test:')
            classifier.evaluate(testx, testy)
            return classifier
        elif classifier_type == ClassifierType.CellNN:
            classifier = init_classifier(classifier_type, (59,100), 2)
            classifier.cuda()
            opt = optim.Adam(classifier.parameters(), lr=0.001)
            for epoch in range(0, 2):
                print("Epoch %d" % epoch)
                train(classifier, epoch, trainx, trainy, testx, testy, opt)
            return classifier

    def offline_processing(self, db_name=Databases.BBCI_DE, feature_type=None, classifier_type=None):
        my_dict = self._proc._create_bbci_de_db(1)

        both_hands_features = np.empty((100,59,100))
        both_legs_features = np.empty((100,59,100))

        for i in range(99):
            np.append(both_hands_features, my_dict['both hands'][i][0][0])
            np.append(both_legs_features, my_dict['both legs'][i][0][0])

        features = np.concatenate([both_hands_features, both_legs_features])

        def our_generator():
            for i in range(200):
                x = features[i]
                if i < 100:
                    y = 0
                else:
                    y = 1
                yield x, y

        dataset = tensorflow.data.Dataset.from_generator(our_generator, (tensorflow.float32, tensorflow.int16))
        dataset = dataset.batch(200)
        dataset = dataset.shuffle(buffer_size=1000)

        train_size = int(0.7 * 200)
        test_size = int(0.15 * 200)

        train_dataset = dataset.take(train_size)
        test_dataset = dataset.take(test_size)

        iterator = tensorflow.compat.v1.data.make_one_shot_iterator(train_dataset)
        x, y = iterator.get_next()
        iterator2 = tensorflow.compat.v1.data.make_one_shot_iterator(test_dataset)
        testx, testy = iterator2.get_next()

        classifier = self._train_classifier(x, y, testx, testy, classifier_type)

if __name__ == '__main__':
    bci = BCISystem()
    bci.offline_processing(
        db_name=Databases.BBCI_DE,
        feature_type=FeatureType.RAW,
        classifier_type=ClassifierType.CNN)