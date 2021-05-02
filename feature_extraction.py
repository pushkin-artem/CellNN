from enum import Enum, auto

import numpy as np

# features
class FeatureType(Enum):
    RAW = auto()

class FeatureExtractor:

    def __init__(self, feature_type, fs=None,
                 fft_low=None, fft_high=None, fft_step=2, fft_width=2, fft_ranges=None,
                 channel_list=None, info=None, crop=True):
        self.feature_type = feature_type
        self.fs = fs

        self.fft_low = fft_low
        self.fft_high = fft_high
        self.fft_step = fft_step
        self.fft_width = fft_width
        self.fft_ranges = fft_ranges

        self.channel_list = channel_list
        self._crop = crop
        self._info = info  # todo: change back info to interp when it is possible...

    def run(self, data):

        if len(data.shape) == 2:
            data = np.expand_dims(data, 0)
        if self.channel_list is not None:
            data = data[:, self.channel_list, :]
            print('It is assumed that the reference electrode is POz!!!')

        elif self.feature_type == FeatureType.RAW:
            feature = data
        else:
            raise NotImplementedError('{} feature is not implemented'.format(self.feature_type))

        return feature