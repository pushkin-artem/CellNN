from enum import Enum
from pathlib import Path
import mne
import numpy as np
from config import BBCI_DE
from feature_extraction import FeatureType, FeatureExtractor

EPOCH_DB = 'preprocessed_database'

class SourceDB(Enum):
    SUBJECTS = 'subjects'
    DATA_SOURCE = 'data_source'
    INFO = 'epoch_info'
    FEATURE_SHAPE = 'feature_shape'

# db selection options
class Databases(Enum):
    BBCI_DE = 'bbci_de'

def init_base_config():
    base_directory = 'BBCI.DE'
    return base_directory

def _generate_filenames_for_subject(file_path, subject, subject_format_str, runs, run_format_str):
    if type(runs) is not list:
        runs = [runs]

    for i in runs:
        f = file_path
        f = f.replace('{subj}', subject_format_str.format(subject))
        f = f.replace('{rec}', run_format_str.format(i))
        yield f

def generate_bbci_de_filenames(file_path, subject, runs=1):
    return _generate_filenames_for_subject(file_path, subject, '{:02d}', runs, '{:02d}')

def get_epochs_from_bbci_raw(raw, task_dict, epoch_tmin=-0.2, epoch_tmax=0.5, baseline=None, event_id='auto', preload=True,
                        my_event_file=None):
    """Generate epochs from bbci raw data."""

    events = mne.read_events(my_event_file)

    baseline = tuple([None, epoch_tmin + 0.1])  # if self._epoch_tmin > 0 else (None, 0)
    epochs = mne.Epochs(raw, events, baseline=baseline, event_id=task_dict, tmin=epoch_tmin,
                        tmax=epoch_tmax, preload=preload, on_missing='warning')
    return epochs

def get_epochs_from_bbci_files(filenames, task_dict, epoch_tmin=-0.2, epoch_tmax=0.5, baseline=None, event_id='auto', preload=False):

    """Generate epochs from bbci files."""

    info_file = open(str(filenames[2]), 'r')
    lines = info_file.read().split('\n')
    info_d = dict()
    for l in lines:
        if l != '':
            pair = l.split(': ')
            if str(pair[0]) == 'fs':
                info_d[str(pair[0])] = int(pair[1])
            elif str(pair[0]) == 'classes':
                info_d[str(pair[0])] = pair[1].split(', ')
            elif str(pair[0]) == 'clab':
                info_d[str(pair[0])] = pair[1].split(', ')
            elif str(pair[0]) == 'xpos':
                info_d[str(pair[0])] = [float(i) for i in pair[1].split(', ')]
            elif str(pair[0]) == 'ypos':
                info_d[str(pair[0])] = [float(i) for i in pair[1].split(', ')]
    info_file.close()

    import re
    with open(str(filenames[0]), 'r') as f1:
        raw_data = [[0.1 * float(num) for num in re.split(r'\t+', line)] for line in f1]

    info = mne.create_info(ch_names=info_d['clab'], sfreq=info_d['fs'], ch_types='eeg')
    raw = mne.io.RawArray(data=np.array(raw_data).transpose(), info=info)
    xy = np.empty(shape=(len(info_d['xpos']), 2), dtype=float)
    for i in range(len(xy)):
        xy[i][0] = info_d['xpos'][i]
        xy[i][1] = info_d['ypos'][i]

    mne.datasets.eegbci.standardize(raw)
    my_layout = mne.channels.generate_2d_layout(xy=xy, ch_names=info_d['clab'], name='BBCI_DE')
    #my_layout.plot()
    my_event_file = str(filenames[1])

    epochs = get_epochs_from_bbci_raw(raw, task_dict, epoch_tmin, epoch_tmax, baseline, event_id, preload=preload, my_event_file=my_event_file)

    return epochs

class OfflineDataPreprocessor:

    def __init__(self, base_dir, epoch_tmin=0, epoch_tmax=4, window_length=1.0, window_step=0.1, feature_type=FeatureType.RAW):

        self._base_dir = Path(base_dir)
        self._epoch_tmin = epoch_tmin
        self._epoch_tmax = epoch_tmax  # seconds
        self._window_length = window_length  # seconds
        self._window_step = window_step  # seconds
        self._subject_list = list()
        self.feature_type = feature_type

        self.info = mne.Info()
        self.feature_kwargs = dict()
        self._feature_shape = tuple()

        self._data_path = Path(base_dir)
        self._db_type = BBCI_DE

        self.proc_db_path = Path(base_dir)
        self._proc_db_filenames = dict()
        self._proc_db_source = str()

    @property
    def fs(self):
        return self.info['sfreq']

    def generate_mne_epoch(self, data):
        if len(data.shape) == 2:
            data = np.expand_dims(data, 0)
        return mne.EpochsArray(data, self.info)

    def run(self, subject=None, feature_type=None, **feature_kwargs):
        self._subject_list = [subject] if type(subject) is int else subject
        self.feature_type = feature_type
        assert len(feature_kwargs) > 0, 'Feature parameters must be defined!'
        self.feature_kwargs = feature_kwargs

    def get_labels(self, make_binary_classification=False):
        subj = list(self._proc_db_filenames)[0]
        label_list = list(self._proc_db_filenames[subj])
        return label_list

    def get_feature_shape(self):
        return self._feature_shape

    def is_name(self, db_name):
        return db_name in str(self._db_type)

    def _convert_task(self, record_number=None):
        return self._db_type.TRIGGER_TASK_CONVERTER

    def _create_bbci_de_db(self, subj):
        task_dict = self._convert_task()

        cnt_filename = generate_bbci_de_filenames(str(self._data_path.joinpath(self._db_type.CNT_FILE_PATH)), subj)
        mrk_filename = generate_bbci_de_filenames(str(self._data_path.joinpath(self._db_type.MRK_FILE_PATH)), subj)
        nfo_filename = generate_bbci_de_filenames(str(self._data_path.joinpath(self._db_type.NFO_FILE_PATH)), subj)
        filenames = [str(*cnt_filename), str(*mrk_filename), str(*nfo_filename)]

        epochs = get_epochs_from_bbci_files(filenames, task_dict, self._epoch_tmin, self._epoch_tmax)
        self.info = epochs.info
        subject_data = self._get_windowed_features(epochs)
        return subject_data

    def _get_windowed_features(self, epochs, task=None):

        if task is not None:
            epochs = epochs[task]

        epochs.load_data()

        task_set = set([list(epochs[i].event_id.keys())[0] for i in range(len(epochs.selection))])

        window_length = self._window_length - 1 / self.fs  # win length correction
        win_num = int((self._epoch_tmax - self._epoch_tmin - window_length) / self._window_step) \
            if self._window_step > 0 else 1

        feature_extractor = FeatureExtractor(self.feature_type, self.fs, info=self.info, **self.feature_kwargs)

        task_dict = dict()
        for task in task_set:
            tsk_ep = epochs[task]
            win_epochs = {i: list() for i in range(len(tsk_ep))}
            for i in range(win_num):  # cannot speed up here with parallel process...
                ep = tsk_ep.copy()
                ep.crop(ep.tmin + i * self._window_step, ep.tmin + window_length + i * self._window_step)
                feature = feature_extractor.run(ep.get_data())
                f_shape = feature.shape[1:]
                if len(self._feature_shape) > 0:
                    assert f_shape == self._feature_shape, 'Error: Change in feature output shape. prev: {},  ' \
                                                           'current: {}'.format(self._feature_shape, f_shape)
                self._feature_shape = f_shape
                for j in range(len(tsk_ep)):
                    win_epochs[j].append((feature[j], np.array([task])))
            task_dict[task] = win_epochs

        return task_dict