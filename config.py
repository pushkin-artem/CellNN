"""
Configuration for databases
"""
from enum import Enum

# Task types:
BOTH_HANDS = 'both hands'
BOTH_LEGS = 'both legs'
SUBJECT = 'subject'


class ControlCommand(Enum):
    LEFT = 1
    RIGHT = 3
    HEADLIGHT = 2
    STRAIGHT = 0


DIR_FEATURE_DB = 'tmp/'

# Record types:
IMAGINED_MOVEMENT = "imagined"
REAL_MOVEMENT = "real"
BASELINE = 'baseline'

class BBCI_DE:
    DIR = "BBCI.DE/"
    CNT_FILE_PATH = 'BCICIV_1_asc/subject{subj}/BCICIV_calib_ds{subj}_cnt.txt'
    MRK_FILE_PATH = 'BCICIV_1_asc/subject{subj}/BCICIV_calib_ds{subj}_mrk.txt'
    NFO_FILE_PATH = 'BCICIV_1_asc/subject{subj}/BCICIV_calib_ds{subj}_nfo.txt'

    TRIGGER_TASK_CONVERTER = {
        BOTH_HANDS: 1,
        BOTH_LEGS: -1
    }

    TRIGGER_EVENT_ID = {'Stimulus/S ' + (2 - len(str(i + 1))) * ' ' + str(i + 1): i + 1 for i in range(16)}

    DROP_SUBJECTS = []