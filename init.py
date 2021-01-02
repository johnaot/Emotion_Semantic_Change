# initialize
from constants import *
from helpers_data import *
from helpers_compute import *
from helpers_plot import *

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

# initialize historical data
time_start, time_end, time_delta = 1800, 2000, 10

print('Getting English data')
proto = read_table_1987(table1_1987_path)
emotion_words = list(proto.keys())
data, pos_data = get_hamilton_data(time_start, time_end, time_delta, ENG)
freqs_all = pickle.load(open(freq_path_eng, 'rb'), encoding='latin1')

print('Getting French data')
proto_fr = read_table_1998(french_1998_path)
emotion_words_fr = list(proto_fr.keys())
data_fr, pos_data_fr = get_hamilton_data(time_start, time_end, time_delta, FRA)
freqs_all_fr = pickle.load(open(freq_path_fra, 'rb'), encoding='latin1')

# initialize empirical data
print('Getting emotion words and bird names')

words_eng = [w for w in set(emotion_words) if proto[w] > 2.75] + ['awe', 'surprise']
words_eng = [w for w in words_eng if not w in {'abhorrence', 'ire', 'malevolence', 'titillation'}]

fr_words = lambda w: (w in pos_data_fr) and (len(pos_data_fr[w]) > 0) and (pos_data_fr[w][-1] == NOUN)
words_fr = [w for w in emotion_words_fr if fr_words(w)]

proto_rosch = read_table_1987(bird_rosch_1975_path)
words_rosch = list(proto_rosch.keys())
