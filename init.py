# initialize
from constants import *
from helpers import *

import numpy as np
from scipy.stats.stats import pearsonr, spearmanr

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
sns.set()

time_start, time_end, time_delta = 1800, 2000, 10

print('Getting english data')
proto = read_table_1987(table1_1987_path)
emotion_words = list(proto.keys())
data, pos_data = get_hamilton_data(time_start, time_end, time_delta)
freqs_all = pickle.load(open(freq_path_eng, 'rb'), encoding='latin1')

print('Getting french data')
proto_fr = read_table_1998(french_1998_path)
emotion_words_fr = list(proto_fr.keys())
data_fr, pos_data_fr, freqs_all_fr = get_hamilton_data_french(time_start, time_end, time_delta)


# init categories to use
print('Getting emotion words and bird names')

words_eng = [w for w in set(emotion_words) if proto[w] > 2.75] + ['awe', 'surprise']
words_eng = [w for w in words_eng if not w in {'abhorrence', 'ire', 'malevolence', 'titillation'}]

fr_words = lambda w: (w in pos_data_fr) and (len(pos_data_fr[w]) > 0) and (pos_data_fr[w][-1] == NOUN)
words_fr = [w for w in emotion_words_fr if fr_words(w)]

proto_leuven = read_leuven(goodness_ratings_path)['birds']
words_leuven = list(proto_leuven.keys())
proto_rosch = read_table_1987(bird_rosch_1975_path)
proto_rosch = {w: -v for w,v in proto_rosch.items()}
words_rosch = list(proto_rosch.keys())

eng_seeds = sorted(words_eng, key=lambda w:proto[w], reverse=True)
fra_seeds = sorted(words_fr, key=lambda w:proto_fr[w], reverse=True)
leuven_seeds = sorted(words_leuven, key=lambda w:proto_leuven[w], reverse=True)
rosch_seeds = sorted(words_rosch, key=lambda w:proto_rosch[w], reverse=True)

eng_seeds_reverse = eng_seeds[::-1]
fra_seeds_reverse = fra_seeds[::-1]
leuven_seeds_reverse = leuven_seeds[::-1]
rosch_seeds_reverse = rosch_seeds[::-1]

# overlap between rosch and leuven
overlap = set(proto_leuven.keys()) & set(proto_rosch.keys())
print('number of overlapping words between Leuven and Rosch:', len(overlap))
print('correlation between Leuven and Rosch ratings:', pearsonr([proto_leuven[w] for w in overlap], [proto_rosch[w] for w in overlap]))


