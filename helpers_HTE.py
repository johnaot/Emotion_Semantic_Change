import sys, os, pickle
sys.path.append("./HTE Reader/")
from word import Word
from constants import *

from scipy.stats.stats import pearsonr, spearmanr

def hack_sense_objs(s):
  s = s.categories
  pos = ''
  data_str = s[s.index('{'):]
  if 'adj.' in s:
    pos = ADJ
  elif 'vi.' in s or 'vt.' in s:
    pos = VERB
  elif 'n.' in s:
    pos = NOUN
  return(pos, eval(data_str))

def get_senses(sense_file, words, t_to_use):
  senses = {}
  if os.path.isfile(sense_file):
    senses = pickle.load(open(sense_file, 'rb'))
  else:
    get_sense = Word('','')
    senses = {w: get_sense.make_sense_list(w, 'all') for w in words}
    pickle.dump(senses, open(sense_file, 'wb'))

  num_senses, first = {}, {}
  for w in words:
    if not w in senses or len(senses[w]) == 0:
      print('No HTE entry:', w)
      continue
    senses_processed = [hack_sense_objs(s) for s in senses[w]]
    senses_processed = [sense for pos, sense in senses_processed if pos == NOUN]
    processed_times = []
    for i, s in enumerate(senses_processed):
      for j, t in enumerate(s['times']):
        if not 'ending_time' in t:
          t['ending_time'] = t['starting_time']
        processed_time = (t['starting_time'], t['ending_time'])
        processed_times.append(processed_time)
    if len(processed_times) == 0:
      continue
    first[w] = sorted([t[0] for t in processed_times])[0]
    num_senses[w] = len([t for t in processed_times if t[0] <= t_to_use and t[-1] >= t_to_use])
  return(num_senses, first)

def corr_proto_HTE(words, first, proto):
  r, p = pearsonr([first[w] for w in words], [proto[w] for w in words])
  return r, p, len(words)

