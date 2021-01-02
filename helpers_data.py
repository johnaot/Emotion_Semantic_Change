"""
Useful functions
"""
import numpy as np
import pandas as pd
import gensim
from openpyxl import Workbook, load_workbook

import string, pickle, os
from collections import defaultdict

from constants import *

#################### Read data from Niedenthal 1998 ##################
# get french words from appendix
def read_table_1998(p):
  data = defaultdict(int)
  words, ratings = [], []
  with open(p + '/french_edit', 'r') as f:
    lines = f.readlines()
    words = [line.strip() for line in lines if line != '\n']
  with open(p + '/ratings', 'r') as f:
    content = f.read()
    content = content.replace('\n', ' ')[0:-1]
    ratings = content.split(' ')
  for i, w in enumerate(words):
    data[w.lower()] = float(ratings[i])
  return data

def read_table_1998_valence(p):
  data = defaultdict(int)
  words, valence = [], []
  with open(p + '/french_edit', 'r') as f:
    lines = f.readlines()
    words = [line.strip() for line in lines if line != '\n']
  with open(p + '/valence', 'r') as f:
    lines = f.readlines()
    valence = [v.strip() for v in lines]
  for i, w in enumerate(words):
    data[w.lower()] = float(valence[i])
  return data

# get english equivalents
def read_english_1998(p):
  data = defaultdict(int)
  words, eng_words = [], []
  with open(p + '/french_edit', 'r') as f:
    lines = f.readlines()
    words = [line.strip() for line in lines if line != '\n']
  with open(p + '/english', 'r') as f:
    lines = f.readlines()
    eng_words = [line.strip() for line in lines if line != '\n']
  for i, w in enumerate(words):
    data[w.lower()] = eng_words[i].lower()
  return data

#################### Read data from Shaver 1987 ####################
# get emotion words from Table 1
def read_table_1987(p):
  data = defaultdict(int)
  with open(p, 'r') as f:
    lines = f.readlines()
    for line in lines:
      line = line.split(' ')
      if len(line) == 1:
        continue
      data[line[0]] = float(line[1])
  return data

# get emotion words from Table 2
def read_table2_1987(p):
  data = defaultdict(int)
  with open(p, 'r') as f:
    lines = f.readlines()
    i = 0
    words = []
    while i < len(lines):
      line = lines[i].strip()
      if line == 'Evaluation':
        for w in words:
          i += 1
          line = lines[i].strip()
          data[w] = float(line)
      elif line == 'Intensity':
        for w in words:
          i += 1
        words = []
      else:
        words.append(line)
      i += 1
  return data

#################### Read data form HistWords (Hamilton 2016) ####################
def get_hamilton_data(time_start, time_end, time_delta, lang):
  time_range = list(range(time_start, time_end, time_delta))

  npy_path, vocab_path, pos_path = '', '', ''
  if lang == ENG:
    npy_path = os.path.join(npy_path_eng, "%d-w.npy")
    vocab_path = os.path.join(vocab_path_eng, "%d-vocab.pkl")
    pos_path = os.path.join(pos_path_eng, "%d-pos.pkl")
  elif lang == FRA:
    npy_path = os.path.join(npy_path_fra, "%d-w.npy")
    vocab_path = os.path.join(vocab_path_fra, "%d-vocab.pkl")
    pos_path = os.path.join(pos_path_fra, "%d-pos.pkl")

  data = defaultdict(list)
  pos_data = defaultdict(list)
  for t in time_range:
    vocab = pickle.load(open(vocab_path % t, 'rb'), encoding='latin1')
    pos = pickle.load(open(pos_path % t, 'rb'), encoding='latin1')
    vectors = np.load(npy_path % t)

    for i, w in enumerate(vocab):
      data[w].append(vectors[i])
      if w in pos:
        if pos[w] == 'NOUN':
          pos_data[w].append(NOUN)
        elif pos[w] == 'ADJ':
          pos_data[w].append(ADJ)
        elif pos[w] == 'VERB':
          pos_data[w].append(VERB)
        else:
          pos_data[w].append(pos_placeholder)
      else:
        pos_data[w].append(pos_placeholder)

  return(data, pos_data)


