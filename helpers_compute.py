from heapq import nlargest
import pickle

import numpy as np
import pandas as pd

from constants import *

### compute knn rates of change ###
def vectors_filter(w, t, vectors):
  exist = (w in vectors) and (len(vectors[w]) > 0)
  nonzero = exist and not all(vectors[w][t] == 0)
  return nonzero

def get_closest_neighbours(w, pos, pos_data, vectors, t, k):
  other_words = [v for v in vectors if not v == w]
  other_words = [v for v in other_words if pos == pos_data[v][t]]
  other_words = [v for v in other_words if vectors_filter(v, t, vectors)]
  similarities = {v: np.dot(vectors[w][t], vectors[v][t]) for v in other_words}
  return nlargest(k, other_words, key=lambda x:similarities[x])

def nn_measure(w, pos, pos_data, vectors, t1, t2, k):
  nns_t1 = set(get_closest_neighbours(w, pos, pos_data, vectors, t1, k))
  nns_t2 = set(get_closest_neighbours(w, pos, pos_data, vectors, t2, k))
  return 1 - len(nns_t1 & nns_t2) / len(nns_t1 | nns_t2)

def nn_measure_over_words(words, pos, pos_data, vectors, t1, t2, k):
  ret = {}
  for w in words:
    if not vectors_filter(w, t1, vectors):
      continue
    if not vectors_filter(w, t2, vectors):
      continue
    ret[w] = nn_measure(w, pos, pos_data, vectors, t1, t2, k)
  return ret

### compute knn rates of change, using category bounded measure ###
def get_closest_neighbours_filtered(w, others, pos, pos_data, vectors, t, k):
    other_words = [v for v in others if w != v]
    other_words = [v for v in other_words if vectors_filter(v, t, vectors)]
    similarities = {v: np.dot(vectors[w][t], vectors[v][t]) for v in other_words}
    return nlargest(k, other_words, key=lambda x:similarities[x])

def nn_measure_filtered(w, others, pos, pos_data, vectors, t1, t2, k):
    nns_t1 = set(get_closest_neighbours_filtered(w, others, pos, pos_data, vectors, t1, k))
    nns_t2 = set(get_closest_neighbours_filtered(w, others, pos, pos_data, vectors, t2, k))
    return 1 - len(nns_t1 & nns_t2) / len(nns_t1 | nns_t2)

def nn_measure_over_words_filtered(words, pos, pos_data, vectors, t1, t2, k):
    ret = {}
    for w in words:
        if not vectors_filter(w, t1, vectors):
            continue
        if not vectors_filter(w, t2, vectors):
            continue
        ret[w] = nn_measure_filtered(w, words, pos, pos_data, vectors, t1, t2, k)
    return ret

### estimate historical prototypicality ###  
def prototype_density(words, vectors, t):
    prototype = np.mean([vectors[w][t] for w in words], axis=0)
    return {w: (2*np.pi)**0.5 * np.exp(-np.sum((vectors[w][t] - prototype)**2)) for w in words}

def vector_prototype(words, vectors, t):
    prototype = np.mean([vectors[w][t] for w in words], axis=0)
    return {w: np.dot(prototype, vectors[w][t]) for w in words}

def vector_prototype_rosch(words, proto, vectors, t):
    prototypes = sorted(words, key=lambda w:proto[w], reverse=True)[0:1]
    prototype = np.mean([vectors[w][t] for w in prototypes], axis=0)
    return {w: np.dot(prototype, vectors[w][t]) for w in words if not w in prototypes}

def vector_prototype_tversky(words, vectors, t):
    sums = dict()
    for w in words:
        prototype = vectors[w][t]
        sums[w] = np.sum([np.dot(prototype, vectors[v][t]) for v in words])
    prototype = sorted(words, key=lambda w: sums[w], reverse=True)[0]
    return {v: np.dot(vectors[prototype][t], vectors[v][t]) for v in words}

