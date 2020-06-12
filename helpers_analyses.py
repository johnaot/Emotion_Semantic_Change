from heapq import nlargest
import pickle

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
sns.set()
from scipy.stats.stats import pearsonr, spearmanr
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pingouin as pg

from constants import *
from helpers import *

def vectors_filter(w, t, vectors):
  exist = (w in vectors) and (len(vectors[w]) > 0)
  nonzero = exist and not all(vectors[w][t] == 0)
  return nonzero

def estimate_proto(seeds, words, vectors, proto, t=-1, seed_num=5):
  words = [w for w in words if vectors_filter(w, t, vectors)]
  seeds = [w for w in seeds if w in words][0:seed_num]
  print('seeds:', seeds)

  prototype = np.mean([vectors[w][t] for w in seeds], axis=0)
  proto_w = [proto[w] for w in words]
  estimate_w = [similarity(vectors[w][t], prototype) for w in words]
  return(proto_w, estimate_w)

def get_closest_neighbours(w, pos, pos_data, vectors, t, k):
  other_words = [v for v in vectors if not v == w]
  other_words = [v for v in other_words if pos == pos_data[v][t]]
  other_words = [v for v in other_words if vectors_filter(v, t, vectors)]
  similarities = {v: similarity(vectors[w][t], vectors[v][t]) for v in other_words}
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

def regression_results(words, changes, proto, freqs, senses={}, valence={}, print_results=True):
  words = list(words)
  data = {
    'prototypicality': [proto[w] for w in words], 
    'change': [changes[w] for w in words], 
    'freqs': [freqs[w] for w in words], 
    'lens': [len(w) for w in words], 
  }
  formula = 'change ~ prototypicality + freqs + lens'
  
  if len(valence) > 0:
    data['valence'] = [valence[w] for w in words]
    formula += ' + valence'
  if len(senses) > 0:
    data['num_senses'] = [senses[w] for w in words]
    formula += ' + num_senses'
  
  # multiple regression
  df = pd.DataFrame(data) 
  model = smf.ols(formula, data=data)
  results = model.fit()
  
  # partial correlations
  part_corr = pg.partial_corr(df, y='change', x='prototypicality', covar='freqs')
  part_corr_freq = pg.partial_corr(df, y='change', x='freqs', covar='prototypicality')
  
  if print_results:
    print(results.summary(), '\n')
    print('partial correlation, change and prototypicality given frequency')
    print(part_corr)
    print('partial correlation, change and frequency given prototypicality')
    print(part_corr_freq)
  
  pr = part_corr['r'].iloc[0]
  pp = part_corr['p-val'].iloc[0]
  pr2 = part_corr_freq['r'].iloc[0]
  pp2 = part_corr_freq['p-val'].iloc[0]
  return(results, len(words), pr, pp, pr2, pp2)

def plot_change_scatter(ax, words, proto, changes):
  df = pd.DataFrame({
    'prototypicality': [proto[w] for w in words], 
    'change': [changes[w] for w in words], 
  }) 
  ax = sns.regplot(ax=ax, x="prototypicality", y="change", data=df)
  ax.xaxis.set_tick_params(labelsize=20)
  ax.yaxis.set_tick_params(labelsize=20) 
  ax.set_xlabel('Prototypicality Rating', fontsize=30)
  ax.set_ylabel('Degree of Semantic Change', fontsize=30)
  r, p = pearsonr(df['prototypicality'], df['change'])
  rs, ps = spearmanr(df['prototypicality'], df['change'])
  print('Pearson: r=%f, p-value=%f' % (r,  p))
  print('Spearman: r=%f, p-value=%f' % (rs,  ps))
  return ax

def plot_coefficents(ax, results, predictors):   
  ax.tick_params(axis='both', labelsize=24)
  params = {
    'regression coefficients': results.params[1:],
    'predictors': predictors
  }
  ax = sns.barplot(ax=ax, x="regression coefficients", y="predictors", xerr=results.bse[1:], data=params, errwidth=5)
  x = 0.001+max(results.params[1:])
  for i, p in enumerate(results.pvalues[1:]):
    if p < 0.001:
      ax.text(x, i, '***', fontsize=30)
    elif p < 0.01:
      ax.text(x, i,  '**', fontsize=30)
    elif p < 0.05:
      ax.text(x, i, '*', fontsize=30)
    else:
      ax.text(x, i, 'n.s.', fontsize=30)
  ax.set_ylabel('Predictors', fontsize=30)
  ax.set_xlabel('Regression Coefficients', fontsize=30)
  return ax
