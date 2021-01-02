import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
sns.set_style("ticks")
sns.despine()

from helpers_compute import vectors_filter

### create plots for analyses ###  
def plot_line_of_best_fit(ax, x, y, xlabel, ylabel, colour='tab:blue'):
    df = pd.DataFrame({
        'x': x, 
        'y': y, 
    }) 
    ax.scatter(x, y, color=colour, alpha=0.5)
    ax = sns.regplot(ax=ax, x="x", y="y", scatter=False, color=colour, data=df)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20) 
    ax.set_xlabel(xlabel, fontsize=25)
    ax.set_ylabel(ylabel, fontsize=25)
    return ax

def annotate_scatter(ax, example_words, all_words, xs, ys):
    for w in example_words:
        x = xs[all_words.index(w)]
        y = ys[all_words.index(w)]
        ax.annotate(w, xy=(x,y), xytext=(x,y), color='black', fontsize=20)
    return ax

def plot_coefficents(ax, results, predictors, labelsize=30):
    params = {
        'regression coefficients': results.params[1:],
        'predictors': predictors
    }
    ax = sns.barplot(ax=ax, x="regression coefficients", y="predictors", xerr=results.bse[1:], data=params, errwidth=5)
    x = 0.001+max(results.params[1:])
    if x < 0: ax.set_xlim(right=0.001)
    x = max(x, 0.001)
    for i, p in enumerate(results.pvalues[1:]):
        if p < 0.001:
            ax.text(x, i, '***', fontsize=30)
        elif p < 0.01:
            ax.text(x, i,  '**', fontsize=30)
        elif p < 0.05:
            ax.text(x, i, '*', fontsize=30)
        else:
            ax.text(x, i, 'n.s.', fontsize=30)
    ax.tick_params(axis='both', labelsize=24)
    ax.set_ylabel('Predictors', fontsize=labelsize)
    ax.set_xlabel('Regression Coefficients', fontsize=labelsize)
    return ax

### create pca illustration ###
def make_pca_plot(axes, emotion_words, vectors, t1, t2):
    font_size = 35
    linewidth = 8
    anchor_s, example_s = 750, 1500
    anchor_colour = 'tab:purple'
    words_visual_pca = [
        'anger', 
        'love', 
        'fear', 
        'happiness', 
        'sadness', 
        'awe', 
        'disgust', 
        'desire', 
        'sympathy'
    ]
    word_marker = {
        'anger': '_',
        'love': '+',
        'fear': '_',
        'happiness': '+',
        'sadness': '_',
        'awe': '.',
        'disgust': '.',
        'desire': '+',
        'sympathy': '+',
    }
    
    ts = [t1, t2]
    vectors_pca = [
        [vectors[w][t1] for w in emotion_words if vectors_filter(w, t1, vectors) and vectors_filter(w, t2, vectors)],
        [vectors[w][t2] for w in emotion_words if vectors_filter(w, t1, vectors) and vectors_filter(w, t2, vectors)]
    ]
    pca = PCA(n_components=2, svd_solver='full')
    pca.fit(vectors_pca[0])
    for i, X in enumerate(vectors_pca):
        ax = axes[i]
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        for j, w in enumerate(words_visual_pca):
            x, y = pca.transform([vectors[w][ts[i]]])[0]
            if i == 0:
                if w == 'awe':
                    ax.scatter(x, y, zorder=1, label=w, s=example_s, color='tab:brown', marker=word_marker[w])
                    ax.annotate(w, xy=(x,y), xytext=(x,y), color='black', fontsize=font_size, )
                elif w == 'disgust':
                    ax.scatter(x, y, zorder=1, label=w, s=example_s, color='tab:pink', marker=word_marker[w])
                    ax.annotate(w, xy=(x,y-0.025), xytext=(x,y-0.025), color='black', fontsize=font_size, )
                elif w == 'happiness':
                    ax.scatter(x, y, zorder=1, label=w, s=anchor_s, linewidth=linewidth, color=anchor_colour, marker=word_marker[w])
                    ax.annotate(w, xy=(x-0.05,y), xytext=(x-0.26,y), color='black', fontsize=font_size, alpha=0.4)
                else:
                    ax.scatter(x, y, zorder=1, label=w, s=anchor_s, linewidth=linewidth, color=anchor_colour, marker=word_marker[w])
                    ax.annotate(w, xy=(x,y), xytext=(x+0.025,y), color='black', fontsize=font_size, alpha=0.4)
            else:
                if w == 'awe':
                    ax.scatter(x, y, zorder=1, label=w, s=example_s, color='tab:brown', marker=word_marker[w])
                    ax.annotate(w, xy=(x,y), xytext=(x,y), color='black', fontsize=font_size, )
                elif w == 'disgust':
                    ax.scatter(x, y, zorder=1, label=w, s=example_s, color='tab:pink', marker=word_marker[w])
                    ax.annotate(w, xy=(x,y), xytext=(x,y), color='black', fontsize=font_size, )
                elif w == 'happiness':
                    ax.scatter(x, y, zorder=1, label=w, s=anchor_s, linewidth=linewidth, color=anchor_colour, marker=word_marker[w])
                    ax.annotate(w, xy=(x-0.05,y), xytext=(x-0.09,y-0.03), color='black', fontsize=font_size, alpha=0.4)                
                elif w == 'desire':
                    ax.scatter(x, y, zorder=1, label=w, s=anchor_s, linewidth=linewidth, color=anchor_colour, marker=word_marker[w])
                    ax.annotate(w, xy=(x-0.05,y), xytext=(x-0.072,y+0.01), color='black', fontsize=font_size, alpha=0.4)
                else:
                    ax.scatter(x, y, zorder=1, label=w, s=anchor_s, linewidth=linewidth, color=anchor_colour, marker=word_marker[w])
                    ax.annotate(w, xy=(x,y), xytext=(x+0.015,y), color='black', fontsize=font_size, alpha=0.4)                
        if i == 0:
            ax.set_ylabel('Second principal component', fontsize=30)
            ax.set_xlabel('First principal component', fontsize=30)
            ax.set_title('1890s', fontsize=35, weight='bold')
        else:
            ax.set_xlabel('First principal component', fontsize=30)
            ax.set_title('1990s', fontsize=35, weight='bold')
            ax.set_xlim((-0.25,0.25))
            ax.set_ylim((-0.18,0.31))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    return axes

def set_x_time_range(ax, time_range):
    ax.set_xticks(time_range)
    ax.set_xticklabels(['',1900,'','','',1940,'','','',1980], fontsize=20)
    return ax  

### create kde illustration ###  
def estimate_bounded(x, lb, ub, samples, h):
    transform = lambda x: np.log(x - lb) - np.log(ub - x)
    y = transform(x).reshape(-1, 1)
    kde = KernelDensity(kernel='gaussian', bandwidth=h).fit(y)
    ret = np.exp(kde.score_samples(transform(samples).reshape(-1, 1)))
    ret = ret * (1 / (samples - lb) + 1 / (ub - samples))
    return ret

def plot_kde_bounded(x, h, p, ax, label, color, scatter_y):
    samples = np.linspace(0.001, 1 - 0.001, 1000)
    density = estimate_bounded(np.array([y for y in x if y < 1.]), 0, 1, samples, h)
    ax.plot(np.concatenate([[0], samples, [1]]), np.concatenate([[0], density, [0]]), label=label, color=color)
    ax.fill_between(samples, density, step="pre", alpha=0.4, color=color)
    ax.set_xlabel('Degree of change', fontsize=30)
    ax.set_ylabel('Density', fontsize=30)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.scatter(x, [scatter_y] * len(p), color=color, marker='o', s=200*p**2)
    return ax

def plot_kde(x, h, p, ax, label, color, scatter_y):
    ax = sns.kdeplot(x, ax=ax, shade=True, color=color, label=label, kernel='gau')
    ax.set_xlabel('Degree of change', fontsize=30)
    ax.set_ylabel('Density', fontsize=30)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.scatter(x, [scatter_y] * len(p), color=color, marker='o', s=200*p**2)
    return ax

