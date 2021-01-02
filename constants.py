import os

#################### example words in illustrations ##################
eng_examples = [
    'hysteria', 'zest', 'exhilaration',
    'joy', 'anger', 'fear',
]
fra_examples = [
    'hystérie', 'allégresse', 'soupçon',
    'joie', 'colère', 'jalousie'
]
example_birds = [
	'eagle', 'sparrow', 'chicken', 'turkey', 'crow', 'geese'
]

#################### data type ##################
# pos
NOUN = 'n'
ADJ = 'a'
VERB = 'v'
pos_placeholder = ''

# data used
ENG = 'ENG'
FRA = 'FRA'
TABLE1_1987 = 1 # only nouns
FRENCH_1998 = 3
ROSCH_1975 = 8

#################### data and save files ##################
HOME = os.environ['HOME']

npy_path_eng = HOME + "/data/Hamilton/"
vocab_path_eng = HOME + "/data/Hamilton/"
pos_path_eng = HOME + "/data/Hamilton/"
npy_path_fra = HOME + "/data/Hamilton_Fra/sgns/"
vocab_path_fra = HOME + "/data/Hamilton_Fra/sgns/"
pos_path_fra = HOME + "/data/Hamilton_Fra/pos/"
freq_path_eng = "./data/freqs.pkl"
freq_path_fra = HOME + "/data/Hamilton_Fra/freqs.pkl"

table1_1987_path = "./data/1987_ratings"
table2_1987_path = "./data/shaver_valence"
french_1998_path = "./data/french"
english_1998_path = "./data/french"
bird_rosch_1975_path = "./data/1975_birds"

save_folder = "./saved_files/"
HTE_senses = save_folder + "senses_HTE_%s_%d.pkl" # pos, data used

