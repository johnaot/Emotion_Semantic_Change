#################### Options ##################
# pos
NOUN = 'n'
ADJ = 'a'
VERB = 'v'
pos_placeholder = ''

# data used
TABLE1_1987 = 1 # only nouns
FRENCH_1998 = 3
LEUVEN = 4
NORMS_1980 = 7
ROSCH_1975 = 8

# semantic change measures
FIRST_LAST = 0
OVERLAP = 2

#################### Data and save files ##################
emotion_path = "./data/Emotion_list.xlsx"
table1_1987_path = "./data/1987_ratings"
table2_1987_path = "./data/shaver_valence"
freq_path = "./data/freqs.pkl"

french_1998_path = "./data/french"
english_1998_path = "./data/french"

bird_rosch_1975_path = "./data/1975_birds"
goodness_ratings_path = './data/leuven/exemplar judgments/exemplarGoodnessRatings.xls'
norms_1980_path = './data/norms_1980'

save_folder = "./saved_files/"
csv_file = save_folder + "emotions_%s_%d_%d.csv" # pos, data used, semantic change measure
pickle_file = save_folder + "changes_%s_%d_%d.pkl" # pos, data used, semantic change measure

HTE_senses = save_folder + "senses_HTE_%s_%d.pkl" # pos, data used

