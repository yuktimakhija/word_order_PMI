from io import open
import nltk
import pandas as pd
import re
from scipy.stats import spearmanr, rankdata

pmi_df = pd.read_csv('pmi_EWT.csv').set_index('Tags')
df_ranks = pd.read_csv('../../brown-ranks.csv')
df = pd.DataFrame()

# Open the text file with sentences
file = open("../../brown.txt", "r")
dep_index = open('../../brown_index.dep','r')
dep_trees = dep_index.readlines()
lines = file.readlines()

# Finding pmi values from csv
def get_pmi(head_pos,dep_pos):
	return pmi_df.loc[head_pos,dep_pos]
	# for index, row in pmi_df.iterrows():
	# 	if row['h']==head_pos and row['d']==dep_pos :
	# 		return row['pmi']

#Tag converter
def tag(corpus_tag):
	tag  = nltk.map_tag('brown', 'universal', corpus_tag)
	if tag == 'CONJ':
		tag = 'CCONJ'
	if tag == 'PRT' or corpus_tag == 'POS' or corpus_tag == 'UH':
		tag = 'PART'
	if corpus_tag == 'VBP' or corpus_tag == 'VBD|VBN':
		tag = 'VERB'
	if corpus_tag == 'NNP'or corpus_tag == 'NNPS':
		tag = 'PROPN'
	if corpus_tag == 'RBS':
		tag = 'ADV'
	if corpus_tag in ['WP','PRP', 'PRP$']:
		tag = 'PRON'
	if corpus_tag == 'PDT':
		tag = 'DET'
	if corpus_tag == '$' or corpus_tag == 'SYM':
		tag = 'SYM'
	if corpus_tag == 'NN|NNS':
		tag = 'NOUN'

	# TO - PRT	
	# VBP - VERB (NLTK - X)
	return tag

# Check for alphabet in string
def str_alpha(string_1):
	return any(c.isalpha() for c in string_1)

# Finding all PMI features
def pmi_features(idx,label):
	print(idx)
	for i in range(len(dep_trees)):
		# if dep_trees[i] == '\n'+idx+'\n' or dep_trees[i] == idx+'\n':
		if idx in dep_trees[i]:
			index_line = i
			i+=1
			break
	post_words_type, post_lg = [], []
	flag_root=0
	flag_adj = 0
	adj_tag = ''
	adj = ''
	# print(i)
	while(dep_trees[i][0].isnumeric()):
		# print(dep_trees[i])
		split = dep_trees[i].split('\t')
		# print(split)
		split[0] = int(split[0])
		split[6] = int(split[6])
		if 'ROOT' in dep_trees[i]:
			flag_root = 1
			# print('Root found')
			root = split[1]
			# print(root)
			root_tag = split[4]
			# print(root_tag)
			root_pos = split[0]
			# print(root_pos)
			root_line = i
			# print(root_line)
			if tag(root_tag) != 'VERB':
				return 'non-verbal root'
		# if flag_root == 1 and i != root_line and split[1][0].isalpha():
		# if flag_root == 1 and i != root_line and (split[7] != 'P'):
		if flag_root == 1 and i != root_line and str_alpha(split[1]):
			# print('working')
			post_lg.append(split[0]-root_pos)
			post_words_type.append(split[4])
			# if split[6] == root_pos and split[1][0].isalpha():
			# if split[6] == root_pos and split[7] != 'P':
			if split[1][0].isalpha() and flag_adj == 0:
				adj = split[1]
				adj_tag = split[4]
				flag_adj = 1
		i+=1
	for j in range(index_line+1,root_line):
		split = dep_trees[j].split('\t')
		split[0] = int(split[0])
		split[6] = int(split[6])
		# if split[6] == root_pos and split[1][0].isalpha():
		# if split[6] == root_pos and split[7] != 'P':
	if adj_tag == '':
		pmi_adj = 0
	else:
		pmi_adj = get_pmi(tag(root_tag),tag(adj_tag))
	pmi_words =  []
	for p in post_words_type:
		pmi_words.append(get_pmi(tag(root_tag),tag(p)))

	# print(post_words_type)
	# print(post_lg)
	# print(post_dep)
	# print(post_dlg)
	# print(pmi_dep)
	# print(root,' : ',root_pos,' : ',root_tag)
	#Reverse ranking
	pmi_word_rev_rank = []
	if len(pmi_words)>=1:
		# print('ALL WORDS')
		# print(post_words_type)
		# print(pmi_words)
		# print(post_lg)
		for r in rankdata(pmi_words):
			pmi_word_rev_rank.append(len(pmi_words) + 1 -r)
	# print(pmi_word_rev_rank)
	# if len(pmi_word_rev_rank)>1:        
	# 	coef_post_words, p1 = spearmanr(pmi_word_rev_rank, rankdata(post_lg))
	# 	if pd.isna(coef_post_words):
	# 		coef_post_words = 0 
	# else:
	# 	coef_post_words = 0
	coef_post_words, p1 = spearmanr(pmi_word_rev_rank, rankdata(post_lg))
	if pd.isna(coef_post_words):
		coef_post_words = 0 
	
	dict = {'id':idx,'label':label,'root_verb':root, 
			'adjacent_post':adj,'adjacent_tag':adj_tag,'pmi_adjacent':pmi_adj,
			'Spearman_allwords':coef_post_words}
	
	return dict

index, label = [], []
index = list(df_ranks['id'])
label = list(df_ranks['label'])
# index = ['cf01.2.1','cf01.7.1','cf01.11.1']
# index = ['cf06.73.1']
# label = ['pre-adv','var','var']
exceptions = ['cf14.34.','cf29.79.','cg02.12.']
# exceptions = ['cf14.34.','cf29.79.','cg02.12.','cr09.85.']
for i in range(len(index)):
	# if index[i][:2] == 'cf':
	idx_var = index[i]
	type_var = label[i]
	if idx_var[:len(idx_var)-1] in exceptions:
		continue
	idx_ref = idx_var[:len(idx_var)-1]+'0'
	if idx_var[len(idx_var)-1] == '1':
		index.append(idx_ref)
		label.append('ref')
		index.append(idx_var)
		label.append(type_var)
		ref = pmi_features(idx_ref,'ref')
		var = pmi_features(idx_var,type_var)
		if ref == 'non-verbal root' or var == 'non-verbal root':
			continue 
		df = df.append(pmi_features(idx_ref,'ref'), ignore_index=True)
		df = df.append(pmi_features(idx_var,type_var), ignore_index=True)

	else:
		index.append(idx_var)
		label.append(type_var)
		var = pmi_features(idx_var,type_var)
		if var == 'non-verbal root' or idx_ref not in list(df['id']):
			continue 
		df = df.append(pmi_features(idx_var,type_var), ignore_index=True)
	# print(df)

df.to_csv('brown_spearmanallwords_adjword.csv')

# In sentences cf15.15.0 and cf15.15.1, 'a' (position 13) was given the tag ','. Changed it to 'DT'
# cl02.17.0 and it's var changed NN|SYM to SYM
# cn02.153.1 and it's var changed NN|VBG to NN (running)
# cn02.130.1 and it's var changed RB|RP to RP (word: by)
# cn02.145.0 and it's var changed JJ|VBG to JJ (word: cloying)
# cn13.53.0 and it's var changed JJ|VBG to JJ (word: lilting)
# cp16.178.0 and it's var changed CD|RB to JJ (word: one o'clock)
# cg26.63.1 and it's var changed RB|VBG to RB (word: not-knowing)