from io import open
import nltk
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import json
from scipy.stats import spearmanr, rankdata
from sklearn.cluster import KMeans
import pickle
import gen_feature_encoding
from pmi_dep_calculation import pmi_dep_calc
from pmi_allwords_calculations import pmi_allwords_calc

k_clusters = 128
corpora = 'brown' 
# pmi_head_dep_df = pd.read_csv(f'{k_clusters}/pmi_head_dep_EWT.csv')
# pmi_root_allwords_df = pd.read_csv(f'{k_clusters}/pmi_root_allwords_EWT.csv')
df = pd.DataFrame()

lengths = {'ud': 12543, 'wsj': 36437, 'brown': 15582} # number of sentences in each
starts = {'ud': 0, 'wsj': 12543, 'brown': 12543+36437}
ends = {'ud': 12543, 'wsj': 12543+36437, 'brown': 12543+36437+15582}


# lengths = {'ud': 1, 'wsj': 1, 'brown': 1} # number of sentences in each
# starts = {'ud': 0, 'wsj': 1, 'brown': 2}
# ends = {'ud': 1, 'wsj': 2, 'brown': 3}


# Finding pmi values from csv
def get_pmi(pmi_df,head_pos,dep_pos):
	return pmi_df[dep_pos][head_pos]

def preprocess_cluster(train_file):
	if os.path.exists(train_file+'toks.json'):
		print('toks already saved')
		x = json.load(open(train_file+'toks.json'))
		return x['sentences'], x['ids'], x['u'], x['root_idx'], x['relation'], x['tags']
	f = open(train_file)
	sentences = []
	sentence = []
	ids = []
	root_idx = []
	tags, t = [], []
	relation,rel = [],[]
	unique_tags = set()
	for x in f.readlines():
		if x.startswith('# sent_id'):
			ids.append(x.replace('# sent_id = ', '').rstrip())
			continue
		if x.startswith('#'):
			continue
		if x[0].isalpha():
			ids.append(x.rstrip())
			continue
		elif x == '\n':
			sentences.append(sentence)
			relation.append(rel)
			tags.append(t)
			sentence,rel,t = [],[],[]
			continue
		# '3\tGoogle\tGoogle\tPROPN\tNNP\tNumber=Sing\t4\tnsubj\t4:nsubj\t_\n'
		x = x.split('\t')
		if x[7] in ['ROOT','root']:
			root_idx.append(int(x[0])-1)
		sentence.append((x[1],tag(x[4])))
		t.append(tag(x[4]))
		unique_tags.add(tag(x[4]))
		rel.append(x[6])

	x = open(train_file+'toks.json','w')
	json.dump({
		'sentences': sentences,
		'ids':ids,
		'u': list(unique_tags),
		'root_idx': root_idx,
		'relation': relation ,
		'tags': tags
		}, x)
	return sentences, ids, unique_tags, root_idx, relation, tags


def predict_clusters(train_file, k_clusters):
	# kmeans_model = pickle.load(open(f"{k_clusters}/kmeans_.pkl", "rb"))
	train_sents, ids, u, root_idx, relation, tags = preprocess_cluster(train_file)
	print('File preprocd',end='\t')
	clusters_pred=[]
	# w = np.load(f"{k_clusters}/feature_.npy")
	print(len(train_sents), sum(len(x) for x in train_sents))
	
	f = gen_feature_encoding.FeatureExtractor(train_sents)

	word_embs = np.zeros((len(f.featuresets), len(f.encoding.labels())), dtype=np.float16)
	# print(word_embs.shape)
	
	for i,x in tqdm(enumerate(f.featuresets)):
		for j,label in enumerate(f.encoding.labels()):
			v = f.encoding.encode(x[0], label)
			total = np.float16(0.0)
			for (f_id, f_val) in v:
				total += f.weights[f_id]*f_val
			word_embs[i][j] = total
	kmeans_model = KMeans(n_clusters=k_clusters, random_state=42).fit(word_embs)
	with open(f"{k_clusters}/kmeans_.pkl", "wb") as g:
		pickle.dump(kmeans_model, g)
	with open(f"{k_clusters}/word_embs.npy", "wb") as g:
		np.save(g, word_embs)
	clusters_pred = kmeans_model.predict(word_embs)
	clusters_sentence = []
	j,c=-1,0
	print('converting to sentences..')
	for i,x in enumerate(clusters_pred):
		if i == c:
			j += 1
			c += len(train_sents[j])
			clusters_sentence.append([])
		clusters_sentence[j].append(x)
	assert len(clusters_sentence) == len(train_sents)
	return clusters_sentence, clusters_pred, ids, root_idx, relation, tags

#Tag converter
def tag(corpus_tag):
	tag  = nltk.map_tag('wsj', 'universal', corpus_tag)
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
def pmi_features(trainfile):
	if os.path.exists(f'{k_clusters}/word_embeddings_after_training'):
		x = pickle.load(open(f'{k_clusters}/word_embeddings_after_training','rb'))
		sentence_cluster = x['s']
		word_cluster = x['w']
		ids = x['i']
		root_idx = x['roots']
		relation = x['rel']
		tags = x['tags']
	else:
		sentence_cluster, word_cluster, ids, root_idx, relation, tags = predict_clusters(trainfile,k_clusters)
		savefile = open(f'{k_clusters}/word_embeddings_after_training','wb')
		pickle.dump({
			's':sentence_cluster,
			'w':word_cluster,
			'i':ids,
			'roots':root_idx,
			'rel':relation,
			'tags':tags
		}, savefile)
	print('Cluster Predictions made')
	ud_words = sum([len(x) for x in sentence_cluster[starts['ud']:ends['ud']]])
	print('All words starts')
	pmi_root_allwords_df = pmi_allwords_calc(word_cluster[:ud_words],k_clusters)
	print('All words ends, dep calc starts')
	pmi_head_dep_df = pmi_dep_calc(word_cluster[:ud_words], k_clusters)
	print('Dep calc ends')
	for c in ['wsj','brown']:
		print(c,'starts',end='\t')
		corpora = c
		start_ind = starts[c]
		end_ind = ends[c]
		items = {
			'id':ids[start_ind:end_ind],
			'pmi_neardep': [],
			'pmi_adjacent': [],
			'spearman_allwords':[],
			'spearman_dependents':[],
			'root_cluster':[]
		}
		for i in range(start_ind,end_ind):
			root_cluster = sentence_cluster[i][root_idx[i]]
			post_root_dep_clusters, post_root_dep_dist = [],[]
			post_all_root_clusters, post_all_root_dist = [], []
			for j in range(len(relation[i])):
				if (j>root_idx[i] and tags[i][j]!='PUNCT'):
					post_all_root_clusters.append(sentence_cluster[i][j])
					post_all_root_dist.append(j-root_idx[i])
					if ((int(relation[i][j])-1)==root_idx[i]):
						post_root_dep_clusters.append(sentence_cluster[i][j])
						post_root_dep_dist.append(root_idx[i]-j)
			pmi_post_allwords = [get_pmi(pmi_root_allwords_df ,root_cluster,j) for j in post_all_root_clusters]
			pmi_post_root_dep = [get_pmi(pmi_head_dep_df, root_cluster,j) for j in post_root_dep_clusters]
			pmi_word_rev_rank, pmi_dep_rev_rank = [],[]
			if len(pmi_post_root_dep)>=1:
				for r in rankdata(pmi_post_root_dep):
					pmi_dep_rev_rank.append(len(pmi_post_root_dep)+ 1 - r)
				items['pmi_neardep'].append(pmi_post_root_dep[0])
			else:
				items['pmi_neardep'].append(0)
			if len(pmi_post_allwords)>=1:
				for r in rankdata(pmi_post_allwords):
					pmi_word_rev_rank.append(len(pmi_post_allwords) + 1 -r)
				items['pmi_adjacent'].append(pmi_post_allwords[0])
			else:
				items['pmi_adjacent'].append(0)
			coef_post_words, p1 = spearmanr(pmi_word_rev_rank, rankdata(post_all_root_dist))
			if pd.isna(coef_post_words):
				coef_post_words = 0 
			coef_post_dep, p2 = spearmanr(pmi_dep_rev_rank, rankdata(post_root_dep_dist))
			if pd.isna(coef_post_dep):
				coef_post_dep = 0
			items['spearman_allwords'].append(coef_post_words)
			items['spearman_dependents'].append(coef_post_dep)
			items['root_cluster'].append(root_cluster)
			# pmi_neardep, pmi_adjacent, coef_post_dep, coef_post_words, id, root_cluster
		df = pd.DataFrame(items).set_index('id')
		df.to_csv(f'{k_clusters}/{corpora}_pmi_features.csv')
		print(c,'ends')

if __name__ == '__main__':
	pmi_features('all_data.dep')		
# pmi_features('example.tsv')		

# In sentences cf15.15.0 and cf15.15.1, 'a' (position 13) was given the tag ','. Changed it to 'DT'
# cl02.17.0 and it's var changed NN|SYM to SYM
# cn02.153.1 and it's var changed NN|VBG to NN (running)
# cn02.130.1 and it's var changed RB|RP to RP (word: by)
# cn02.145.0 and it's var changed JJ|VBG to JJ (word: cloying)
# cn13.53.0 and it's var changed JJ|VBG to JJ (word: lilting)
# cp16.178.0 and it's var changed CD|RB to JJ (word: one o'clock)
# cg26.63.1 and it's var changed RB|VBG to RB (word: not-knowing)