from tqdm import tqdm 
import numpy as np
import pickle
import gen_feature_encoding
from sklearn.metrics import silhouette_score
np.set_printoptions(suppress=True)

k_clusters = 128
def pmi_dep_calc(clusters_pred, k_clusters=k_clusters):
	clusters = []
	# Dictionary containing the index of each cluster
	cluster_id = {} 
	for i in range(k_clusters):
		# x = "Cluster_"+str(i)
		x=i
		cluster_id[x] = i
		clusters.append(str(x))
	# print(ud_tags)
	# print(cluster_id)
	rows = np.arange(k_clusters) # rows == dependents 
	columns = np.arange(k_clusters) # columns == heads


	# Names of all the corpora in ud-treebankv2.9
	# corpora = ['Atis','ESL','EWT','GUM','GUMReddit','LinES','ParTUT','Pronouns','PUD']
	corpora = ['EWT']
	total_sentence_count = 0
	sentences_lesser_20words = 0
	tdep = 0
	total_count_matrix = np.zeros((k_clusters,k_clusters))
	# load the k-means model
	# kmeans_model = pickle.load(open(str(k_clusters)+f"/kmeans_.pkl", "rb"))
	for corpus in corpora:
		# 2-D Array to count the total number of head-dependent relations present in the corpora 
		path_corpus = "../PMI_Calculation_UD/ud-treebanks-v2.9/UD_English-"+corpus+"/en_"+corpus.lower()+"-ud-"
		path_dict = {}
		count_arr = arr = [[0 for i in columns] for j in rows]
		# Each cell in the matrix is [Dependent][Head]
		# print(count_arr)
		print(corpus)
		# Opening the corpus file (.conllu file)
		test = open("../PMI_Calculation_UD/ud-treebanks-v2.9/UD_English-"+corpus+"/en_"+corpus.lower()+"-ud-test.conllu", "r",encoding='utf-8')
		collective = [test]
		path_dict[test] = path_corpus+"test.conllu"
		if corpus not in ['PUD','Pronouns']:
			train = open("../PMI_Calculation_UD/ud-treebanks-v2.9/UD_English-"+corpus+"/en_"+corpus.lower()+"-ud-train.conllu", "r",encoding='utf-8')
			dev = open("../PMI_Calculation_UD/ud-treebanks-v2.9/UD_English-"+corpus+"/en_"+corpus.lower()+"-ud-dev.conllu", "r",encoding='utf-8')
			path_dict[train] = path_corpus+"train.conllu"
			path_dict[dev] = path_corpus+"dev.conllu"
			collective.append(dev)
			collective.append(train)
		# dev = open(r"ud-treebanks-v2.9\sample_sentence.conllu", "r")
		# collective = [dev]
		heads_sentence = [] # Stores the position of head of each word in a sentence
		# cluster_sentence = [] # Stores cluster of each word in the sentence
		dep_id = 0 # Stores id of the cluster of the dependent 
		head_id = 0 # Stores id of cluster of the head
		total_dep = 0 # Counts the total number of dependents in the dataset
		word_count = 0
		# Using all three corpora for calculation
		for file in collective:
			lines = file.readlines()
			# train_sents,_ = gen_feature_encoding.preprocess(path_dict[file])
			# print(len(train_sents), sum(len(x) for x in train_sents))
			# f = gen_feature_encoding.FeatureExtractor(train_sents)
			# w = np.load(f"{k_clusters}/feature_.npy")

			# word_embs = np.zeros((len(f.featuresets), len(f.encoding.labels())), dtype=np.float16)
			# for i,x in tqdm(enumerate(f.featuresets)):
			# 	for j,label in enumerate(f.encoding.labels()):
			# 		v = f.encoding.encode(x[0], label)
			# 		total = np.float16(0.0)
			# 		for (f_id, f_val) in v:
			# 			total += w[f_id]*f_val
			# 		word_embs[i][j] = total
			# clusters_pred = kmeans_model.predict(word_embs)
			# print(clusters_pred.shape)
			j = 0
			for line in lines:
				# print(line)
				if line[0].isnumeric():
					if line.startswith("1") and line[1] == "\t":
						total_sentence_count+=1 
						# print(True)
						# print(line)
						heads_sentence = []
						cluster_sentence = []
						infl_sentence = []
						tag_sentence = []
						word_count = 0
					# print(line)
					split = line.split("\t")
					# print(split)
					if split[0].isnumeric():
						# print(line)
						word_cluster_label = clusters_pred[j]
						j += 1
						word_count +=1
						cluster_sentence.append(word_cluster_label)
						tag_sentence.append(split[3])
						infl_sentence.append(split[5])
						if split[6] != '-':
							heads_sentence.append(int(split[6]))
						else:
							heads_sentence.append(-2)
				# print('len',len(heads_sentence))
				# print(line[0])
				if (len(heads_sentence) != 0) and word_count<20  and not (line[0].isnumeric()):
					# print ('if worked!')
					# print('---------------', line)
					sentences_lesser_20words +=1 
					for i in range(len(heads_sentence)):
						# print(heads_sentence)
						if (heads_sentence[i] != 0) and (heads_sentence[i] != -2) and (tag_sentence[i]!='PUNCT') and (tag_sentence[heads_sentence[i]-1] != 'PUNCT'): 
							dep_id = cluster_sentence[i]
							head_id = cluster_sentence[heads_sentence[i]-1]
							
							count_arr[dep_id][head_id] += 1
							total_dep += 1
					heads_sentence = []
		tdep += total_dep
		# Print count_arr (list of list)
		# for x in count_arr:
		# 	for i in x:
		# 		print(i,end ='\t')
		# 	print()

		# converting count_arr to numpy matrix
		count_arr = np.array(count_arr)
		# print(count_arr)
		# print(count_arr)
		total_count_matrix = total_count_matrix + count_arr
		# print(total_count_matrix)
		col_sum = count_arr.sum(axis=0) # Stores the total number of relation where a given POS Tag is the head
		row_sum = count_arr.sum(axis=1) # Stores the total number of relation where a given POS Tag is the dependent 
		pmi_matrix = np.zeros((k_clusters,k_clusters))
		# print(total_count_matrix)
		# print(col_sum)
		# print(row_sum)
		# Calculating PMI values for each pair of head-dependent pairs
		for i in range(k_clusters):
			for j in range(k_clusters):
				# print(pmi_matrix[(i,j)])
				if (col_sum[j] != 0 and row_sum[i]!=0):  
					pmi_matrix[i,j] += np.log2((count_arr[i,j]/col_sum[j])*(total_dep/row_sum[i]))
					pmi_matrix[i,j] = f'{pmi_matrix[i,j]:.3f}'
				else:
					pmi_matrix[i,j] = np.nan
		np.savetxt(f"{k_clusters}/pmi_head_dep_{corpus}.csv", pmi_matrix, delimiter=',', header=','.join(clusters), fmt='%.4f')
				# print(pmi_matrix[(i,j)])
		return pmi_matrix
	# 	# print(count_arr)
	# 	# print(pmi_matrix)
	# 	# print(count_arr[(1,2)])
	# # print(total_count_matrix)
	# print(tdep, 'tdep')
	# print(total_sentence_count)
	# print(sentences_lesser_20words)
	# col_sum = total_count_matrix.sum(axis=0) # Stores the total number of relation where a given POS Tag is the head
	# row_sum = total_count_matrix.sum(axis=1) # Stores the total number of relation where a given POS Tag is the dependent 
	# pmi_matrix = np.zeros((k_clusters,k_clusters))
	# for i in range(k_clusters):
	# 	for j in range(k_clusters):
	# 		# print(pmi_matrix[(i,j)])
	# 		if (col_sum[j] != 0 and row_sum[i]!=0):  
	# 			pmi_matrix[i,j] = np.log2((total_count_matrix[i,j]*tdep)/(col_sum[j]*row_sum[i]))
	# 			# pmi_matrix[(i,j)] = f'{pmi_matrix[(i,j)]:.3f}'
	# 		else:
	# 			pmi_matrix[i,j] = np.inf
	# print(pmi_matrix)
	# # print(col_sum)
	# # print(row_sum)
	# # np.savetxt("pmi_all_datasets_combined.csv", pmi_matrix, delimiter=',', header=','.join(ud_tags))
	# # Total sentences in all corpora = 45120
	# # Total sentence with lesser than 20 words = 