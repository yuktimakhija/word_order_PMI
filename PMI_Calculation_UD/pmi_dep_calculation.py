from tqdm import tqdm 
import numpy as np
np.set_printoptions(suppress=True)

# List of all UD POS Tags
ud_tags = ['ADJ','ADP','ADV','AUX','CCONJ','DET','INTJ','NOUN','NUM','PART','PRON','PROPN',
			'PUNCT','SCONJ','SYM','VERB','X'] 
# Dictionary containing the index of each tag
tag_id = {} 
for i in range(len(ud_tags)):
	tag_id[ud_tags[i]] = i
# print(ud_tags)
# print(tag_id)
rows = ud_tags # rows == dependents 
columns = ud_tags # columns == heads


# Names of all the corpora in ud-treebankv2.9
# corpora = ['Atis','ESL','EWT','GUM','GUMReddit','LinES','ParTUT','Pronouns','PUD']
corpora = ['EWT']
total_sentence_count = 0
sentences_lesser_20words = 0
tdep = 0
total_count_matrix = np.zeros((17,17))
for corpus in corpora:
	# 2-D Array to count the total number of head-dependent relations present in the corpora 
	count_arr = arr = [[0 for i in columns] for j in rows]
	# Each cell in the matrix is [Dependent][Head]
	# print(count_arr)
	print(corpus)
	# Opening the corpus file (.conllu file)
	test = open(r"ud-treebanks-v2.9\UD_English-"+corpus+"\\en_"+corpus.lower()+"-ud-test.conllu", "r",encoding='utf-8')
	collective = [test]
	if corpus not in ['PUD','Pronouns']:
		train = open(r"ud-treebanks-v2.9\UD_English-"+corpus+"\\en_"+corpus.lower()+"-ud-train.conllu", "r",encoding='utf-8')
		dev = open(r"ud-treebanks-v2.9\UD_English-"+corpus+"\\en_"+corpus.lower()+"-ud-dev.conllu", "r",encoding='utf-8')
		collective.append(dev)
		collective.append(train)
	# dev = open(r"ud-treebanks-v2.9\sample_sentence.conllu", "r")
	# collective = [dev]
	heads_sentence = [] # Stores the position of head of each word in a sentence
	tag_sentence = [] # Stores UD POS Tag of each word in the sentence
	dep_id = 0 # Stores id of the POS Tag of the dependent 
	head_id = 0 # Stores id of POS Tag of the head
	total_dep = 0 # Counts the toal number of dependents in the dataset

	# Using all three corpora for calculation
	for file in collective:
		lines = file.readlines()
		for line in lines:
			# print(line)
			if line[0].isnumeric():
				if line.startswith("1") and line[1] == "\t":
					total_sentence_count+=1 
					# print(True)
					# print(line)
					heads_sentence = []
					tag_sentence = []
					infl_sentence = []
					word_count = 0
				# print(line)
				word_count +=1
				split = line.split("\t")
				# print(split)
				if split[0].isnumeric() :
					# print(line)
					tag_sentence.append(split[3])
					infl_sentence.append(split[5])
					heads_sentence.append(int(split[6]))
			# print('len',len(heads_sentence))
			# print(line[0])
			if (len(heads_sentence) != 0) and word_count<20  and not (line[0].isnumeric()):
				# print ('if worked!')
				# print('---------------', line)
				sentences_lesser_20words +=1 
				for i in range(len(heads_sentence)):
					# print(heads_sentence)
					if (heads_sentence[i] != 0) and (heads_sentence[i] != '-') and (tag_sentence[i]!='PUNCT') and (tag_sentence[heads_sentence[i]-1] != 'PUNCT'): 
						dep_id = tag_id[tag_sentence[i]]
						head_id = tag_id[tag_sentence[heads_sentence[i]-1]]
						if dep_id == 16:
							if "=" not in infl_sentence[i]:
								infls = {}
							else: 
								infls = dict(kv.split("=") for kv in infl_sentence[i].split("|"))
							if 'prontype' in infls:
								dep_id = 10 #tag_id['PRON']

							elif 'subpos' in infls and infls['subpos'] == 'det':
								dep_id = tag_id['DET']
							elif 'pos' in infls:
								dep_id = tag_id[infls['pos'].upper()]
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
	pmi_matrix = np.zeros((17,17))
	# print(total_count_matrix)
	# print(col_sum)
	# print(row_sum)
	# Calculating PMI values for each pair of head-dependent pairs
	for i in range(17):
		for j in range(17):
			# print(pmi_matrix[(i,j)])
			if (col_sum[j] != 0 and row_sum[i]!=0):  
				pmi_matrix[i,j] += np.log2((count_arr[i,j]/col_sum[j])*(total_dep/row_sum[i]))
				pmi_matrix[i,j] = f'{pmi_matrix[i,j]:.3f}'
			else:
				pmi_matrix[i,j] = np.nan
	np.savetxt(f"pmi_{corpus}.csv", pmi_matrix, delimiter=',', header=','.join(ud_tags))
			# print(pmi_matrix[(i,j)])
	# print(count_arr)
	# print(pmi_matrix)
	# print(count_arr[(1,2)])
# print(total_count_matrix)
print(tdep, 'tdep')
print(total_sentence_count)
print(sentences_lesser_20words)
col_sum = total_count_matrix.sum(axis=0) # Stores the total number of relation where a given POS Tag is the head
row_sum = total_count_matrix.sum(axis=1) # Stores the total number of relation where a given POS Tag is the dependent 
pmi_matrix = np.zeros((17,17))
for i in range(17):
	for j in range(17):
		# print(pmi_matrix[(i,j)])
		if (col_sum[j] != 0 and row_sum[i]!=0):  
			pmi_matrix[i,j] = np.log2((total_count_matrix[i,j]*tdep)/(col_sum[j]*row_sum[i]))
			# pmi_matrix[(i,j)] = f'{pmi_matrix[(i,j)]:.3f}'
		else:
			pmi_matrix[i,j] = np.inf
print(pmi_matrix)
# print(col_sum)
# print(row_sum)
# np.savetxt("pmi_all_datasets_combined.csv", pmi_matrix, delimiter=',', header=','.join(ud_tags))
# Total sentences in all corpora = 45120
# Total sentence with lesser than 20 words = 