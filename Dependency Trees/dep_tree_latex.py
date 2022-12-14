import pandas as pd
from tqdm import tqdm
import nltk
# import numpy as np

pmi_df = pd.read_csv('pmi_ud.csv')
# Finding pmi values from csv
def get_pmi(head_pos,dep_pos):
	for index, row in pmi_df.iterrows():
		if row['h']==head_pos and row['d']==dep_pos :
			return row['pmi']

def extract_WORDs(sentence):
	return str(' '.join(sentence['WORD']))

def extract_pos(sentence):
	tags = {}
	for i in sentence.index:
		tags[sentence.loc[i,'WORD']] = tag(sentence.loc[i,'POS2'])
	return tags

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
	if tag == '.':
		return 'X'

	# TO - PRT	
	# VBP - VERB (NLTK - X)
	return tag

def find_connections(sentence):
	conn = {}
	for i in sentence.index:
		conn[str(sentence.loc[i,'id'])] = [str(sentence.loc[i,'connected_to']), sentence.loc[i,'connection']]
	conn = dict(sorted(conn.items(), key= lambda x: int(x[1][0])))
	return conn

starter = '''\\resizebox{\\linewidth}{!}{
\\usetikzlibrary{matrix}
	\\tikzset{row 2/.style={nodes={font=\\it}}}
	\\tikzset{row 3/.style={nodes={font=\\ttfamily}}}
\\begin{dependency}[arc edge]
	\\begin{deptext}[row sep=.1ex]
'''

def to_latex(sentence, only_root=True):
	latex = starter
	# latex = ''
	latex += "~ \\& "*len(sentence)
	latex = latex[:-2] + "\\"
	latex += '\n'
	# adding WORDs
	for i in sentence.index:
		latex += sentence.loc[i,'WORD'].replace('%','\\%').replace('$','\\$') + " \\& "
	latex = latex[:-2] + "\\"
	latex += '\n'
	# adding pos
	for i in sentence.index:
		latex += tag(sentence.loc[i,'POS2']) + " \\& "
	latex = latex[:-2] + "\\"
	latex += '\n'
	# Adding PMI between each word and root
	for i in sentence.index:
		# print(tag(sentence.loc[i,'POS2']))
		latex += str(round(get_pmi('VERB',tag(sentence.loc[i,'POS2'])),2)) + " \\& "
	latex = latex[:-2] + "\\"
	latex += '\n'
	latex += "\\end{deptext}\n"
	connections = find_connections(sentence)
	# root = connections[]
	for i in connections:
		a = connections[i][0]
		b = connections[i][1]
		if a == '0':
			latex += "\t\\deproot[edge below]{"+i+"}{"+b+"}\n"
			continue
		if only_root:
			if connections[a][1] in ['root','ROOT'] :
				latex += "\t\\depedge{"+i+"}{"+a+"}{"+b+"}\n"
		else:
			latex += "\t\\depedge{"+i+"}{"+a+"}{"+b+"}\n"

		# if connections[a][1] == 'root':
		# 	a,b = b,a
	latex += "\\end{dependency}\n}\n\n"

	return latex

def pmi_latex(sid):
	latex = ''
	latex += sid.replace('_','\_')
	latex += '\n'
	if sid[-1] == '0':
		latex += '\\begin{center} \\large Reference Sentence\\\\'
		latex += '\n'
		latex += "\\small spearman\\ allwords\\ =\\ "
		latex += str(round(pmi_features.loc[sid,'Spearman_allwords'],2))
		# latex += "{\\small spearman_allwords:}"
		# latex += str(pmi_features.loc[sid,'Spearman_allwords'])
		latex += '\n'
		latex += '\\end{center}\n'
	else:
		latex += '\\begin{center} \\large Variant Sentence\\\\'
		latex += '\n'
		latex += "\\small spearman\\ allwords\\ =\\ "
		latex += str(round(pmi_features.loc[sid,'Spearman_allwords'],2))
		# latex += "{\\small spearman_allwords:}"
		# latex += str(pmi_features.loc[sid,'Spearman_allwords'])
		latex += '\n'
		# if pmi_ranked.loc[sid,'class'] == 0:
		# 	latex += "\\small dlg (ref-var):"
		# 	latex += str(round(-1*pmi_ranked.loc[sid,'dlg'],2))
		# 	latex += '\t'
		# 	latex += "\\small wtembdep (ref-var):"
		# 	latex += str(round(-1*pmi_ranked.loc[sid,'wtembdep'],2))
		# 	latex += '\t'
		# 	latex += "\\small ldep (ref-var):"
		# 	latex += str(round(-1*pmi_ranked.loc[sid,'ldep'],2))
		# else:
		# 	latex += "\\small dlg (ref-var):"
		# 	latex += str(round(pmi_ranked.loc[sid,'dlg'],2))
		# 	latex +='\t'
		# 	latex += "\\small wtembdep (ref-var):"
		# 	latex += str(round(pmi_ranked.loc[sid,'wtembdep'],2))
		# 	latex +='\t'
		# 	latex += "\\small ldep (ref-var):"
		# 	latex += str(round(pmi_ranked.loc[sid,'ldep'],2))
		# latex += '\n'
		latex += '\\end{center}\n'
	return latex

def pmi_diff_ranked(sid):
	latex = ''
	latex += '\\begin{center} \\large $\\phi$(ref) - $\\phi$(var):\\ '
	if pmi_ranked.loc[sid,'class'] == 0:
		latex += "\\small dlg\\ =\\ "
		latex += str(round(-1*pmi_ranked.loc[sid,'dlg'],2))
		latex += '\\quad'
		latex += "\\small wtembdep\\ =\\ "
		latex += str(round(-1*pmi_ranked.loc[sid,'wtembdep'],2))
		latex += '\\quad'
		latex += "\\small ldep\\ =\\ "
		latex += str(round(-1*pmi_ranked.loc[sid,'ldep'],2))
	else:
		latex += "\\small dlg\\ =\\ "
		latex += str(round(pmi_ranked.loc[sid,'dlg'],2))
		latex +='\\quad'
		latex += "\\small wtembdep\\ =\\ "
		latex += str(round(pmi_ranked.loc[sid,'wtembdep'],2))
		latex +='\\quad'
		latex += "\\small ldep\\ =\\ "
		latex += str(round(pmi_ranked.loc[sid,'ldep'],2))
	latex += '\n'
	latex += '\\end{center}\n\n'
	return latex

pmi_features = pd.read_csv('wsj_pmi_features.csv')
pmi_features = pmi_features.set_index('id')
pmi_ranked = pd.read_csv('wsj-ranked.csv')
pmi_ranked = pmi_ranked.set_index('id')
ranks_old = pd.read_csv('wsj-ranks.csv')
ranks_old = ranks_old.set_index('id')
# print(ranks_old)
# print(pmi_features)
df = pd.read_csv('wsj_index.dep', sep='\t',names = ['id','WORD','word','POS','POS2','descr','connected_to','connection','ex','ex2'])
f = open('wsj/Model_Comparison_Memory_PMI/ids.txt')
w = open('wsj/Model_Comparison_Memory_PMI/quote/latex_converted_quote.txt','w')

for curr_id in tqdm(f.readlines()):
	curr_id = curr_id.rstrip() # remove \n at end
	sentence = pd.DataFrame()
	# Extract sentence in a dataframe
	reached = False
	for a in df.index:
		if reached:
			if (df.loc[a, 'id'].startswith('w')): # CHANGE THIS TO WHATEVER LETTER
				break
			sentence = sentence.append(df.loc[a])
		if df.loc[a,'id'] == curr_id:
			reached = True
	sentence.set_index('id')
	sentence['connected_to'] = sentence['connected_to'].astype(int)
	# print(sentence)
	result = to_latex(sentence)
	pmi = pmi_latex(curr_id)
	constr = 'quote'
	if (curr_id[-1] == '0'):
		# ranks_old.index.str.find((curr_id[:-1]))
		try:
			if (ranks_old.loc[(curr_id[:-1]+'1'),'constr'] == constr):
				w.write(pmi)
				w.write(result)
		except:
			pass 

	# if (curr_id[-1] != '0' and ranks_old.loc[curr_id,'constr'] == constr):
	if (curr_id[-1] == '1' and ranks_old.loc[curr_id,'constr'] == constr):
		w.write(pmi)
		w.write(result)
		w.write(pmi_diff_ranked(curr_id))


f.close()
w.close()

