import pandas as pd

pmi_features = pd.read_csv('PMI_Calculation_UD/PMI_root_allwords/wsj_spearmanallwords_adjword.csv')
pmi_features = pmi_features.set_index('id')
diff_df = pd.DataFrame()
rank_df = pd.DataFrame()
baseline_df = pd.read_csv('wsj-ranks.csv')
baseline_df = baseline_df.set_index('id')
baseline = ['dlg','wtembdep','ldep','lm','bkpsl']
# pmi = ['pmi_adjacent','pmi_neardep','Spearman_allwords','Spearman_dependents']
pmi = ['Spearman_allwords']
all_features = ['dlg','wtembdep','ldep','lm','bkpsl','Spearman_allwords'
				# ,'pmi_adjacent','pmi_neardep','Spearman_dependents'
				]

for i in pmi_features.index:
	if i[len(i)-1] != '0':
		idx_var = i
		idx_ref = idx_var[:len(idx_var)-1]+'0'
		row_var = pmi_features.loc[i]
		row_ref = pmi_features.loc[idx_ref]
		dict = {}
		dict['id'] = idx_var
		for j in pmi:
			dict[j] = pmi_features.loc[idx_ref,j] - pmi_features.loc[idx_var,j]
		diff_df = diff_df.append(dict,ignore_index=True)
diff_df = diff_df.set_index('id')
rank = 0
for i in diff_df.index:
	choice = baseline_df.loc[i,'choice']
	dict = {}
	dict['id'] = i
	dict['class'] = rank
	for j in baseline:
		if choice == rank:
			dict[j] = baseline_df.loc[i,j]
		else:
			dict[j] = -1*baseline_df.loc[i,j]
	for j in pmi:
		if rank == 1:
			dict[j] = diff_df.loc[i,j]
		else:
			dict[j] = -1*diff_df.loc[i,j]
	if rank == 0:
		rank = 1
	else:
		rank = 0
	rank_df = rank_df.append(dict,ignore_index=True)

rank_df.to_csv('PMI_Calculation_UD/PMI_root_allwords/wsj-ranked.csv')
