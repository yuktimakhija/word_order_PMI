import pandas as pd
from tqdm import tqdm

k_clusters = [16,32,64,128]
corpora = ['brown','wsj']
for k in k_clusters:
    for corpus in corpora:
        pmi_features = pd.read_csv(f'{k}/{corpus}_pmi_features.csv').set_index('id')
        df_ranks = pd.read_csv(f'../{corpus}-ranked.csv').set_index('id')
        baseline = ['dlg','wtembdep','ldep','lm','bkpsl']
        pmi = ['pmi_neardep','pmi_adjacent','spearman_allwords','spearman_dependents']
        target = []
        ids = []
        flag=0
        df = pd.DataFrame()
        rows = []
        for id in tqdm(df_ranks.index, leave=False):
            if id[-1]!='0':
                var = id
                ref = var[:-1]+'0'
                ids.append(var)
                target.append(flag)
                if flag == 0:
                    row = pmi_features.loc[var]-pmi_features.loc[ref]
                    if df_ranks.loc[id,'class'] != 0:
                        row = pd.concat([row, -df_ranks.loc[id,baseline]])
                    else:
                        row = pd.concat([row, df_ranks.loc[id,baseline]])
                    flag=1
                else:
                    row = pmi_features.loc[ref]-pmi_features.loc[var]
                    if df_ranks.loc[id,'class'] != 1:
                        row = pd.concat([row, -df_ranks.loc[id,baseline]])
                    else:
                        row = pd.concat([row, df_ranks.loc[id,baseline]])
                    flag=0
                rows.append(row)

        df = pd.concat(rows, axis=1).transpose()
        df = df.drop(columns=['root_cluster'])
        df['id']=ids
        df['target']=target
        df = df.set_index('id')
        # print(df)
        # assert 1==0
        df.to_csv(f'{k}/{corpus}_ranked.csv')
