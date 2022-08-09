import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn import preprocessing

k_clusters = [16,32,64,128]
corpora =  ['brown', 'wsj']
for k in k_clusters:
	for corpus in corpora:
		df = pd.read_csv(f'{k}/{corpus}_ranked.csv').set_index('id')
		y = df['target'].values
		pmi = ['pmi_adjacent','pmi_neardep','spearman_allwords','spearman_dependents']
		df_results = pd.DataFrame()
		for i in pmi:
			surprisal = ['lm','bkpsl']
			memory = ['dlg','wtembdep','ldep']
			baseline = ['dlg','wtembdep','ldep','lm','bkpsl']
			results_todf = []
			for j in [surprisal,memory,baseline]:
				features = [i]
				for l in j:
					features.append(l)
				# print(i,j)
				result = {}
				result['K']=k
				result['Corpus']=corpus
				result['Baseline Features'] = j
				result['PMI Feature'] = i
				result['Features']=features
				# features.append(i)
				# print(features)
				dict_coef = {}
				x = df[features].fillna(0).replace([-np.inf, np.inf], 0)
				# print(x.max())
				#Standardization of data
				scaler = preprocessing.StandardScaler().fit(x)
				X_scaled = scaler.transform(x)
				# assert 1==0
				# sklearn - Regression and kfold model
				kfold = model_selection.KFold(n_splits=10, random_state=100, shuffle=True)
				model_kfold = LogisticRegression(penalty='none')
				results_kfold = model_selection.cross_val_score(model_kfold, X_scaled, y, cv=kfold)
				results=cross_validate(model_kfold, X_scaled, y, cv=kfold, return_estimator=True, return_train_score=True)
				result['Training Accuracy'] = results['train_score'].mean()*100.0
				result['Test Accuracy'] = results_kfold.mean()*100.0
				# Coef and Bias of the model
				model = LogisticRegression(penalty='none')
				model.fit(X_scaled, y)
				coef = model.coef_[0]
				result['Bias'] = model.intercept_[0]
				for ii in range(len(features)):
					dict_coef[features[ii]] = coef[ii]
				result['Coefficients'] = dict_coef
				print(result)
				results_todf.append(result)
			# df_results = pd.concat(results)
			df_results = pd.DataFrame(results_todf)
			# print(df_results)
			# assert 1==0
		# df_results.to_csv(f'{k}/{corpus}_models.csv')