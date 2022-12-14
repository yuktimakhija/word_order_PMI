from conllu import T
import pandas as pd
from sklearn import model_selection
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate,cross_val_predict
from sklearn import preprocessing
from tqdm import tqdm

df = pd.read_csv('brown-ranked.csv') 
df = df.set_index('id')
df_old = pd.read_csv('brown-ranks.csv') #only importing this to get the construction type for ref-var
df_old = df_old.set_index('id')
y = df['class'].values
pmi = ['pmi_adjacent','pmi_neardep','Spearman_allwords','Spearman_dependents']
df_results = pd.DataFrame()
def contruction_accuracies(predictions,y):
	tdat, dat, post, tpost, quote, tquote, iquote, tiquote, pre, tpre = 0,0,0,0,0,0,0,0,0,0
	for i in range(len(predictions)):
		id = df.index[i]
		constr = df_old.loc[id,'constr']
		if constr == 'dat':
			tdat += 1
		elif constr == 'post':
			tpost += 1
		elif constr == 'pre':
			tpre += 1
		elif constr == 'quote':
			tquote +=1
		else:
			tiquote += 1
		if (predictions[i] == y[i]):
			if constr == 'dat':
				dat += 1
			elif constr == 'post':
				post += 1
			elif constr == 'pre':
				pre += 1
			elif constr == 'quote':
				quote +=1
			else:
				iquote += 1
	construction = {'dat':(dat/tdat)*100, 'post': (post/tpost)*100, 'pre':(pre/tpre)*100, 
					'quote':(quote/tquote)*100, 'iquote':(iquote/tiquote)*100}
	return construction

pmi = ['pmi_adjacent']
for i in tqdm(pmi):
	surprisal = ['lm','bkpsl']
	memory = ['dlg','wtembdep','ldep']
	baseline = ['dlg','wtembdep','ldep','lm','bkpsl']
	for j in tqdm([baseline,surprisal,memory], leave=False):
		features = []
		for l in j:
			features.append(l)
		# print(i,j)
		result = {}
		result['Baseline Features'] = j
		result['PMI Feature'] = i
		features.append('Spearman_allwords')
		features.append('Spearman_dependents')
		# features.append(i)
		# print(features)
		# dict_coef = {}
		x = df[features]
		#Standardization of data
		scaler = preprocessing.StandardScaler().fit(x)
		X_scaled = scaler.transform(x)
		# sklearn - SVM and kfold model
		kfold = model_selection.KFold(n_splits=10, random_state=100, shuffle=True)
		model_kfold = SVC(kernel = 'rbf')
		results_kfold = model_selection.cross_val_score(model_kfold, X_scaled, y, cv=kfold)
		test_predictions = cross_val_predict(model_kfold,X_scaled,y,cv=kfold,method="predict")
		results=cross_validate(model_kfold, X_scaled, y, cv=kfold, return_estimator=True, return_train_score=True)
		result['Training Accuracy'] = results['train_score'].mean()*100.0
		result['Test Accuracy'] = results_kfold.mean()*100.0
		# Coef and Bias of the model
		# model = SVC(kernel ='rbf')
		# model.fit(X_scaled, y)
		# coef = model.coef_[0]
		# # print(model.get_params())
		# result['Intercept'] = model.intercept_[0]
		# for k in range(len(features)):
		# 	dict_coef[features[k]] = coef[k]
		# result['Coefficients'] = dict_coef
		result['Construction Accuracies'] = contruction_accuracies(test_predictions,y)
		# print(dict_coef)
		df_results = df_results.append(result,ignore_index=True)
		print(result)
# df_results.to_csv('SVM_Results/brown_models_rbf.csv')