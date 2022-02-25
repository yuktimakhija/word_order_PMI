import pandas as pd
import plotly.express as px

df1 = pd.read_excel("pmi_values_result.xlsx", sheet_name=0).set_index('Tags').drop(['PUNCT'],axis=1).drop(['PUNCT'],axis=0).drop(['X'],axis=1).drop(['X'],axis=0)
df2 = pd.read_excel("pmi_values_result.xlsx", sheet_name=2).set_index('Tags').drop(['PUNCT'],axis=1).drop(['PUNCT'],axis=0).drop(['X'],axis=1).drop(['X'],axis=0)
# df3 = pd.read_excel('PMI_root_allwords/pmi_all_words.xlsx',sheet_name=3).set_index('Tags').drop(['PUNCT'],axis=1).drop(['PUNCT'],axis=0)
df3 = pd.read_csv('PMI_root_allwords/pmi_EWT.csv').set_index('Tags').drop(['PUNCT'],axis=1).drop(['PUNCT'],axis=0).drop(['X'],axis=1).drop(['X'],axis=0)

def htmp(df):
	fig = px.imshow(df, color_continuous_scale=[(0, "#ff0077"), (0.5, "#ffffff"), (1, "#36e1e6")], color_continuous_midpoint=0)
	fig.update_layout(paper_bgcolor='#fff', plot_bgcolor='#888',
		width=1200,
		xaxis_title='Heads', yaxis_title='Dependents',
		# xaxis_title='POS Tag 2', yaxis_title='POS Tag 1',
		font={'size':20},
		xaxis={'showgrid':False},yaxis={'showgrid':False},
		)
	fig.show()

# htmp(df1)
htmp(df2)
# htmp(df3)
