import pandas as pd
import plotly.express as px

k_clusters = [16,32,64,128]
for k in k_clusters:
	df1 = pd.read_csv(f"{k}/pmi_head_dep_EWT.csv")
	df2 = pd.read_csv(f"{k}/pmi_root_allwords_EWT.csv")

	def htmp(df, xtitle, ytitle):
		fig = px.imshow(df, color_continuous_scale=[(0, "#ff0077"), (0.5, "#ffffff"), (1, "#36e1e6")], color_continuous_midpoint=0)
		fig.update_layout(paper_bgcolor='#fff', plot_bgcolor='#888',
			width=1200,
			# rows: Dependents and columns: Heads
			# xaxis_title='Heads', yaxis_title='Dependents',
			# rows: second columns: First 
			# xaxis_title='Cluster 1', yaxis_title='CLuster 2',
			xaxis_title=xtitle, yaxis_title=ytitle,
			font={'size':40},
			xaxis={'showgrid':False},yaxis={'showgrid':False},
			)
		fig.show()

	# htmp(df1)
	htmp(df1,'Heads','Dependents')
	htmp(df2,'Cluster 1','Cluster 2')

