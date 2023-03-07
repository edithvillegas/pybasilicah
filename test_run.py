import pybasilica
import pandas as pd

file = r'pybasilica\data\degasperi\counts_all.tsv' 
data = pd.read_csv(file, index_col=0, sep='\t')

#filter data
data = data.iloc[0:50]
#make categories for groups
data.organ = pd.Categorical(data.organ)
data['groups'] = data.organ.cat.codes
groups = data.groups.to_list()
#mutation values
values = data.drop(['cohort', 'organ', 'groups'], axis=1)

# #without groups
# r = pybasilica.fit(values, k_list=10)
# r.beta_denovo

#with groups
r2 = pybasilica.fit(values, k_list=2, groups=groups)
r2.beta_denovo


