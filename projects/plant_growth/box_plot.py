import pandas as pd
datafile="PlantGrowth.csv"
data = pd.read_csv(datafile)
 
#Create a boxplot
data.boxplot('weight', by='group', figsize=(12, 8))
 
ctrl = data['weight'][data.group == 'ctrl']
 
grps = pd.unique(data.group.values)
d_data = {grp:data['weight'][data.group == grp] \
    for grp in pd.unique(data.group.values)}
 
k = len(pd.unique(data.group))  # number of conditions
N = len(data.values)  # conditions times participants
n = data.groupby('group').size()[0] #Participants in each condition

from pyvttbl import DataFrame
 
df=DataFrame()
df.read_tbl(datafile)
aov_pyvttbl = df.anova1way('weight', 'group')
print(aov_pyvttbl)