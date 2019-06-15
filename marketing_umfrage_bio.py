# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 14:13:14 2019

@author: chise
"""

### Import libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from numpy.polynomial.polynomial import polyfit
import statsmodels.stats.multicomp as multi


#matplotlib inline
sns.set_style('ticks')
### Functions

def unique_sentences(answers_survey):
    answers=[]
    for row in range(len(answers_survey)-1):
        for col in range(len(answers_survey.columns)-1):
            if not any(x in str(answers_survey.loc[row,col]) for x in answers):
                answers.append(str(answers_survey.loc[row,col]))

    return answers

def create_bos(answers, answers_survey):
    bos=np.zeros((len(answers_survey),len(answers)), dtype=int)
    for row in range(len(answers_survey)-1):
        for col in range(len(answers_survey.columns)-1):
           bos[row,answers.index(str(answers_survey.loc[row,col]))] = 1
    bos=pd.DataFrame(bos).drop(answers.index('None'), axis=1)
    del answers[answers.index('None')]
    bos.columns=answers
    return bos

def create_bos_s(answers, answers_survey):
    bos=np.zeros((len(answers_survey),len(answers)), dtype=int)
    for row in range(len(answers_survey)-1):
        bos[row,np.where(str(answers_survey.loc[row])==answers)] = 1
    bos=pd.DataFrame(bos)
    bos.columns=answers
    return bos

### Change directory

os.chdir('C:/Users/chise/Downloads')
# Create target Directory
#os.mkdir('C:/Users/chise/Downloads/luisa')
### Import data "Konsumentenbefragung zum Thema Einflussfaktoren und Anreizsysteme nachhaltigen Konsums in der Lebensmittelbranche"

df=pd.read_csv('Konsumentenbefragung.csv', sep=';',encoding = 'unicode_escape')
os.chdir('C:/Users/chise/Downloads/luisa')
## Check data
df.head()

## Check data
print('The survey was performed between {} and {}'.format(df[df.columns[0]].min(axis=0),df[df.columns[0]].max(axis=0))) 

##Get stats from numeric part
statis = df.describe()

##Easy plots from numerical data in pandas dataframe
df.hist()
#pd.options.display.mpl_style = 'default'
df.boxplot()

## Save original questions and change column names to more addressable ones
original_questions=pd.DataFrame(df.columns)
df.columns=['Time','Gender','Age','Country','Job','Education','Children','Income','Product','Frequency','Where','Consequences','Consequence_relevance','Stamp','Criteria','Criteria-bio','Constraints','When_bio','Criteria_expensive','Marketing','Marketing4me','Image_bio','Proposals']
## Making pie charts according to survey demographics
gender_grouping = df.groupby('Gender').agg('count')
age_grouping = df.groupby('Age').agg('count')
country_grouping = df.groupby('Country').agg('count')
where_grouping = df.groupby('Where').agg('count')
age_grouping[df.columns[0]].plot(kind='pie')

##Convert from Categorical to numerical
df['Age'][df['Age']=='<20']=18
df['Age'][df['Age']=='21-30']=25
df['Age'][df['Age']=='31-40']=35
df['Age'][df['Age']=='41-50']=45
df['Age'][df['Age']=='51-60']=55
df['Age'][df['Age']=='>60']=70
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')

# Plot all in a for loop
#fig, axes = plt.subplots(nrows=1, ncols=4)
#for i in range(1,4):
    df.groupby(df.columns[i]).agg('count')[df.columns[0]].plot(kind='pie',ax=axes[i-1])

#Two groups can be seen with low and high income. Girls maybe less paid and Divers at extremes
df.groupby('Constraints').When_bio.hist(alpha=0.4)
df.groupby('Constraints').agg('count')['When_bio'].plot(kind='pie')

#Scatter matrix for visualization of correlations
#Subset one answer questions
one_answer=df[df.columns[[1,2,3,4,5,6,7,9,11,12,13,19,20]]]

#No strong correlations seems to be apparent
scatter_matrix(one_answer, alpha=0.2, figsize=(13, 13), diagonal='kde')
sns.heatmap(one_answer.corr(),annot=True)

## Create one answer questions one hot encoding
bos2 = create_bos_s(one_answer[one_answer.columns[0]].unique(),one_answer[one_answer.columns[0]])
bos3 = create_bos_s(one_answer[one_answer.columns[1]].unique(),one_answer[one_answer.columns[1]])
bos4 = create_bos_s(one_answer[one_answer.columns[2]].unique(),one_answer[one_answer.columns[2]])
bos5 = create_bos_s(one_answer[one_answer.columns[3]].unique(),one_answer[one_answer.columns[3]])
bos6 = create_bos_s(one_answer[one_answer.columns[4]].unique(),one_answer[one_answer.columns[4]])
bos7 = create_bos_s(one_answer[one_answer.columns[5]].unique(),one_answer[one_answer.columns[5]])
# Numerical: bos8 = create_bos_s(one_answer[one_answer.columns[6]].unique(),one_answer[one_answer.columns[6]])
bos10 = create_bos_s(one_answer[one_answer.columns[7]].unique(),one_answer[one_answer.columns[7]])
bos12 = create_bos_s(one_answer[one_answer.columns[8]].unique(),one_answer[one_answer.columns[8]])
# Numerical: bos13 = create_bos_s(one_answer[one_answer.columns[9]].unique(),one_answer[one_answer.columns[9]])
# Numerical: bos14 = create_bos_s(one_answer[one_answer.columns[10]].unique(),one_answer[one_answer.columns[10]])
bos20 = create_bos_s(one_answer[one_answer.columns[11]].unique(),one_answer[one_answer.columns[11]])
# Numerical: bos21 = create_bos_s(one_answer[one_answer.columns[12]].unique(),one_answer[one_answer.columns[12]])
## One answer: 1-8,10,12-14,20,21   More answers: 9,11,15-18  Comments: 19,22,23
## For more answers: Get correlations and plot them
#Question 9
multi_split_9 = df[df.columns[8]].str.split(";", n = 3, expand = True)
answers9 = unique_sentences(multi_split_9)
bos9 = create_bos(answers9, multi_split_9)
corr_9=bos9.corr()

fig, ax = plt.subplots(figsize=(15, 10))
sns.heatmap(corr_9, ax = ax, annot=True)   #, vmin=-1, vmax=1
ax.set_title(df.columns[8])
plt.show()
fig.savefig('Correlation_multi9.pdf')
#Question 11
multi_split_11 = df[df.columns[10]].str.split(";", n = 3, expand = True)
answers11 = unique_sentences(multi_split_11)
bos11 = create_bos(answers11, multi_split_11)
corr_11=bos11.corr()

fig, ax = plt.subplots(figsize=(15, 10))
sns.heatmap(corr_11, ax = ax, annot=True)   #, vmin=-1, vmax=1
ax.set_title(df.columns[10])
plt.show()
fig.savefig('Correlation_multi11.pdf')
#Question 15
multi_split_15 = df[df.columns[14]].str.split(";", n = 3, expand = True)
answers15 = unique_sentences(multi_split_15)
bos15 = create_bos(answers15, multi_split_15)
corr_15=bos15.corr()

fig, ax = plt.subplots(figsize=(15, 10))
sns.heatmap(corr_15, ax = ax, annot=True)   #, vmin=-1, vmax=1
ax.set_title(df.columns[14])
plt.show()
fig.savefig('Correlation_multi15.pdf')
#Question 16  
multi_split_16 = df[df.columns[15]].str.split(";", n = 3, expand = True)
answers16 = unique_sentences(multi_split_16)
bos16 = create_bos(answers16, multi_split_16)
corr_16=bos16.corr()

fig, ax = plt.subplots(figsize=(15, 10))
sns.heatmap(corr_16, ax = ax, annot=True)   #, vmin=-1, vmax=1
ax.set_title(df.columns[15])
plt.show()
fig.savefig('Correlation_multi16.pdf')
#Question 17     
multi_split_17 = df[df.columns[16]].str.split(";", n = 3, expand = True)
answers17 = unique_sentences(multi_split_17)
bos17 = create_bos(answers17, multi_split_17)
corr_17=bos17.corr()

fig, ax = plt.subplots(figsize=(15, 10))
sns.heatmap(corr_17, ax = ax, annot=True)   #, vmin=-1, vmax=1
ax.set_title(df.columns[16])
plt.show()
fig.savefig('Correlation_multi17.pdf')
#Question 18
multi_split_18 = df[df.columns[17]].str.split(";", n = 3, expand = True)
answers18 = unique_sentences(multi_split_18)
bos18 = create_bos(answers18, multi_split_18)
corr_18=bos18.corr()

fig, ax = plt.subplots(figsize=(15, 10))
sns.heatmap(corr_18, ax = ax, annot=True)   #, vmin=-1, vmax=1
ax.set_title(df.columns[17])
plt.show() 
fig.savefig('Correlation_multi18.pdf')
## Something to do with the coments
#Put all coments from the same question together to sumarize
comments_19=[]
comments_19.append(df.loc[~df[df.columns[18]].isnull(),df.columns[18]])
comments_22=[]
comments_22.append(df.loc[~df[df.columns[21]].isnull(),df.columns[21]])
comments_23=[]
comments_23.append(df.loc[~df[df.columns[22]].isnull(),df.columns[22]])

## Find personalities
# PCA to extract features from loadings. However categorical data makes this much more difficult.
# There are several options: nonlinear PCA, A priori, vectorize categorical data, MFA, Ordinal PCA,
# Polychoric correlation

#Combine multi-answers for PCA
multi_answers=np.concatenate((bos2,bos3,bos4,bos5,bos6,bos7,pd.DataFrame(df[df.columns[7]]),bos9,bos10,bos11,bos12,pd.DataFrame(df[df.columns[12]]),pd.DataFrame(df[df.columns[13]]),bos15,bos16,bos17,bos18,bos20,pd.DataFrame(df[df.columns[20]])),axis=1)
multi_answers_names=pd.DataFrame(np.concatenate((list(bos2.columns),list(bos3.columns),list(bos4.columns),list(bos5.columns),list(bos6.columns),list(bos7.columns),[df.columns[7]],list(bos9.columns),list(bos10.columns),list(bos11.columns),list(bos12.columns),[df.columns[12]],[df.columns[13]],list(bos15.columns),list(bos16.columns),list(bos17.columns),list(bos18.columns),list(bos20.columns),[df.columns[20]]),axis=0))
#multi_answers=np.concatenate((bos9,bos11,bos15,bos16,bos17,bos18),axis=1)
#multi_answers_names=pd.DataFrame(np.concatenate((bos9.columns,bos11.columns,bos15.columns,bos16.columns,bos17.columns,bos18.columns),axis=0))

# Change nans in income to zero. Asuming that no answer means no income
multi_answers[np.where(np.isnan(multi_answers))]=0
#PCA
#Scaling the values and doing PCA
X = scale(multi_answers)
pca = PCA(n_components=6)
pca.fit(X)

principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2','principal component 3', 'principal component 4','principal component 5', 'principal component 6'])

#The amount of variance that each PC explains & cumulative
var= pca.explained_variance_ratio_
plt.plot(var)
var_cum=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
plt.plot(var_cum)
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.title('PCA Analysis')
#plt.ylim(30,100.5)
plt.show()

# summarize components
PCs=pd.DataFrame(pca.components_)
PCs.columns=multi_answers_names

#Plot and save PCA figure
fig, ax = plt.subplots(figsize=(20, 10))
ax.set_xlim([-7,11])
ax.set_ylim([-4,10])
ax.scatter(principalDf['principal component 1'], principalDf['principal component 3'],
            edgecolor='none', alpha=0.7,s=60)
factor=25
for i, txt in enumerate(np.array(multi_answers_names)):
    ax.annotate(txt, (factor*PCs.iloc[0,i], factor*PCs.iloc[2,i]))
    ax.arrow(0, 0, factor*PCs.iloc[0,i], factor*PCs.iloc[2,i], head_width=0.1, head_length=0.1, fc='k', ec='k', alpha=0.1)
plt.xlabel('Persönlichkeit 1')
plt.ylabel('Persönlichkeit 3')
fig.savefig('Personlichkeit13_obs.pdf')

fig, ax = plt.subplots(figsize=(20, 10))
ax.set_xlim([-7,11])
ax.set_ylim([-4,10])
ax.scatter(principalDf['principal component 2'], principalDf['principal component 3'],
            edgecolor='none', alpha=0.7,s=60)
factor=25
for i, txt in enumerate(np.array(multi_answers_names)):
    ax.annotate(txt, (factor*PCs.iloc[1,i], factor*PCs.iloc[2,i]))
    ax.arrow(0, 0, factor*PCs.iloc[1,i], factor*PCs.iloc[2,i], head_width=0.1, head_length=0.1, fc='k', ec='k', alpha=0.1)
plt.xlabel('Persönlichkeit 2')
plt.ylabel('Persönlichkeit 3')
fig.savefig('Personlichkeit23_obs.pdf')

## Quantify nachhaltig/sostenible scores in people with a mix of factors: Product,Frequency, Criteria, Criteria_Bio
#Negative in PC1 & PC2 and positive in PC3
people_bio=principalDf['principal component 3']-principalDf['principal component 1']-principalDf['principal component 2']

df['Bio_personality']=people_bio

## Find specific associations of two questions (categorical-categorical) with Chi-Squared statistic
#Image_bio-When_bio/Criteria_expensive(Willing2pay) Categorical-Categorical 
gender_price = pd.concat([bos17['Preisniveau'],df['Gender']],axis=1)
chi_comparison = pd.crosstab(index=gender_price['Preisniveau'],columns=gender_price['Gender'])
#chi_comparison = df.groupby(['Gender','Income']).agg('count')['Time']
#chi_comparison = df.groupby(['Consequence_relevance','Frequency']).agg('count')['Time']
# Is there an effect?
stats.chisquare(chi_comparison)
# Are variables independent?
stats.chi2_contingency(observed= chi_comparison)

chi_comparison.plot(kind='pie')
df.groupby('Gender').When_bio.hist(alpha=0.4)

## Find specific associations of two questions (categorical-continuous) with ANOVA statistic
#Create a boxplot
df.boxplot('Bio_personality', by='Gender', figsize=(12, 8))
#Bio_männlich = df['Bio_personality'][df.Gender == 'Männlich']

#1Nachhaltig-Gender
grps = pd.unique(df.Gender.values)
d_data = {grp:df['Bio_personality'][df.Gender == grp] for grp in grps}
#k = len(pd.unique(df.Gender))  # number of conditions
#N = len(df.values)  # conditions times participants
#n = df.groupby('Gender').size()[0] #Participants in each condition
F, p = stats.f_oneway(d_data[grps[0]], d_data[grps[1]])
fig, ax = plt.subplots()
ax = sns.boxplot(x='Gender', y='Bio_personality', data=df)
# Add jitter with the swarmplot function.
ax = sns.swarmplot(x='Gender', y='Bio_personality', data=df, color="grey")
fig.savefig('Bio_personality_Gender.pdf')
#2Nachhaltig-Children
grps = pd.unique(df.Children.values)
d_data = {grp:df['Bio_personality'][df.Children == grp] for grp in grps}
F, p = stats.f_oneway(d_data[grps[0]], d_data[grps[1]])
# Usual boxplot
fig, ax = plt.subplots()
ax = sns.boxplot(x='Children', y='Bio_personality', data=df)
# Add jitter with the swarmplot function.
ax = sns.swarmplot(x='Children', y='Bio_personality', data=df, color="grey")
fig.savefig('Bio_personality_Children.pdf')
#3Nachhaltig-Sozial&Umwelt wichtig
grps = pd.unique(df.Consequence_relevance.values)
d_data = {grp:df['Bio_personality'][df.Consequence_relevance == grp] for grp in grps}
F, p = stats.f_oneway(d_data[grps[0]], d_data[grps[1]], d_data[grps[2]], d_data[grps[3]], d_data[grps[4]], d_data[grps[5]], d_data[grps[6]], d_data[grps[7]], d_data[grps[8]], d_data[grps[9]])
#Post-hoc analysis
posthoc=multi.MultiComparison(df['Bio_personality'],df['Consequence_relevance'])
res1=posthoc.tukeyhsd()
print(res1.summary())
# Usual boxplot
fig, ax = plt.subplots()
ax = sns.boxplot(x='Consequence_relevance', y='Bio_personality', data=df)
# Add jitter with the swarmplot function.
ax = sns.swarmplot(x='Consequence_relevance', y='Bio_personality', data=df, color="grey")
fig.savefig('Bio_personality_Consequence_relevance2.pdf')
# Scatterplot
fig, ax = plt.subplots()
ax.scatter(df['Consequence_relevance'], df['Bio_personality'],
            edgecolor='none', alpha=0.7,s=60)
# Fit with polyfit
b, m = polyfit(df['Consequence_relevance'], df['Bio_personality'], 1)
plt.plot(df['Consequence_relevance'], b + m * df['Consequence_relevance'], '-')
plt.show()
fig.savefig('Bio_personality_consequence_relevance.pdf')
#Nachhaltig-Siegel vertrauen
grps = pd.unique(df.Stamp.values)
d_data = {grp:df['Bio_personality'][df.Stamp == grp] for grp in grps}
F, p = stats.f_oneway(d_data[grps[0]], d_data[grps[1]], d_data[grps[2]], d_data[grps[3]], d_data[grps[4]], d_data[grps[5]], d_data[grps[6]], d_data[grps[7]], d_data[grps[8]], d_data[grps[9]])
#Tukey posthoc
posthoc=multi.MultiComparison(df['Bio_personality'],df['Stamp'])
res1=posthoc.tukeyhsd()
print(res1.summary())
#Plot
fig, ax = plt.subplots()
ax = sns.boxplot(x='Stamp', y='Bio_personality', data=df)
# Add jitter with the swarmplot function.
ax = sns.swarmplot(x='Stamp', y='Bio_personality', data=df, color="grey")
plt.savefig('Bio_personality_Stamp.pdf')
#Image_bio-When_bio/Criteria_expensive(Willing2pay) Categorical-Categorical -> Chi2
#Nachhaltig-Consequences/Consequence_relevance(Knowledge&consequences)
grps = pd.unique(df.Consequences.values)
d_data = {grp:df['Bio_personality'][df.Consequences == grp] for grp in grps}
F, p = stats.f_oneway(d_data[grps[0]], d_data[grps[1]], d_data[grps[2]])
#Tukey posthoc
posthoc=multi.MultiComparison(df['Bio_personality'],df['Consequences'])
res1=posthoc.tukeyhsd()
print(res1.summary())
#Plot
fig, ax = plt.subplots()
ax = sns.boxplot(x='Consequences', y='Bio_personality', data=df)
# Add jitter with the swarmplot function.
ax = sns.swarmplot(x='Consequences', y='Bio_personality', data=df, color="grey")
fig.savefig('Bio_personality_Consequences.pdf')

################# Other alternatives to linear PCA ########################

#Nonlinear PCA
from sklearn.decomposition import PCA, KernelPCA
kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
X_kpca = kpca.fit_transform(X)
X_back = kpca.inverse_transform(X_kpca)

# Plot results
plt.figure()
plt.subplot(2, 1, 1, aspect='equal')
plt.scatter(X_kpca[:, 0], X_kpca[:, 1], s=20, edgecolor='k')
plt.title("Projection by KPCA")
plt.xlabel(r"1st principal component in space induced by $\phi$")
plt.ylabel("2nd component")

plt.subplot(2, 1, 2, aspect='equal')
plt.scatter(X_back[:, 0], X_back[:, 1], s=20, edgecolor='k')
plt.title("Original space after inverse transform")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")

plt.tight_layout()
plt.show()   

#Preprocessing
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
lb = preprocessing.LabelBinarizer()
lb.fit(df['Product'])
le.fit(df['Product'])
le.classes_
le.transform(le.classes_[0])
lb.classes_
lb.transform(lb.classes_[0])

#Apriori
from apyori import apriori
#Preprocessing to convert to a list of lists for apriori function
def data_generator(filename):
  """
  Data generator, needs to return a generator to be called several times.
  """
  def data_gen():
    with open(filename) as file:
      for line in file:
        yield tuple(k.strip() for k in line.split(','))      

  return data_gen

transactions = data_generator('Konsumentenbefragung.csv')
itemsets, rules = apriori(transactions, min_support=0.9, min_confidence=0.6)
records = []  
for i in range(0, len(df)):  
    records.append([str(one_answer) for j in range(0, len(df.columns))])
#apply apriori
itemsets, rules = apriori(one_answer, min_support=0.2,  min_confidence=1)

association_rules = apriori(one_answer, min_support=0.045, min_confidence=0.2, min_lift=3, min_length=2)  
association_results = list(association_rules)
for item in association_rules:

    # first index of the inner list
    # Contains base item and add item
    pair = item[0] 
    items = [x for x in pair]
    print("Rule: " + items[0] + " -> " + items[1])

    #second index of the inner list
    print("Support: " + str(item[1]))

    #third index of the list located at 0th
    #of the third index of the inner list

    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")

#Multiple Correspondence Analysis
import mca
mca_ben = mca.MCA(one_answer, ncols=len(one_answer.columns))
mca_ind = mca.MCA(one_answer, ncols=len(one_answer.columns), benzecri=False) 
