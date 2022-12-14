from io import open
from conllu import parse_tree_incr
from conllu import parse_incr
import csv  
import nltk
import pandas
import csv
import math
from scipy.stats import spearmanr
from scipy.stats import rankdata
from scipy.stats import pearsonr
from matplotlib import pyplot as plt

#Used for obtaining the comparison stats of Spearman's coefficient of PMI/PPMI and distance of all words from the root verb of reference-variant pairs for the corpora/genre entered
#In order to get the PPMI make changes everywhere PMI is being used/calculated

#Extracting PMI values from look up table
df1 = pandas.read_csv('hdpmi_pos2.csv')
def get_pmi(head_pos,dep_pos):
    for index, row in df1.iterrows():
        if row['h']==head_pos and row['d']==dep_pos :
            return row['pmi']

#Extracting PPMI values from look up table
df1 = pandas.read_csv('hdpmi_pos2.csv')
def get_ppmi(head_pos,dep_pos):
    for index, row in df1.iterrows():
        if row['h']==head_pos and row['d']==dep_pos :
            if (row['pmi']>0):
               return row['pmi']
            else: 
               return 0        
        
def spearman_stats(var, ref):
    prev_dis=[]
    prer_dis=[]
    postv_dis=[]
    postr_dis=[]
    prev_pmi=[]
    prer_pmi=[]
    postv_pmi=[]
    postr_pmi=[]
    

    for t in var:
        if t['deprel']=='ROOT':
            vid=t['id']
            vroot=t
    for t in ref:
        if t['deprel']=='ROOT':
            rid=t['id']
            rroot=t
    
    prev=0
    postv=0
    prer=0
    postr=0
    
    #Obtaining PMI/PPMI and distance of all words before and after the root verb for ref-var pair           
    for t in var:
        if t['id']<vid:
            prev=prev+1            
            pmi=get_pmi(nltk.map_tag('brown', 'universal', vroot["xpos"]),nltk.map_tag('brown', 'universal', t["xpos"]))
            if pmi:
                prev_dis.append(vroot["id"]-t["id"])
                prev_pmi.append(pmi)
             
                
        if t['id']>vid:
            postv=postv+1
            pmi=get_pmi(nltk.map_tag('brown', 'universal', vroot["xpos"]),nltk.map_tag('brown', 'universal', t["xpos"]))
            if pmi:
                postv_dis.append(t["id"]-vroot["id"])
                postv_pmi.append(pmi)
              
    
    for t in ref:
        if t['id']<rid:
            prer=prer+1
            pmi=get_pmi(nltk.map_tag('brown', 'universal', rroot["xpos"]),nltk.map_tag('brown', 'universal', t["xpos"]))
            if pmi:
                prer_dis.append(rroot["id"]-t["id"])
                prer_pmi.append(pmi)
              
        if t['id']>rid:
            postr=postr+1
            pmi=get_pmi(nltk.map_tag('brown', 'universal', rroot["xpos"]),nltk.map_tag('brown', 'universal', t["xpos"]))
            if pmi:
                postr_dis.append(t["id"]-rroot["id"])
                postr_pmi.append(pmi)
               
    #Storing data of number of words before and after the root verb in ref-var pairs
    precountv.append(prev)
    postcountv.append(postv)
    precountr.append(prer)
    postcountr.append(postr)
    
    #Calculating Spearman's coefficient
    coef_prev, p1 = spearmanr(rankdata(prev_dis), rankdata(prev_pmi)) 
    coef_prer, p2 = spearmanr(rankdata(prer_dis), rankdata(prer_pmi)) 
    coef_postv, p3 = spearmanr(rankdata(postv_dis), rankdata(postv_pmi)) 
    coef_postr, p4 = spearmanr(rankdata(postr_dis), rankdata(postr_pmi))
    
    #Comparing Spearman's coefficients of ref-var pair
    flagpre_nan=0
    flagpost_nan=0
    if (math.isnan(coef_prev) or math.isnan(coef_prer)):
        flagpre_nan=1
    if (math.isnan(coef_postv) or math.isnan(coef_postr)):
        flagpost_nan=1
        ranks.append(0)
    else:
         ranks.append(coef_postr-coef_postv)
        
    flagpre_lessequal=0
    flagpost_lessequal=0
    if coef_prev <= coef_prer and not(math.isnan(coef_prev) or math.isnan(coef_prer)):
        flagpre_lessequal=1
    if coef_postv <= coef_postr and not(math.isnan(coef_postv) or math.isnan(coef_postr)):
        flagpost_lessequal=1
        
    flagpre_lessthan=0
    flagpost_lessthan=0
    if coef_prev < coef_prer and not(math.isnan(coef_prev) or math.isnan(coef_prer)):
        flagpre_lessthan=1
    if coef_postv < coef_postr and not(math.isnan(coef_postv) or math.isnan(coef_postr)):
        flagpost_lessthan=1
        
    flagpre_greaterthan=0
    flagpost_greaterthan=0
    if coef_prev > coef_prer and not(math.isnan(coef_prev) or math.isnan(coef_prer)):
        flagpre_greaterthan=1
    if coef_postv > coef_postr and not(math.isnan(coef_postv) or math.isnan(coef_postr)):
        flagpost_greaterthan=1
    
    return (flagpre_nan, flagpost_nan,flagpre_lessequal, flagpost_lessequal,flagpre_lessthan, flagpost_lessthan,flagpre_greaterthan, flagpost_greaterthan)    
     
      

precountv=[]
precountr=[]
postcountv=[]
postcountr=[] 
ranks=[] 

#Enter the file name for which data needs to be obtained 
file = open("brown.txt", "r")
lines = file.readlines()
data_file = open("brown_numbered.dep", "r", encoding="utf-8")
reflist=[]
sentence_list=[]

for line in lines:
    if "\tref\t" in line:
        reflist.append(1)
    else:
        if "\tpost-" in line:
            reflist.append(0)
        else:
            if "\tpre-" in line:
                reflist.append(3)
            else:
                reflist.append(2)
                
i=0;
cntpairs=0
cntref=0
cntpre_nan=0
cntpost_nan=0
cntpre_lessequal=0
cntpost_lessequal=0
cntpre_lessthan=0
cntpost_lessthan=0
cntpre_greaterthan=0
cntpost_greaterthan=0

#Obtaining the stats of comparison of Spearman's coefficients of ref-var pairs
for listtoken in parse_incr(data_file):
           if reflist[i]==1:    
                ref=listtoken
                cntref=cntref+1
                refno=i+1
           else:
                flagpre_nan, flagpost_nan,flagpre_lessequal, flagpost_lessequal,flagpre_lessthan, flagpost_lessthan,flagpre_greaterthan, flagpost_greaterthan=spearman_stats(listtoken, ref)
                cntpre_nan=cntpre_nan+flagpre_nan
                cntpost_nan=cntpost_nan+flagpost_nan
                cntpre_lessequal=cntpre_lessequal+flagpre_lessequal
                cntpost_lessequal=cntpost_lessequal+flagpost_lessequal
                cntpre_lessthan=cntpre_lessthan+flagpre_lessthan
                cntpost_lessthan=cntpost_lessthan+flagpost_lessthan
                cntpre_greaterthan=cntpre_greaterthan+flagpre_greaterthan
                cntpost_greaterthan=cntpost_greaterthan+flagpost_greaterthan
                cntpairs=cntpairs+1
           i=i+1

print ('Number of cases where Spearmans coefficient was NAN for either ref or var sentence, Pre Case: {}, Post Case: {}'.format(cntpre_nan, cntpost_nan))
print ('Number of cases where Spearmans coefficient of variant sentence was less than or equal to that of ref sentence, Pre Case: {}, Post Case: {}'.format(cntpre_lessequal, cntpost_lessequal))
print ('Number of cases where Spearmans coefficient of variant sentence was less than that of ref sentence, Pre Case: {}, Post Case: {}'.format(cntpre_lessthan, cntpost_lessthan))
print ('Number of cases where Spearmans coefficient of variant sentence was greater than that of ref sentence, Pre Case: {}, Post Case: {}'.format(cntpre_greaterthan, cntpost_greaterthan))


#Histograms of number of words before and after the root verb in reference and variant sentences
fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(precountv, bins = [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60])
plt.show()

fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(precountr, bins = [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60])
plt.show()

fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(postcountv, bins = [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70])
plt.show()

fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(postcountr, bins = [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70])
plt.show()

#Used for adding Spearman's coefficient between PMI/PPMI and distance between all words and root verb in the form of ranks
# df = pandas.read_csv('brown-ranks_intermediate.csv')
df['Spearman_allwords']=ranks
print(ranks)
# df.to_csv('brown-ranks_intermediate.csv')