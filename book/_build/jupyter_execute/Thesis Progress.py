#!/usr/bin/env python
# coding: utf-8

# ## Thesis Progress
# 
# 

# <hr>
# 
# #### Current Worries
# 
# - The **label frequencies** of the twemlab goldstandard training dataset. This is the distribution of the overall 1625 goldstandard emotion labels:

# In[1]:


# Load TwEmLab Goldstandard for Birmingham
tree1 = ET.parse('../Data/twemlab_goldstandards_original/birmingham_labels.xml')
root1 = tree1.getroot()

# check contents
#root1[0][1].text

# create dataframe from xml file
data1 = []
for tweet in root1.findall('Tweet'):
    id = tweet.find('ID').text
    label = tweet.find('Label').text
    data1.append((id, label))

df1 = pd.DataFrame(data1,columns=['id','label'])
 # df1.head()
    
# Load TwEmLab Birmingham Tweets
tree2 = ET.parse('../Data/twemlab_goldstandards_original/birmingham_tweets.xml')
root2 = tree2.getroot()

# check contents
# root2[0][1].text

# create dataframe from xml file
data2 = []
for tweet in root2.findall('Tweet'):
    id = tweet.find('ID').text
    text = tweet.find('text').text
    goldstandard = tweet.attrib.get("goldstandard")
    data2.append((id, text, goldstandard))

df2 = pd.DataFrame(data2,columns=['id','text', 'goldstandard'])
# df2.head()

 # merge the two separate dataframes based on id columns
merge = pd.merge(df1, df2, on='id')

# keep only the tweets that are part of the goldstandard
twemlab = merge[merge['goldstandard'] == 'yes']
print(f'Number of tweets in goldstandard: {len(twemlab)}')

emotions = []
# assign emotion label (happiness, anger, sadness, fear)
for index, row in twemlab.iterrows():
    if row['label'] == 'beauty' or row['label'] == 'happiness':
        emotions.append('happiness')
    elif row['label'] == 'anger/disgust':
        emotions.append('anger')
    elif row['label'] == 'sadness':
        emotions.append('sadness')
    elif row['label'] == 'fear':
        emotions.append('fear')
    else: 
        emotions.append('none')
        
twemlab['emotion'] = emotions

twemlab_birmingham = twemlab[['id','text','emotion']]

# check dataset
# twemlab_birmingham.head(20)

readfile = pd.read_csv('../Data/twemlab_goldstandards_original/boston_goldstandard.csv')
twemlab_boston = readfile[['Tweet_ID', 'Tweet_timestamp', 'Tweet_text', 'Tweet_goldstandard_attribute', 'Tweet_longitude','Tweet_latitude','Tweet_timestamp','Emotion']]
# use only rows that have text in them
twemlab_boston = twemlab_boston[0:631]
# twemlab_boston.head()

emotions = []
# assign emotion label (happiness, anger, sadness, fear)
for index, row in twemlab_boston.iterrows():
    if row['Emotion'] == 'beauty' or row['Emotion'] == 'happiness':
        emotions.append('happiness')
    elif row['Emotion'] == 'anger/disgust':
        emotions.append('anger')
    elif row['Emotion'] == 'sadness':
        emotions.append('sadness')
    elif row['Emotion'] == 'fear':
        emotions.append('fear')
    else: 
        emotions.append('none')
        
twemlab_boston['emotion'] = emotions

twemlab_boston = twemlab_boston[['Tweet_ID','Tweet_text','emotion']]

# check dataset
# twemlab_boston.head(20)


# In[96]:


# extract the emotion column from both dfs and merge
brim_emo = twemlab_birmingham[['emotion']]
bost_emo = twemlab_boston[['emotion']]

emotions_in_twemlab_all = brim_emo.append(bost_emo, ignore_index=True)
print(len(emotions_in_twemlab_all))


# In[97]:


#value_counts = twemlab['label'].value_counts()
value_counts = emotions_in_twemlab_all['emotion'].value_counts().reset_index()
value_counts = value_counts.rename(columns={'index': 'Value', 'emotion': 'Frequency'})
df_value_counts = pd.DataFrame(value_counts)

dem_cols = df_value_counts[['Value', 'Frequency']]
dem_cols


# - **Size and standard of training data**: The overall size of the training data for twemlab is 1625 and its aspect terms have been annotated by one individual "ad hoc" as per "Annotating Twemlab Goldstandard Files to Include Aspect Term Labels".</li>
# 
# - **Mearsuring performance**: a robust testing dataset is needed. Above, I have shown a makeshift performance measure on the same dataset that the model was trained on. Drawing any meaningful conclusions based on this is precarious.
# 
# - **Use different training data vs. annotate?** [SemEval2018 Task1 Affect in Tweets](https://competitions.codalab.org/competitions/17751#learn_the_details-datasets) (anger: 1700, fear: 2200, joy: 1600, sadness: 1500)
# 
# 
# |ID	|Tweet	|Affect Dimension	|Intensity Score|
# |-----|-----|------|-----|
# |2017-En-10264	|@xandraaa5 @amayaallyn6 shut up hashtags are cool #offended	|anger|	0.562|
# |2017-En-10072|	it makes me so fucking irate jesus. nobody is calling ppl who like hajime abusive stop with the strawmen lmao	|anger	|0.750|
# |2017-En-11383	|Lol Adam the Bull with his fake outrage...	|anger	|0.417|
# |2017-En-11102|	@THATSSHAWTYLO passed away early this morning in a fast and furious styled car crash as he was leaving an ATL strip club. That's rough stuff|	anger	|0.354|
# |2017-En-20968|	@RockSolidShow @Pat_Francis #revolting cocks if you think I'm sexy!|	fear	|0.292|
# |2017-En-21816|	@Its_just_Huong I will beat you !!! Always thought id be gryffindor so this is a whole new world for me 😨😨😨 #excited #afraid	|fear	|0.667|
# |2017-En-40023|	This the most depressing shit ever|	sadness	|0.861|
# |2017-En-30793|	@david_garrett Quite saddened.....no US dates, no joyous anticipation of attending a DG concert (since 2014). Happy you are keeping busy.	|joy	|0.140|
# 
# - **Training capacities**: Batch size reduced due to out-of-memory errors. GRACE training is memory intensive (the authors use a nvidia tesla v100 gpu). Potential options: reduce float point precision? Currently having issues installing conda package for apex to do so. 
# 
# #### Current Questions
# 
# - **Model optimisation**? The GRACE model uses GeLU (an "advanced" activation function), the standard BERT nn.embeddings layer, 12 transformer encoder layers and 2 decoder layers. On top of that it has two classification heads (both nn.Linear). During training the model uses additional functions for virtaul adversarial training and gradient harmonized loss calculation.
# 
# - **Transfer Learning**? Arguments for: fine-tuning is much more accurate than feature-extraction. And the most efficient way of fine-tuning a model that will likely need to be fine-tuned again and again is transfer learning (adapters)
# 
# #<img src="https://github.com/Christina1281995/demo-repo/blob/main/transfernlp2.JPG?raw=true">
# <!-- <img src="https://github.com/Christina1281995/demo-repo/blob/main/transfernlp2.JPG?raw=true" align="left" width="49%"> -->
# 
# <img src="https://github.com/Christina1281995/demo-repo/blob/main/transfernlp.JPG?raw=true">
# <!-- <img src="https://github.com/Christina1281995/demo-repo/blob/main/transfernlp.JPG?raw=true" align="right" width="49%"> -->
# 
# <br>
# <br>
# 
# - **Create pipline for entire workflow**? 
# 
# <img src="https://github.com/Christina1281995/demo-repo/blob/main/piplineworkflow.png?raw=true" width="70%">
# 
# 
# 

# #### Potential Approach to Thesis
# 
# 
# **“Pitch”**
# 
# Aspect-based sentiment analysis (ABSA) has garnered attention in recent years due to its fine-grained approach to sentiment analysis. ABSA enables sentiment analysis at the entity or aspect level, instead of the entire document, resulting in better insights. ABSA has four sub-tasks, each identifying a different token-level piece of information. While numerous models have been developed for ABSA, there are only a handful of methods that have been applied to social media (Twitter) data for aspect term and sentiment extraction.
# 
# However, current ABSA research has rarely explored the application of emotion detection for aspect-level sentiment analysis and it is not known that emotion detection has been applied to the End to End ABSA task. To date, there is also lacking research into the geographical distribution of emotions related to aspect terms. Additionally, there appears to be a lack of publicly available training data for aspect-level emotions on Twitter.
# 
# Therefore, the thesis aims to generate a publicly accessible dataset for aspect-level emotions and investigate whether a model optimized for social media end-to-end ABSA can be trained on this more fine-grained data with comparable accuracy. The thesis will also explore whether this fine-grained approach, coupled with geographical analysis, reveals deeper and more varied insights into public opinion on specific aspect terms. The hypothesis is that a geographical analysis of aspect-based emotions provides a more nuanced geospatial view of the otherwise simplified "negative" and "positive" labels.
# 
# <hr>
# 
# <img src="https://github.com/Christina1281995/demo-repo/blob/main/Picture3.png?raw=true" width="80%">
# 
# <hr>
# 
# I. Introduction
# - Background and motivation for the research (need for fine-grained geographical, semantic and sentiment related information in real-world use cases such as COVID-19)
# - Research question, aims, and objectives
# - Hypothesis (a geographical analysis of aspect-based emotions provides a more nuanced view of the otherwise simplified "negative" and "positive" labels)
# - Outline of the thesis
# 
# II. Literature Review
# - Overview of sentiment analysis and ABSA
# - Review of related work in aspect-level sentiment analysis
# - Review of related work in emotion detection for ABSA 
# - Review of related work in geographical sentiment analysis
# 
# III. Methodology
# - Description of the proposed methodology for generating a dataset for aspect-level emotions (labelling standard, data collection and preprocessing)
# - Description of the proposed methodology for training and testing the End to End ABSA model on the dataset (GRACE in some detail, performance metrics)
# - Description of the proposed methodology for performing geographical analysis on the aspect-based emotions (hot spot analysis –Getis-Ord Gi*)
# 
# IV. Results
# - Description of the dataset and its properties (label distribution, count, stats of annotated twemlab goldstandard or SemEval 2018)
# - Evaluation of the End to End ABSA model on the dataset (performance metrics)
# - Analysis of the geographical distribution of aspect-based emotions (maps displaying aspect-term emotions)
# - Comparison with geographical results of other methodologies (maps displaying sentiments: doc-level sentiment analysis and original ABSA model)
# 
# V. Discussion
# - Discussion of the results in relation to the research aims and hypothesis (does emotion ABSA show more detailed, nuanced view of the case study?) 
# - Reflection on the limitations of the research (transferability, training capacities, …)
# 
# VI. Conclusion and Future Work
# - Summary of the main findings
# - Suggestions for future research (model optimisation, transfer learning, pipelining the approach …)
# 
# VII. References
# - List of cited works in the thesis
# 
# VIII. Appendix
# - Description of the dataset and its format
# - Description of the End to End ABSA model (GRACE) and its parameters used for training etc.
# 
# <hr>
# 
# **Potential Research questions**
# 
# 1. Can a model optimised for social media end-to-end ABSA be trained on a more fine-grained set of emotions with comparable accuracy? 
# 2. How do the insights gained from analysing the emotions associated with aspect terms differ from those derived from traditional aspect-based sentiment analysis? (How can these insights be used to improve public opinion analysis?)
# 3. What new insights can be gained from analysing the geographical distribution of aspect-based emotions, and how can this be used to inform decision-making processes?
# 
# 
