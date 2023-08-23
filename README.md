# Viralityt_of_Tweet

```
%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier

from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score,accuracy_score
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from datetime import date
from google.colab import files
```


```
df = pd.read_json('Train Data.json', lines=True)
df.head()
```





  <div id="df-45f8a0ce-7b87-4a5e-822d-39542fbc5735" class="colab-df-container">
    <div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>created_at</th>
      <th>id</th>
      <th>id_str</th>
      <th>text</th>
      <th>truncated</th>
      <th>entities</th>
      <th>metadata</th>
      <th>source</th>
      <th>in_reply_to_status_id</th>
      <th>in_reply_to_status_id_str</th>
      <th>...</th>
      <th>favorite_count</th>
      <th>favorited</th>
      <th>retweeted</th>
      <th>lang</th>
      <th>possibly_sensitive</th>
      <th>quoted_status_id</th>
      <th>quoted_status_id_str</th>
      <th>extended_entities</th>
      <th>quoted_status</th>
      <th>withheld_in_countries</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2018-07-31 13:34:40+00:00</td>
      <td>1024287229525598210</td>
      <td>1024287229525598208</td>
      <td>RT @KWWLStormTrack7: We are more than a month ...</td>
      <td>False</td>
      <td>{'hashtags': [], 'symbols': [], 'user_mentions...</td>
      <td>{'iso_language_code': 'en', 'result_type': 're...</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
      <td>en</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018-07-31 13:34:40+00:00</td>
      <td>1024287229512953856</td>
      <td>1024287229512953856</td>
      <td>@hail_ee23 Thanks love its just the feeling of...</td>
      <td>False</td>
      <td>{'hashtags': [], 'symbols': [], 'user_mentions...</td>
      <td>{'iso_language_code': 'en', 'result_type': 're...</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>1.024128e+18</td>
      <td>1.024128e+18</td>
      <td>...</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
      <td>en</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018-07-31 13:34:40+00:00</td>
      <td>1024287229504569344</td>
      <td>1024287229504569344</td>
      <td>RT @TransMediaWatch: Pink News has more on the...</td>
      <td>False</td>
      <td>{'hashtags': [], 'symbols': [], 'user_mentions...</td>
      <td>{'iso_language_code': 'en', 'result_type': 're...</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
      <td>en</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018-07-31 13:34:40+00:00</td>
      <td>1024287229496029190</td>
      <td>1024287229496029184</td>
      <td>RT @realDonaldTrump: One of the reasons we nee...</td>
      <td>False</td>
      <td>{'hashtags': [], 'symbols': [], 'user_mentions...</td>
      <td>{'iso_language_code': 'en', 'result_type': 're...</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
      <td>en</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018-07-31 13:34:40+00:00</td>
      <td>1024287229492031490</td>
      <td>1024287229492031488</td>
      <td>RT @First5App: This hearing of His Word doesn’...</td>
      <td>False</td>
      <td>{'hashtags': [], 'symbols': [], 'user_mentions...</td>
      <td>{'iso_language_code': 'en', 'result_type': 're...</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
      <td>en</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-45f8a0ce-7b87-4a5e-822d-39542fbc5735')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  </div>


<div id="df-f4f00d2b-08f5-46fd-be11-6c5c549e206b">
  <button class="colab-df-quickchart" onclick="quickchart('df-f4f00d2b-08f5-46fd-be11-6c5c549e206b')"
            title="Suggest charts."
            style="display:none;">


  </button>

</div>
    </div>
  </div>





```
null_counts = df.isnull().sum()
#null
print(null_counts[null_counts > 0].sort_values(ascending=False))
```

    contributors                 11099
    withheld_in_countries        11097
    geo                          11082
    coordinates                  11082
    place                        10943
    quoted_status                10772
    quoted_status_id              9945
    quoted_status_id_str          9945
    extended_entities             9900
    in_reply_to_status_id         9697
    in_reply_to_status_id_str     9697
    in_reply_to_user_id           9596
    in_reply_to_user_id_str       9596
    in_reply_to_screen_name       9596
    possibly_sensitive            7907
    retweeted_status              3727
    dtype: int64



```
#percent
print((null_counts[null_counts > 0].sort_values(ascending=False)/len(df))*100)
```

    contributors                 100.000000
    withheld_in_countries         99.981980
    geo                           99.846833
    coordinates                   99.846833
    place                         98.594468
    quoted_status                 97.053789
    quoted_status_id              89.602667
    quoted_status_id_str          89.602667
    extended_entities             89.197225
    in_reply_to_status_id         87.368231
    in_reply_to_status_id_str     87.368231
    in_reply_to_user_id           86.458239
    in_reply_to_user_id_str       86.458239
    in_reply_to_screen_name       86.458239
    possibly_sensitive            71.240652
    retweeted_status              33.579602
    dtype: float64



```
median_retweets = np.median(df['retweet_count'])
df['is_viral'] = np.where(df['retweet_count'] >= median_retweets, 1, 0)

df['tweet_length'] = df.apply(lambda tweet: len(tweet['text']), axis=1)
df['followers_count'] = df.apply(lambda tweet: tweet['user']['followers_count'], axis=1)
df['friends_count'] = df.apply(lambda tweet: tweet['user']['friends_count'], axis=1)
df['favourites_count'] = df.apply(lambda tweet: tweet['user']['favourites_count'], axis=1)
df['statuses_count'] = df.apply(lambda tweet: tweet['user']['statuses_count'], axis=1)
df['listed_count']= df.apply(lambda tweet: tweet['user']['listed_count'], axis=1)
df['hashtag_count'] = df.apply(lambda tweet: tweet['text'].count('#'), axis=1)
df['http_count']= df.apply(lambda tweet: tweet['text'].count('http'), axis=1)
df['sign']= df.apply(lambda tweet: tweet['text'].count('@'),axis=1)
df['verified']= df.apply(lambda tweet: tweet['user']['verified'], axis=1)
```


```
def month_converter(month):
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    return months.index(month) + 1
```


```
df['Days_count']=0

df['date']=df.user.apply(lambda x:x['created_at'].split( ))
for i in range(len(df['date'])):
  df['date'][i][1]=month_converter(df['date'][i][1])

w=len(df['date'])
Matrix = [[0 for x in range(3)] for y in range(w)]
d= date(2020, 7, 7)
for i in range(len(df['date'])):
  Matrix[i][0] = int(df['date'][i][5])
  Matrix[i][1] = int(df['date'][i][1])
  Matrix[i][2] = int(df['date'][i][2])
  df['Days_count'][i]=(d-date(Matrix[i][0],Matrix[i][1],Matrix[i][2])).days
```

    <ipython-input-89-6f6eb7250377>:14: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df['Days_count'][i]=(d-date(Matrix[i][0],Matrix[i][1],Matrix[i][2])).days


**Define x , y**


```
y = df['is_viral']
x = df[['tweet_length' , 'followers_count', 'friends_count','favourites_count','statuses_count',
        'hashtag_count', 'http_count', 'sign', 'verified','listed_count','Days_count','truncated']]

```

**Describe Data**


```
x.head()
```





  <div id="df-844664d0-b24d-41e8-88af-6ae1a72c7ad1" class="colab-df-container">
    <div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tweet_length</th>
      <th>followers_count</th>
      <th>friends_count</th>
      <th>favourites_count</th>
      <th>statuses_count</th>
      <th>hashtag_count</th>
      <th>http_count</th>
      <th>sign</th>
      <th>verified</th>
      <th>listed_count</th>
      <th>Days_count</th>
      <th>truncated</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>140</td>
      <td>215</td>
      <td>335</td>
      <td>3419</td>
      <td>4475</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>False</td>
      <td>2</td>
      <td>3703</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>77</td>
      <td>199</td>
      <td>203</td>
      <td>2136</td>
      <td>3922</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>False</td>
      <td>1</td>
      <td>2308</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>140</td>
      <td>196</td>
      <td>558</td>
      <td>62560</td>
      <td>11546</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>False</td>
      <td>0</td>
      <td>1046</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>140</td>
      <td>3313</td>
      <td>2272</td>
      <td>51818</td>
      <td>26609</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>False</td>
      <td>41</td>
      <td>4138</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>140</td>
      <td>125</td>
      <td>273</td>
      <td>1332</td>
      <td>519</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>False</td>
      <td>3</td>
      <td>2795</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-844664d0-b24d-41e8-88af-6ae1a72c7ad1')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">


  </div>


<div id="df-889ef14c-29c3-4942-ac0e-4f9ed98d86ab">
  <button class="colab-df-quickchart" onclick="quickchart('df-889ef14c-29c3-4942-ac0e-4f9ed98d86ab')"
            title="Suggest charts."
            style="display:none;">


</div>
    </div>
  </div>





```
print(x.shape)
print(x.columns)
x.describe()
```

    (11099, 12)
    Index(['tweet_length', 'followers_count', 'friends_count', 'favourites_count',
           'statuses_count', 'hashtag_count', 'http_count', 'sign', 'verified',
           'listed_count', 'Days_count', 'truncated'],
          dtype='object')






  <div id="df-b9d3b797-af00-4f86-a244-69be2fee9a0a" class="colab-df-container">
    <div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tweet_length</th>
      <th>followers_count</th>
      <th>friends_count</th>
      <th>favourites_count</th>
      <th>statuses_count</th>
      <th>hashtag_count</th>
      <th>http_count</th>
      <th>sign</th>
      <th>listed_count</th>
      <th>Days_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>11099.000000</td>
      <td>1.109900e+04</td>
      <td>11099.000000</td>
      <td>11099.000000</td>
      <td>1.109900e+04</td>
      <td>11099.000000</td>
      <td>11099.000000</td>
      <td>11099.000000</td>
      <td>11099.000000</td>
      <td>11099.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>122.833589</td>
      <td>6.009168e+03</td>
      <td>1442.336337</td>
      <td>19413.978286</td>
      <td>3.476254e+04</td>
      <td>0.232543</td>
      <td>0.412379</td>
      <td>1.085233</td>
      <td>47.017479</td>
      <td>2376.525813</td>
    </tr>
    <tr>
      <th>std</th>
      <td>27.850477</td>
      <td>2.013144e+05</td>
      <td>7645.949991</td>
      <td>39144.906425</td>
      <td>8.879138e+04</td>
      <td>0.725709</td>
      <td>0.525913</td>
      <td>0.970616</td>
      <td>254.953725</td>
      <td>1120.136156</td>
    </tr>
    <tr>
      <th>min</th>
      <td>9.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>707.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>110.000000</td>
      <td>1.310000e+02</td>
      <td>194.000000</td>
      <td>1052.000000</td>
      <td>2.543000e+03</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1300.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>140.000000</td>
      <td>4.030000e+02</td>
      <td>442.000000</td>
      <td>5538.000000</td>
      <td>9.943000e+03</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>2386.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>140.000000</td>
      <td>1.249000e+03</td>
      <td>1116.000000</td>
      <td>19576.500000</td>
      <td>3.418700e+04</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>19.000000</td>
      <td>3319.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>155.000000</td>
      <td>2.021186e+07</td>
      <td>510292.000000</td>
      <td>635920.000000</td>
      <td>2.848360e+06</td>
      <td>10.000000</td>
      <td>4.000000</td>
      <td>12.000000</td>
      <td>12895.000000</td>
      <td>4977.000000</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-b9d3b797-af00-4f86-a244-69be2fee9a0a')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  </div>


<div id="df-ebaa55c7-e6fd-4411-b320-8b1413e5da81">
  <button class="colab-df-quickchart" onclick="quickchart('df-ebaa55c7-e6fd-4411-b320-8b1413e5da81')"
            title="Suggest charts."
            style="display:none;">


</div>
    </div>
  </div>





```
f, ax = plt.subplots(figsize = [10,9])
sns.heatmap(x.corr(),linewidths = .5, annot = True, cmap = 'YlGnBu', square = True, vmin=-1, vmax=1)
```


![1](https://github.com/Zejabati/Viralityt_of_Tweet/assets/65095428/aaa5c85b-f189-4f67-916d-c8ef28bdcc94)




```
list_columns=['tweet_length', 'followers_count', 'friends_count', 'favourites_count',
       'statuses_count', 'hashtag_count', 'http_count', 'sign','listed_count', 'Days_count']

fig = plt.figure()
fig, axs = plt.subplots(2,5,figsize=(15,5))
plt.subplots_adjust(wspace=1,hspace=1)

for i in range(1,11):
  plt.subplot(2, 5, i)
  plt.scatter(x[list_columns[i-1]],df['retweet_count'],s=2)
  plt.xlabel(list_columns[i-1])
  plt.ylabel('retweet_count')

plt.show()
```


![2](https://github.com/Zejabati/Viralityt_of_Tweet/assets/65095428/6fb2e2a9-e5a1-4b2b-8478-57b013c7f14c)



```
null_counts = x.isnull().sum()
#null
print(null_counts[null_counts > 0].sort_values(ascending=False))
#percent
print((null_counts[null_counts > 0].sort_values(ascending=False)/len(x))*100)
```

    Series([], dtype: int64)
    Series([], dtype: float64)



```
x_boxplot=x.copy()
```


```
non_numerical_columns = ['verified','truncated']

for i in non_numerical_columns:
    x1 = pd.get_dummies(x[i])
    x = x.join(x1,lsuffix='_l',rsuffix='_r')
    x.drop(i,axis=1,inplace=True)
```


```
data=x.copy()
data['is_viral']=y
```


```
x.head()
```





  <div id="df-94721e64-060a-4979-a97f-6a0b6d6a6edc" class="colab-df-container">
    <div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tweet_length</th>
      <th>followers_count</th>
      <th>friends_count</th>
      <th>favourites_count</th>
      <th>statuses_count</th>
      <th>hashtag_count</th>
      <th>http_count</th>
      <th>sign</th>
      <th>listed_count</th>
      <th>Days_count</th>
      <th>False_l</th>
      <th>True_l</th>
      <th>False_r</th>
      <th>True_r</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>140</td>
      <td>215</td>
      <td>335</td>
      <td>3419</td>
      <td>4475</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3703</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>77</td>
      <td>199</td>
      <td>203</td>
      <td>2136</td>
      <td>3922</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2308</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>140</td>
      <td>196</td>
      <td>558</td>
      <td>62560</td>
      <td>11546</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1046</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>140</td>
      <td>3313</td>
      <td>2272</td>
      <td>51818</td>
      <td>26609</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>41</td>
      <td>4138</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>140</td>
      <td>125</td>
      <td>273</td>
      <td>1332</td>
      <td>519</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>2795</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-94721e64-060a-4979-a97f-6a0b6d6a6edc')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">


    </button>



  </div>


<div id="df-9caf7a07-5260-4439-98dc-f28a65426073">
  <button class="colab-df-quickchart" onclick="quickchart('df-9caf7a07-5260-4439-98dc-f28a65426073')"
            title="Suggest charts."
            style="display:none;">


</div>
    </div>
  </div>


```
x.columns
```




    Index(['tweet_length', 'followers_count', 'friends_count', 'favourites_count',
           'statuses_count', 'hashtag_count', 'http_count', 'sign', 'listed_count',
           'Days_count', 'False_l', 'True_l', 'False_r', 'True_r'],
          dtype='object')



```
x_scaled = StandardScaler().fit_transform(x)
x_scaled=pd.DataFrame(x_scaled)
x_scaled.columns =['tweet_length',  'followers_count','friends_count','favourites_count','statuses_count','hashtag_count',
             'http_count', 'sign','listed_count','Days_count','False_l', 'True_l', 'False_r', 'True_r']
x_scaled.head()
```





  <div id="df-d136f0f0-dc58-4cee-9d4f-163f0ebc3072" class="colab-df-container">
    <div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tweet_length</th>
      <th>followers_count</th>
      <th>friends_count</th>
      <th>favourites_count</th>
      <th>statuses_count</th>
      <th>hashtag_count</th>
      <th>http_count</th>
      <th>sign</th>
      <th>listed_count</th>
      <th>Days_count</th>
      <th>False_l</th>
      <th>True_l</th>
      <th>False_r</th>
      <th>True_r</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.616405</td>
      <td>-0.028783</td>
      <td>-0.144833</td>
      <td>-0.408628</td>
      <td>-0.341124</td>
      <td>-0.320451</td>
      <td>-0.784156</td>
      <td>-0.087817</td>
      <td>-0.176579</td>
      <td>1.184261</td>
      <td>0.126569</td>
      <td>-0.126569</td>
      <td>0.411611</td>
      <td>-0.411611</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.645776</td>
      <td>-0.028862</td>
      <td>-0.162098</td>
      <td>-0.441405</td>
      <td>-0.347353</td>
      <td>-0.320451</td>
      <td>-0.784156</td>
      <td>-0.087817</td>
      <td>-0.180502</td>
      <td>-0.061179</td>
      <td>0.126569</td>
      <td>-0.126569</td>
      <td>0.411611</td>
      <td>-0.411611</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.616405</td>
      <td>-0.028877</td>
      <td>-0.115666</td>
      <td>1.102263</td>
      <td>-0.261485</td>
      <td>-0.320451</td>
      <td>-0.784156</td>
      <td>-0.087817</td>
      <td>-0.184424</td>
      <td>-1.187879</td>
      <td>0.126569</td>
      <td>-0.126569</td>
      <td>0.411611</td>
      <td>-0.411611</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.616405</td>
      <td>-0.013393</td>
      <td>0.108515</td>
      <td>0.827834</td>
      <td>-0.091832</td>
      <td>-0.320451</td>
      <td>-0.784156</td>
      <td>-0.087817</td>
      <td>-0.023603</td>
      <td>1.572624</td>
      <td>0.126569</td>
      <td>-0.126569</td>
      <td>0.411611</td>
      <td>-0.411611</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.616405</td>
      <td>-0.029230</td>
      <td>-0.152942</td>
      <td>-0.461945</td>
      <td>-0.385680</td>
      <td>-0.320451</td>
      <td>-0.784156</td>
      <td>-0.087817</td>
      <td>-0.172657</td>
      <td>0.373609</td>
      <td>0.126569</td>
      <td>-0.126569</td>
      <td>0.411611</td>
      <td>-0.411611</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-d136f0f0-dc58-4cee-9d4f-163f0ebc3072')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">


  </div>


<div id="df-e33251ca-2389-49ad-bedd-742fe8c67a28">
  <button class="colab-df-quickchart" onclick="quickchart('df-e33251ca-2389-49ad-bedd-742fe8c67a28')"
            title="Suggest charts."
            style="display:none;">


</div>
    </div>
  </div>




**Detecting Outliers usind DBSCAN**


```
A = []
B = []
C = []

for i in np.linspace(0.1,5,30):
    db = DBSCAN(eps=i, min_samples=3).fit(x_scaled)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    sum = 0
    for t in labels:
        if t == -1:
            sum = sum + 1
            sumpercent=(sum/len(x_scaled))*100
    C.append(sumpercent)
    A.append(i)
    B.append(int(n_clusters_))
```


```
results = pd.DataFrame([A,B,C]).T
results.columns = ['distance','Number of clusters','Percent of outliers']
ax=results.plot(x='distance',y='Percent of outliers',figsize=(10,6),)
ax.set_xlabel('distance')
ax.set_ylabel('Percent of outliers')
ax.set_title('Percent of outliers/distance')

```




    Text(0.5, 1.0, 'Percent of outliers/distance')

    
![3](https://github.com/Zejabati/Viralityt_of_Tweet/assets/65095428/77a999ec-be50-4a3c-83a8-8fafec16d1d2)



```
M=1000
for i in range(len(C)):
  if 3<=C[i]<=4 and M==1000:
    eps=A[i]
    M=0
print(eps)
```

    1.4517241379310346



```
x_scaled.shape
```




    (11099, 14)




```
clustering = DBSCAN(eps=eps, min_samples=3).fit(x_scaled)
predict = clustering.labels_
#list(predict).count(-1)
x_scaled = x_scaled[~np.where(predict == -1, True, False)]
y = y[~np.where(predict == -1, True, False)]
```


```
x_scaled.shape
```




    (10701, 14)



**Detecting Outlier using Boxplot**


```
data.head()

```





  <div id="df-7c1890a9-fdd1-444c-8e5e-e9ad8acfbb9a" class="colab-df-container">
    <div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tweet_length</th>
      <th>followers_count</th>
      <th>friends_count</th>
      <th>favourites_count</th>
      <th>statuses_count</th>
      <th>hashtag_count</th>
      <th>http_count</th>
      <th>sign</th>
      <th>listed_count</th>
      <th>Days_count</th>
      <th>False_l</th>
      <th>True_l</th>
      <th>False_r</th>
      <th>True_r</th>
      <th>is_viral</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>140</td>
      <td>215</td>
      <td>335</td>
      <td>3419</td>
      <td>4475</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3703</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>77</td>
      <td>199</td>
      <td>203</td>
      <td>2136</td>
      <td>3922</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2308</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>140</td>
      <td>196</td>
      <td>558</td>
      <td>62560</td>
      <td>11546</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1046</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>140</td>
      <td>3313</td>
      <td>2272</td>
      <td>51818</td>
      <td>26609</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>41</td>
      <td>4138</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>140</td>
      <td>125</td>
      <td>273</td>
      <td>1332</td>
      <td>519</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>2795</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-7c1890a9-fdd1-444c-8e5e-e9ad8acfbb9a')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">



  </div>


<div id="df-8bde347e-edae-4e72-a421-653f0096786b">
  <button class="colab-df-quickchart" onclick="quickchart('df-8bde347e-edae-4e72-a421-653f0096786b')"
            title="Suggest charts."
            style="display:none;">



</div>
    </div>
  </div>





```
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
print(IQR)

print((data< (Q1 - 1.5 * IQR)) |(data > (Q3 + 1.5 * IQR)))
```

    tweet_length           30.0
    followers_count      1118.0
    friends_count         922.0
    favourites_count    18524.5
    statuses_count      31644.0
    hashtag_count           0.0
    http_count              1.0
    sign                    0.0
    listed_count           19.0
    Days_count           2019.0
    False_l                 0.0
    True_l                  0.0
    False_r                 0.0
    True_r                  0.0
    is_viral                1.0
    dtype: float64
           tweet_length  followers_count  friends_count  favourites_count  \
    0             False            False          False             False   
    1             False            False          False             False   
    2             False            False          False              True   
    3             False             True          False              True   
    4             False            False          False             False   
    ...             ...              ...            ...               ...   
    11094         False            False          False             False   
    11095         False            False          False             False   
    11096         False            False          False             False   
    11097         False            False          False             False   
    11098         False            False          False             False   
    
           statuses_count  hashtag_count  http_count   sign  listed_count  \
    0               False          False       False  False         False   
    1               False          False       False  False         False   
    2               False          False       False  False         False   
    3               False          False       False  False         False   
    4               False          False       False  False         False   
    ...               ...            ...         ...    ...           ...   
    11094           False          False       False  False         False   
    11095           False          False       False  False         False   
    11096           False          False       False   True         False   
    11097           False          False       False  False         False   
    11098           False          False       False  False         False   
    
           Days_count  False_l  True_l  False_r  True_r  is_viral  
    0           False    False   False    False   False     False  
    1           False    False   False    False   False     False  
    2           False    False   False    False   False     False  
    3           False    False   False    False   False     False  
    4           False    False   False    False   False     False  
    ...           ...      ...     ...      ...     ...       ...  
    11094       False    False   False    False   False     False  
    11095       False    False   False    False   False     False  
    11096       False    False   False     True    True     False  
    11097       False    False   False    False   False     False  
    11098       False    False   False    False   False     False  
    
    [11099 rows x 15 columns]



```
x_boxplot.head()
```





  <div id="df-de5a68db-d006-4aaf-8753-81ad557cd692" class="colab-df-container">
    <div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tweet_length</th>
      <th>followers_count</th>
      <th>friends_count</th>
      <th>favourites_count</th>
      <th>statuses_count</th>
      <th>hashtag_count</th>
      <th>http_count</th>
      <th>sign</th>
      <th>verified</th>
      <th>listed_count</th>
      <th>Days_count</th>
      <th>truncated</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>140</td>
      <td>215</td>
      <td>335</td>
      <td>3419</td>
      <td>4475</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>False</td>
      <td>2</td>
      <td>3703</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>77</td>
      <td>199</td>
      <td>203</td>
      <td>2136</td>
      <td>3922</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>False</td>
      <td>1</td>
      <td>2308</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>140</td>
      <td>196</td>
      <td>558</td>
      <td>62560</td>
      <td>11546</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>False</td>
      <td>0</td>
      <td>1046</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>140</td>
      <td>3313</td>
      <td>2272</td>
      <td>51818</td>
      <td>26609</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>False</td>
      <td>41</td>
      <td>4138</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>140</td>
      <td>125</td>
      <td>273</td>
      <td>1332</td>
      <td>519</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>False</td>
      <td>3</td>
      <td>2795</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-de5a68db-d006-4aaf-8753-81ad557cd692')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">


  </div>


<div id="df-75a53ec9-f05f-4dbd-b668-2418c6c73896">
  <button class="colab-df-quickchart" onclick="quickchart('df-75a53ec9-f05f-4dbd-b668-2418c6c73896')"
            title="Suggest charts."
            style="display:none;">


</div>
    </div>
  </div>





```
continous=x_boxplot.drop(columns=['verified','truncated'])
x_out = continous[~((continous < (Q1 - 1.5 * IQR)) |(continous > (Q3 + 1.5 * IQR))).any(axis=1)]
print(x_out.shape)
```

    (4234, 10)


    <ipython-input-111-30413c4a085c>:2: FutureWarning: Automatic reindexing on DataFrame vs Series comparisons is deprecated and will raise ValueError in a future version. Do `left, right = left.align(right, axis=1, copy=False)` before e.g. `left == right`
      x_out = continous[~((continous < (Q1 - 1.5 * IQR)) |(continous > (Q3 + 1.5 * IQR))).any(axis=1)]


**Decision Tree**


```
X_train, X_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2)
print(x_scaled.shape)
x_scaled.head()
```

    (10701, 14)






  <div id="df-76e20e05-d5c2-444e-b25a-84f63d679e4b" class="colab-df-container">
    <div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tweet_length</th>
      <th>followers_count</th>
      <th>friends_count</th>
      <th>favourites_count</th>
      <th>statuses_count</th>
      <th>hashtag_count</th>
      <th>http_count</th>
      <th>sign</th>
      <th>listed_count</th>
      <th>Days_count</th>
      <th>False_l</th>
      <th>True_l</th>
      <th>False_r</th>
      <th>True_r</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.616405</td>
      <td>-0.028783</td>
      <td>-0.144833</td>
      <td>-0.408628</td>
      <td>-0.341124</td>
      <td>-0.320451</td>
      <td>-0.784156</td>
      <td>-0.087817</td>
      <td>-0.176579</td>
      <td>1.184261</td>
      <td>0.126569</td>
      <td>-0.126569</td>
      <td>0.411611</td>
      <td>-0.411611</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.645776</td>
      <td>-0.028862</td>
      <td>-0.162098</td>
      <td>-0.441405</td>
      <td>-0.347353</td>
      <td>-0.320451</td>
      <td>-0.784156</td>
      <td>-0.087817</td>
      <td>-0.180502</td>
      <td>-0.061179</td>
      <td>0.126569</td>
      <td>-0.126569</td>
      <td>0.411611</td>
      <td>-0.411611</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.616405</td>
      <td>-0.028877</td>
      <td>-0.115666</td>
      <td>1.102263</td>
      <td>-0.261485</td>
      <td>-0.320451</td>
      <td>-0.784156</td>
      <td>-0.087817</td>
      <td>-0.184424</td>
      <td>-1.187879</td>
      <td>0.126569</td>
      <td>-0.126569</td>
      <td>0.411611</td>
      <td>-0.411611</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.616405</td>
      <td>-0.013393</td>
      <td>0.108515</td>
      <td>0.827834</td>
      <td>-0.091832</td>
      <td>-0.320451</td>
      <td>-0.784156</td>
      <td>-0.087817</td>
      <td>-0.023603</td>
      <td>1.572624</td>
      <td>0.126569</td>
      <td>-0.126569</td>
      <td>0.411611</td>
      <td>-0.411611</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.616405</td>
      <td>-0.029230</td>
      <td>-0.152942</td>
      <td>-0.461945</td>
      <td>-0.385680</td>
      <td>-0.320451</td>
      <td>-0.784156</td>
      <td>-0.087817</td>
      <td>-0.172657</td>
      <td>0.373609</td>
      <td>0.126569</td>
      <td>-0.126569</td>
      <td>0.411611</td>
      <td>-0.411611</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-76e20e05-d5c2-444e-b25a-84f63d679e4b')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">


  </div>


<div id="df-a0f204f8-b1b2-4e3d-abad-1d4d0c5f7f51">
  <button class="colab-df-quickchart" onclick="quickchart('df-a0f204f8-b1b2-4e3d-abad-1d4d0c5f7f51')"
            title="Suggest charts."
            style="display:none;">


</div>
    </div>
  </div>





```
dectree = tree.DecisionTreeClassifier(max_depth=5)
dectree.fit(X_train, y_train)
tree.plot_tree(dectree.fit(X_train, y_train))
plt.show()

print('accuracy:',dectree.score(X_test, y_test))
print('recall:',recall_score(y_test, dectree.predict(X_test)))
print('precision:',precision_score(y_test, dectree.predict(X_test)))
print('f1_score:',f1_score(y_test, dectree.predict(X_test)))
print('confusion_matrix:','\n',confusion_matrix(y_test, dectree.predict(X_test)))
```


    
![image](https://github.com/Zejabati/Viralityt_of_Tweet/assets/65095428/8f1343df-826b-49a0-ae7d-e1e588f42edf)

    


    accuracy: 0.8014946286781878
    recall: 0.8760484622553588
    precision: 0.762987012987013
    f1_score: 0.8156182212581344
    confusion_matrix: 
     [[776 292]
     [133 940]]



```
score_dectree = []
for i in np.arange(3, 15, 1):
  dectree = tree.DecisionTreeClassifier(max_depth=i)
  dectree.fit(X_train, y_train)
  score_dectree.append([i, np.mean(cross_val_score(dectree, X_train, y_train,scoring='accuracy',cv=5))])

score_dectree = pd.DataFrame(score_dectree)
score_dectree = score_dectree.sort_values(by=1, ascending=False).reset_index()
i=score_dectree[0][0]
print('best parameter:','max_depth:',i)

dectree = tree.DecisionTreeClassifier(max_depth=i)
dectree.fit(X_train, y_train)
tree.plot_tree(dectree.fit(X_train, y_train))
plt.show()
print('accuracy:',dectree.score(X_test, y_test))
print('recall:',recall_score(y_test, dectree.predict(X_test)))
print('precision:',precision_score(y_test, dectree.predict(X_test)))
print('f1_score:',f1_score(y_test, dectree.predict(X_test)))
print('confusion_matrix:','\n',confusion_matrix(y_test, dectree.predict(X_test)))
```

    best parameter: max_depth: 8



    
![png](Project_NHZ_files/Project_NHZ_36_1.png)
    


    accuracy: 0.7991592713685194
    recall: 0.8909599254426841
    precision: 0.7533490937746257
    f1_score: 0.816396242527754
    confusion_matrix: 
     [[755 313]
     [117 956]]


**Bagging**


```
bag = BaggingClassifier(n_estimators=100)
bag.fit(X_train, y_train)

print('accuracy:',bag.score(X_test, y_test))
print('recall:',recall_score(y_test, bag.predict(X_test)))
print('precision:',precision_score(y_test, bag.predict(X_test)))
print('f1_score:',f1_score(y_test, bag.predict(X_test)))
print('confusion_matrix:','\n',confusion_matrix(y_test, bag.predict(X_test)))
```

    accuracy: 0.8234469873890705
    recall: 0.896551724137931
    precision: 0.7827502034174125
    f1_score: 0.8357949609035621
    confusion_matrix: 
     [[801 267]
     [111 962]]



```
score_bag = []
for i in np.arange(5, 110, 5):
  bag = BaggingClassifier(n_estimators=i)
  bag.fit(X_train, y_train)
  score_bag.append([i, np.mean(cross_val_score(bag , X_train, y_train,scoring='accuracy',cv=5))])

score_bag = pd.DataFrame(score_bag)
score_bag = score_bag.sort_values(by=1, ascending=False).reset_index()
i=score_bag[0][0]
print('best parameter:','n_estimators:',i)
bag = BaggingClassifier(n_estimators=i)
bag.fit(X_train, y_train)

print('accuracy:',bag.score(X_test, y_test))
print('recall:',recall_score(y_test, bag.predict(X_test)))
print('precision:',precision_score(y_test, bag.predict(X_test)))
print('f1_score:',f1_score(y_test, bag.predict(X_test)))
print('confusion_matrix:','\n',confusion_matrix(y_test, bag.predict(X_test)))
```

    best parameter: n_estimators: 105
    accuracy: 0.8243811303129379
    recall: 0.9030754892823858
    precision: 0.7808219178082192
    f1_score: 0.8375108038029386
    confusion_matrix: 
     [[796 272]
     [104 969]]


**Random Forest**


```
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

print('accuracy:',rf.score(X_test, y_test))
print('recall:',recall_score(y_test, rf.predict(X_test)))
print('precision:',precision_score(y_test, rf.predict(X_test)))
print('f1_score:',f1_score(y_test, rf.predict(X_test)))
print('confusion_matrix:','\n',confusion_matrix(y_test, rf.predict(X_test)))
```

    accuracy: 0.8229799159271368
    recall: 0.9049394221808015
    precision: 0.7780448717948718
    f1_score: 0.8367083153813011
    confusion_matrix: 
     [[791 277]
     [102 971]]



```
score_rf = []
max_features = ['auto', 'sqrt']

for i in np.arange(100, 501, 100):
  for j in np.arange(5, 21 , 5):
    for k in max_features:
      rf = RandomForestClassifier()
      rf.fit(X_train, y_train)
      score_rf.append([i, j, k, np.mean(cross_val_score(rf, X_train, y_train, scoring='accuracy',cv=5))])

score_rf = pd.DataFrame(score_rf)
score_rf = score_rf.sort_values(by=3, ascending=False).reset_index()
i=score_rf[0][0]
j=score_rf[1][0]
k=score_rf[2][0]
print('best parameters:','n_estimators:',i ,'max_depth:',j ,'max_features:',k)

rf = RandomForestClassifier(n_estimators=i, max_depth=j, max_features=k)
rf.fit(X_train,y_train)

print('accuracy:',rf.score(X_test, y_test))
print('recall:',recall_score(y_test, rf.predict(X_test)))
print('precision:',precision_score(y_test, rf.predict(X_test)))
print('f1_score:',f1_score(y_test, rf.predict(X_test)))
print('confusion_matrix:','\n',confusion_matrix(y_test, rf.predict(X_test)))
```

    best parameters: n_estimators: 400 max_depth: 20 max_features: auto


    /usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_forest.py:424: FutureWarning: `max_features='auto'` has been deprecated in 1.1 and will be removed in 1.3. To keep the past behaviour, explicitly set `max_features='sqrt'` or remove this parameter as it is also the default value for RandomForestClassifiers and ExtraTreesClassifiers.
      warn(


    accuracy: 0.8257823446987389
    recall: 0.9226467847157502
    precision: 0.7734375
    f1_score: 0.8414789630259243
    confusion_matrix: 
     [[778 290]
     [ 83 990]]



```
df_test = pd.read_json('Test_Data.json', lines=True)
df_test.head()
```





  <div id="df-cbbc8c9f-0ce8-45ee-8e80-a5f3ea5057f4" class="colab-df-container">
    <div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>created_at</th>
      <th>id</th>
      <th>id_str</th>
      <th>text</th>
      <th>truncated</th>
      <th>entities</th>
      <th>metadata</th>
      <th>source</th>
      <th>in_reply_to_status_id</th>
      <th>in_reply_to_status_id_str</th>
      <th>...</th>
      <th>favorite_count</th>
      <th>favorited</th>
      <th>retweeted</th>
      <th>lang</th>
      <th>possibly_sensitive</th>
      <th>quoted_status_id</th>
      <th>quoted_status_id_str</th>
      <th>extended_entities</th>
      <th>quoted_status</th>
      <th>withheld_in_countries</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2018-07-31 13:34:18</td>
      <td>1024287138433691650</td>
      <td>1024287138433691648</td>
      <td>RT @Rschooley: Has anyone been held to the Al ...</td>
      <td>False</td>
      <td>{'hashtags': [], 'symbols': [], 'user_mentions...</td>
      <td>{'iso_language_code': 'en', 'result_type': 're...</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
      <td>en</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018-07-31 13:34:13</td>
      <td>1024287116241641473</td>
      <td>1024287116241641472</td>
      <td>RT @seuIsgis: when red flavor comes on shuffle...</td>
      <td>False</td>
      <td>{'hashtags': [], 'symbols': [], 'user_mentions...</td>
      <td>{'iso_language_code': 'en', 'result_type': 're...</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
      <td>en</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>{'media': [{'id': 1023603946344960002, 'id_str...</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018-07-31 13:34:24</td>
      <td>1024287163637092354</td>
      <td>1024287163637092352</td>
      <td>the world is his runway https://t.co/3sLst44JOT</td>
      <td>False</td>
      <td>{'hashtags': [], 'symbols': [], 'user_mentions...</td>
      <td>{'iso_language_code': 'en', 'result_type': 're...</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
      <td>en</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>{'media': [{'id': 1024282644933095426, 'id_str...</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018-07-31 13:34:26</td>
      <td>1024287169861509120</td>
      <td>1024287169861509120</td>
      <td>RT @lexie_marie5: 12:46 AM and I cannot stop t...</td>
      <td>False</td>
      <td>{'hashtags': [], 'symbols': [], 'user_mentions...</td>
      <td>{'iso_language_code': 'en', 'result_type': 're...</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
      <td>en</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018-07-31 13:34:35</td>
      <td>1024287208772198400</td>
      <td>1024287208772198400</td>
      <td>@RationalPanic No surprise at what the council...</td>
      <td>True</td>
      <td>{'hashtags': [], 'symbols': [], 'user_mentions...</td>
      <td>{'iso_language_code': 'en', 'result_type': 're...</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>1.023992e+18</td>
      <td>1.023992e+18</td>
      <td>...</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
      <td>en</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 30 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-cbbc8c9f-0ce8-45ee-8e80-a5f3ea5057f4')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

 
  </div>


<div id="df-2fa7a268-eda4-45d7-bde2-df0cee76b533">
  <button class="colab-df-quickchart" onclick="quickchart('df-2fa7a268-eda4-45d7-bde2-df0cee76b533')"
            title="Suggest charts."
            style="display:none;">


</div>
    </div>
  </div>





```
df_test.shape
```




    (1000, 30)




```
df_test['tweet_length'] = df_test.apply(lambda tweet: len(tweet['text']), axis=1)
df_test['followers_count'] = df_test.apply(lambda tweet: tweet['user']['followers_count'], axis=1)
df_test['friends_count'] = df_test.apply(lambda tweet: tweet['user']['friends_count'], axis=1)
df_test['favourites_count'] = df_test.apply(lambda tweet: tweet['user']['favourites_count'], axis=1)
df_test['statuses_count'] = df_test.apply(lambda tweet: tweet['user']['statuses_count'], axis=1)
df_test['listed_count']= df_test.apply(lambda tweet: tweet['user']['listed_count'], axis=1)
df_test['hashtag_count'] = df_test.apply(lambda tweet: tweet['text'].count('#'), axis=1)
df_test['http_count']= df_test.apply(lambda tweet: tweet['text'].count('http'), axis=1)
df_test['sign']= df_test.apply(lambda tweet: tweet['text'].count('@'),axis=1)
df_test['verified']= df_test.apply(lambda tweet: tweet['user']['verified'], axis=1)
```


```
df_test['Days_count']=0

df_test['date']=df_test.user.apply(lambda x:x['created_at'].split( ))
for i in range(len(df_test['date'])):
  df_test['date'][i][1]=month_converter(df_test['date'][i][1])

w=len(df_test['date'])
Matrix = [[0 for x in range(3)] for y in range(w)]
d= date(2020, 7, 7)
for i in range(len(df_test['date'])):
  Matrix[i][0] = int(df_test['date'][i][5])
  Matrix[i][1] = int(df_test['date'][i][1])
  Matrix[i][2] = int(df_test['date'][i][2])
  df_test['Days_count'][i]=(d-date(Matrix[i][0],Matrix[i][1],Matrix[i][2])).days

df_test['Days_count'].head()
```

    <ipython-input-122-475d892e7c8c>:14: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df_test['Days_count'][i]=(d-date(Matrix[i][0],Matrix[i][1],Matrix[i][2])).days





    0    1989
    1    2817
    2    1163
    3     725
    4     874
    Name: Days_count, dtype: int64




```
x_test = df_test[['tweet_length' , 'followers_count', 'friends_count','favourites_count','statuses_count',
                  'hashtag_count', 'http_count', 'sign', 'verified','listed_count','Days_count','truncated']]
non_numerical_columns = ['verified','truncated']

for i in non_numerical_columns:
    x2 = pd.get_dummies(x_test[i])
    x_test = x_test.join(x2,lsuffix='_l',rsuffix='_r')
    x_test.drop(i,axis=1,inplace=True)
```


```
x_test_scaled = StandardScaler().fit_transform(x_test)
x_test_scaled=pd.DataFrame(x_test_scaled)
x_test_scaled.columns =['tweet_length',  'followers_count','friends_count','favourites_count','statuses_count','hashtag_count',
             'http_count', 'sign','listed_count','Days_count','False_l', 'True_l', 'False_r', 'True_r']
x_test_scaled.head()
```





  <div id="df-67f8cce3-5815-4e5e-a481-63b37a1cb2c6" class="colab-df-container">
    <div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tweet_length</th>
      <th>followers_count</th>
      <th>friends_count</th>
      <th>favourites_count</th>
      <th>statuses_count</th>
      <th>hashtag_count</th>
      <th>http_count</th>
      <th>sign</th>
      <th>listed_count</th>
      <th>Days_count</th>
      <th>False_l</th>
      <th>True_l</th>
      <th>False_r</th>
      <th>True_r</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.559086</td>
      <td>-0.064637</td>
      <td>-0.129676</td>
      <td>0.260932</td>
      <td>-0.389227</td>
      <td>-0.320255</td>
      <td>-0.791265</td>
      <td>-0.082309</td>
      <td>-0.203424</td>
      <td>-0.408901</td>
      <td>0.153432</td>
      <td>-0.153432</td>
      <td>0.415130</td>
      <td>-0.415130</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.324393</td>
      <td>0.029803</td>
      <td>-0.213809</td>
      <td>-0.203293</td>
      <td>0.914798</td>
      <td>-0.320255</td>
      <td>1.124630</td>
      <td>-0.082309</td>
      <td>0.527802</td>
      <td>0.325799</td>
      <td>0.153432</td>
      <td>-0.153432</td>
      <td>0.415130</td>
      <td>-0.415130</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-2.754371</td>
      <td>-0.062918</td>
      <td>-0.199485</td>
      <td>-0.317103</td>
      <td>-0.424399</td>
      <td>-0.320255</td>
      <td>1.124630</td>
      <td>-1.098473</td>
      <td>-0.191817</td>
      <td>-1.141827</td>
      <td>0.153432</td>
      <td>-0.153432</td>
      <td>0.415130</td>
      <td>-0.415130</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.614159</td>
      <td>-0.065075</td>
      <td>-0.212603</td>
      <td>-0.495150</td>
      <td>-0.474844</td>
      <td>-0.320255</td>
      <td>-0.791265</td>
      <td>-0.082309</td>
      <td>-0.203424</td>
      <td>-1.530472</td>
      <td>0.153432</td>
      <td>-0.153432</td>
      <td>0.415130</td>
      <td>-0.415130</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.577938</td>
      <td>-0.065266</td>
      <td>-0.230696</td>
      <td>-0.451774</td>
      <td>-0.456709</td>
      <td>-0.320255</td>
      <td>1.124630</td>
      <td>-0.082309</td>
      <td>-0.203424</td>
      <td>-1.398262</td>
      <td>0.153432</td>
      <td>-0.153432</td>
      <td>-2.408884</td>
      <td>2.408884</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-67f8cce3-5815-4e5e-a481-63b37a1cb2c6')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">



  </div>


<div id="df-2aec67a2-a987-436b-b0e2-1ad0ce8bdf0b">
  <button class="colab-df-quickchart" onclick="quickchart('df-2aec67a2-a987-436b-b0e2-1ad0ce8bdf0b')"
            title="Suggest charts."
            style="display:none;">


</div>
    </div>
  </div>





```
df_test['Viral_prediction']=rf.predict(x_test_scaled)
df_test['Viral_prediction']
```




    0      0
    1      1
    2      0
    3      0
    4      0
          ..
    995    1
    996    1
    997    0
    998    1
    999    0
    Name: Viral_prediction, Length: 1000, dtype: int64




```
df_test.to_csv('test_viral_prediction.csv')
df_test.to_json('test_viral_prediction.json')
```
