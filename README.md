
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/philippbayer/Goodreads_visualization/master?filepath=README.ipynb)

# Goodreads visualization

A Jupyter notebook to play around with Goodreads data and make some seaborn visualizations, learn more about scikit-learn, my own playground!

You can use it with your own data - go [here](https://www.goodreads.com/review/import) and press "Export your library" to get your own csv.

The text you're reading is generated from a jupyter notebook by the Makefile. If you want to run it yourself, clone the repository then run

    jupyter notebook your_file.ipynb
    
to get the interactive version. In there, replace the path to my Goodreads exported file by yours in the ipynb file, and then run click on Cell -> Run All.

** WARNING **
It seems that there's currently a bug on Goodreads' end with the export of data, as many recently 'read' books have a read-date which is shown on the web page but doesn't show up in the CSV.

## Dependencies

* Python (3! rpy2 doesn't work under Python2 any more)
* Jupyter
* R (for rpy2)

### Python packages

* seaborn
* pandas
* wordcloud
* nltk
* networkx
* pymarkovchain
* scikit-learn
* distance
* image (PIL inside python for some weird reason)
* gender_guesser
* rpy2

To install all:

    pip install seaborn wordcloud nltk networkx pymarkovchain image sklearn distance gender_guesser rpy2
    
Under Windows and anaconda you instead need to run 

    conda install rpy2
   
instead of using pip to install rpy2.

## Licenses

License for reviews: CC-BY-SA 4.0
Code: MIT

OK, let's start!

## Setting up the notebook


```python
%pylab inline

# for most plots
import numpy as np
import pandas as pd
import seaborn as sns
from collections import defaultdict, Counter, OrderedDict

# for stats
import scipy.stats

# for time-related plots
import datetime
import calendar

# for word cloud
import re
import string
from nltk.corpus import stopwords
from wordcloud import WordCloud

# for Markov chain
from pymarkovchain import MarkovChain
import pickle
import networkx as nx

# for shelf clustering
import distance
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

sns.set_palette("coolwarm")

# for plotting images
from IPython.display import Image

import gender_guesser.detector as gender

# for R
import pandas
from rpy2 import robjects 
# conda install -c r rpy2 on Windows

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 5]
```

    Populating the interactive namespace from numpy and matplotlib


## Loading the data


```python
df = pd.read_csv('./goodreads_library_export.csv')
# keep only books that have a rating (unrated books have a rating of 0, we don't need that)
cleaned_df = df[df["My Rating"] != 0]

# get rid of noise in 2012
cleaned_df = cleaned_df[(cleaned_df['Date Added'] > '2013-01-01')]
```

# Score distribution
With a score scale of 1-5, you'd expect that the average score is ~~2.5~~ 3 (since 0 is not counted) after a few hundred books (in other words, is it a normal distribution?)


```python
g = sns.distplot(cleaned_df["My Rating"], kde=False)
"Average: %.2f"%cleaned_df["My Rating"].mean(), "Median: %s"%cleaned_df["My Rating"].median()
```




    ('Average: 3.54', 'Median: 4.0')




![png](README_files/README_5_1.png)


That doesn't look normally distributed to me - let's ask Shapiro-Wilk (null hypothesis: data is drawn from normal distribution):


```python
W, p_value = scipy.stats.shapiro(cleaned_df["My Rating"])
if p_value < 0.05:
    print("Rejecting null hypothesis - data does not come from a normal distribution (p=%s)"%p_value)
else:
    print("Cannot reject null hypothesis (p=%s)"%p_value)
```

    Rejecting null hypothesis - data does not come from a normal distribution (p=8.048559751530179e-22)


In my case, the data is not normally distributed (in other words, the book scores are not evenly distributed around the middle). If you think about it, this makes sense: most readers don't read perfectly randomly, I avoid books I believe I'd dislike, and choose books that I prefer. I rate those books higher than average, therefore, my curve of scores is slanted towards the right.

## plot Pages vs Ratings

Do I give longer books better scores? A minor tendency but nothing special (it's confounded by having just 5 possible numbers in ratings)


```python
g = sns.jointplot("Number of Pages", "My Rating", data=cleaned_df, kind="reg", height=7, ylim=[0.5,5.5])
g.annotate(scipy.stats.pearsonr)
```

    C:\Users\00089503\AppData\Local\Continuum\anaconda3\lib\site-packages\seaborn\axisgrid.py:1847: UserWarning: JointGrid annotation is deprecated and will be removed in a future release.
      warnings.warn(UserWarning(msg))





    <seaborn.axisgrid.JointGrid at 0x1f90b514080>




![png](README_files/README_10_2.png)


I seem to mostly read books at around 200 to 300 pages so it's hard to tell whether I give longer books better ratings. 
It's a nice example that in regards to linear regression, a p-value as tiny as this one doesn't mean much, the r-value is still bad.

***

## plot Ratings vs Bookshelves

Let's parse ratings for books and make a violin plot for the 7 categories with the most rated books!


```python
CATEGORIES = 7 # number of most crowded categories to plot

# we have to fiddle a bit - we have to count the ratings by category, 
# since each book can have several comma-delimited categories
# TODO: find a pandas-like way to do this

shelves_ratings = defaultdict(list) # key: shelf-name, value: list of ratings
shelves_counter = Counter() # counts how many books on each shelf
shelves_to_names = defaultdict(list) # key: shelf-name, value: list of book names
for index, row in cleaned_df.iterrows():
    my_rating = row["My Rating"]
    if my_rating == 0:
        continue
    if pd.isnull(row["Bookshelves"]):
        continue

    shelves = row["Bookshelves"].split(",")

    for s in shelves:
        # empty shelf?
        if not s: continue
        s = s.strip() # I had "non-fiction" and " non-fiction"
        shelves_ratings[s].append(my_rating)
        shelves_counter[s] += 10
        shelves_to_names[s].append(row.Title)

names = []
ratings = []
for name, _ in shelves_counter.most_common(CATEGORIES):
    for number in shelves_ratings[name]:
        names.append(name)
        ratings.append(number)

full_table = pd.DataFrame({"Category":names, "Rating":ratings})

# if we don't use scale=count here then each violin has the same area
sns.violinplot(x = "Category", y = "Rating", data=full_table, scale='count')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1f90b6e8278>




![png](README_files/README_12_1.png)


There is some *bad* SF out there.

At this point I wonder - since we can assign multiple 'shelves' (tags) to each book, do I have some tags that appear more often together than not? Let's use R!



```python
%load_ext rpy2.ipython
```


```python
all_shelves = shelves_counter.keys()

names_dict = {} # key: shelf name, value: robjects.StrVector of names
for c in all_shelves:
    names_dict[c] = robjects.StrVector(shelves_to_names[c])

names_dict = robjects.ListVector(names_dict)    
```


```r
%%R -i names_dict -r 150 -w 900 -h 600
library(UpSetR)
names_dict <- fromList(names_dict)
# by default, only 5 sets are considered, so change nsets
upset(names_dict, nsets = 9)
```


![png](README_files/README_16_0.png)


Most shelves are 'alone', but 'essays + non-fiction', 'sci-fi + sf' (should clean that up...), 'biography + non-fiction' show the biggest overlap.

I may have messed up the categories, let's cluster them! Typos should cluster together


```python
# get the Levenshtein distance between all shelf titles, normalise the distance by string length
X = np.array([[float(distance.levenshtein(shelf_1,shelf_2))/max(len(shelf_1), len(shelf_2)) \
               for shelf_1 in all_shelves] for shelf_2 in all_shelves])
# scale for clustering
X = StandardScaler().fit_transform(X)

# after careful fiddling I'm settling on eps=10
clusters = DBSCAN(eps=10, min_samples=1).fit_predict(X)
print('DBSCAN made %s clusters for %s shelves/tags.'%(len(set(clusters)), len(all_shelves)))

cluster_dict = defaultdict(list)
assert len(clusters) == len(all_shelves)
for cluster_label, element in zip(clusters, all_shelves):
    cluster_dict[cluster_label].append(element)
    
print('Clusters with more than one member:')
for k in sorted(cluster_dict):
    if len(cluster_dict[k]) > 1:
        print(k, cluster_dict[k])
```

    DBSCAN made 166 clusters for 184 shelves/tags.
    Clusters with more than one member:
    1 ['fiction', 'action']
    2 ['russia', 'russian']
    12 ['latin-america', 'native-american']
    24 ['ww1', 'ww2']
    32 ['humble-bundle2', 'humble-bundle-jpsf']
    47 ['essays', 'essay']
    49 ['on-living', 'on-writing', 'on-thinking']
    50 ['history-of-biology', 'history-of-maths', 'history-of-cs', 'history-of-philosophy']
    53 ['greek', 'greece']
    66 ['iceland', 'ireland']
    88 ['mythology', 'psychology', 'sociology', 'theology']
    116 ['philosophy', 'pop-philosophy']
    126 ['letters', 'lectures']


Some clusters are problematic due to too-short label names (arab/iraq), some other clusters are good and show me that I made some mistakes in labeling! French and France should be together, Greece and Greek too. *Neat!*

(Without normalising the distance by string length clusters like horror/body-horror don't appear.)

## plotHistogramDistanceRead.py

Let's check the "dates read" for each book read and plot the distance between books read in days - shows you how quickly you hop from book to book.

I didn't use Goodreads in 2012 much so let's see how it looks like without 2012:


```python
# first, transform to datetype and get rid of all invalid dates
#dates = pd.to_datetime(cleaned_df["Date Read"])
dates = pd.to_datetime(cleaned_df["Date Added"])

dates = dates.dropna()
sorted_dates = sorted(dates)

last_date = None
differences = []
all_days = []
all_days_without_2012 = [] # not much goodreads usage in 2012 - remove that year
for date in sorted_dates:
    if not last_date:
        last_date = date
        if date.year != 2012:
            last_date_not_2012 = date
    difference = date - last_date
    
    days = difference.days
    all_days.append(days)
    if date.year != 2012:
        all_days_without_2012.append(days)
    last_date = date

sns.distplot(all_days_without_2012, axlabel="Distance in days between books read")
pylab.show()
```


![png](README_files/README_21_0.png)


***

## plot Heatmap of dates read

Parses the "dates read" for each book read, bins them by month, and makes a heatmap to show in which months I read more than in others. Also makes a lineplot for books read, split up by year.

NOTE: There is a very strange bug in Goodreads for about a year now. The exported CSV does not correctly track the date read.


```python
# we need a dataframe in this format:
# year months books_read
# I am sure there's some magic pandas function for this

read_dict = defaultdict(int) # key: (year, month), value: count of books read
for date in sorted_dates:
    this_year = date.year
    this_month = date.month
    read_dict[ (this_year, this_month) ] += 1

first_date = sorted_dates[0]

first_year = first_date.year
first_month = first_date.month

todays_date = datetime.datetime.today()
todays_year = todays_date.year
todays_month = todays_date.month

all_years = []
all_months = []
all_counts = []
for year in range(first_year, todays_year+1):
    for month in range(1, 13):
        if (year == todays_year) and month > todays_month:
            # don't count future months
            break
        this_count = read_dict[ (year, month) ]
        all_years.append(year)
        all_months.append(month)
        all_counts.append(this_count)

# now get it in the format heatmap() wants
df = pd.DataFrame( { "month":all_months, "year":all_years, "books_read":all_counts } )
dfp = df.pivot("month", "year", "books_read")

fig, ax = plt.subplots(figsize=(10,10))
# now make the heatmap
ax = sns.heatmap(dfp, annot=True, ax=ax, square= True)
```


![png](README_files/README_23_0.png)


What happened in May 2014?

Update in 2018 - currently the 'date_read' column doesn't accurately track which books were actually read, this is a bug on Goodreads' end, see for example https://help.goodreads.com/s/question/0D51H00004ADr7o/i-have-exported-my-library-and-some-books-do-not-have-any-information-listed-for-date-read

***

## Plot books read by year


```python
g = sns.FacetGrid(df, col="year", sharey=True, sharex=True, col_wrap=4)
g.map(plt.scatter, "month", "books_read")
g.set_ylabels("Books read")
g.set_xlabels("Month")
pylab.xlim(1, 12)
pylab.show()
```


![png](README_files/README_25_0.png)


It's nice how reading behaviour (Goodreads usage) connects over the months - it slowly in 2013, stays constant in 2014/2015, and now goes down again. You can see when my first son was born!

(Solution: 2016-8-25)

(all other >2018 books are still missing their date_read dates...)


***

## Guessing authors' genders

Let's check whether I read mostly male or female authors using the gender-guesser package!


```python
first_names = cleaned_df['Author'].str.split(' ',expand=True)[0]
d = gender.Detector(case_sensitive=False)

genders = [d.get_gender(name) for name in first_names]
print(list(zip(genders[:5], first_names[:5])))
# let's also add those few 'mostly_female' and 'mostly_male' into the main grou
genders = pd.Series([x.replace('mostly_female','female').replace('mostly_male','male') for x in genders])
```

    [('male', 'Don'), ('male', 'Daniil'), ('male', 'William'), ('unknown', 'E.T.A.'), ('male', 'John')]



```python
gender_ratios = genders.value_counts()
print(gender_ratios)
_ = gender_ratios.plot(kind='bar')
```

    male       423
    unknown     67
    female      56
    andy         3
    dtype: int64



![png](README_files/README_28_1.png)


Now THAT'S gender bias. Do I rate the genders differently?


```python
cleaned_df['Gender'] = genders

male_scores = cleaned_df[cleaned_df['Gender'] == 'male']['My Rating'].values
female_scores = cleaned_df[cleaned_df['Gender'] == 'female']['My Rating'].values

_ = plt.hist([male_scores, female_scores], color=['r','b'], alpha=0.5)
```


![png](README_files/README_30_0.png)


Hard to tell any difference since there are so fewer women authors here - let's split them up into different plots


```python
fig, axes = plt.subplots(2,1)

axes[0].hist(male_scores, color='r', alpha=0.5, bins=10)
axes[0].set_xlabel('Scores')
# Make the y-axis label, ticks and tick labels match the line color.
axes[0].set_ylabel('male scores')

axes[1].hist(female_scores, color='b', alpha=0.5, bins=10)
axes[1].set_ylabel('female scores')

fig.tight_layout()
```


![png](README_files/README_32_0.png)


Are these two samples from the same distribution? Hard to tell since their size is so different, but let's ask Kolmogorov-Smirnov (null hypothesis: they are from the same distribution)


```python
scipy.stats.ks_2samp(male_scores, female_scores)
```




    Ks_2sampResult(statistic=0.22018779342723005, pvalue=0.13257156821934568)



We cannot reject the null hypthesis as the p-value is very, very high. (but again, there are so few female scores...)


***

## Compare with Goodreads 10k


A helpful soul has uploaded ratings and stats for the 10,000 books with most ratings on Goodreads (https://github.com/zygmuntz/goodbooks-10k). Let's compare those with my ratings!

(You may have to run 

    git submodule update
    
to get the 10k submodule)


```python
other = pd.read_csv('./goodbooks-10k/books.csv')
print(other.columns)
other.head(3)
```

    Index(['book_id', 'goodreads_book_id', 'best_book_id', 'work_id',
           'books_count', 'isbn', 'isbn13', 'authors', 'original_publication_year',
           'original_title', 'title', 'language_code', 'average_rating',
           'ratings_count', 'work_ratings_count', 'work_text_reviews_count',
           'ratings_1', 'ratings_2', 'ratings_3', 'ratings_4', 'ratings_5',
           'image_url', 'small_image_url'],
          dtype='object')





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>book_id</th>
      <th>goodreads_book_id</th>
      <th>best_book_id</th>
      <th>work_id</th>
      <th>books_count</th>
      <th>isbn</th>
      <th>isbn13</th>
      <th>authors</th>
      <th>original_publication_year</th>
      <th>original_title</th>
      <th>...</th>
      <th>ratings_count</th>
      <th>work_ratings_count</th>
      <th>work_text_reviews_count</th>
      <th>ratings_1</th>
      <th>ratings_2</th>
      <th>ratings_3</th>
      <th>ratings_4</th>
      <th>ratings_5</th>
      <th>image_url</th>
      <th>small_image_url</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2767052</td>
      <td>2767052</td>
      <td>2792775</td>
      <td>272</td>
      <td>439023483</td>
      <td>9.780439e+12</td>
      <td>Suzanne Collins</td>
      <td>2008.0</td>
      <td>The Hunger Games</td>
      <td>...</td>
      <td>4780653</td>
      <td>4942365</td>
      <td>155254</td>
      <td>66715</td>
      <td>127936</td>
      <td>560092</td>
      <td>1481305</td>
      <td>2706317</td>
      <td>https://images.gr-assets.com/books/1447303603m...</td>
      <td>https://images.gr-assets.com/books/1447303603s...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>4640799</td>
      <td>491</td>
      <td>439554934</td>
      <td>9.780440e+12</td>
      <td>J.K. Rowling, Mary GrandPré</td>
      <td>1997.0</td>
      <td>Harry Potter and the Philosopher's Stone</td>
      <td>...</td>
      <td>4602479</td>
      <td>4800065</td>
      <td>75867</td>
      <td>75504</td>
      <td>101676</td>
      <td>455024</td>
      <td>1156318</td>
      <td>3011543</td>
      <td>https://images.gr-assets.com/books/1474154022m...</td>
      <td>https://images.gr-assets.com/books/1474154022s...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>41865</td>
      <td>41865</td>
      <td>3212258</td>
      <td>226</td>
      <td>316015849</td>
      <td>9.780316e+12</td>
      <td>Stephenie Meyer</td>
      <td>2005.0</td>
      <td>Twilight</td>
      <td>...</td>
      <td>3866839</td>
      <td>3916824</td>
      <td>95009</td>
      <td>456191</td>
      <td>436802</td>
      <td>793319</td>
      <td>875073</td>
      <td>1355439</td>
      <td>https://images.gr-assets.com/books/1361039443m...</td>
      <td>https://images.gr-assets.com/books/1361039443s...</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 23 columns</p>
</div>



What's the gender ratio here?


```python

other_first_names = other.authors.str.split(' ',expand=True)[0]
for index, x in enumerate(other_first_names):
    if x == 'J.R.R.':
        other_first_names[index] = 'John'
    elif x == 'J.K.':
        other_first_names[index] = 'Joanne'
    elif x == 'F.':
        other_first_names[index] = 'Francis'
    elif x == 'C.S.':
        other_first_names[index] = 'Clive'
    elif x == 'J.D.':
        other_first_names[index] = 'Jerome'
        
other_genders = pd.Series([d.get_gender(name) for name in other_first_names])
gender_ratios = other_genders.value_counts()
print(gender_ratios)
_ = gender_ratios.plot(kind='bar')

other['Gender'] = other_genders
```

    male             4669
    female           3424
    unknown          1180
    mostly_male       347
    mostly_female     332
    andy               48
    dtype: int64



![png](README_files/README_39_1.png)


A bit better than my own reviews! I should multiply the authors with their numbers_read, then J. K. Rowling will probably blow everybody else out of the water


```python
male_scores = other[other['Gender'] == 'male']['average_rating'].values
female_scores = other[other['Gender'] == 'female']['average_rating'].values

fig, axes = plt.subplots(2,1)

axes[0].hist(male_scores, color='r', alpha=0.5, bins=25)
axes[0].set_xlabel('Scores')
# Make the y-axis label, ticks and tick labels match the line color.
axes[0].set_ylabel('male scores')

axes[1].hist(female_scores, color='b', alpha=0.5, bins=25)
axes[1].set_ylabel('female scores')

fig.tight_layout()
```


![png](README_files/README_41_0.png)


Very similar, again, with a slight shift to the right in the 'female scores'

Is my 'Book Id' the same as the other's table 'goodreads_book_id'?


```python
both = other.merge(cleaned_df, how='inner', left_on='goodreads_book_id', right_on='Book Id')
print('My reviews: %s, 10k Reviews: %s, Intersection: %s'%(cleaned_df.shape, other.shape, both.shape))
```

    My reviews: (549, 32), 10k Reviews: (10000, 24), Intersection: (131, 56)


Looks good! Now check which is the most common and the most obscure book in my list


```python
Image(both.sort_values(by='ratings_count').head(1).image_url.iloc[0])
```




![jpeg](README_files/README_46_0.jpeg)



Too Loud A Solitude, a wonderful book, you should read it!


```python
Image(both.sort_values(by='ratings_count').tail(1).image_url.iloc[0])
```




![jpeg](README_files/README_48_0.jpeg)



For which book does my rating have the highest difference in score?


```python
my_rating = cleaned_df['My Rating']
other_ratings = cleaned_df['Average Rating']
cleaned_df['Difference Rating'] = np.abs(my_rating - other_ratings)
ten_biggest_diff = cleaned_df.sort_values(by='Difference Rating').tail(10)

for x in ten_biggest_diff.iterrows():
    book_id = x[1]['Book Id']
    ten_thousand_books_info = other.where(other['goodreads_book_id'] == book_id).dropna()
    try:
        this_image_url = ten_thousand_books_info.image_url.iloc[0]
    except IndexError:
        # not found in big table
        continue
    display(Image(this_image_url))
    details = x[1]
    print('Book: %s, My rating: %s Global average rating: %s'%(details['Title'], details['My Rating'], details['Average Rating'] ))
```


![jpeg](README_files/README_50_0.jpeg)


    Book: The Perks of Being a Wallflower, My rating: 2 Global average rating: 4.2



![jpeg](README_files/README_50_2.jpeg)


    Book: The Martian, My rating: 2 Global average rating: 4.4



![png](README_files/README_50_4.png)


    Book: The Dice Man, My rating: 1 Global average rating: 3.57



![jpeg](README_files/README_50_6.jpeg)


    Book: Stranger in a Strange Land, My rating: 1 Global average rating: 3.92



![jpeg](README_files/README_50_8.jpeg)


    Book: To Your Scattered Bodies Go (Riverworld, #1), My rating: 1 Global average rating: 3.95


Do I have many differences in how I rate my book when compared with the community?


```python
sns.distplot(cleaned_df['Difference Rating'], kde=False)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1f912dc06d8>




![png](README_files/README_52_1.png)


Not really, mostly 0 and 1 difference.

## plot Word Cloud


This one removes noisy words and creates a word-cloud of most commonly used words in the reviews.


```python
def replace_by_space(word):
    new = []
    for letter in word:
        if letter in REMOVE:
            new.append(' ')
        else:
            new.append(letter)
    return ''.join(new)

STOP = stopwords.words("english")
html_clean = re.compile('<.*?>')
gr_clean = re.compile('\[.*?\]')
PRINTABLE = string.printable
REMOVE = set(["!","(",")",":",".",";",",",'"',"?","-",">","_"])

all_my_words = []
all_my_words_with_stop_words = []

reviews = cleaned_df["My Review"]

num_reviews = 0
num_words = 0
for row in reviews:
    if pd.isnull(row):
        continue
    review = row.lower()
    if not review:
        # empty review
        continue
    # clean strings
    cleaned_review = re.sub(html_clean, '', review)
    cleaned_review = re.sub(gr_clean, '', cleaned_review)
    all_my_words_with_stop_words += cleaned_review
    cleaned_review = replace_by_space(cleaned_review)
    cleaned_review = "".join(filter(lambda x: x in PRINTABLE, cleaned_review))
    # clean words
    cleaned_review = cleaned_review.split()
    cleaned_review = list(filter(lambda x: x not in STOP, cleaned_review))
    num_words += len(cleaned_review)
    all_my_words += cleaned_review
    num_reviews += 1

print("You have %s words in %s reviews"%(num_words, num_reviews))

# we need all words later for the Markov chain
all_my_words_with_stop_words = ''.join(all_my_words_with_stop_words)

# WordCloud takes only string, no list/set
wordcloud = WordCloud(max_font_size=200, width=800, height=500).generate(' '.join(all_my_words))
pylab.imshow(wordcloud)
pylab.axis("off")
pylab.show()
```

    You have 83304 words in 491 reviews



![png](README_files/README_55_1.png)


***

## plot books read vs. week-day

Let's parse the weekday a "book read" has been added and count them


```python
# initialize the dict in the correct order
read_dict = OrderedDict() # key: weekday, value: count of books read
for day in range(0,7):
    read_dict[calendar.day_name[day]] = 0

for date in sorted_dates:
    weekday_name = calendar.day_name[date.weekday()]  # Sunday
    read_dict[weekday_name] += 1

full_table = pd.DataFrame({"Weekday":list(read_dict.keys()), "Books read":list(read_dict.values())})

sns.barplot(x="Weekday", y="Books read", data=full_table)
plt.tight_layout()
plt.show()

```


![png](README_files/README_57_0.png)


Monday is procrastination day.

***

## Generate Reviews

Tiny script that uses a simple Markov Chain and the review text as created by plotWordCloud.py to generate new reviews.
Some examples:

* “natural” death, almost by definition, means something slow, smelly and painful
* a kind of cyborg, saved by the master was plagued in his work - for that i'm getting angry again just typing this - some are of exactly the opposite, and of black holes
* american actress wikipedia tells me) once said: "a critic never fights the battle; they just read, focus on his own goshawk 50 years
* he always wanted to do something, and i don't know how accurate he is
* not recommended for: people who, if they can't be reduced to a small essay
* machiavelli summarises quite a bit like reading a 120 pages summary of the helmet of horror
* - no supervisor, no grant attached to a beautiful suicide and now i cleared my mind of circe's orders -cramping my style, urging me not to write the paper
* not being focused on useless mobile apps, but on medical companies that treat death as a sign of dissent
* the harassment of irs-personnel to get into the dark cave
* they're doing "good"
* i think it's supposed to be the worst essay is a vampire: "interview with a strong voice and judges the poem by the use of might (hitler is referenced several times) - the 4 alternating voices quickly blur into one network of states
* one or two minor problems: you need to murder his children

*why does this work so well*

This script also creates a graph of probabilities for word connections for the word "translation", the thicker the edge between the nodes, the higher the probability.


```python
mc = MarkovChain(dbFilePath='./markov_db')
mc.generateDatabase(all_my_words_with_stop_words)

print(mc.generateString())

mc.dumpdb()

# a key in the datbase looks like:
# ('when', 'you') defaultdict(<function _one at 0x7f5c843a4500>, 
# {'just': 0.06250000000059731, 'feel': 0.06250000000059731, 'had': 0.06250000000059731, 'accidentally': 0.06250000000059731, ''love': 0.06250000000059731, 'read': 0.06250000000059731, 'see': 0.06250000000059731, 'base': 0.06250000000059731, 'know': 0.12499999999641617, 'have': 0.12499999999641617, 'were': 0.06250000000059731, 'come': 0.06250000000059731, 'can't': 0.06250000000059731, 'are': 0.06250000000059731})
# so 'just' follows after 'when you' with 6% probability

db = pickle.load(open('./markov_db', 'rb'))
# let's get a good node
#for key in db:
#    # has in between 5 and 10 connections
#    if len(db[key]) > 5 and (len(db[key]) < 10):
#        if len(set(db[key].values())) > 2:
#            print key, set(db[key].values())

# manually chosen from above
good_key = ('translation',)
values = db[good_key]

# create the graph

G = nx.DiGraph()
good_key = str(good_key[0])
G.add_node(good_key)
G.add_nodes_from(values.keys())
# get the graph for one of the connected nodes
# we go only one step deep - anything more and we'd better use recursion (but graph gets ugly then anyway)
for v in values:
    if (v,) in db and (len(db[(v,)]) < 20):
        G.add_nodes_from(db[(v,)].keys())
        for partner in db[(v,)]:
            edge_weight = db[(v,)][partner]
            G.add_weighted_edges_from([ (v, partner, edge_weight) ])
        # for now, only add one
        break

# now add the edges of the "original" graph around "translation"
for partner in values:
    edge_weight = values[partner]
    G.add_weighted_edges_from([ (good_key, partner, edge_weight) ])

pos = nx.shell_layout(G)

nx.draw_networkx_nodes(G, pos, node_color = 'white', node_size = 2500)

# width of edges is based on probability * 10
for edge in G.edges(data=True):
    nx.draw_networkx_edges(G, pos, edgelist = [(edge[0], edge[1])], width = edge[2]['weight']*10)

nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
pylab.axis('off')
pylab.show()
```

    we read poetry



![png](README_files/README_59_1.png)


***

## Some other ideas

- Some people on goodreads have complained that their reviews disappear and I feel (but don't know) that I lost at least one, this tracks my exported CSV to check whether it actually happens. So far I haven't observed it.
- ~~Write automated parser that exports reviews to html/epub/tumblr/blogger/wordpress etc.~~ support for this was added to goodreads)
- ~~cron job which automatically pulls exported CSV from https://www.goodreads.com/review_porter/goodreads_export.csv (login a bit weird esp. with Facebook login, use API instead? Needs dev key, but easier to do /review/list.xml=USERID than to play Red Queen with Facebook's oauth)~~ see github.com/philippbayer/Goodreads_to_Tumblr
- various visualization things in regards to language use
- RNN to write automated reviews, similar to the Markov one. also have to look at embeddings to predict category of book?


```python

```
