# before first run, need to install stopwords:
# import nltk
# nltk.download()
# and select "stopwords"

import re
import string
from nltk.corpus import stopwords
from wordcloud import WordCloud
import csv

STOP = stopwords.words("english")
html_clean = re.compile('<.*?>')
gr_clean = re.compile('\[.*?\]')
PRINTABLE = string.printable
REMOVE = set(["!","(",")",":",".",";",",",'"',"?","-",">","_"])

def replace_by_space(word):
    new = []
    for letter in word:
        if letter in REMOVE:
            new.append(' ')
        else:
            new.append(letter)
    return ''.join(new)

all_my_words = []
all_my_words_with_stop_words = []

fh = open('./goodreads_export.csv')
reader = csv.reader(fh)
header = reader.next()
position = header.index('My Review')

reviews = 0
words = 0
for ll in reader:
    review = ll[position].lower()
    if not review:
        # empty review
        continue
    # clean strings
    cleaned_review = re.sub(html_clean, '', review)
    cleaned_review = re.sub(gr_clean, '', cleaned_review)
    all_my_words_with_stop_words += cleaned_review
    cleaned_review = replace_by_space(cleaned_review)
    cleaned_review = filter(lambda x: x in PRINTABLE, cleaned_review)
    # clean words
    cleaned_review = cleaned_review.split()
    cleaned_review = filter(lambda x: x not in STOP, cleaned_review)
    words += len(cleaned_review)
    all_my_words += cleaned_review
    reviews += 1

print("You have %s words in %s reviews"%(words, reviews))

# write out word to disk for Markov chain
all_my_words_with_stop_words = ''.join(all_my_words_with_stop_words)
with open("All_review_words.txt","w") as f:
    f.write(all_my_words_with_stop_words + "\n")

# WordCloud takes only string, no list/set
wordcloud = WordCloud(max_font_size=200, width=800, height=500).generate(' '.join(all_my_words))
wordcloud.to_file("GR_wordcloud.png")

fh.close()

