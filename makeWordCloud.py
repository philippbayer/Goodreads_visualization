# before first run, need to install stopwords:
# import nltk
# nltk.download()
# and select "stopwords"

import re
import string
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt

STOP = stopwords.words("english")
html_clean = re.compile('<.*?>')
PRINTABLE = string.printable
REMOVE = set(["!","(",")",":",".",";",",",'"',"?","-",">","_"])

def replace_by_space(word):
    new = []
    for letter in word:
        if letter in REMOVE:
            new.append(" ")
        else:
            new.append(letter)
    return "".join(new)

all_my_words = set()

fh = open("./goodreads_export.csv")
header = fh.readline().rstrip().split("\t")
position = header.index("My Review")
for line in fh:
    ll = line.split("\t")
    review = ll[position]
    if not review:
        # empty review
        continue
    cleaned_review = re.sub(html_clean, '', review)
    cleaned_review = replace_by_space(cleaned_review)
    cleaned_review = filter(lambda x: x in PRINTABLE, cleaned_review)
    for w in cleaned_review.lower().split():
        if w not in STOP:
            all_my_words.add(w)

# WordCloud takes only string, no list/set
wordcloud = WordCloud(max_font_size=40).generate(" ".join(all_my_words))
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('GR_wordcloud.png')

