# -*- coding: utf-8 -*-
import logging
import pandas as pd
import pytumblr
import translitcodec
import unicodedata
import re
import editdistance

logging.basicConfig(level=logging.INFO)
# Go to https://www.tumblr.com/settings/apps and create an app,
# and/or copy OAuth Consumer Key and OAuth Consumer Scret
# then go to https://api.tumblr.com/console/calls/user/info
# and enter those to receive TOKEN and TOKEN_SECRET

CONSUMER_KEY = 'SNIP'
CONSUMER_SECRET = 'SNIP'
TOKEN = 'SNIP'
TOKEN_SECRET = 'SNIP'

client = pytumblr.TumblrRestClient(
        CONSUMER_KEY,
        CONSUMER_SECRET,
        TOKEN,
        TOKEN_SECRET
)

# Make the request
print(client.info())

FILENAME = './goodreads_export.csv'
MAX = 40
TAGS = ['book', 'review', 'books']
ROWS_TO_GET = ['Title', 'Author', 'My Review', 'My Rating',
               'Date Read', 'Bookshelves']
BLOGNAME = 'biggestfool'
STATE = 'queue'  # alternatives: draft, private, published
# queue: posts get published automatically two times per day
GOODREADS_URL = 'https://www.goodreads.com/review/list/8325737'
# Goodreads links look like [b:TEXT]
GOODREADS_LINK = re.compile("\[b:(.*)\]")

all_titles = set()
# Get all posts of user so that we don't duplicate
all_posts = client.posts(BLOGNAME)['posts']
# just check titles
for p in all_posts:
    try:
        all_titles.add(p['title'])
    except KeyError:
        # some blog posts unrelated to this script have no title
        continue
# also get everything on the queue
all_queue = client.queue(BLOGNAME)['posts']
for p in all_queue:
    try:
        all_titles.add(p['title'])
    except KeyError:
        continue

df = pd.read_csv(FILENAME)
cleaned_df = df[df["My Rating"] != 0]

to_upload = df[ROWS_TO_GET]
to_upload = to_upload.dropna(subset=['My Review']).head(MAX)

# Now make a post for each row in the csv
for index, row in to_upload.iterrows():
    title, author, review, rating, date, bookshelves = row[ROWS_TO_GET]
    review = 'Rating: {0} out of 5<br/><br/>{1}<br/><br/>\
              Originally posted on {2} at Goodreads:{3}<br/><br/>'\
              .format(rating, review, date, GOODREADS_URL)

    # Fix Japanese/Greek author and book titles
    title = title.decode('utf-8').encode('translit/long')
    # Kenkō becomes Kenko
    author = author.decode('utf-8').encode('translit/long')
    # still some weird Greek strings left, drop those
    author = ''.join([i if ord(i) < 128 else '' for i in author])
    title = ''.join([i if ord(i) < 128 else '' for i in title])

    post_title = 'Review: {0} - {1}'.format(author, title)
 

    # Remove Goodreads specific links from the review text
    while '[b:' in review:
        matches = re.search(GOODREADS_LINK, review).groups()
        if matches:
            for s in matches:
                this_title = s.split('|')[0]
                review = review.replace(s, this_title).\
                         replace('[b:','').replace(']','')

    # the slug is the hardcoded end of the post's URL
    # The following is based on Django's slugify
    # https://github.com/django/django/blob/master/django/utils/text.py#L417
    # can't be bothered to import all of Django for this
    slug = unicode('review_{0}_{1}'.format(
           author.lower().replace(' ', '_'),
           title.lower().replace(' ', '_')))

    slug = unicodedata.normalize('NFKD', slug).\
                       encode('ascii', 'ignore').\
                       decode('ascii')

    slug = re.sub(r'[-\s]+', '-',
                  re.sub(r'[^\w\s-]', '', slug).
                  strip().
                  lower())

    # what follows is a hilariously over-engineered solution to the tumblr API
    # not always returning exactly what I upload
    # Could be cleanup of different strings, could be Python2 unicode wreckage
    # there is no word for 'over-engineered' in German so I'm doing this
    skip_me = False
    for t in all_titles:
        # Get the Levenshtein distance between this title and all titles
        edit_distance = editdistance.eval(post_title, t)
        # longer titles can have larger distances
        both_lengths = len(t), len(title)
        longer = float(max(both_lengths))
        distance = edit_distance/longer*100
        if distance < 10:
            skip_me = True
            break

    if skip_me:
        logging.info('Skipping post for {0} because title is already uploaded'.
                     format(title))
        continue

    this_tags = list(TAGS) + [author, title]
    # GR bookshelves work like tags so let's just copy them
    if not pd.isnull(bookshelves):
        this_tags += bookshelves.strip().split(',')

    logging.info('Uploading post for {0}'.format(title))
    client.create_text(BLOGNAME, state=STATE,
                       slug=slug, title=post_title,
                       body=review, tags=this_tags)
