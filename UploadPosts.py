import logging
import pandas as pd
import pytumblr

logging.basicConfig(level=logging.INFO)
# Go to https://www.tumblr.com/settings/apps and create an app, or copy OAuth Consumer Key and OAuth Consumer Scret
# then go to https://api.tumblr.com/console/calls/user/info and enter those to receive TOKEN and TOKEN_SECRET

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
print client.info()

FILENAME = './goodreads_export.csv'
MAX = 30
TAGS = ['book', 'review', 'books']
ROWS_TO_GET = ['Title', 'Author', 'My Review', 'My Rating', 'Date Read', 'Bookshelves']
BLOGNAME = 'biggestfool'
STATE = 'queue' # alternatives: draft, private, published
# draft is nice, those posts get published automatically one each two times per day
GOODREADS_URL = 'https://www.goodreads.com/review/list/8325737'

# Get all posts of user so that we don't duplicate
all_posts = client.posts(BLOGNAME)['posts']
all_titles = set()
# PROBLEM: This doesn't check for draft posts - no clue right now how to get those
for p in all_posts:
    try:
        all_titles.add(p['title'])
    except KeyError:
        # some blog posts unrelated to this script have no title
        continue

df = pd.read_csv(FILENAME)
cleaned_df = df[df["My Rating"] != 0]

to_upload = df[ROWS_TO_GET]
to_upload = to_upload.dropna(subset = ['My Review']).head(MAX)
for index, row in to_upload.iterrows():
    title, author, review, rating, date, bookshelves = row[ROWS_TO_GET]
    review = 'Rating: %s out of 5<br/><br/>%s<br/><br/>Originally posted on %s at Goodreads: %s<br/><br/>'%(rating, review, date, GOODREADS_URL)
    post_title = 'Review: %s - %s'%(author, title)
    # the slug is the hardcoded end of the post's URL
    slug = 'review_%s_%s'%(author.lower().replace(' ','_'),title.lower().replace(' ','_'))
    slug = slug.replace('?','').replace(':','').replace('%','').replace('#','').replace('.','').replace("'",'').replace('"','').replace('(','').replace(')','')
    if title in all_titles:
        logging.info('Skipping post for %s because title is already uploaded'%(title))
        continue

    this_tags = list(TAGS) + [author, title]
    # GR bookshelves work like tags so let's just copy them
    if not pd.isnull(bookshelves):
        this_tags +=  bookshelves.strip().split(',')

    logging.info('Uploading post for %s'%title)
    client.create_text(BLOGNAME, state = STATE, slug = slug, title = post_title, body = review, tags = this_tags)

