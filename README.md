A collection of scripts to play around with goodreads data

You can use it with your own data - go to [here](https://www.goodreads.com/review/import) and press "Export your library" to get your own csv, it should work out of the box with these scripts

# Scripts included:

## plotWordCloud.py

This one parses goodreads_export.csv, removes noisy words and creates a word-cloud of most commonly used words:

![wordcloud](https://raw.github.com/philippbayer/my_goodreads_shelves/master/GR_wordcloud.png)

## plotRatingsVsBookshelves.py

Parses ratings for books, makes a violin plot for the 7 categories with the most rated books:

![Ratings_by_Shelves](https://raw.github.com/philippbayer/my_goodreads_shelves/master/categories_violinplot.png)

There is some *bad* SF out there.

## plotHistogramDistanceRead.py

Parses the dates read for each book read, plots the distance between books read in days - shows you how quickly you hop from book to book.

![Distance in days between books](https://raw.github.com/philippbayer/my_goodreads_shelves/master/Histogram_Days_Read_Distance.png)

Some other ideas for things to do with this:

- Some people on goodreads have complained that their reviews disappear and I feel (but don't know) that I lost at least one, this tracks my exported CSV to check whether it actually happens. So far I haven't observed it.
- Write automated parser that exports reviews to html/epub/tumblr/blogger/wordpress etc.
- cron job which automatically pulls exported CSV from https://www.goodreads.com/review_porter/goodreads_export.csv (login a bit weird esp. with Facebook login, use API instead? Needs dev key, but easier to do /review/list.xml=USERID than to play Red Queen with Facebook's oauth)
- various visualization things in regards to language use

License for reviews: CC-BY-SA 4.0
Code: MIT
