A collection of scripts to play around with goodreads data

You can use it with your own data - go to [here](https://www.goodreads.com/review/import) and press "Export your library" to get your own csv, it should work out of the box with these scripts. Each script expects a "goodreads_export.csv" in the current folder (this may change), you can just replace that one with yours from goodreads, and then run any of these scripts.

# Scripts included:

## plotPagesVsRating.py

Do I give longer books better scores? A minor tendency but nothing special (it's confounded by having just 5 possible numbers in ratings)

![Number pages versus ratings](https://raw.github.com/philippbayer/Goodreads_visualization/master/Pages_vs_Ratings.png)

I seem to mostly read books at around 200 to 300 pages so it's hard to tell whether I give longer books better ratings. It's also a nice example that in regards to linear regression, a p-value as tiny as this one doesn't mean much, the r-value is still bad.

## plotRatingsVsBookshelves.py

Parses ratings for books, makes a violin plot for the 7 categories with the most rated books:

![Ratings_by_Shelves](https://raw.github.com/philippbayer/Goodreads_visualization/master/categories_violinplot.png)

There is some *bad* SF out there.

## plotHistogramDistanceRead.py

Parses the "dates read" for each book read, plots the distance between books read in days - shows you how quickly you hop from book to book.

![Distance in days between books](https://raw.github.com/philippbayer/Goodreads_visualization/master/Histogram_Days_Read_Distance.png)

Of course, sometimes I just add several at once and guesstimate the correct "date read".

## plotHeatmapDatesRead.py

Parses the "dates read" for each book read, bins them by month, and makes a heatmap to show in which months I read more than in others. Also makes a lineplot for books read, split up by year.

![Heatmap books read per month](https://raw.github.com/philippbayer/Goodreads_visualization/master/Heatmap_Books_Read_Per_Month.png)

What happened in May 2014?

![Lineplot books read per month by year](https://raw.github.com/philippbayer/Goodreads_visualization/master/Lineplot_Books_Read_Per_Month_split_up_by_year.png)

It's nice how reading behaviour (Goodreads usage) connects over the months - it slowly in 2013, stays constant in 2014/2015, and now goes down again.

## plotWordCloud.py

This one removes noisy words and creates a word-cloud of most commonly used words in the reviews:

![wordcloud](https://raw.github.com/philippbayer/Goodreads_visualization/master/GR_wordcloud.png)

## plotBooksreadVsWeekday.py

Parses the weekday a "book read" has been added, counts them

![wordcloud](https://raw.github.com/philippbayer/Goodreads_visualization/master/Books_read_by_weekday.png)

Monday is procrastination day.


Some other ideas for things to do with this:

- A Python-library that interacts with the goodreads csv - a lot of code in this repo repeats itself slightly
- Some people on goodreads have complained that their reviews disappear and I feel (but don't know) that I lost at least one, this tracks my exported CSV to check whether it actually happens. So far I haven't observed it.
- Write automated parser that exports reviews to html/epub/tumblr/blogger/wordpress etc.
- cron job which automatically pulls exported CSV from https://www.goodreads.com/review_porter/goodreads_export.csv (login a bit weird esp. with Facebook login, use API instead? Needs dev key, but easier to do /review/list.xml=USERID than to play Red Queen with Facebook's oauth)
- various visualization things in regards to language use

License for reviews: CC-BY-SA 4.0
Code: MIT
