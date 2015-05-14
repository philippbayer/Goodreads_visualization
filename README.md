Some people on goodreads have complained that their reviews disappear and I feel (but don't know) that I lost at least one, this tracks my exported CSV to check whether it actually happens

Some ideas for things to do with this:
- Write automated parser that exports reviews to html/epub/tumblr/blogger/wordpress etc.
- cron job which automatically pulls exported CSV from https://www.goodreads.com/review_porter/goodreads_export.csv (login a bit weird esp. with Facebook login, use API instead? Needs dev key, but easier to do /review/list.xml=USERID than to play Red Queen with Facebook's oauth)
- various visualization things like word-clouds, stats for correlations of ratings with, for example, shelf etc.

HOW TO OPEN:
The format is a bit weird, as it's a comma-delimited file with double quotes enclosing each column, so you can't just split by comma, you'll split the review text. If all else fails, use Libreoffice (Text Delimiter -> ", Separated by -> Comma) to insert tabs.

License for reviews and code and whatnot: CC-BY-SA 4.0


