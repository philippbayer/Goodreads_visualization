import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from datetime import datetime

if __name__ == "__main__":
    df = pd.read_csv("./goodreads_export.csv")
    dates = df.xs("Date Read",axis=1).dropna()
    dates = pd.to_datetime(dates)

    # we need a dataframe in this format:
    # year months books_read
    # I am sure there's some magic pandas function for this

    read_dict = defaultdict(int) # key: (year, month), value: count of books read
    for date in dates:
        this_year = date.year
        this_month = date.month
        read_dict[ (this_year, this_month) ] += 1

    first_date = sorted(read_dict)[0] # 2012, 3
    first_year = first_date[0]
    first_month = first_date[1]

    todays_date = datetime.today()
    todays_year = todays_date.year
    todays_month = todays_date.month

    all_years = []
    all_months = []
    all_counts = []
    for year in range(first_year, todays_year+1):
        for month in range(1, 13):
            if (year == todays_year) and month > todays_month:
                # don't count future months
                # it's 2015-12 now so a bit hard to test
                break
            this_count = read_dict[ (year, month) ]
            all_years.append(year)
            all_months.append(month)
            all_counts.append(this_count)

    # now get it in the format heatmap() wants
    df = pd.DataFrame( { "month":all_months, "year":all_years, "books_read":all_counts } )
    dfp = df.pivot("month", "year", "books_read")

    # now make the heatmap
    ax = sns.heatmap(dfp, annot=True)
    plt.savefig("Heatmap_Books_Read_Per_Month.png")
    plt.show()
    
    # now make histogram by year
    g = sns.FacetGrid(df, col="year", sharey=True, sharex=True) 
    g.map(plt.plot, "month", "books_read")
    g.set_ylabels("Books read")
    g.set_xlabels("Month")
    plt.xlim(1, 12)
    plt.savefig("Lineplot_Books_Read_Per_Month_split_up_by_year.png")
    plt.show()
