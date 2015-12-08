import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
from collections import defaultdict

if __name__ == "__main__":
    df = pd.read_csv("./goodreads_export.csv")
    dates = df.xs("Date Read",axis=1).dropna()
    dates = pd.to_datetime(dates)

    read_dict = defaultdict(int) # key: weekday, value: count of books read
    for date in dates:
        weekday_name = calendar.day_name[date.weekday()]  # Sunday
        read_dict[weekday_name] += 1

    full_table = pd.DataFrame({"Weekday":read_dict.keys(), "Books read":read_dict.values()})

    sns.barplot(x="Weekday", y="Books read", data=full_table)
    plt.tight_layout()
    plt.savefig("Books_read_by_weekday.png")
    plt.show()