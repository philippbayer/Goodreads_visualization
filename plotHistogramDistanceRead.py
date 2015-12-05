import csv
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    fh = open('./goodreads_export.csv')
    reader = csv.reader(fh)
    header = reader.next()
    position = header.index('Date Read')

    all_dates = []
    for ll in reader:
        date = ll[position].lower()
        if not date: continue
        date = datetime.strptime(date, "%Y/%m/%d")
        all_dates.append(date)

    assert all_dates
    all_dates = sorted(all_dates)
    last_date = ""
    differences = []
    all_days = []
    for date in all_dates:
        if not last_date:
            last_date = date
        difference = date - last_date
        days = difference.days
        all_days.append(days)
        last_date = date

    sns.distplot(all_days, axlabel="Distance in days between books read")
    plt.tight_layout()
    plt.savefig("Histogram_Days_Read_Distance.png")
    plt.show()
