import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import csv

CATEGORIES = 7 # number of most crowded categories to plot

if __name__ == '__main__':
    fh = open('./goodreads_export.csv')
    reader = csv.reader(fh)
    header = reader.next()
    print header
    position_rating = header.index('My Rating')
    position_pages = header.index('Number of Pages')

    all_ratings = []
    all_pages = []
    for ll in reader:
        my_rating = int(ll[position_rating])
        if my_rating == 0:
            # no rating
            continue
        try:
            pages = int(ll[position_pages])
        except ValueError:
            # some page-numbers are missing
            continue
        all_ratings.append(my_rating)
        all_pages.append(pages)

    data = pd.DataFrame({"Ratings":all_ratings, "Pages":all_pages})
    g = sns.jointplot("Pages", "Ratings", data=data, color="r", kind="reg")
    plt.tight_layout()
    plt.savefig("Pages_vs_Ratings.png")
    plt.show()

