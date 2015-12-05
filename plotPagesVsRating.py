from collections import defaultdict, Counter
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import csv

CATEGORIES = 7 # number of most crowded categories to plot

if __name__ == '__main__':
    shelves_ratings = defaultdict(list) # key: shelf-name, value: list of ratings
    shelves_counter = Counter() # counts how many books on each shelf

    fh = open('./goodreads_export.csv')
    reader = csv.reader(fh)
    header = reader.next()
    position = header.index('My Rating')
    position_pages = header.index('Number of Pages')

    all_ratings = []
    all_pages = []
    for ll in reader:
        review = ll[position].lower()
        my_rating = int(ll[position])
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
    g = sns.jointplot("Ratings", "Pages", data=data, color="r", kind="reg")
    plt.tight_layout()
    plt.savefig("Pages_vs_Ratings.png")
    plt.show()

