from collections import defaultdict, Counter
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import csv

CATEGORIES = 7 # number of most crowded categories to plot

if __name__ == "__main__":
    shelves_ratings = defaultdict(list) # key: shelf-name, value: list of ratings
    shelves_counter = Counter() # counts how many books on each shelf

    fh = open('./goodreads_export.csv')
    reader = csv.reader(fh)
    header = reader.next()
    position = header.index('My Review')

    for ll in reader:
        review = ll[position].lower()
        my_rating = int(ll[7])
        if my_rating == 0:
            # no rating
            continue
        shelves = ll[16].strip().split(",")
        for s in shelves:
            # empty shelf?
            if not s: continue
            s = s.strip() # I had "non-fiction" and " non-fiction"
            shelves_ratings[s].append(my_rating)
            shelves_counter[s] += 1

    data_to_plot = []
    x_axis_labels = []
    pos = range(CATEGORIES)
    # i am sure there's some magic pandas function which does all of this for you
    names = []
    ratings = []
    for name, _ in shelves_counter.most_common(CATEGORIES):
        for number in shelves_ratings[name]:
            names.append(name)
            ratings.append(number)

    full_table = pd.DataFrame({"Category":names, "Rating":ratings})

    sns.violinplot(x="Category", y="Rating", data=full_table)
    plt.savefig("categories_violinplot.png")
    plt.show()

