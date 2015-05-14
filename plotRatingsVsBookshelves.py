# slightly modified violinplotting from 
# http://pyinsci.blogspot.com.br/2009/09/violin-plot-with-matplotlib.html
# all I did is overwrite the xticks and add tight_layout
from collections import defaultdict, Counter
from matplotlib.pyplot import figure, show, xticks, tight_layout, savefig
from scipy.stats import gaussian_kde
from numpy import arange

CATEGORIES = 10 # number of most crowded categories to plot

def violin_plot(ax,data,pos, bp=False):
    '''
    create violin plots on an axis
    '''
    dist = max(pos)-min(pos)
    w = min(0.15*max(dist,1.0),0.5)
    for d,p in zip(data,pos):
        k = gaussian_kde(d) #calculates the kernel density
        m = k.dataset.min() #lower bound of violin
        M = k.dataset.max() #upper bound of violin
        x = arange(m,M,(M-m)/100.) # support for violin
        v = k.evaluate(x) #violin profile (density curve)
        v = v/v.max()*w #scaling the violin to the available space
        ax.fill_betweenx(x,p,v+p,facecolor='y',alpha=0.3)
        ax.fill_betweenx(x,p,-v+p,facecolor='y',alpha=0.3)
    if bp:
        ax.boxplot(data,notch=1,positions=pos,vert=1)

# Header:
# Book Id,Title,Author,Author l-f,Additional Authors,ISBN,ISBN13,My Rating,Average Rating,Publisher,Binding,Number of Pages,Year Published,Original Publication Year,Date Read,Date Added,Bookshelves,Bookshelves with positions,Exclusive Shelf,My Review,Spoiler,Private Notes,Read Count,Recommended For,Recommended By,Owned Copies,Original Purchase Date,Original Purchase Location,Condition,Condition Description,BCID

# to get only_read.tsv:
# grep '"read"' goodreads_export.csv > only_read.csv
# use libreoffice calc to add tabs to only_read.tsv
if __name__ == "__main__":
    shelves_ratings = defaultdict(list) # key: shelf-name, value: list of ratings
    shelves_counter = Counter() # counts how many books on each shelf
    with open("./only_read.tsv") as read:
        for line in read:
            ll = line.rstrip().split('\t')
            my_rating = int(ll[7])
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
        for name, _ in shelves_counter.most_common(CATEGORIES):
            data_to_plot.append(shelves_ratings[name])
            x_axis_labels.append("{0} ({1})".format(name, shelves_counter[name]))

        fig = figure()
        ax = fig.add_subplot(111)
        violin_plot(ax, data_to_plot, pos, bp=False)
        # overwrite the x-axis labels
        xticks(range(CATEGORIES), x_axis_labels, size="small", rotation=90)
        tight_layout()
        savefig("categories_boxplot.png")
        show()
