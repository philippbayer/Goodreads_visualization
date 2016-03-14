from pymarkovchain import MarkovChain

with open('All_review_words.txt') as f:
    all_text = ''.join(f.readlines())

mc = MarkovChain(dbFilePath='./markov_db')
mc.generateDatabase(all_text)

print mc.generateString()

mc.dumpdb()

# check out word probabilities for visualization?
# looks like:
# ('when', 'you') defaultdict(<function _one at 0x7f5c843a4500>, 
# {'just': 0.06250000000059731, 'feel': 0.06250000000059731, 'had': 0.06250000000059731, 'accidentally': 0.06250000000059731, '"love': 0.06250000000059731, 'read': 0.06250000000059731, 'see': 0.06250000000059731, 'base': 0.06250000000059731, 'know': 0.12499999999641617, 'have': 0.12499999999641617, 'were': 0.06250000000059731, 'come': 0.06250000000059731, "can't": 0.06250000000059731, 'are': 0.06250000000059731})
# so "just" follows after "when you" with 6% probability

#import pickle
#db = pickle.load(open("./markov_db"))
#for key in db:
#    if len(db[key]) > 5:
#        print key, db[key]
