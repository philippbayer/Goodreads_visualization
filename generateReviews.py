

with open('All_review_words.txt') as f:
    all_text = ''.join(f.readlines())

from pymarkovchain import MarkovChain
mc = MarkovChain('./markov_db')
mc.generateDatabase(all_text)
print mc.generateString()
