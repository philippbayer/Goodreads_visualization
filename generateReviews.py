from pymarkovchain import MarkovChain
import pickle
import networkx as nx
import matplotlib.pyplot as plt

with open('All_review_words.txt') as f:
    all_text = ''.join(f.readlines())

mc = MarkovChain(dbFilePath='./markov_db')
mc.generateDatabase(all_text)

print mc.generateString()

mc.dumpdb()

# a key in the datbase looks like:
# ('when', 'you') defaultdict(<function _one at 0x7f5c843a4500>, 
# {'just': 0.06250000000059731, 'feel': 0.06250000000059731, 'had': 0.06250000000059731, 'accidentally': 0.06250000000059731, ''love': 0.06250000000059731, 'read': 0.06250000000059731, 'see': 0.06250000000059731, 'base': 0.06250000000059731, 'know': 0.12499999999641617, 'have': 0.12499999999641617, 'were': 0.06250000000059731, 'come': 0.06250000000059731, 'can't': 0.06250000000059731, 'are': 0.06250000000059731})
# so 'just' follows after 'when you' with 6% probability

db = pickle.load(open('./markov_db'))
# let's get a good node
#for key in db:
#    # has in between 5 and 10 connections
#    if len(db[key]) > 5 and (len(db[key]) < 10):
#        if len(set(db[key].values())) > 2:
#            print key, set(db[key].values())

# manually chosen from above
good_key = ('translation',)
values = db[good_key]

# create the graph

G = nx.DiGraph()
good_key = str(good_key[0])
G.add_node(good_key)
G.add_nodes_from(values.keys())

all_edges = []
for partner in values:
    edge_weight = values[partner]
    all_edges.append(edge_weight)
    G.add_weighted_edges_from([ (good_key, partner, edge_weight) ])

pos = nx.circular_layout(G)
nx.draw_networkx_nodes(G, pos, node_color = 'white', node_size = 5000)

# width of edges is based on probability * 10
for edge in G.edges(data=True):
    nx.draw_networkx_edges(G, pos, edgelist = [(edge[0], edge[1])], width = edge[2]['weight']*10)

nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
plt.axis('off')
plt.tight_layout()
plt.savefig('Markov_graph.png')
plt.show()
