import networkx as nx


def parseData(path):
    file = open(path)
    data = []
    line = file.readline()
    while line:
        data.append(line)
        line = file.readline()
    file.close()
    return data


def raws_to_tuple(raws):
    tuples = []
    for r in raws:
        tuples.append((int(r.split()[0]), int(r.split()[1]), int(r.split()[2]), int(r.split()[3])))
    return tuples


def generate_bipartite_graph(raws_file=None):
    if raws_file is not None:
        print("[NETWORK INIT] Read network from " + raws_file)
        raws = parseData(raws_file)
        raws = raws_to_tuple(raws)
        raws.sort()
        G = nx.Graph()
        for e in raws:
            i = e[0]
            j = e[1] + 1000
            w = e[2]
            t = e[3]
            if i not in G.nodes:
                G.add_node(i, bipartite=0)
            if j not in G.nodes:
                G.add_node(j, bipartite=1)
            G.add_edge(i, j, weight=w, timestamp=t)
        print("[NETWORK INIT] Bipartite Graph generate")
    else:
        print("[ERROR] Specify source file")
    return G


def normalize_matrix(W):
    Wmax, Wmin = W.max(), W.min()
    W = (W - Wmin) / (Wmax - Wmin)
    return W


def get_column(i, R):
    return [row[i] for row in R]

