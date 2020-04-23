import networkx as nx
from util import generate_bipartite_graph, get_column, normalize_matrix
from networkx.algorithms import bipartite
from collections import Counter
from math import log2
import numpy as np
from scipy.spatial.distance import cosine as cos_dist
import progressbar


class RS:

    def __init__(self, data):
        '''
        Init system with bipartite graph
        :param G: bipartite Graph, generated from util.generate_bipartite_graph
        :var users: user ids in G
        :var items: item ids in G
        :var R: adjacency matrix
        :var U, S, V: SVD decomposition for user vectorization
        :var R_abs_i: absolute difference matrix (dictionary) between user i and all others users j.
                      R_abs_i[j][it] abs difference between i and j over common item it
        :var target: id of target user
        '''
        self.G = generate_bipartite_graph(data)
        self.users, self.items = bipartite.sets(self.G)
        self.R = bipartite.matrix.biadjacency_matrix(self.G, self.items).toarray().tolist()
        self.T = bipartite.matrix.biadjacency_matrix(self.G, self.items, weight='timestamp').toarray().tolist()
        self.T = normalize_matrix(np.array(self.T))
        self.U, self.S, self.V = np.linalg.svd(self.R)
        self.R_abs_i = None
        self.T_abs_i = None
        self.target = None

    def info(self):
        '''
        Print System Information (Matrix dimensions, users_ids, item_ids...)
        :return:
        '''
        print()
        print("---- SYSTEM INFO ----")
        print("Bipartite: " + str(nx.is_connected(self.G)))
        print(
            "Users: " + str(len(self.users)) + " labels: [" + str(min(self.users)) + ", " + str(max(self.users)) + "]")
        print(
            "Items: " + str(len(self.items)) + " labels: [" + str(min(self.items)) + ", " + str(max(self.items)) + "]")
        print("R: (" + str(len(self.R)) + ", " + str(len(self.R[0])) + ") raws: [" + str(min(self.items)) +
              ", " + str(max(self.items)) + "] cols: [" + str(min(self.users)) + ", " + str(max(self.users)) + "]")
        print()

    def commons_items(self, node_i, node_j):
        '''
        :param node_i: user_i id
        :param node_j: user_j id
        :return: list of common rated items between i and j
        '''
        l = [x for x in nx.common_neighbors(self.G, node_i, node_j)]
        l.sort()
        return l

    def set_target_user(self, user):
        '''
        See also abs_difference_rating_matr
        :param user: target user id
        :return:
        '''
        self.target = user
        self.abs_difference_rating_matr(self.target)
        self.abs_difference_time_matr(self.target)

    def vectorize_users(self, user, dim=25):
        '''
        :param user: user id to vectorize
        :param dim: dimension of result vector.
                    SVD organize features importance in decreasing order. V columns collect user representation with
                    most important features in the first elements of vectors (this why dim = 3, it's like vectorize
                    users in 3-dimensional space). More dimension are taken, more no relevant features are used.
                    For cosine similarity is necessary compute similarity among important features since taking all
                    the whole vectors increse the precision of a single representation but returns bad similarity
                    comparing unimportant features.
        :return: vector of V that represent user.
        '''
        return get_column(user - 1, self.V)[0:dim]

    def abs_difference_rating_matr(self, user_i):
        '''
        Fill R_abs_i matrix for user i. Call in target user setter.
        :param user_i: user i id
        :return:
        '''
        self.R_abs_i = {u: {} for u in self.users if u != user_i}
        for user_j in self.users:
            if user_j != user_i:
                commons = self.commons_items(user_i, user_j)
                dict = {}
                for c in commons:
                    dict[c] = abs(self.R[c - min(self.items)][user_i - 1] - self.R[c - min(self.items)][user_j - 1])
                dict = {k: v for k, v in sorted(dict.items(), key=lambda item: item[1])}  # ORDERING DICT IMPORTANT
                self.R_abs_i[user_j] = dict

    def abs_difference_time_matr(self, user_i):
        '''
        Fill T_abs_i matrix for user i. Call in target user setter.
        :param user_i: user i id
        :return:
        '''
        self.T_abs_i = {u: {} for u in self.users if u != user_i}
        for user_j in self.users:
            if user_j != user_i:
                commons = self.commons_items(user_i, user_j)
                dict = {}
                for c in commons:
                    dict[c] = abs(self.T[c - min(self.items)][user_i - 1] - self.T[c - min(self.items)][user_j - 1])
                dict = {k: v for k, v in sorted(dict.items(), key=lambda item: item[1])}  # ORDERING DICT IMPORTANT
                self.T_abs_i[user_j] = dict

    def count_abs_rating_difference(self, user_j):
        '''
        Util function for Equation 2 in paper.
        :param user_j:
        :return: a dictionary that contains for each abs_rate_difference (0 (i.e 5-5) to 4 (i.e 5-1)) the number of
                 items between target and user_j with abs_rate_difference.
        '''
        dict = {}
        for i in range(5):
            count = Counter(self.R_abs_i[user_j].values())[i]
            dict[i] = count
        return dict

    def weighted_sum(self, user_i):
        '''
        :param user_i: user id (target)
        :return: a dictionary with key values all user_j != target and values the weighted sum p(i,j)
                 as defined in Equation 2
        '''
        dict = {}
        for user_j in self.users:
            if user_j != user_i:
                count = self.count_abs_rating_difference(user_j)
                dict[user_j] = 1 * count[0] + 0.8 * count[1] + 0.6 * count[2] + 0.4 * count[3] + 0.2 * count[4]
        return dict

    def temporal_weighted_sum(self, user_i):
        '''
        :param user_i: user id (target)
        :return: a dictionary with key values all user_j != target and values the weighted sum p(i,j)
                 as defined in Equation 2
        '''
        w = [1, 0.8, 0.6, 0.4, 0.2]
        dict = {}
        for user_j in self.users:
            sum = 0
            if user_j != user_i:
                for i in self.R_abs_i[user_j]:
                    value = int(self.R_abs_i[user_j][i])
                    sum += w[value] * (1 - self.T_abs_i[user_j][i])
                dict[user_j] = sum
        return dict

    def avg_rate(self, user_id):
        '''
        :param user_id: user id
        :return: average rate based on user_id rates
        '''
        rates = [x for x in get_column(user_id - 1, self.R) if x != 0]
        return sum(rates) / len(rates)

    def de(self, user_i, user_j, item):
        '''
        :param user_i:
        :param user_j:
        :param item:
        :return: value defined in Equation 5.
        '''
        return abs(
            (self.R[item - min(self.items)][user_i - 1] - self.avg_rate(user_i)) - (
                    self.R[item - min(self.items)][user_j - 1] - self.avg_rate(user_j)))

    def deviation(self, user_i, user_j, I):
        '''
        :param user_i:
        :param user_j:
        :param I: cardinality of filtered common items based on weighted sum
        :return: Deviation across all the co-rated items for the user U i and U j is represented as in Eq. (6)
        '''
        commons = self.commons_items(user_i, user_j)
        s = 0
        for i in range(0, I):
            s += self.de(user_i, user_j, commons[i])
        if s != 0:
            return s

    def probability_function(self, user_i, user_j, item_k, I):
        '''
        :param user_i:
        :param user_j:
        :param item_k:
        :param I:
        :return: p function Equation 7
        '''
        return self.de(user_i, user_j, item_k) / self.deviation(user_i, user_j, I)

    def entropy(self, user_i, user_j, I):
        '''
        :param user_i:
        :param user_j:
        :param I:
        :return: entropy given in Equation 8
        '''
        commons = list(self.R_abs_i[user_j].keys())
        H = 0
        for i in range(0, I):
            pk = self.probability_function(user_i, user_j, commons[i], I)
            H += pk * log2(pk)
        return -H

    def simE(self, user_i, user_j, I):
        '''
        :param user_i:
        :param user_j:
        :param I:
        :return: simE given in Equation 9
        '''
        return 1 - (self.entropy(user_i, user_j, I) / log2(I))

    def simC(self, user_i, user_j, vector_type):
        '''
        :param user_i:
        :param user_j:
        :return: simC
        '''
        if vector_type == 'svd':
            ui = self.vectorize_users(user_i)
            uj = self.vectorize_users(user_j)
            sim = 1 - cos_dist(ui, uj)
            return sim
        elif vector_type == 'chen':
            num = 0
            sum1 = 0
            sum2 = 0
            for it in self.items:
                num += self.R[it - 1001][user_i - 1] * self.R[it - 1001][user_j - 1]
                sum1 += np.power(self.R[it - 1001][user_i - 1], 2)
                sum2 += np.power(self.R[it - 1001][user_j - 1], 2)
            den = np.sqrt(sum1) * np.sqrt(sum2)
            return num / den

    def hybrid_similarity(self, user_i, user_j, I, beta, cos_vector_type, verbose):
        '''
        :param cos_vector_type:
        :param user_i:
        :param user_j:
        :param I:
        :param beta:
        :param verbose:
        :return: Hybrid similarity proposed in the paper, Equation 10
        '''
        sim_c = self.simC(user_i, user_j, cos_vector_type)
        sim_e = self.simE(user_i, user_j, I)
        if verbose:
            print("Cosine: " + str(sim_c))
            print("Entropy: " + str(sim_e))
        return (beta * sim_c) + ((1 - beta) * sim_e)

    def compute_similar_users(self, thresh, beta, cos_vector_type='chen', filtering='temporal', nodes=None,
                              verbose=True):
        '''
        :param cos_vector_type: Type of vectorization for cosine: 'svd' for SVD Decomposition, 'chen' for Chen2013 prop.
        :param thresh:
        :param beta:
        :param nodes:
        :param verbose:
        :return: Compute similar users . Similar users have hybrid_similarity > thresh
        '''
        if filtering == 'temporal':
            Wsum = self.temporal_weighted_sum(self.target)
        elif filtering == 'count':
            Wsum = self.weighted_sum(self.target)
        similar_users = []
        similarty_values = []
        if nodes is None:
            users_to_compute = self.users
        else:
            users_to_compute = nodes
        if verbose:
            bar = list
        else:
            bar = progressbar.ProgressBar()
        for j in bar(users_to_compute):
            try:
                if j != self.target:
                    I = int(Wsum[j])
                    hs = self.hybrid_similarity(self.target, j, I, beta, cos_vector_type, verbose)
                    if verbose:
                        print("Similarity(" + str(self.target) + "," + str(j) + ") = " + str(hs))
                        print()
                    similarty_values.append(hs)
                    if hs > thresh:
                        similar_users.append(j)
            except Exception:
                pass
        return similar_users

    def predict(self, similar_users, item):
        '''
        :param similar_users:
        :param item:
        :return: Predicted rates based on similar users (mean of rates)
        '''
        r = 0
        c = 0
        for j in similar_users:
            rate = self.R[item - min(self.items)][j - 1]
            if rate != 0:
                r += self.R[item - min(self.items)][j - 1]
                c += 1
        if c != 0:
            return r / c
        else:
            return 0

    def topN(self, N, thresh, beta, cos_vector_type, filtering, verbose=False):
        print("Target: "+str(self.target))
        similars = self.compute_similar_users(thresh, beta, cos_vector_type=cos_vector_type, filtering=filtering,
                                              verbose=verbose)
        predictions = []
        for i in self.items:
            predictions.append((i, self.predict(similars, i)))
        predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
        return predictions[:N]

    def evaluation(self, topN, t=3.5):
        relevant = [x[0] for x in topN if self.R[x[0] - min(self.items)][self.target - 1] > t]
        recommended = [x[0] for x in topN if x[1] > t]
        intersection = len(set(relevant) & set(recommended))
        precision = intersection / len(recommended)
        recall = intersection / len(relevant)
        return precision, recall

    def eval_test_set(self, test_set, filter='count', cosine_vect_type='chen'):
        print("TEST_SET: " + str(test_set))
        degree = [self.G.degree(x) for x in test_set]
        print("TEST_SET_DEGREE: " + str(degree))
        prec = []
        recall = []
        for u in test_set:
            self.set_target_user(u)
            topN = self.topN(1685, 0.4, 0.56, cosine_vect_type, filtering=filter, verbose=False)
            p, r = self.evaluation(topN)
            prec.append(p)
            recall.append(r)
        return sum(prec) / len(prec), sum(recall) / len(recall)
