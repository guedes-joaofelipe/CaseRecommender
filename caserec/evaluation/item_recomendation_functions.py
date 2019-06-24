# coding=utf-8
""""
    These functions are responsible for evaluate item recommendation algorithms (rankings).

    They are used by evaluation/item_recommendation.py

"""

# © 2018. Case Recommender (MIT License)

import numpy as np

__author__ = 'Arthur Fortes <fortes.arthur@gmail.com>'


def precision_at_k(ranking, k):
    """
    Score is precision @ k
    Relevance is binary (nonzero is relevant).

    :param ranking: Relevance scores (list or numpy) in rank order (first element is the first item)
    :type ranking: list, np.array

    :param k: length of ranking
    :type k: int

    :return: Precision @ k
    :rtype: float

    """

    assert k >= 1
    ranking = np.asarray(ranking)[:k] != 0
    if ranking.size != k:
        raise ValueError('Relevance score length ({}) < k ({})'.format(ranking.size, k))
    return np.mean(ranking)


def average_precision(ranking):
    """
    Score is average precision (area under PR curve). Relevance is binary (nonzero is relevant).

    :param ranking: Relevance scores (list or numpy) in rank order (first element is the first item)
    :type ranking: list, np.array

    :return: Average precision
    :rtype: float

    """

    ranking = np.asarray(ranking) != 0
    out = [precision_at_k(ranking, k + 1) for k in range(ranking.size) if ranking[k]]
    if not out:
        return 0.
    return np.mean(out)


def mean_average_precision(ranking):
    """
    Score is mean average precision. Relevance is binary (nonzero is relevant).

    :param ranking: Relevance scores (list or numpy) in rank order (first element is the first item)
    :type ranking: list, np.array

    :return: Mean average precision
    :rtype: float
    """

    return np.mean([average_precision(r) for r in ranking])


def ndcg_at_k(ranking, k = None):
    """
    Score is normalized discounted cumulative gain (ndcg). Relevance is positive real values.  Can use binary
    as the previous methods.

    :param ranking: ranking to evaluate in dcg format [0, 0, 1], where 1 is correct info
    :type ranking: list

    :return: Normalized discounted cumulative gain
    :rtype: float

    """

    k = len(ranking) if k is None else k 
    assert k >= 1
    # if ranking.size != k:
    #     raise ValueError('Relevance score length ({}) < k ({})'.format(ranking.size, k))

    ranking = np.asfarray(ranking)[:k] 
    r_ideal = np.asfarray(sorted(ranking, reverse=True))
    dcg_ideal = r_ideal[0] + np.sum(r_ideal[1:] / np.log2(np.arange(2, r_ideal.size + 1)))
    dcg_ranking = ranking[0] + np.sum(ranking[1:] / np.log2(np.arange(2, ranking.size + 1)))

    return dcg_ranking / dcg_ideal


def reciprocal_rank(ranking, k=None):
    """
        Score is reciprocal of the rank of the first relevant item. 
        First element is rank 1. Relevance is binary (nonzero is relevant).

    :param ranking: Relevance scores (list or numpy) in rank order (first element is the first item)
    :type ranking: list, np.array

    :return: reciprocal rank of a ranking list
    :rtype: float
    """
    k = len(ranking) if k is None else k
    assert k >= 1

    ranking = np.asfarray(ranking)[:k]
    index_found_ranks = np.where(ranking.astype(int) == 1)[0]
    if index_found_ranks.size > 0:
        rank = index_found_ranks[0] + 1
        return 1/float(rank)    
    return 0

def mean_reciprocal_rank(rankings, k=None):
    """
        Score is the mean of reciprocal ranks on an array of rankings. 

        :param rankings: array of rankings, where ranking is a relevance score 
            (list of numpy) in rank order (first element is the most relevant)
        :type ranking: list, np.array

        :return: mean reciprocal rank of a list of rankings 
        :rtype: float
    """

    return np.mean([reciprocal_rank(ranking, k=k) for ranking in rankings])

# def rank_accuracy(ranking):
#   FUNCTION IN DEVELOPMENT
#   ISSUE: when all correct items are in the ranking but all in the incorrect order, I want a 0.5 score
#
#     """
#     Score is rank accuracy . Relevance is positive real values.  Can use binary
#     as the previous methods.

#     For a N-size list:
#         If item is outside the sequence: 0 score
#         Elif item is in sequence but in the wrong position: 1/N score
#         Else (item in the sequence and in the right position): 1 score

#     """    
#     ranking = np.asfarray(ranking)
#     ranking_ideal = np.asfarray(sorted(ranking, reverse=True))    
#     ideal_index = np.argsort(-ranking_ideal)

#     score = 0
#     ranking_length = len(ranking)

#     for item_index in np.arange(ranking_length):
#         if ranking[item_index] == 0:
#             item_score = 0
#         elif item_index == ideal_index[item_index]:
#             item_score = 1
#         else:
#             item_score = float(1/ranking_length)        
#         score += item_score

#     print ("Rank Acc for {}: {}".format(ranking, float(score/ranking_length)))
#     return float(score/ranking_length)
    
# def mean_rank_accuracy(rankings):
#     """
#         Score is the mean rank accuracy of a list of rankings. Relevance is positive real values.  Can use binary
#         as the previous methods.
        
#         :param rankings: a list of rankings
#         :ptype: [list, np.array]
        
#         :return: mean rank accuracy
#         :rtype: float
#     """
#     return np.mean([rank_accuracy(ranking) for ranking in rankings])


if __name__ == "__main__":
    
    rankings = [[1, 1, 1, 1, 1, 1], # Totally right ranking
                [0, 0, 0, 0, 0, 0], # Totally wrong ranking
                [0, 0, 1, 0, 0, 1]] # Partially right ranking
    k = 2

    print ("-"*10)
    print ("Precision@{}: {}".format(k, precision_at_k(rankings[-1], k)))

    print ("-"*10)
    print ("Reciprocal Rank: ", reciprocal_rank(rankings[-1]))
    print ("Reciprocal Rank@{}: {}".format(k, reciprocal_rank(rankings[-1], k)))
    print ("Mean Reciprocal Rank: ", mean_reciprocal_rank(rankings))
    print ("Mean Reciprocal Rank@{}: {}".format(k, mean_reciprocal_rank(rankings, k)))

    # print ("-"*10)    
    # rankings = [[0.6, 0.5, 0.4, 0.3, 0.2, 0.1], # Totally right ranking
    #             [0, 0, 0, 0, 0, 0], # Totally wrong ranking
    #             [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]] # Partially right ranking (wrong order)
    # print ("Rankings: ", rankings)
    # print ("Rank Accuracy: ", rank_accuracy(rankings[0]))    
    # print ("Mean Rank Accuracy: ", mean_rank_accuracy(rankings))