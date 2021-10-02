import numpy as np



def item_based_CF(rating_matrix, neig_size, u, i):
    """
    :param rating_matrix: user-item matrix
    :param neig_size: neighbor size
    :param u: user id
    :param i: item id
    :return: the rating by u for i
    """

    #### remove the items who are not rated by user u
    item_list = list(np.where(rating_matrix[u, :]>0)[0])
    if i in item_list:
        item_list.remove(i)
    num_item = len(item_list)



    #### find nearest neighbors/users using cosine similarity ######
    similarities = np.zeros(num_item)
    item_list = np.array(item_list)
    Ii = rating_matrix[:, i]
    Ii = np.delete(Ii, u) # remove the row for the target user
    for p, j in enumerate(item_list):
        Ij = rating_matrix[:, j]
        Ij = np.delete(Ij, u)
        similarities[p] = np.dot(Ii, Ij)/(np.dot(Ii, Ii) * np.dot(Ij, Ij))
    tmp = np.argsort(similarities)[::-1][:neig_size]
    nearest_items = item_list[tmp]



    #### average ratings of the k-nearest users ######
    avg_ratings = np.mean(rating_matrix[:, nearest_items], axis=0)
    avg_i = np.mean(Ii, axis=0)



    ### prediction ###
    numerator = 0
    denominator = 0
    for p, j in enumerate(nearest_items):
        sim_i_j = similarities[tmp[p]]
        avg_j = avg_ratings[p]
        rating_uj = rating_matrix[u, j]
        numerator += sim_i_j * (rating_uj - avg_j)
        denominator += sim_i_j
    rating_ui = avg_i + numerator/denominator



    return rating_ui


