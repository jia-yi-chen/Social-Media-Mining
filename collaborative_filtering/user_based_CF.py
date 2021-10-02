"""
Created on April 2021
author: Jiayi Chen
"""
import numpy as np




def user_based_CF(rating_matrix, neig_size, u, i):
    """
    :param rating_matrix: user-item matrix
    :param neig_size: neighbor size
    :param u: user id
    :param i: item id
    :return: the rating by u for i
    """

    #### remove the users who did not rate item i
    user_list = list(np.where(rating_matrix[:,i]>0)[0])
    if u in user_list:
        user_list.remove(u)
    num_user = len(user_list)



    #### find nearest neighbors/users using cosine similarity ######
    similarities = np.zeros(num_user)
    user_list = np.array(user_list)
    Uu = rating_matrix[u, :]
    Uu = np.delete(Uu, i) # remove the column for the target item
    for j, v in enumerate(user_list):
        Uv = rating_matrix[v, :]
        Uv = np.delete(Uv, i)
        similarities[j] = np.dot(Uv, Uu)/(np.dot(Uu, Uu) * np.dot(Uv, Uv))
    tmp = np.argsort(similarities)[::-1][:neig_size]
    nearest_users = user_list[tmp]



    #### average ratings of the k-nearest users ######
    avg_ratings = np.mean(rating_matrix[nearest_users, :], axis=1)
    avg_u = np.mean(Uu, axis=0)



    ### prediction ###
    numerator = 0
    denominator = 0
    for j, v in enumerate(nearest_users):
        sim_u_v = similarities[tmp[j]]
        avg_v = avg_ratings[j]
        rating_vi = rating_matrix[v, i]
        numerator += sim_u_v * (rating_vi - avg_v)
        denominator += sim_u_v
    rating_ui = avg_u + numerator/denominator



    return rating_ui


