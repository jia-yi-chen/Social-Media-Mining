from user_based_CF import user_based_CF
from item_based_CF import item_based_CF
import pandas as pd




###### Obtain the full user-item matrix
rating_cols = ['user_id', 'item_id', 'rating','timestamp']
ratings = pd.read_csv('./ml-100k/u.data', sep='\t', names=rating_cols)
ratings = ratings.pivot_table(index=['user_id'],columns=['item_id'],values='rating').reset_index(drop=True)
ratings.fillna( 0, inplace = True )
user_item_matrix = ratings.to_numpy()




##### other inputs #####
neighborhood_size = 10
user_id = 234
item_id = 988




###### Predict the rating
R_ui1 = user_based_CF(user_item_matrix, neighborhood_size, user_id, item_id )
R_ui2 = item_based_CF(user_item_matrix, neighborhood_size, user_id, item_id )
print("The predicted rating for user u=", user_id,"item i=",item_id,"is")
print("Rating(u,i)=", R_ui1, "using User-based CF.")
print("Rating(u,i)=", R_ui2, "using Item-based CF.")
if user_item_matrix[user_id, item_id]>0:
    print("Ground truth for Rating(u,i) is", user_item_matrix[user_id, item_id])
else:
    print("Ground truth for Rating(u,i) is missing, i.e.,", "NaN")

