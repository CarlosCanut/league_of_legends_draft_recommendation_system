from helpers import *


########### main() ###########

def main():
    
    ####################################################################################################
    # Data Cleaning
    ####################################################################################################
    total_games_raw = pd.read_excel("./data/competitive/total_games.xlsx")

    total_games_raw = total_games_raw[total_games_raw['gameEndedInEarlySurrender'] == False]
    total_games = total_games_raw.drop(['item0', 'item1', 'item2', 'item3', 'item4', 'item5', 'item6', 
                                    'gameEndedInEarlySurrender', 'gameEndedInSurrender', 'teamId',
                                'game_id', 'league', 'blue_team', 'red_team', 'summonerId', 'participantId', 'summonerName',
                                'team', 'team_vs', 'rune1', 'rune2', 'rune3', 'rune4', 'rune5', 'Unnamed: 0'], axis=1)
    top_games = total_games[total_games['teamPosition'] == "Top"].drop(['teamPosition'], axis=1)
    jungle_games = total_games[total_games['teamPosition'] == "Jungle"].drop(['teamPosition'], axis=1)
    mid_games = total_games[total_games['teamPosition'] == "Mid"].drop(['teamPosition'], axis=1)
    adc_games = total_games[total_games['teamPosition'] == "Adc"].drop(['teamPosition'], axis=1)
    support_games = total_games[total_games['teamPosition'] == "Support"].drop(['teamPosition'], axis=1)

    ####################################################################################################
    # Clustering
    ####################################################################################################
    
    ########### Top stats ###########
    x_top, y_top = standarize_df(top_games)
    results_dict_top = get_best_clustering(x_top, pca_params = [0.95, 0.90, 0.85], umap_params = [2, 3, 4, 5, 6, 7, 8, 9], kmeans_params = [3, 4, 5, 6, 7, 8, 9], optics_params = [3, 4, 5, 6, 7, 8, 9])
    # print("results_dict_top:")
    # print(results_dict_top)
    top, top_champions_list, top_principal_components = umap_kmeans(x_top, y_top, n_comps=6, k = 5 )
    # y_top.to_excel("../data/competitive/clustering/top_clustering.xlsx")

    ########### Jungle stats ###########
    x_jungle, y_jungle = standarize_df(jungle_games)
    results_dict_jungle = get_best_clustering(x_jungle, pca_params = [0.95, 0.90, 0.85], umap_params = [2, 3, 4, 5, 6, 7, 8, 9], kmeans_params = [3, 4, 5, 6, 7, 8, 9], optics_params = [3, 4, 5, 6, 7, 8, 9])
    # print("results_dict_jungle:")
    # print(results_dict_jungle)
    jungle, top_champions_list, top_principal_components = umap_kmeans(x_jungle, y_jungle, n_comps=2, k = 8 )
    # y_jungle.to_excel("../data/competitive/clustering/jungle_clustering.xlsx")

    ########### Mid stats ###########
    x_mid, y_mid = standarize_df(mid_games)
    results_dict_mid = get_best_clustering(x_mid, pca_params = [0.95, 0.90, 0.85], umap_params = [2, 3, 4, 5, 6, 7, 8, 9], kmeans_params = [3, 4, 5, 6, 7, 8, 9], optics_params = [3, 4, 5, 6, 7, 8, 9])
    # print("results_dict_mid:")
    # print(results_dict_mid)
    mid, top_champions_list, top_principal_components = umap_kmeans(x_mid, y_mid, n_comps=2, k = 5 )
    # y_mid.to_excel("../data/competitive/clustering/mid_clustering.xlsx")

    ########### Adc stats ###########
    x_adc, y_adc = standarize_df(adc_games)
    results_dict_adc = get_best_clustering(x_adc, pca_params = [0.95, 0.90, 0.85], umap_params = [2, 3, 4, 5, 6, 7, 8, 9], kmeans_params = [3, 4, 5, 6, 7, 8, 9], optics_params = [3, 4, 5, 6, 7, 8, 9])
    # print("results_dict_adc:")
    # print(results_dict_adc)
    adc, top_champions_list, top_principal_components = umap_kmeans(x_adc, y_adc, n_comps=2, k = 4 )
    # y_adc.to_excel("../data/competitive/clustering/adc_clustering.xlsx")

    ########### Support stats ###########
    x_support, y_support = standarize_df(support_games)
    results_dict_support = get_best_clustering(x_support, pca_params = [0.95, 0.90, 0.85], umap_params = [2, 3, 4, 5, 6, 7, 8, 9], kmeans_params = [3, 4, 5, 6, 7, 8, 9], optics_params = [3, 4, 5, 6, 7, 8, 9])
    # print("results_dict_support:")
    # print(results_dict_support)
    support, top_champions_list, top_principal_components = umap_kmeans(x_support, y_support, n_comps=2, k = 4 )
    # y_support.to_excel("../data/competitive/clustering/support_clustering.xlsx")

    # umap_optics(x_support, y_support, n_comps=2, min_samples = 5 )

    ####################################################################################################
    # Prep dataset with groups
    ####################################################################################################

    total_games_reduced = total_games_raw[["game_id", "teamId", "teamPosition", "win", "championName", "championId"]]
    top_competitive = total_games_reduced[total_games_reduced['teamPosition'] == "Top"]
    jungle_competitive = total_games_reduced[total_games_reduced['teamPosition'] == "Jungle"]
    mid_competitive = total_games_reduced[total_games_reduced['teamPosition'] == "Mid"]
    adc_competitive = total_games_reduced[total_games_reduced['teamPosition'] == "Adc"]
    support_competitive = total_games_reduced[total_games_reduced['teamPosition'] == "Support"]



    w_top_competitive = insert_group(total_games_reduced, top, "Top", "w_top_group", True)[["game_id", "win", "teamId", "teamPosition", "w_top_group"]]
    l_top_competitive = insert_group(total_games_reduced, top, "Top", "l_top_group", False)[["game_id", "win", "teamId", "teamPosition", "l_top_group"]]

    w_jungle_competitive = insert_group(total_games_reduced, jungle, "Jungle", "w_jungle_group", True)[["game_id", "win", "teamId", "teamPosition", "w_jungle_group"]]
    l_jungle_competitive = insert_group(total_games_reduced, jungle, "Jungle", "l_jungle_group", False)[["game_id", "win", "teamId", "teamPosition", "l_jungle_group"]]

    w_mid_competitive = insert_group(total_games_reduced, mid, "Mid", "w_mid_group", True)[["game_id", "win", "teamId", "teamPosition", "w_mid_group"]]
    l_mid_competitive = insert_group(total_games_reduced, mid, "Mid", "l_mid_group", False)[["game_id", "win", "teamId", "teamPosition", "l_mid_group"]]

    w_adc_competitive = insert_group(total_games_reduced, adc, "Adc", "w_adc_group", True)[["game_id", "win", "teamId", "teamPosition", "w_adc_group"]]
    l_adc_competitive = insert_group(total_games_reduced, adc, "Adc", "l_adc_group", False)[["game_id", "win", "teamId", "teamPosition", "l_adc_group"]]

    w_support_competitive = insert_group(total_games_reduced, support, "Support", "w_support_group", True)[["game_id", "win", "teamId", "teamPosition", "w_support_group"]]
    l_support_competitive = insert_group(total_games_reduced, support, "Support", "l_support_group", False)[["game_id", "win", "teamId", "teamPosition", "l_support_group"]]

    total_top_competitive = insert_all_groups(total_games_reduced, top, "Top", "top_group")[["game_id", "win", "teamId", "teamPosition", "top_group"]]
    total_top_competitive.to_excel("top_groups.xlsx")
    
    

    top_competitive = pd.merge(left=w_top_competitive, right=l_top_competitive, on="game_id")[["game_id", "teamId_x", "w_top_group", "l_top_group"]]
    jungle_competitive = pd.merge(left=w_jungle_competitive, right=l_jungle_competitive, left_on="game_id", right_on="game_id")[["game_id", "w_jungle_group", "l_jungle_group"]]
    mid_competitive = pd.merge(left=w_mid_competitive, right=l_mid_competitive, left_on="game_id", right_on="game_id")[["game_id", "w_mid_group", "l_mid_group"]]
    adc_competitive = pd.merge(left=w_adc_competitive, right=l_adc_competitive, left_on="game_id", right_on="game_id")[["game_id", "w_adc_group", "l_adc_group"]]
    support_competitive = pd.merge(left=w_support_competitive, right=l_support_competitive, left_on="game_id", right_on="game_id")[["game_id", "w_support_group", "l_support_group"]]

    dataframes = [top_competitive, jungle_competitive, mid_competitive, adc_competitive, support_competitive]
    grouped_competitive = reduce(lambda left, right: pd.merge(left, right, on="game_id"), dataframes)


    grouped_competitive = grouped_competitive.drop_duplicates("game_id")
    grouped_competitive['winner_side'] = grouped_competitive.apply(lambda row: 100 if row['teamId_x'] == "Blue" else 200, axis=1)
    grouped_competitive = grouped_competitive.drop("teamId_x", axis=1)
    grouped_competitive = grouped_competitive[["game_id", "winner_side", 
                                            "w_top_group", 
                                            "w_jungle_group", 
                                            "w_mid_group", 
                                            "w_adc_group", 
                                            "w_support_group",
                                            "l_top_group",                               
                                            "l_jungle_group",                               
                                            "l_mid_group",                               
                                            "l_adc_group",                               
                                            "l_support_group"]]

    grouped_competitive.to_excel("./data/competitive/games_with_cluster_by_lane.xlsx")


if __name__ == '__main__':
    main()

