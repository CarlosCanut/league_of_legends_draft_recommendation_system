
from helpers import *

########### main() ###########

def main():
    
    ####################################################################################################
    # Data Cleaning
    ####################################################################################################
    soloq_games_euw = pd.read_csv("./data/soloq/Europe_stats.csv")
    soloq_games_kr = pd.read_csv("./data/soloq/Asia_stats.csv")

    soloq_games = pd.concat([soloq_games_euw, soloq_games_kr])

    soloq_games = soloq_games.dropna()
    # delete games with < 15 mins
    soloq_games = soloq_games[soloq_games['gameEndedInEarlySurrender'] == False]
    # creates a patch column
    soloq_games['patch'] = soloq_games.apply(lambda x: str(x['gameVersion'].split('.')[0] + '.' + x['gameVersion'].split('.')[1]), axis=1 )

    relevant_cols = [
        "patch", "teamPosition", "championId", "championName", "gameDuration", "win",
        "neutralMinionsKilled", "totalMinionsKilled", "cs_diff_at_15",
        "champExperience", "xp_diff", "xp_diff_per_min", "xp_per_min_3_15",
        "damageDealtToBuildings", "damageDealtToObjectives", "damageDealtToTurrets", "damageSelfMitigated", "magicDamageDealt", "magicDamageDealtToChampions", "magicDamageTaken",
        "physicalDamageDealt", "physicalDamageDealtToChampions", "physicalDamageTaken", "totalDamageDealt", "totalDamageDealtToChampions", "totalDamageShieldedOnTeammates",
        "totalDamageTaken", "totalHeal", "totalHealsOnTeammates", "totalUnitsHealed", "trueDamageDealt", "trueDamageDealtToChampions", "trueDamageTaken",
        "totalTimeCCDealt", "timeCCingOthers", "totalTimeSpentDead", "dmg_per_minute_diff", "dmg_per_minute_diff_15", "kills", "deaths", "assists", "kill_share", "kill_participation",
        "doubleKills", "tripleKills", "quadraKills", "pentaKills", "firstBloodAssist", "firstBloodKill", "killingSprees", "largestKillingSpree", "largestMultiKill",
        "goldEarned", "goldSpent", "gold_share", "gold_earned_per_min", "gold_diff_15", "gold_10k_time",
        "inhibitorKills", "inhibitorTakedowns", "inhibitorsLost", 
        "itemsPurchased", "consumablesPurchased",
        "largestCriticalStrike", "longestTimeSpentLiving",
        "firstTowerAssist", "firstTowerKill", "objectivesStolen", "objectivesStolenAssists", "turretKills", "turretTakedowns", "turretsLost",
        "sightWardsBoughtInGame", "visionScore", "visionWardsBoughtInGame", "detectorWardsPlaced", "wardsKilled", "wardsPlaced",
        "spell1Casts", "spell2Casts", "spell3Casts", "spell4Casts", "summoner1Casts", "summoner2Casts",
        "lane_proximity", "jungle_proximity", "percent_mid_lane", "percent_side_lanes", "forward_percentage", "counter_jungle_time_percentage",
    ]


    # select only relevant cols
    soloq = soloq_games[ relevant_cols ]
    
    general_soloq = clean_data(soloq, role="None", patch="12.5", stratified_sampling = False)

    top_soloq = clean_data(soloq, role="TOP", patch="12.5", stratified_sampling = False)
    jungle_soloq = clean_data(soloq, role="JUNGLE", patch="12.5", stratified_sampling = False)
    mid_soloq = clean_data(soloq, role="MIDDLE", patch="12.5", stratified_sampling = False)
    bottom_soloq = clean_data(soloq, role="BOTTOM", patch="12.5", stratified_sampling = False)
    utility_soloq = clean_data(soloq, role="UTILITY", patch="12.5", stratified_sampling = False)
    
    # general_soloq.to_csv("../data/soloq/clean/general_soloq.csv")
    # top_soloq.to_csv("../data/soloq/clean/top_soloq.csv")
    # jungle_soloq.to_csv("../data/soloq/clean/jungle_soloq.csv")
    # mid_soloq.to_csv("../data/soloq/clean/mid_soloq.csv")
    # bottom_soloq.to_csv("../data/soloq/clean/bottom_soloq.csv")
    # utility_soloq.to_csv("../data/soloq/clean/utility_soloq.csv")
    
    ####################################################################################################
    # Clustering
    ####################################################################################################
    
    ########### General stats ###########
    x_general, y_general = standarize_df(general_soloq)
    # results_dict = get_best_clustering(
    #                                     x_general, 
    #                                     pca_params = [0.95, 0.90, 0.85], 
    #                                     umap_params = [2, 3, 4, 5, 6, 7, 8, 9, 10], 
    #                                     kmeans_params = [2, 3, 4, 5, 6, 7, 8, 9, 10], 
    #                                     optics_params = [2, 3, 4, 5, 6, 7, 8, 9, 10])

    x_general, y_general = standarize_df(general_soloq)
    y_general, general_champions_list, general_principal_components = umap_kmeans(x_general, y_general, n_comps=4, k = 4 )
    general = y_general
    # y_general.to_excel("../data/soloq/clustering/general_clustering.xlsx")
    
    ########### Top stats ###########
    x_top, y_top = standarize_df(top_soloq)
    # top_results_dict = get_best_clustering(
    #                                         x_top, 
    #                                         pca_params = [0.85], 
    #                                         umap_params = [2, 3, 4, 5, 6, 7, 8, 9], 
    #                                         kmeans_params = [3, 4, 5, 6, 7, 8, 9, 10], 
    #                                         optics_params = [3, 4, 5])
    # print("top_results_dict")
    # print(top_results_dict)

    x_top, y_top = standarize_df(top_soloq)
    y_top, top_champions_list, top_principal_components = umap_optics(x_top, y_top, n_comps=4, min_samples = 5 )
    top = y_top
    # y_top.to_excel("../data/soloq/clustering/top_clustering.xlsx")
    
    ########### Jungle stats ###########
    x_jungle, y_jungle = standarize_df(jungle_soloq)
    # jungle_results_dict = get_best_clustering(
    #                                             x_jungle, 
    #                                             pca_params = [0.85], 
    #                                             umap_params = [2, 3, 4, 5, 6, 7, 8, 9], 
    #                                             kmeans_params = [3, 4, 5, 6, 7, 8, 9, 10], 
    #                                             optics_params = [3, 4, 5])
    # print("jungle_results_dict")
    # print(jungle_results_dict)

    x_jungle, y_jungle = standarize_df(jungle_soloq)
    y_jungle, jungle_champions_list, jungle_principal_components = umap_kmeans(x_jungle, y_jungle, n_comps=2, k = 3 )
    jungle = y_jungle
    # y_jungle.to_excel("../data/soloq/clustering/jungle_clustering.xlsx")
    
    ########### Mid stats ###########
    x_mid, y_mid = standarize_df(mid_soloq)
    # mid_results_dict = get_best_clustering(
    #                                         x_mid, 
    #                                         pca_params = [0.85, 0.9], 
    #                                         umap_params = [2, 3, 4, 5, 6, 7, 8], 
    #                                         kmeans_params = [3, 4, 5, 6, 7, 8, 9, 10], 
    #                                         optics_params = [3, 4, 5, 6, 7])
    # print("mid_results_dict")
    # print(mid_results_dict)

    x_mid, y_mid = standarize_df(mid_soloq)
    y_mid, mid_champions_list, mid_principal_components = umap_kmeans(x_mid, y_mid, n_comps=2, k = 4 )
    mid = y_mid
    # y_mid.to_excel("../data/soloq/clustering/mid_clustering.xlsx")
    
    ########### Adc stats ###########
    x_bottom, y_bottom = standarize_df(bottom_soloq)
    # bottom_results_dict = get_best_clustering(
    #                                             x_bottom, 
    #                                             pca_params = [0.85, 0.9], 
    #                                             umap_params = [2, 3, 4, 5, 6, 7, 8], 
    #                                             kmeans_params = [3, 4, 5, 6, 7, 8, 9], 
    #                                             optics_params = [3, 4])
    # print("bottom_results_dict")
    # print(bottom_results_dict)

    x_bottom, y_bottom = standarize_df(bottom_soloq)
    y_bottom, bottom_champions_list, bottom_principal_components = umap_kmeans(x_bottom, y_bottom, n_comps=2, k = 3 )
    bottom = y_bottom
    # y_bottom.to_excel("../data/soloq/clustering/bottom_clustering.xlsx")
    
    ########### Support stats ###########
    x_utility, y_utility = standarize_df(utility_soloq)
    # utility_results_dict = get_best_clustering(
    #                                             x_utility, 
    #                                             pca_params = [0.85], 
    #                                             umap_params = [2, 3, 4, 5, 6, 7, 8], 
    #                                             kmeans_params = [3, 4, 5, 6, 7, 8, 9, 10], 
    #                                             optics_params = [3, 4, 5, 6, 7, 8, 9])
    # print("utility_results_dict")
    # print(utility_results_dict)

    x_utility, y_utility = standarize_df(utility_soloq)
    y_utility, utility_champions_list, utility_principal_components = umap_optics(x_utility, y_utility, n_comps=7, min_samples = 9 )
    utility = y_utility
    # y_utility.to_excel("../data/soloq/clustering/utility_clustering.xlsx")
    
    
    ####################################################################################################
    # Grouped data
    ####################################################################################################
    clean_soloq_games = soloq_games[["game_id", "teamId", "teamPosition", "win", "championName", "championId"]]

    
    w_top_soloq = insert_group_by_result(clean_soloq_games, top, "TOP", "w_top_group", True)[["game_id", "win", "teamId", "teamPosition", "w_top_group"]]
    l_top_soloq = insert_group_by_result(clean_soloq_games, top, "TOP", "l_top_group", False)[["game_id", "win", "teamId", "teamPosition", "l_top_group"]]
    blue_top_soloq = insert_group_by_side(clean_soloq_games, top, "TOP", "blue_top_group", "Blue")[["game_id", "team_win", "win", "teamId", "teamPosition", "blue_top_group"]]
    red_top_soloq = insert_group_by_side(clean_soloq_games, top, "TOP", "red_top_group", "Red")[["game_id", "team_win", "win", "teamId", "teamPosition", "red_top_group"]]

    w_jungle_soloq = insert_group_by_result(clean_soloq_games, jungle, "JUNGLE", "w_jungle_group", True)[["game_id", "win", "teamId", "teamPosition", "w_jungle_group"]]
    l_jungle_soloq = insert_group_by_result(clean_soloq_games, jungle, "JUNGLE", "l_jungle_group", False)[["game_id", "win", "teamId", "teamPosition", "l_jungle_group"]]
    blue_jungle_soloq = insert_group_by_side(clean_soloq_games, jungle, "JUNGLE", "blue_jungle_group", "Blue")[["game_id", "team_win", "win", "teamId", "teamPosition", "blue_jungle_group"]]
    red_jungle_soloq = insert_group_by_side(clean_soloq_games, jungle, "JUNGLE", "red_jungle_group", "Red")[["game_id", "team_win", "win", "teamId", "teamPosition", "red_jungle_group"]]

    w_mid_soloq = insert_group_by_result(clean_soloq_games, mid, "MIDDLE", "w_mid_group", True)[["game_id", "win", "teamId", "teamPosition", "w_mid_group"]]
    l_mid_soloq = insert_group_by_result(clean_soloq_games, mid, "MIDDLE", "l_mid_group", False)[["game_id", "win", "teamId", "teamPosition", "l_mid_group"]]
    blue_mid_soloq = insert_group_by_side(clean_soloq_games, mid, "MIDDLE", "blue_mid_group", "Blue")[["game_id", "team_win", "win", "teamId", "teamPosition", "blue_mid_group"]]
    red_mid_soloq = insert_group_by_side(clean_soloq_games, mid, "MIDDLE", "red_mid_group", "Red")[["game_id", "team_win", "win", "teamId", "teamPosition", "red_mid_group"]]

    w_bottom_soloq = insert_group_by_result(clean_soloq_games, bottom, "BOTTOM", "w_bottom_group", True)[["game_id", "win", "teamId", "teamPosition", "w_bottom_group"]]
    l_bottom_soloq = insert_group_by_result(clean_soloq_games, bottom, "BOTTOM", "l_bottom_group", False)[["game_id", "win", "teamId", "teamPosition", "l_bottom_group"]]
    blue_bottom_soloq = insert_group_by_side(clean_soloq_games, bottom, "BOTTOM", "blue_bottom_group", "Blue")[["game_id", "team_win", "win", "teamId", "teamPosition", "blue_bottom_group"]]
    red_bottom_soloq = insert_group_by_side(clean_soloq_games, bottom, "BOTTOM", "red_bottom_group", "Red")[["game_id", "team_win", "win", "teamId", "teamPosition", "red_bottom_group"]]

    w_utility_soloq = insert_group_by_result(clean_soloq_games, utility, "UTILITY", "w_utility_group", True)[["game_id", "win", "teamId", "teamPosition", "w_utility_group"]]
    l_utility_soloq = insert_group_by_result(clean_soloq_games, utility, "UTILITY", "l_utility_group", False)[["game_id", "win", "teamId", "teamPosition", "l_utility_group"]]
    blue_utility_soloq = insert_group_by_side(clean_soloq_games, utility, "UTILITY", "blue_utility_group", "Blue")[["game_id", "team_win", "win", "teamId", "teamPosition", "blue_utility_group"]]
    red_utility_soloq = insert_group_by_side(clean_soloq_games, utility, "UTILITY", "red_utility_group", "Red")[["game_id", "team_win", "win", "teamId", "teamPosition", "red_utility_group"]]
    
    
    ########### Create groups by wins ###########
    top_soloq = pd.merge(left=w_top_soloq, right=l_top_soloq, on="game_id")[["game_id", "teamId_x", "w_top_group", "l_top_group"]]
    jungle_soloq = pd.merge(left=w_jungle_soloq, right=l_jungle_soloq, left_on="game_id", right_on="game_id")[["game_id", "w_jungle_group", "l_jungle_group"]]
    mid_soloq = pd.merge(left=w_mid_soloq, right=l_mid_soloq, left_on="game_id", right_on="game_id")[["game_id", "w_mid_group", "l_mid_group"]]
    bottom_soloq = pd.merge(left=w_bottom_soloq, right=l_bottom_soloq, left_on="game_id", right_on="game_id")[["game_id", "w_bottom_group", "l_bottom_group"]]
    utility_soloq = pd.merge(left=w_utility_soloq, right=l_utility_soloq, left_on="game_id", right_on="game_id")[["game_id", "w_utility_group", "l_utility_group"]]

    dataframes = [top_soloq, jungle_soloq, mid_soloq, bottom_soloq, utility_soloq]
    grouped_soloq_by_result = reduce(lambda left, right: pd.merge(left, right, on="game_id"), dataframes)

    grouped_soloq_by_result = grouped_soloq_by_result.drop_duplicates("game_id")
    grouped_soloq_by_result['winner_side'] = grouped_soloq_by_result.apply(lambda row: 100 if row['teamId_x'] == "Blue" else 200, axis=1)
    grouped_soloq_by_result = grouped_soloq_by_result.drop("teamId_x", axis=1)
    grouped_soloq_by_result = grouped_soloq_by_result[["game_id", "winner_side", 
                                "w_top_group", 
                                "w_jungle_group", 
                                "w_mid_group", 
                                "w_bottom_group", 
                                "w_utility_group", 
                                "l_top_group",
                                "l_jungle_group",
                                "l_mid_group",
                                "l_bottom_group",
                                "l_utility_group"
                                ]]
    grouped_soloq_by_result.to_excel("./data/soloq/games_by_result_with_cluster_by_lane.xlsx")
    
    ########### Create groups by side ###########
    top_soloq_by_side = pd.merge(left=blue_top_soloq, right=red_top_soloq, on="game_id")[["game_id", "team_win_x", "blue_top_group", "red_top_group"]]
    top_soloq_by_side

    jungle_soloq_by_side = pd.merge(left=blue_jungle_soloq, right=red_jungle_soloq, left_on="game_id", right_on="game_id")[["game_id", "blue_jungle_group", "red_jungle_group"]]
    mid_soloq_by_side = pd.merge(left=blue_mid_soloq, right=red_mid_soloq, left_on="game_id", right_on="game_id")[["game_id", "blue_mid_group", "red_mid_group"]]
    bottom_soloq_by_side = pd.merge(left=blue_bottom_soloq, right=red_bottom_soloq, left_on="game_id", right_on="game_id")[["game_id", "blue_bottom_group", "red_bottom_group"]]
    utility_soloq_by_side = pd.merge(left=blue_utility_soloq, right=red_utility_soloq, left_on="game_id", right_on="game_id")[["game_id", "blue_utility_group", "red_utility_group"]]

    dataframes = [top_soloq_by_side, jungle_soloq_by_side, mid_soloq_by_side, bottom_soloq_by_side, utility_soloq_by_side]
    grouped_soloq_by_side = reduce(lambda left, right: pd.merge(left, right, on="game_id"), dataframes)
    grouped_soloq_by_side
    grouped_soloq_by_side = grouped_soloq_by_side.drop_duplicates("game_id")
    grouped_soloq_by_side['winner_side'] = grouped_soloq_by_side.apply(lambda row: 100 if row['team_win_x'] == "Blue" else 200, axis=1)
    grouped_soloq_by_side = grouped_soloq_by_side.drop("team_win_x", axis=1)
    grouped_soloq_by_side = grouped_soloq_by_side[["game_id", "winner_side", 
                                "blue_top_group", 
                                "blue_jungle_group", 
                                "blue_mid_group", 
                                "blue_bottom_group", 
                                "blue_utility_group", 
                                "red_top_group",
                                "red_jungle_group",
                                "red_mid_group",
                                "red_bottom_group",
                                "red_utility_group"
                                ]]
    grouped_soloq_by_side.to_excel("./data/soloq/games_by_side_with_cluster_by_lane.xlsx")
    
if __name__ == '__main__':
    main()
