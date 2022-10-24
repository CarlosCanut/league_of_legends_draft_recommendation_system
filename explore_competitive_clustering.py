from helpers import *


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

x_jungle, y_jungle = standarize_df(jungle_games)
x_mid, y_mid = standarize_df(mid_games)
x_adc, y_adc = standarize_df(adc_games)
x_support, y_support = standarize_df(support_games)

jungle_results_dict = get_best_clustering(x_jungle, pca_params = [0.95, 0.90, 0.85], umap_params = [2, 3, 4, 5, 6, 7, 8, 9], kmeans_params = [3, 4, 5, 6, 7, 8, 9], optics_params = [3, 4, 5, 6, 7, 8, 9])
jungle_results_dict.to_excel("../jungle_cluster_models.xlsx")

mid_results_dict = get_best_clustering(x_mid, pca_params = [0.95, 0.90, 0.85], umap_params = [2, 3, 4, 5, 6, 7, 8, 9], kmeans_params = [3, 4, 5, 6, 7, 8, 9], optics_params = [3, 4, 5, 6, 7, 8, 9])
mid_results_dict.to_excel("../mid_cluster_models.xlsx")

adc_results_dict = get_best_clustering(x_adc, pca_params = [0.95, 0.90, 0.85], umap_params = [2, 3, 4, 5, 6, 7, 8, 9], kmeans_params = [3, 4, 5, 6, 7, 8, 9], optics_params = [3, 4, 5, 6, 7, 8, 9])
adc_results_dict.to_excel("../adc_cluster_models.xlsx")

supp_results_dict = get_best_clustering(x_support, pca_params = [0.95, 0.90, 0.85], umap_params = [2, 3, 4, 5, 6, 7, 8, 9], kmeans_params = [3, 4, 5, 6, 7, 8, 9], optics_params = [3, 4, 5, 6, 7, 8, 9])
supp_results_dict.to_excel("../supp_cluster_models.xlsx")

