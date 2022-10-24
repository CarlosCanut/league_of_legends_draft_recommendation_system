import mwclient
import pandas as pd
import pprint
from datetime import date as date_x
from datetime import timedelta
import time
from collections import OrderedDict
import json
import logging

from riot_helpers import get_timeline_stats, get_postgame_stats_v5, get_postgame_stats_v4


##############################
def main():
    leagues = [
    "LCS 2022 Spring",
    "LCS 2022 Spring Playoffs",
    "LCS 2022 Summer",
    "LCS 2022 Championship",
    "NA Academy 2022 Spring",
    "LCS Proving Grounds 2022 Spring",
    "NA Academy 2022 Summer",
    "LCS Proving Grounds 2022 Summer",
    "LCS Proving Grounds 2022 CQ 2",
    "CBLOL 2022 Split 1",
    "CBLOL 2022 Split 1 Playoffs",
    "CBLOL 2022 Split 2",
    "CBLOL 2022 Split 2 Playoffs",
    "LLA 2022 Opening Promotion",
    "LLA 2022 Opening",
    "LLA 2022 Opening Playoffs",
    "LLA 2022 Closing",
    "LLA 2022 Closing Playoffs",
    "LCO 2022 Split 1",
    "LCO 2022 Split 1 Playoffs",
    "LCO 2022 Split 2",
    "LCO 2022 Split 2 Playoffs",
    "LVP SL 2022 Spring",
    "LVP SL 2022 Spring Playoffs",
    "LVP SL 2022 Summer",
    "LVP SL 2022 Summer Playoffs",
    "LFL 2022 Spring",
    "LFL 2022 Spring Playoffs",
    "LFL 2022 Summer",
    "LFL 2022 Summer Playoffs",
    "PRM 1st Division 2022 Spring",
    "PRM 1st Division 2022 Spring Playoffs",
    "PRM 1st Division 2022 Summer",
    "PRM 1st Division 2022 Summer Playoffs",
    "EM 2022 Spring Play-In",
    "EM 2022 Spring Main Event",
    "European Masters 2022 Summer",
    "EM 2022 Summer Play-In",
    "EM 2022 Summer Main Event",
    "LEC 2022 Spring",
    "LEC 2022 Spring Playoffs",
    "LEC 2022 Summer",
    "LEC 2022 Summer Playoffs",
    "LCK 2022 Spring",
    "LCK 2022 Spring Playoffs",
    "LCK 2022 Summer",
    "LCK 2022 Summer Playoffs",
    "LCK 2022 Regional Finals",
    "LCK CL 2022 Spring",
    "LCK CL 2022 Spring Playoffs",
    "LCK CL 2022 Summer",
    "LCK CL 2022 Summer Playoffs",
    "LCK AS 2022 1st Championship",
    "LCK AS 2022 1st Championship Playoffs",
    "LCK AS 2022 2nd Championship",
    "LCK AS 2022 2nd Championship Playoffs",
    "LPL 2022 Spring",
    "LPL 2022 Spring Playoffs",
    "LPL 2022 Summer",
    "LPL 2022 Summer Playoffs",
    "LPL 2022 Regional Finals",
    "LDL 2022 Spring",
    "LDL 2022 Spring Playoffs",
    "LDL 2022 Summer",
    "LDL 2022 Summer Playoffs",
    "LJL 2022 Spring",
    "LJL 2022 Spring Playoffs",
    "LJL 2022 Summer",
    "LJL 2022 Summer Playoffs",
    "TCL 2022 Winter",
    "TCL 2022 Winter Playoffs",
    "TCL 2022 Summer",
    "TCL 2022 Summer Playoffs",
    "Turkey Academy 2022 Winter",
    "VCS 2022 Spring",
    "VCS 2022 Spring Playoffs",
    "VCS 2022 Summer Promotion",
    "VCS 2022 Summer",
    "VCS 2022 Summer Playoffs",
    "PCS 2022 Spring",
    "PCS 2022 Spring Playoffs",
    "PCS 2022 Summer",
    "PCS 2022 Summer Playoffs",
    "NLC 2022 Spring",
    "NLC 2022 Spring Playoffs"
    "NLC 2022 Summer",
    "NLC 2022 Summer Playoffs",
    "NLC 2nd Division 2022 Spring",
    "EM 2022 Summer Main Event",
    "EM 2022 Summer Play-In",    
    "LFL Division 2 2022 Spring",
    "LFL Division 2 2022 Spring Playoffs",
    "LFL Division 2 2022 Summer",
    "LFL Division 2 2022 Summer Playoffs",
    "LFL Division 2 2023 Promotion",    
    "LVP SL 2nd Division 2022 Spring",
    "LVP SL 2nd Division 2022 Spring Playoffs",
    "LVP SL 2nd Division 2022 Summer",
    "LVP SL 2nd Division 2022 Summer Playoffs",
    "LVP SL 2nd Division 2023 Promotion",
    "Liga Nexo 2022 Split 1",
    "Liga Nexo 2022 Split 1 Playoffs",
    "Liga Nexo 2022 Relegations",    
    "PG Nationals 2022 Spring",
    "PG Nationals 2022 Spring Playoffs",
    "PG Nationals 2022 Summer",
    "PG Nationals 2022 Summer Playoffs",
    "PG Nationals 2023 Promotion",   
    "PG Proving Grounds 2022 Spring Playoffs",
    "PG Proving Grounds 2022 Spring", 
    "Ultraliga Season 7",
    "Ultraliga Season 7 Playoffs",
    "Ultraliga Season 8",
    "Ultraliga Season 8 Playoffs",
    "Ultraliga Season 9 Promotion",    
    "LPLOL 2022 Spring",
    "LPLOL 2022 Spring Playoffs",
    "LPLOL 2022 Summer",
    "LPLOL 2022 Summer Playoffs",
    "LPLOL 2023 Promotion",    
    "PRM 2nd Division 2022 Spring",
    "PRM 2nd Division 2022 Spring Playoffs",
    "PRM 2nd Division 2022 Summer Promotion",
    "PRM 2nd Division 2022 Summer",
    "PRM 2nd Division 2022 Summer Playoffs",
    "PRM 2nd Division 2023 Spring Promotion"
    "GLWL 2022 Spring Playoffs",
    "GLWL 2022 Spring",
    "LMF 2022 Opening",
    "LMF 2022 Opening Playoffs",
    "LMF 2022 Closing",
    "LMF 2022 Closing Playoffs",
    "VL 2022 Opening",
    "VL 2022 Opening Playoffs",
    "VL 2022 Closing",
    "VL 2022 Closing Playoffs",
    "VL 2022 Super Final",
    "UEM 2022 European League",
    "GLL 2022 Spring",
    "GLL 2022 Spring Playoffs",
    "GLL 2022 Summer",
    "GLL 2022 Summer Playoffs",
    "Dutch Tour 2022 Spring Split",
    "Dutch Tour 2022 Spring Playoffs",
    "Dutch Tour 2022 Summer Qualifiers",
    "Dutch Tour 2022 Summer",
    "Dutch Tour 2022 Summer Playoffs",
    "DDH 2022 Opening",
    "DDH 2022 Opening Playoffs",
    "DDH 2022 Closing",
    "DDH 2022 Closing Playoffs",
    "Elite Series 2022 Spring",
    "Elite Series 2022 Spring Playoffs",
    "Elite Series 2022 Summer",
    "Elite Series 2022 Summer Playoffs",
    "Elite Series 2022 Pro-Am"
    "Belgian Tour 2022 Spring Split",
    "Belgian Tour 2022 Spring Playoffs",
    "Belgian Tour 2022 Summer Qualifiers",
    "Belgian Tour 2022 Summer",
    "Belgian Tour 2022 Summer Playoffs",
    "Luxembourg Tour 2022 Spring",
    "Luxembourg Tour 2022 Spring Playoffs",
    "Luxembourg Tour 2022 Summer Qualifiers",
    "Luxembourg Tour 2022 Summer",
    "Luxembourg Tour 2022 Summer Playoffs",
    "EBL 2022 Spring",
    "EBL 2022 Spring Playoffs",
    "EBL 2022 Summer",
    "EBL 2022 Summer Playoffs",
    # "EBL 2022 Pro-Am",
    "GL 2022 Opening",
    "GL 2022 Opening Playoffs",
    "GL 2022 Closing",
    "GL 2022 Closing Playoffs",
    "Regional Norte 2022",
    "Regional Sur 2022",
    "EL 2022 Opening",
    "EL 2022 Opening Playoffs",
    "EL 2022 Closing",
    "EL 2022 Closing Playoffs",
    "LHE 2022 Opening",
    "LHE 2022 Opening Playoffs",
    "LHE 2022 Closing",
    "LHE 2022 Closing Playoffs",
    "SL 2022 Opening",
    "SL 2022 Opening Playoffs",
    "SL 2022 Closing",
    "SL 2022 Closing Playoffs",
    "LCL 2022 Spring",
    "Hitpoint Masters 2022 Spring",
    "Hitpoint Masters 2022 Spring Playoffs",
    "Hitpoint Masters 2022 Summer",
    "Hitpoint Masters 2022 Summer Playoffs",
    "LPLOL 2nd Division 2022 Spring",
    "LPLOL 2nd Division 2022 Spring Playoffs",
    "LPLOL 2nd Division 2022 Summer",
    "LPLOL 2nd Division 2022 Summer Playoffs",
    "Sea Games 31 - Vietnam Qualifier",
    "Sea Games 31"
    ]
    # league = "TCL 2022 Winter"
    site = mwclient.Site('lol.fandom.com',path='/')
    ##############################


    ############################################################
    ###################### Champions ID ########################
    ############################################################
    champions_response = site.api('cargoquery',
        limit = 'max',
        tables = "Champions=C",
        fields = "C.Name , C.KeyInteger"
    )
    champions = {}
    for champion in champions_response['cargoquery']:
        champions[champion['title']['Name']] = champion['title']['KeyInteger']

    def get_champ_id(row, column):
        try:
            return champions[row[column]]
        except:
            return ""

    total_leagues_games = []
    for league in leagues:
        ############################################################
        ######################## Patches ###########################
        ############################################################
        try:
            print(league)
            last_date = "2000-01-01"
            old_date = "0"

            patch_response = []
            while True:
                new_response = site.api('cargoquery',
                    limit = 'max',
                    tables = "Tournaments=T, ScoreboardTeams=ST, ScoreboardGames=SG",
                    join_on = "ST.OverviewPage = T.OverviewPage, ST.GameId = SG.GameId",
                    fields = "ST.GameId, SG.Patch, SG.DateTime_UTC",
                    where = "SG.DateTime_UTC > '" + str(last_date) + "' AND T.Name = '" + league + "'",
                    order_by = "SG.DateTime_UTC"
                )
                if old_date == last_date:
                    break
                old_date = last_date
                patch_response = (patch_response + list(new_response['cargoquery']))
                last_date = str(new_response['cargoquery'][-1]['title']['DateTime UTC']).split(" ")[0]


            patches = {}
            for match in patch_response:
                new_patch = str(match['title']['Patch']).replace(",",".")
                patches[match['title']['GameId']] = new_patch


            def get_patch(row):
                return patches[row['GameId']]

            ############################################################
            ############### Player's data extraction ###################
            ############################################################
            last_date = "2000-01-01"
            old_date = "0"

            # player's query
            total_postgame_data = []
            players_response = []
            for x in range(25):
                new_response = site.api('cargoquery',
                    limit = 'max',
                    tables = "Tournaments=T, ScoreboardPlayers=SP",
                    join_on = "SP.OverviewPage = T.OverviewPage",
                    fields = "SP.OverviewPage, SP.GameTeamId, SP.GameId, SP.DateTime_UTC, SP.Link, SP.PlayerWin, SP.Champion, SP.Role, SP.Side , SP.Runes, SP.Team, SP.TeamVs, SP.KeystoneMastery, SP.KeystoneRune, SP.PrimaryTree, SP.SecondaryTree, SP.Items, SP.Trinket, SP.SummonerSpells",
                    where = "SP.DateTime_UTC > '" + str(last_date) + "' AND T.name='" + str(league) + "'",
                    order_by = "SP.DateTime_UTC"
                )
                if old_date == last_date:
                    break
                old_date = last_date
                players_response = (players_response + list(new_response['cargoquery']))
                last_date = str(new_response['cargoquery'][-1]['title']['DateTime UTC']).split(" ")[0]


            total_games = [ game['title'] for game in players_response]
            league_games_df = pd.DataFrame(total_games)



            stats_page_games = []
            unique_league_games = pd.unique(league_games_df['GameId'])
            for game_id in unique_league_games:
                
                match_response = site.api(
                    action = 'cargoquery',
                    limit = 'max',
                    tables = "MatchScheduleGame=MSG, PostgameJsonMetadata=PJM",
                    fields = "MSG.RiotPlatformGameId, MSG.GameId, MSG.Blue, MSG.Red, PJM.StatsPage, PJM.TimelinePage",
                    where= "MSG.GameId='" + game_id + "'",
                    join_on = "MSG.RiotPlatformGameId = PJM.RiotPlatformGameId"
                )
                
                game_info = match_response['cargoquery'][0]
                stats_page_games.append(game_info)

            stats_games = [ game['title'] for game in stats_page_games]
            stats_page_games_df = pd.DataFrame(stats_games)
                
            games_df = stats_page_games_df.merge(league_games_df, how="inner", on="GameId")
            # games_df.to_excel("test.xlsx")

            for index, unique_game in (games_df.drop_duplicates(subset=["GameId"])).iterrows():
                query_titles = (unique_game['TimelinePage'] + "|" + unique_game['StatsPage'])
                stats = site.api(
                    action = "query",
                    format = "json",
                    prop = "revisions",
                    titles = query_titles,
                    rvprop = "content",
                    rvslots = "main"
                )
                timeline_data = {}
                postgame_data = {}
                for page in stats['query']['pages']:
                    pages = stats['query']['pages']
                    title = pages[page]['title']
                    try:
                        data = json.loads(pages[page]['revisions'][0]['slots']['main']['*'])
                        data['blue_team'] = unique_game['Blue']
                        data['red_team'] = unique_game['Red']
                        data['game_id'] = unique_game['RiotPlatformGameId']
                        data['league'] = unique_game['OverviewPage']
                        if "Timeline" in title:
                            timeline_data = get_timeline_stats(data)
                        else:
                            if "V5" in title:
                                postgame_data = get_postgame_stats_v5(data)
                            elif "V4" in title:
                                postgame_data = get_postgame_stats_v4(data)
                    except Exception as e:
                        print("Error: ",e)
                
                
                final_data = []
                if timeline_data:
                    for post_game_stats_participant in postgame_data:
                        participant_stats = post_game_stats_participant
                        participant_stats['cs_diff_15'] = timeline_data[str(post_game_stats_participant['participantId'])]['cs_diff_15']
                        participant_stats['dmg_diff_15'] = timeline_data[str(post_game_stats_participant['participantId'])]['dmg_diff_15']
                        participant_stats['gold_diff_15'] = timeline_data[str(post_game_stats_participant['participantId'])]['gold_diff_15']
                        participant_stats['xp_diff_15'] = timeline_data[str(post_game_stats_participant['participantId'])]['xp_diff_15']
                        
                        participant_stats['cs_team_15'] = timeline_data[str(post_game_stats_participant['participantId'])]['cs_team_15']
                        participant_stats['cs_share_15'] = timeline_data[str(post_game_stats_participant['participantId'])]['cs_share_15']
                        
                        participant_stats['gold_team_15'] = timeline_data[str(post_game_stats_participant['participantId'])]['gold_team_15']
                        participant_stats['gold_share_15'] = timeline_data[str(post_game_stats_participant['participantId'])]['gold_share_15']
                        
                        participant_stats['dmg_team_15'] = timeline_data[str(post_game_stats_participant['participantId'])]['dmg_team_15']
                        participant_stats['dmg_share_15'] = timeline_data[str(post_game_stats_participant['participantId'])]['dmg_share_15']
                        
                        final_data.append(participant_stats)
                else:
                    final_data = postgame_data
                    
                total_postgame_data = total_postgame_data + final_data
                total_leagues_games = total_leagues_games + final_data
                    
                    
            post_game_player_df = pd.DataFrame(total_postgame_data)
            post_game_player_df.sort_values(['game_id', 'participantId'])
            output = league
            print(post_game_player_df)
            print()
            post_game_player_df.to_excel("../data/competitive/aux_datasets/" + output +".xlsx")
        except Exception as e:
            print(league + " - ERROR: ", e)
            print()

    total_leagues_games_df = pd.DataFrame(total_leagues_games)
    total_leagues_games_df.sort_values(['game_id', 'participantId'])
    total_leagues_games_df.to_excel("../data/competitive/total_games.xlsx")


if __name__ == '__main__':
    main()