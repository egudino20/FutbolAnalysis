# import necessary packages

# data manipulation and analysis
import pandas as pd
import numpy as np

# data visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
from mplsoccer import Pitch, VerticalPitch, FontManager
import matplotlib.colors as mcolors
from matplotlib.colors import to_rgba
from matplotlib.patches import ConnectionPatch
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize
import matplotlib.image as mpimg
import matplotlib.ticker as ticker
import matplotlib.patheffects as path_effects
import matplotlib.patches as patches
from matplotlib import rcParams
from matplotlib import cm

# web scraping
from selenium import webdriver

# text and annotation
from itertools import combinations
from highlight_text import fig_text, ax_text
from highlight_text import fig_text, ax_text, HighlightText
from adjustText import adjust_text

# machine learning / statiscal analysis
from scipy.stats import zscore, norm
import scipy.stats as st
from scipy.stats import poisson
from scipy.interpolate import interp1d, make_interp_spline
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp2d
import pickle
from itertools import islice

# embedded packages
import main
from main import insert_ball_carries
import visuals
from visuals import progressive_pass, progressive_carry, pass_into_box, carry_into_box
from visuals import calc_xg, calc_xa, data_preparation, data_preparation_xA, xThreat

# file handling and parsing
import os
import json

# other
import datetime

def player_advanced_stats(events_df, matches_data, comp):

    # apply progressive stat methods to calc each respective progressive stat
    events_df['progressivePass'] = events_df.apply(lambda x: progressive_pass(x, inplay=True, successful_only=True), axis=1)
    events_df['progressiveCarry'] = events_df.apply(lambda x: progressive_carry(x, successful_only=True), axis=1)
    events_df['carryIntoBox'] = events_df.apply(lambda x: carry_into_box(x, successful_only=True), axis=1)
    events_df['passIntoBox'] = events_df.apply(lambda x: pass_into_box(x, inplay=True, successful_only=True), axis=1)

    # this set of code withh loop through matches_data, pull aggregate stats for each player and append to a list of dfs for each match
    players_df = []

    for match in matches_data:
        # getting players dataframe
        match_players_df = pd.DataFrame()
        team_Ids = []
        opp_Ids = []
        team_names = []
        opp_names = []
        player_names = []
        player_ids = []
        player_pos = []
        player_kit_number = []
        player_eleven = []
        pass_count = []
        touch_count = []
        pass_completed = []
        shots = []
        shots_on_target = []
        interceptions = []
        shots_blocked = []
        fouls_committed = []
        dribbles_won = []
        dribbles_attempted = []
        dribbles_lost = []
        dispossessed = []
        clearances = []
        passes_key = []
        aerials = []
        aerials_won = []
        tackles = []
        tackles_successful = []
        tackle_success = []
        subbed_out_min = []
        subbed_in_min = []
        mins_played = []
        max_min = match['periodEndMinutes']['2']
        
        home_df = match['home']
        home_Id = home_df['teamId']
        home_team = home_df['name']
        
        away_df = match['away']
        away_team = away_df['name']
        away_Id = away_df['teamId']
        
        for player in home_df['players']:
            team_Ids.append(home_Id)
            opp_Ids.append(away_Id)
            team_names.append(home_team)
            opp_names.append(away_team)
            player_names.append(player['name'])
            player_ids.append(player['playerId'])
            player_pos.append(player['position'])
            player_kit_number.append(player['shirtNo'])
            try:
                player_eleven.append(player['isFirstEleven'])
            except KeyError:
                player_eleven.append(False)
            try:
                pass_count.append(sum(player['stats']['passesTotal'].values()))
            except KeyError:
                pass_count.append(0)
            try:
                touch_count.append(sum(player['stats']['touches'].values()))
            except KeyError:
                touch_count.append(0)
            try:
                pass_completed.append(sum(player['stats']['passesAccurate'].values()))
            except KeyError:
                pass_completed.append(0)
            try:
                shots.append(sum(player['stats']['shotsTotal'].values()))
            except KeyError:
                shots.append(0)
            try:
                shots_on_target.append(sum(player['stats']['shotsOnTarget'].values()))
            except KeyError:
                shots_on_target.append(0)
            try:
                interceptions.append(sum(player['stats']['interceptions'].values()))
            except KeyError:
                interceptions.append(0)
            try:
                shots_blocked.append(sum(player['stats']['shotsBlocked'].values()))
            except KeyError:
                shots_blocked.append(0)
            try:
                fouls_committed.append(sum(player['stats']['foulsCommitted'].values()))
            except KeyError:
                fouls_committed.append(0)
            try:
                clearances.append(sum(player['stats']['clearances'].values()))
            except KeyError:
                clearances.append(0)
            try:
                dribbles_won.append(sum(player['stats']['dribblesWon'].values()))
            except KeyError:
                dribbles_won.append(0)
            try:
                dribbles_attempted.append(sum(player['stats']['dribblesAttempted'].values()))
            except KeyError:
                dribbles_attempted.append(0)
            try:
                dribbles_lost.append(sum(player['stats']['dribblesLost'].values()))
            except KeyError:
                dribbles_lost.append(0)
            try:
                dispossessed.append(sum(player['stats']['dispossessed'].values()))
            except KeyError:
                dispossessed.append(0)
            try:
                passes_key.append(sum(player['stats']['passesKey'].values()))
            except KeyError:
                passes_key.append(0)
            try:
                aerials.append(sum(player['stats']['aerialsTotal'].values()))
            except KeyError:
                aerials.append(0)
            try:
                aerials_won.append(sum(player['stats']['aerialsWon'].values()))
            except KeyError:
                aerials_won.append(0)
            try:
                tackles.append(sum(player['stats']['tacklesTotal'].values()))
            except KeyError:
                tackles.append(0)
            try:
                tackles_successful.append(sum(player['stats']['tackleSuccessful'].values()))
            except KeyError:
                tackles_successful.append(0)
            try:
                tackle_success.append(sum(player['stats']['tackleSuccess'].values()))
            except KeyError:
                tackle_success.append(0)
            try:
                subbed_out_min.append(player['subbedOutExpandedMinute'])
            except KeyError:
                subbed_out_min.append(max_min)
            try:
                subbed_in_min.append(player['subbedInExpandedMinute'])
            except KeyError:
                subbed_in_min.append(0)
                
        for player in away_df['players']:
            team_Ids.append(away_Id)
            opp_Ids.append(home_Id)
            team_names.append(away_team)
            opp_names.append(home_team)
            player_names.append(player['name'])
            player_ids.append(player['playerId'])
            player_pos.append(player['position'])
            player_kit_number.append(player['shirtNo'])
            try:
                player_eleven.append(player['isFirstEleven'])
            except KeyError:
                player_eleven.append(False)
            try:
                pass_count.append(sum(player['stats']['passesTotal'].values()))
            except KeyError:
                pass_count.append(0)
            try:
                touch_count.append(sum(player['stats']['touches'].values()))
            except KeyError:
                touch_count.append(0)
            try:
                pass_completed.append(sum(player['stats']['passesAccurate'].values()))
            except KeyError:
                pass_completed.append(0)
            try:
                shots.append(sum(player['stats']['shotsTotal'].values()))
            except KeyError:
                shots.append(0)
            try:
                shots_on_target.append(sum(player['stats']['shotsOnTarget'].values()))
            except KeyError:
                shots_on_target.append(0)
            try:
                interceptions.append(sum(player['stats']['interceptions'].values()))
            except KeyError:
                interceptions.append(0)
            try:
                shots_blocked.append(sum(player['stats']['shotsBlocked'].values()))
            except KeyError:
                shots_blocked.append(0)
            try:
                fouls_committed.append(sum(player['stats']['foulsCommitted'].values()))
            except KeyError:
                fouls_committed.append(0)
            try:
                clearances.append(sum(player['stats']['clearances'].values()))
            except KeyError:
                clearances.append(0)
            try:
                dribbles_won.append(sum(player['stats']['dribblesWon'].values()))
            except KeyError:
                dribbles_won.append(0)
            try:
                dribbles_attempted.append(sum(player['stats']['dribblesAttempted'].values()))
            except KeyError:
                dribbles_attempted.append(0)
            try:
                dribbles_lost.append(sum(player['stats']['dribblesLost'].values()))
            except KeyError:
                dribbles_lost.append(0)
            try:
                dispossessed.append(sum(player['stats']['dispossessed'].values()))
            except KeyError:
                dispossessed.append(0)
            try:
                passes_key.append(sum(player['stats']['passesKey'].values()))
            except KeyError:
                passes_key.append(0)
            try:
                aerials.append(sum(player['stats']['aerialsTotal'].values()))
            except KeyError:
                aerials.append(0)
            try:
                aerials_won.append(sum(player['stats']['aerialsWon'].values()))
            except KeyError:
                aerials_won.append(0)
            try:
                tackles.append(sum(player['stats']['tacklesTotal'].values()))
            except KeyError:
                tackles.append(0)
            try:
                tackles_successful.append(sum(player['stats']['tackleSuccessful'].values()))
            except KeyError:
                tackles_successful.append(0)
            try:
                tackle_success.append(sum(player['stats']['tackleSuccess'].values()))
            except KeyError:
                tackle_success.append(0)
            try:
                subbed_out_min.append(player['subbedOutExpandedMinute'])
            except KeyError:
                subbed_out_min.append(max_min)
            try:
                subbed_in_min.append(player['subbedInExpandedMinute'])
            except KeyError:
                subbed_in_min.append(0)
            
        match_players_df['teamId'] = team_Ids  
        match_players_df['oppId'] = opp_Ids
        match_players_df['teamName'] = team_names
        match_players_df['oppName'] = opp_names
        match_players_df['playerId'] = player_ids
        match_players_df['playerName'] = player_names
        match_players_df['playerPos'] = player_pos
        match_players_df['playerKitNumber'] = player_kit_number
        match_players_df['isFirstEleven'] = player_eleven
        match_players_df['passCount'] = pass_count
        match_players_df['touchCount'] = touch_count
        match_players_df['passesCompleted'] = pass_completed
        match_players_df['shots'] = shots
        match_players_df['shotsOnTarget'] = shots_on_target
        match_players_df['interceptions'] = interceptions
        match_players_df['keyPasses'] = passes_key
        match_players_df['aerials'] = aerials
        match_players_df['aerialsWon'] = aerials_won
        match_players_df['tackles'] = tackles
        match_players_df['tacklesSuccessful'] = tackles_successful
        match_players_df['tackleSuccess'] = tackle_success
        match_players_df['clearances'] = clearances 
        match_players_df['subbedOutMin'] = subbed_out_min
        match_players_df['subbedInMin'] = subbed_in_min
        match_players_df['shotsBlocked'] = shots_blocked
        match_players_df['foulsCommitted'] = fouls_committed
        match_players_df['dribblesWon'] = dribbles_won
        match_players_df['dribblesAttempted'] = dribbles_attempted
        match_players_df['dribblesLost'] = dribbles_lost
        match_players_df['dispossessed'] = dispossessed
        match_players_df.loc[(match_players_df['isFirstEleven'] == False) & (match_players_df['subbedInMin'] == 0), 'subbedOutMin'] = 0
        match_players_df['timePlayed'] = match_players_df['subbedOutMin'] - match_players_df['subbedInMin']  
        
        players_df.append(match_players_df)

    # concatenate list of all matches to one single df of all stats to date
    df = pd.concat(players_df, axis=0)
    df = df.drop(columns=['subbedOutMin', 'subbedInMin'])

    # define a custom function to find the most common position
    def find_most_common_pos(x):
        try:
            # check if the most common position is "Sub"
            if x.value_counts().index[0] == 'Sub':
                # return the second most common position
                return x.value_counts().index[1]
            else:
                # return the most common position
                return x.value_counts().index[0]
        except IndexError:
            # handle the case where the Series has only one element
            return x.iloc[0]

    # group the dataframe by playerName and apply the custom function
    positions = df.groupby('playerName')['playerPos'].agg(find_most_common_pos).reset_index()
    teams = df.groupby(['playerName', 'teamName']).size().reset_index()

    # group by player and player/team ID with the total sum of each player's stat
    df = df.groupby(['playerName', 'playerId', 'teamId']).agg({'passCount': ['sum'],
                                                               'touchCount': ['sum'],
                                                               'passesCompleted': ['sum'],
                                                               'shots': ['sum'],
                                                               'shotsOnTarget': ['sum'],
                                                               'interceptions': ['sum'],
                                                               'keyPasses': ['sum'],
                                                               'tackles': ['sum'],
                                                               'tacklesSuccessful': ['sum'],
                                                               'timePlayed': ['sum'],
                                                               'shotsBlocked': ['sum'],
                                                               'clearances': ['sum'],
                                                               'aerials': ['sum'],
                                                               'aerialsWon': ['sum'],
                                                               'foulsCommitted': ['sum'],
                                                               'dribblesWon': ['sum'],
                                                               'dribblesAttempted': ['sum'],
                                                               'dribblesLost': ['sum'],
                                                               'dispossessed': ['sum']})

    # rename columns
    df.columns = ['passes', 'touches', 'passesComp', 'shots',
                  'shotsOnTarget', 'interceptions', 'keyPasses', 
                  'tackles', 'tacklesSuccessful', 'timePlayed',
                  'shotsBlocked', 'clearances', 'aerials', 'aerialsWon',
                  'foulsCommitted', 'dribblesWon', 'dribblesAttempted',
                  'dribblesLost', 'dispossessed']
    df = df.reset_index()

    # merge df with the positions and teams df to match each player's team and position
    df = pd.merge(positions, df, on='playerName', how='left')
    df = pd.merge(df, teams, on='playerName', how='left')
    # remove any players that have not played
    df = df[df['timePlayed'] > 0]

    # apply the same concept as above to group the players_df list by team names to get team total stats
    df_teams = pd.concat(players_df, axis=0)
    df_teams = df_teams.drop(columns=['subbedOutMin', 'subbedInMin'])

    # group the dataframe by teamName and Id
    df_teams = df_teams.groupby(['teamName', 'teamId']).agg({'passCount': ['sum'],
                                                             'touchCount': ['sum'],
                                                             'passesCompleted': ['sum'],
                                                             'shots': ['sum'],
                                                             'shotsOnTarget': ['sum'],
                                                             'interceptions': ['sum'],
                                                             'keyPasses': ['sum'],
                                                             'tackles': ['sum'],
                                                             'tacklesSuccessful': ['sum'],
                                                             'timePlayed': ['sum'],
                                                             'shotsBlocked': ['sum'],
                                                             'clearances': ['sum'],
                                                             'aerials': ['sum'],
                                                             'aerialsWon': ['sum'],
                                                             'foulsCommitted': ['sum'],
                                                             'dribblesWon': ['sum'],
                                                             'dribblesAttempted': ['sum'],
                                                             'dribblesLost': ['sum'],
                                                             'dispossessed': ['sum']})

    # rename columns
    df_teams.columns = ['passes', 'touches', 'passesComp', 'shots',
                        'shotsOnTarget', 'interceptions', 'keyPasses', 
                        'tackles', 'tacklesSuccessful', 'timePlayed',
                        'shotsBlocked', 'clearances', 'aerials', 'aerialsWon',
                        'foulsCommitted', 'dribblesWon', 'dribblesAttempted',
                        'dribblesLost', 'dispossessed']
    df_teams = df_teams.reset_index()

    # same concept as above but the new df will have columns for touches and passes each team conceded
    df_teams_ag = pd.concat(players_df, axis=0)
    df_teams_ag = df_teams_ag.drop(columns=['subbedOutMin', 'subbedInMin'])
    # group the dataframe by teamName and Id and apply the custom function
    df_teams_ag = df_teams_ag.groupby(['oppName']).agg({'passCount': ['sum'],
                                                        'touchCount': ['sum']})
    df_teams_ag.columns = ['oppPasses', 'oppTouches']
    df_teams_ag = df_teams_ag.reset_index()
    df_teams_ag = df_teams_ag.rename(columns={'oppName': 'teamName'})

    # merge to get a final teams df
    df_teams_final = pd.merge(df_teams, df_teams_ag, on='teamName', how='left')

    # calc possession based on passes each team made and conceded
    df_teams_final['possession'] = df_teams_final['passes'] / (df_teams_final['passes'] + df_teams_final['oppPasses']) 
    # matches played
    df_teams_final['90s'] = len(matches_data) * 2 / 28
    # calc number of touches each team gets per 90 min
    df_teams_final['teamTouches90'] = df_teams_final['touches'] / df_teams_final['90s']
    
    # apply import functions to create single df for each new stat (goals, shots, assisted xA, etc.)
    shots = calc_xg(events_df)
    pass_xA = calc_xa(events_df)
    goals = shots[shots['isGoal'] == 1]
    passes_carries = xThreat(events_df)
    prog_passes = passes_carries[passes_carries['progressivePass'] == True]
    prog_carries = passes_carries[passes_carries['progressiveCarry'] == True]
    box_passes = passes_carries[passes_carries['passIntoBox'] == True]
    box_carries = passes_carries[passes_carries['carryIntoBox'] == True]

    # 
    keyPassList = ['keyPassLong', 'keyPassShort', 'keyPassCross', 'keyPassCorner', 'keyPassThroughball',
                'keyPassFreekick', 'keyPassThrowin', 'keyPassOther', 'assistCross', 'assistCorner',
                'assistThroughball', 'assistFreekick', 'assistOther', 'assist']

    # filter df to include only key passes, not including throwIns
    key_passes = events_df[(events_df['throwIn'] != True) & ((events_df['type'] == 'Pass'))]

    # Create a boolean mask to select rows where at least one column is True
    mask = key_passes[keyPassList].any(axis=1)

    key_passes = key_passes[mask]

    # create shots df and concat
    #shots = calc_xg(events_df)

    shots_key_passes = pd.concat([key_passes, shots])

    # Group the dataframe by matchId and store each group in a list
    shots_kp_list = [group for _, group in shots_key_passes.groupby('matchId')]

    # sort each dataframe for each match by cumulative minutes
    for match in shots_kp_list:

        match.sort_values(by='cumulative_mins', ascending=True, inplace=True)
        match.reset_index(drop=True, inplace=True)
        
        # Create a new column to store xG values for passes
        match['assistedxA'] = np.nan

        # Iterate over each row in the dataframe
        for i, row in match.iterrows():
            if row['type'] == 'Pass':
                # Find the next shot event
                next_shot = match[(match['cumulative_mins'] > row['cumulative_mins']) & (match['type'] != 'Pass')].head(1)
                if not next_shot.empty:
                    # Assign the xG value of the shot to the corresponding pass
                    match.at[i, 'assistedxA'] = next_shot['xG'].values[0]
                    
    player_kp = pd.concat(shots_kp_list)
    player_kp = player_kp[player_kp['type'] == 'Pass']

    # merge xA with events_df
    xA_full = pass_xA['xA']
    merged_xA = pd.merge(xA_full, events_df, how='outer', left_index=True, right_index=True)

    # Group the dataframe by matchId and store each group in a list
    xA_received_list = [group for _, group in merged_xA.groupby('matchId')]

    # loop through the new list and create a new column, xA received, that assigns the xA value of each pass to the receiver
    for match in xA_received_list:

        match.sort_values(by='cumulative_mins', ascending=True, inplace=True)
        match.reset_index(drop=True, inplace=True)
        
        # Create a new column to store receivedxA values for pass recipients
        match['receivedxA'] = np.nan

        # Iterate over each row in the dataframe
        for i, row in match.iterrows():
            if row['type'] == 'Pass':
                # Find the next pass event
                next_pass = match[(match['cumulative_mins'] > row['cumulative_mins']) & (match['type'] == 'Pass')].head(1)
                if not next_pass.empty:
                    # Assign the xA value of the pass to the corresponding recipient
                    match.at[next_pass.index[0], 'receivedxA'] = row['xA']

    # concatenate list into one sinle df
    player_xA_received = pd.concat(xA_received_list)

    # the next few lines of code aggregate each stat and groups by playername
    passes_carries = passes_carries[(passes_carries['type'].isin(['Carry', 'Pass'])) &
                                (passes_carries['outcomeType'] == 'Successful')]
    passes = passes_carries[passes_carries['type'] == 'Pass']
    carries = passes_carries[passes_carries['type'] == 'Carry']

    xG = shots.groupby(['playerName']).agg({'xG': ['sum']})
    xG.columns = ['xG']
    xG = xG.reset_index()

    xA = pass_xA.groupby(['playerName']).agg({'xA': ['sum']})
    xA.columns = ['xA']
    xA = xA.reset_index()

    receivedxA = player_xA_received.groupby(['playerName']).agg({'receivedxA': ['sum']})
    receivedxA.columns = ['receivedxA']
    receivedxA = receivedxA.reset_index()

    assistedxA = player_kp.groupby(['playerName']).agg({'assistedxA': ['sum']})
    assistedxA.columns = ['assistedxA']
    assistedxA = assistedxA.reset_index()

    np_shots = shots[shots['isPenalty'] == 0]
    npxG = np_shots.groupby(['playerName']).agg({'xG': ['sum']})
    npxG.columns = ['npxG']
    npxG = npxG.reset_index()

    goal_stats = goals.groupby(['playerName']).agg({'isGoal': ['sum']})
    goal_stats.columns = ['G']
    goal_stats = goal_stats.reset_index()

    np_goal_stats = goals[goals['isPenalty'] == 0]
    np_goal_stats = np_goal_stats.groupby(['playerName']).agg({'isGoal': ['sum']})
    np_goal_stats.columns = ['npG']
    np_goal_stats = np_goal_stats.reset_index()

    xT = passes_carries.groupby(['playerName']).agg({'xT': ['sum']})
    xT.columns = ['xT']
    xT = xT.reset_index()

    xT_gen = passes_carries.groupby(['playerName']).agg({'xT_gen': ['sum']})
    xT_gen.columns = ['xT_gen']
    xT_gen = xT_gen.reset_index()

    xT_carries = carries.groupby(['playerName']).agg({'xT': ['sum']})
    xT_carries.columns = ['xTCarries']
    xT_carries = xT_carries.reset_index()

    xT_carries_gen = carries.groupby(['playerName']).agg({'xT_gen': ['sum']})
    xT_carries_gen.columns = ['xTCarriesGen']
    xT_carries_gen = xT_carries_gen.reset_index()

    xT_passes = passes.groupby(['playerName']).agg({'xT': ['sum']})
    xT_passes.columns = ['xTPasses']
    xT_passes = xT_passes.reset_index()

    xT_passes_gen = passes.groupby(['playerName']).agg({'xT_gen': ['sum']})
    xT_passes_gen.columns = ['xTPassesGen']
    xT_passes_gen = xT_passes_gen.reset_index()

    prog_pass_stats = prog_passes.groupby('playerName').size().reset_index(name='progressivePasses')
    carries_stats = carries.groupby('playerName').size().reset_index(name='carries')
    prog_carry_stats = prog_carries.groupby('playerName').size().reset_index(name='progressiveCarries')
    box_pass_stats = box_passes.groupby('playerName').size().reset_index(name='passesIntoBox')
    box_carry_stats = box_carries.groupby('playerName').size().reset_index(name='carriesIntoBox')

    # merge all the dataframes on "playerName" using outer join
    player_adv_stats = pd.merge(xG, npxG, on='playerName', how='outer')
    player_adv_stats = pd.merge(player_adv_stats, xA, on='playerName', how='outer')
    player_adv_stats = pd.merge(player_adv_stats, assistedxA, on='playerName', how='outer')
    player_adv_stats = pd.merge(player_adv_stats, receivedxA, on='playerName', how='outer')
    player_adv_stats = pd.merge(player_adv_stats, goal_stats, on='playerName', how='outer')
    player_adv_stats = pd.merge(player_adv_stats, np_goal_stats, on='playerName', how='outer')
    player_adv_stats = pd.merge(player_adv_stats, xT, on='playerName', how='outer')
    player_adv_stats = pd.merge(player_adv_stats, xT_gen, on='playerName', how='outer')
    player_adv_stats = pd.merge(player_adv_stats, xT_carries, on='playerName', how='outer')
    player_adv_stats = pd.merge(player_adv_stats, xT_carries_gen, on='playerName', how='outer')
    player_adv_stats = pd.merge(player_adv_stats, xT_passes, on='playerName', how='outer')
    player_adv_stats = pd.merge(player_adv_stats, xT_passes_gen, on='playerName', how='outer')
    player_adv_stats = pd.merge(player_adv_stats, prog_pass_stats, on='playerName', how='outer')
    player_adv_stats = pd.merge(player_adv_stats, prog_carry_stats, on='playerName', how='outer')
    player_adv_stats = pd.merge(player_adv_stats, box_pass_stats, on='playerName', how='outer')
    player_adv_stats = pd.merge(player_adv_stats, box_carry_stats, on='playerName', how='outer')
    player_adv_stats = pd.merge(player_adv_stats, carries_stats, on='playerName', how='outer')
    player_adv_stats = player_adv_stats.fillna(0)
    player_stats_df = pd.merge(df, player_adv_stats, on='playerName', how='outer')
    player_stats_df = player_stats_df.fillna(0)

    # create expected offensive value added column
    player_stats_df['xOVA'] = (player_stats_df['xG'] + player_stats_df['xA']) - player_stats_df['receivedxA']

    # create a column for 90s played and select only necessary columns
    player_stats_df['90s'] = player_stats_df['timePlayed'] / 90
    player_stats_df = player_stats_df[['playerName', 'playerPos', 'playerId', 'teamName', 'teamId', 'timePlayed', 
                                       'passes', 'touches', 'carries', 'passesComp', 'shots', 'shotsOnTarget', 'interceptions', 
                                       'keyPasses', 'tackles', 'tacklesSuccessful', 'shotsBlocked', 'clearances', 
                                       'xG', 'xA', 'assistedxA', 'receivedxA', 'xOVA', 'aerials', 'aerialsWon', 'foulsCommitted', 'dribblesWon', 'dribblesAttempted', 
                                       'dribblesLost', 'dispossessed', 'npxG', 'G', 'npG', 'xT', 'xT_gen', 'xTCarries', 
                                       'xTCarriesGen', 'xTPasses', 'xTPassesGen', 'progressivePasses', 'progressiveCarries',
                                       'passesIntoBox', 'carriesIntoBox', '90s']]
    # round advanced stats to 2 decimals and adjust all stats that reflect count of events to integer
    player_stats_df[['playerId', 'teamId', 'carries', 'timePlayed', 'shots', 'shotsOnTarget', 'interceptions', 'keyPasses', 'tackles', 'tacklesSuccessful', 'shotsBlocked', 'clearances', 'aerials', 'aerialsWon', 'foulsCommitted', 'dribblesWon', 'dribblesAttempted', 'dribblesLost', 'dispossessed', 'G', 'npG', 'passes', 'touches', 'passesComp', 'progressivePasses', 'progressiveCarries', 'passesIntoBox', 'carriesIntoBox']] = player_stats_df[['playerId', 'teamId', 'carries', 'timePlayed', 'shots', 'shotsOnTarget', 'interceptions', 'keyPasses', 'tackles', 'tacklesSuccessful', 'shotsBlocked', 'clearances', 'aerials', 'aerialsWon', 'foulsCommitted', 'dribblesWon', 'dribblesAttempted', 'dribblesLost', 'dispossessed', 'G', 'npG', 'passes', 'touches', 'passesComp', 'progressivePasses', 'progressiveCarries', 'passesIntoBox', 'carriesIntoBox']].astype(int)
    player_stats_df[['xG', 'xA', 'assistedxA', 'receivedxA', 'xOVA', 'npxG', 'xT', 'xT_gen', 'xTCarries', 'xTCarriesGen', 'xTPasses', 'xTPassesGen', '90s']] = player_stats_df[['xG', 'xA', 'assistedxA', 'receivedxA', 'xOVA', 'npxG', 'xT', 'xT_gen', 'xTCarries', 'xTCarriesGen', 'xTPasses', 'xTPassesGen', '90s']].applymap('{:.2f}'.format).astype(float)

    player_stats_df_90s = player_stats_df.copy()
    # duplicate each column that reflects and stat, divide by number of 90s played, and rename each new duplicate to include "per90"
    for i in range(6,40):
        player_stats_df_90s.iloc[:,i] = player_stats_df_90s.iloc[:,i]/player_stats_df_90s['90s']
    player_stats_df_90s = player_stats_df_90s.iloc[:,6:].add_suffix('Per90').applymap('{:.2f}'.format).astype(float)
    # join 90s df and total
    df_new = player_stats_df.join(player_stats_df_90s)
    df_new = df_new.drop(columns=['90sPer90'])
    df_new = df_new.dropna()

    # touch and possession adjustments
    df_new['AvgTeamPoss'] = float(0.0)
    df_new['OppTouches'] = int(1)
    df_new['TeamMins'] = int(1)
    df_new['TeamTouches90'] = float(0.0)

    player_list = list(df_new['playerName'])

    for i in range(len(player_list)):
        team_name = df_new[df_new['playerName']==player_list[i]]['teamName'].values[0]
        team_poss = df_teams_final[df_teams_final['teamName']==team_name]['possession'].values[0]
        opp_touch = df_teams_final[df_teams_final['teamName']==team_name]['oppTouches'].values[0]
        team_mins = df_teams_final[df_teams_final['teamName']==team_name]['timePlayed'].values[0]
        team_touches = df_teams_final[df_teams_final['teamName']==team_name]['teamTouches90'].values[0]
        df_new.at[i, 'AvgTeamPoss'] = team_poss
        df_new.at[i, 'OppTouches'] = opp_touch
        df_new.at[i, 'TeamMins'] = team_mins
        df_new.at[i, 'TeamTouches90'] = team_touches
        
    # All of these are the possession-adjusted columns. A couple touch-adjusted ones at the bottom
    df_new['pAdjTklPer90'] = (df_new['tacklesPer90']/(100-df_new['AvgTeamPoss']))*50
    df_new['pAdjIntPer90'] = (df_new['interceptionsPer90']/(100-df_new['AvgTeamPoss']))*50
    df_new['pAdjClrPer90'] = (df_new['clearancesPer90']/(100-df_new['AvgTeamPoss']))*50
    df_new['pAdjShBlocksPer90'] = (df_new['shotsBlockedPer90']/(100-df_new['AvgTeamPoss']))*50
    #df_new['pAdjPassBlocksPer90'] = (df_new['PassBlocksPer90']/(100-df_new['AvgTeamPoss']))*50
    #df_new['pAdjDrbTklPer90'] = (df_new['DrbTklPer90']/(100-df_new['AvgTeamPoss']))*50
    #df_new['pAdjTklWinPossPer90'] = (df_new['DrbTklPer90']/(100-df_new['AvgTeamPoss']))*50
    #df_new['pAdjDrbPastPer90'] = (df_new['DrbPastPer90']/(100-df_new['AvgTeamPoss']))*50
    df_new['pAdjAerialWinsPer90'] = (df_new['aerialsWonPer90']/(100-df_new['AvgTeamPoss']))*50
    #df_new['pAdjAerialLossPer90'] = (df_new['AerialLossPer90']/(100-df_new['AvgTeamPoss']))*50
    #df_new['pAdjDrbPastAttPer90'] = (df_new['DrbPastAttPer90']/(100-df_new['AvgTeamPoss']))*50
    df_new['TouchCentrality'] = (df_new['touchesPer90']/df_new['TeamTouches90'])*100
    # df['pAdj#OPAPer90'] =(df['#OPAPer90']/(100-df['AvgTeamPoss']))*50
    #df_new['TklPer600OppTouch'] = df_new['tackles'] /(df_new['OppTouches']*(df_new['Min']/df_new['TeamMins']))*600
    #df_new['IntPer600OppTouch'] = df_new['interceptions'] /(df_new['OppTouches']*(df_new['Min']/df_new['TeamMins']))*600
    df_new['pAdjTouchesPer90'] = (df_new['touchesPer90']/(df_new['AvgTeamPoss']))*50
    df_new['CarriesPer50Touches'] = df_new['carries'] / df_new['touches'] * 50
    df_new['ProgCarriesPer50Touches'] = df_new['progressiveCarries'] / df_new['touches'] * 50
    df_new['ProgPassesPer50CmpPasses'] = df_new['progressivePasses'] / df_new['passesComp'] * 50

    df_new[['AvgTeamPoss', 'TeamTouches90', 'pAdjTklPer90', 'pAdjIntPer90', 'pAdjClrPer90', 'pAdjShBlocksPer90', 'pAdjAerialWinsPer90', 'TouchCentrality', 'pAdjTouchesPer90', 'CarriesPer50Touches', 'ProgCarriesPer50Touches', 'ProgPassesPer50CmpPasses']] = df_new[['AvgTeamPoss', 'TeamTouches90', 'pAdjTklPer90', 'pAdjIntPer90', 'pAdjClrPer90', 'pAdjShBlocksPer90', 'pAdjAerialWinsPer90', 'TouchCentrality', 'pAdjTouchesPer90', 'CarriesPer50Touches', 'ProgCarriesPer50Touches', 'ProgPassesPer50CmpPasses']].applymap('{:.2f}'.format).astype(float)

    # save the data in the folder for each respective comp, depening on matches data

    # drop any duplicates and save to folder
    df_new = df_new.drop_duplicates(subset='playerName')
    df_new.to_csv(f'Data/{comp}/{season[5:]}/player-stats/player-advanced-stats.csv')

    return df_new

def team_advanced_stats(events_df, matches_data, comp):

    # apply progressive stat methods to calc each respective progressive stat
    events_df['progressivePass'] = events_df.apply(lambda x: progressive_pass(x, inplay=True, successful_only=True), axis=1)
    events_df['progressiveCarry'] = events_df.apply(lambda x: progressive_carry(x, successful_only=True), axis=1)
    events_df['carryIntoBox'] = events_df.apply(lambda x: carry_into_box(x, successful_only=True), axis=1)
    events_df['passIntoBox'] = events_df.apply(lambda x: pass_into_box(x, inplay=True, successful_only=True), axis=1)

    # this set of code withh loop through matches_data, pull aggregate stats for each team and append to a list of dfs for each match
    teams_df = []

    for match in matches_data:
        # getting teams dataframe
        match_players_df = pd.DataFrame()
        team_Ids = []
        opp_Ids = []
        team_names = []
        opp_names = []
        pass_count = []
        touch_count = []
        pass_completed = []
        shots = []
        shots_on_target = []
        interceptions = []
        shots_blocked = []
        fouls_committed = []
        dribbles_won = []
        dribbles_attempted = []
        dribbles_lost = []
        dispossessed = []
        clearances = []
        passes_key = []
        aerials = []
        aerials_won = []
        tackles = []
        tackles_successful = []
        tackle_success = []
        subbed_out_min = []
        subbed_in_min = []
        mins_played = []
        max_min = match['periodEndMinutes']['2']
        
        home_df = match['home']
        home_Id = home_df['teamId']
        home_team = home_df['name']
        
        away_df = match['away']
        away_team = away_df['name']
        away_Id = away_df['teamId']
        
        for player in home_df['players']:
            team_Ids.append(home_Id)
            opp_Ids.append(away_Id)
            team_names.append(home_team)
            opp_names.append(away_team)
            player_names.append(player['name'])
            player_ids.append(player['playerId'])
            player_pos.append(player['position'])
            player_kit_number.append(player['shirtNo'])
            try:
                player_eleven.append(player['isFirstEleven'])
            except KeyError:
                player_eleven.append(False)
            try:
                pass_count.append(sum(player['stats']['passesTotal'].values()))
            except KeyError:
                pass_count.append(0)
            try:
                touch_count.append(sum(player['stats']['touches'].values()))
            except KeyError:
                touch_count.append(0)
            try:
                pass_completed.append(sum(player['stats']['passesAccurate'].values()))
            except KeyError:
                pass_completed.append(0)
            try:
                shots.append(sum(player['stats']['shotsTotal'].values()))
            except KeyError:
                shots.append(0)
            try:
                shots_on_target.append(sum(player['stats']['shotsOnTarget'].values()))
            except KeyError:
                shots_on_target.append(0)
            try:
                interceptions.append(sum(player['stats']['interceptions'].values()))
            except KeyError:
                interceptions.append(0)
            try:
                shots_blocked.append(sum(player['stats']['shotsBlocked'].values()))
            except KeyError:
                shots_blocked.append(0)
            try:
                fouls_committed.append(sum(player['stats']['foulsCommitted'].values()))
            except KeyError:
                fouls_committed.append(0)
            try:
                clearances.append(sum(player['stats']['clearances'].values()))
            except KeyError:
                clearances.append(0)
            try:
                dribbles_won.append(sum(player['stats']['dribblesWon'].values()))
            except KeyError:
                dribbles_won.append(0)
            try:
                dribbles_attempted.append(sum(player['stats']['dribblesAttempted'].values()))
            except KeyError:
                dribbles_attempted.append(0)
            try:
                dribbles_lost.append(sum(player['stats']['dribblesLost'].values()))
            except KeyError:
                dribbles_lost.append(0)
            try:
                dispossessed.append(sum(player['stats']['dispossessed'].values()))
            except KeyError:
                dispossessed.append(0)
            try:
                passes_key.append(sum(player['stats']['passesKey'].values()))
            except KeyError:
                passes_key.append(0)
            try:
                aerials.append(sum(player['stats']['aerialsTotal'].values()))
            except KeyError:
                aerials.append(0)
            try:
                aerials_won.append(sum(player['stats']['aerialsWon'].values()))
            except KeyError:
                aerials_won.append(0)
            try:
                tackles.append(sum(player['stats']['tacklesTotal'].values()))
            except KeyError:
                tackles.append(0)
            try:
                tackles_successful.append(sum(player['stats']['tackleSuccessful'].values()))
            except KeyError:
                tackles_successful.append(0)
            try:
                tackle_success.append(sum(player['stats']['tackleSuccess'].values()))
            except KeyError:
                tackle_success.append(0)
            try:
                subbed_out_min.append(player['subbedOutExpandedMinute'])
            except KeyError:
                subbed_out_min.append(max_min)
            try:
                subbed_in_min.append(player['subbedInExpandedMinute'])
            except KeyError:
                subbed_in_min.append(0)
                
        for player in away_df['players']:
            team_Ids.append(away_Id)
            opp_Ids.append(home_Id)
            team_names.append(away_team)
            opp_names.append(home_team)
            player_names.append(player['name'])
            player_ids.append(player['playerId'])
            player_pos.append(player['position'])
            player_kit_number.append(player['shirtNo'])
            try:
                player_eleven.append(player['isFirstEleven'])
            except KeyError:
                player_eleven.append(False)
            try:
                pass_count.append(sum(player['stats']['passesTotal'].values()))
            except KeyError:
                pass_count.append(0)
            try:
                touch_count.append(sum(player['stats']['touches'].values()))
            except KeyError:
                touch_count.append(0)
            try:
                pass_completed.append(sum(player['stats']['passesAccurate'].values()))
            except KeyError:
                pass_completed.append(0)
            try:
                shots.append(sum(player['stats']['shotsTotal'].values()))
            except KeyError:
                shots.append(0)
            try:
                shots_on_target.append(sum(player['stats']['shotsOnTarget'].values()))
            except KeyError:
                shots_on_target.append(0)
            try:
                interceptions.append(sum(player['stats']['interceptions'].values()))
            except KeyError:
                interceptions.append(0)
            try:
                shots_blocked.append(sum(player['stats']['shotsBlocked'].values()))
            except KeyError:
                shots_blocked.append(0)
            try:
                fouls_committed.append(sum(player['stats']['foulsCommitted'].values()))
            except KeyError:
                fouls_committed.append(0)
            try:
                clearances.append(sum(player['stats']['clearances'].values()))
            except KeyError:
                clearances.append(0)
            try:
                dribbles_won.append(sum(player['stats']['dribblesWon'].values()))
            except KeyError:
                dribbles_won.append(0)
            try:
                dribbles_attempted.append(sum(player['stats']['dribblesAttempted'].values()))
            except KeyError:
                dribbles_attempted.append(0)
            try:
                dribbles_lost.append(sum(player['stats']['dribblesLost'].values()))
            except KeyError:
                dribbles_lost.append(0)
            try:
                dispossessed.append(sum(player['stats']['dispossessed'].values()))
            except KeyError:
                dispossessed.append(0)
            try:
                passes_key.append(sum(player['stats']['passesKey'].values()))
            except KeyError:
                passes_key.append(0)
            try:
                aerials.append(sum(player['stats']['aerialsTotal'].values()))
            except KeyError:
                aerials.append(0)
            try:
                aerials_won.append(sum(player['stats']['aerialsWon'].values()))
            except KeyError:
                aerials_won.append(0)
            try:
                tackles.append(sum(player['stats']['tacklesTotal'].values()))
            except KeyError:
                tackles.append(0)
            try:
                tackles_successful.append(sum(player['stats']['tackleSuccessful'].values()))
            except KeyError:
                tackles_successful.append(0)
            try:
                tackle_success.append(sum(player['stats']['tackleSuccess'].values()))
            except KeyError:
                tackle_success.append(0)
            try:
                subbed_out_min.append(player['subbedOutExpandedMinute'])
            except KeyError:
                subbed_out_min.append(max_min)
            try:
                subbed_in_min.append(player['subbedInExpandedMinute'])
            except KeyError:
                subbed_in_min.append(0)
            
        match_players_df['teamId'] = team_Ids  
        match_players_df['oppId'] = opp_Ids
        match_players_df['teamName'] = team_names
        match_players_df['oppName'] = opp_names
        match_players_df['playerId'] = player_ids
        match_players_df['playerName'] = player_names
        match_players_df['playerPos'] = player_pos
        match_players_df['playerKitNumber'] = player_kit_number
        match_players_df['isFirstEleven'] = player_eleven
        match_players_df['passCount'] = pass_count
        match_players_df['touchCount'] = touch_count
        match_players_df['passesCompleted'] = pass_completed
        match_players_df['shots'] = shots
        match_players_df['shotsOnTarget'] = shots_on_target
        match_players_df['interceptions'] = interceptions
        match_players_df['keyPasses'] = passes_key
        match_players_df['aerials'] = aerials
        match_players_df['aerialsWon'] = aerials_won
        match_players_df['tackles'] = tackles
        match_players_df['tacklesSuccessful'] = tackles_successful
        match_players_df['tackleSuccess'] = tackle_success
        match_players_df['clearances'] = clearances 
        match_players_df['subbedOutMin'] = subbed_out_min
        match_players_df['subbedInMin'] = subbed_in_min
        match_players_df['shotsBlocked'] = shots_blocked
        match_players_df['foulsCommitted'] = fouls_committed
        match_players_df['dribblesWon'] = dribbles_won
        match_players_df['dribblesAttempted'] = dribbles_attempted
        match_players_df['dribblesLost'] = dribbles_lost
        match_players_df['dispossessed'] = dispossessed
        match_players_df.loc[(match_players_df['isFirstEleven'] == False) & (match_players_df['subbedInMin'] == 0), 'subbedOutMin'] = 0
        match_players_df['timePlayed'] = match_players_df['subbedOutMin'] - match_players_df['subbedInMin']  
        
        players_df.append(match_players_df)

    # concatenate list of all matches to one single df of all stats to date
    df = pd.concat(players_df, axis=0)
    df = df.drop(columns=['subbedOutMin', 'subbedInMin'])

    # define a custom function to find the most common position
    def find_most_common_pos(x):
        try:
            # check if the most common position is "Sub"
            if x.value_counts().index[0] == 'Sub':
                # return the second most common position
                return x.value_counts().index[1]
            else:
                # return the most common position
                return x.value_counts().index[0]
        except IndexError:
            # handle the case where the Series has only one element
            return x.iloc[0]

    # group the dataframe by playerName and apply the custom function
    positions = df.groupby('playerName')['playerPos'].agg(find_most_common_pos).reset_index()
    teams = df.groupby(['playerName', 'teamName']).size().reset_index()

    # group by player and player/team ID with the total sum of each player's stat
    df = df.groupby(['playerName', 'playerId', 'teamId']).agg({'passCount': ['sum'],
                                                               'touchCount': ['sum'],
                                                               'passesCompleted': ['sum'],
                                                               'shots': ['sum'],
                                                               'shotsOnTarget': ['sum'],
                                                               'interceptions': ['sum'],
                                                               'keyPasses': ['sum'],
                                                               'tackles': ['sum'],
                                                               'tacklesSuccessful': ['sum'],
                                                               'timePlayed': ['sum'],
                                                               'shotsBlocked': ['sum'],
                                                               'clearances': ['sum'],
                                                               'aerials': ['sum'],
                                                               'aerialsWon': ['sum'],
                                                               'foulsCommitted': ['sum'],
                                                               'dribblesWon': ['sum'],
                                                               'dribblesAttempted': ['sum'],
                                                               'dribblesLost': ['sum'],
                                                               'dispossessed': ['sum']})

    # rename columns
    df.columns = ['passes', 'touches', 'passesComp', 'shots',
                  'shotsOnTarget', 'interceptions', 'keyPasses', 
                  'tackles', 'tacklesSuccessful', 'timePlayed',
                  'shotsBlocked', 'clearances', 'aerials', 'aerialsWon',
                  'foulsCommitted', 'dribblesWon', 'dribblesAttempted',
                  'dribblesLost', 'dispossessed']
    df = df.reset_index()

    # merge df with the positions and teams df to match each player's team and position
    df = pd.merge(positions, df, on='playerName', how='left')
    df = pd.merge(df, teams, on='playerName', how='left')
    # remove any players that have not played
    df = df[df['timePlayed'] > 0]

    # apply the same concept as above to group the players_df list by team names to get team total stats
    df_teams = pd.concat(players_df, axis=0)
    df_teams = df_teams.drop(columns=['subbedOutMin', 'subbedInMin'])

    # group the dataframe by teamName and Id
    df_teams = df_teams.groupby(['teamName', 'teamId']).agg({'passCount': ['sum'],
                                                             'touchCount': ['sum'],
                                                             'passesCompleted': ['sum'],
                                                             'shots': ['sum'],
                                                             'shotsOnTarget': ['sum'],
                                                             'interceptions': ['sum'],
                                                             'keyPasses': ['sum'],
                                                             'tackles': ['sum'],
                                                             'tacklesSuccessful': ['sum'],
                                                             'timePlayed': ['sum'],
                                                             'shotsBlocked': ['sum'],
                                                             'clearances': ['sum'],
                                                             'aerials': ['sum'],
                                                             'aerialsWon': ['sum'],
                                                             'foulsCommitted': ['sum'],
                                                             'dribblesWon': ['sum'],
                                                             'dribblesAttempted': ['sum'],
                                                             'dribblesLost': ['sum'],
                                                             'dispossessed': ['sum']})

    # rename columns
    df_teams.columns = ['passes', 'touches', 'passesComp', 'shots',
                        'shotsOnTarget', 'interceptions', 'keyPasses', 
                        'tackles', 'tacklesSuccessful', 'timePlayed',
                        'shotsBlocked', 'clearances', 'aerials', 'aerialsWon',
                        'foulsCommitted', 'dribblesWon', 'dribblesAttempted',
                        'dribblesLost', 'dispossessed']
    df_teams = df_teams.reset_index()

    # same concept as above but the new df will have columns for touches and passes each team conceded
    df_teams_ag = pd.concat(players_df, axis=0)
    df_teams_ag = df_teams_ag.drop(columns=['subbedOutMin', 'subbedInMin'])
    # group the dataframe by teamName and Id and apply the custom function
    df_teams_ag = df_teams_ag.groupby(['oppName']).agg({'passCount': ['sum'],
                                                        'touchCount': ['sum']})
    df_teams_ag.columns = ['oppPasses', 'oppTouches']
    df_teams_ag = df_teams_ag.reset_index()
    df_teams_ag = df_teams_ag.rename(columns={'oppName': 'teamName'})

    # merge to get a final teams df
    df_teams_final = pd.merge(df_teams, df_teams_ag, on='teamName', how='left')

    # calc possession based on passes each team made and conceded
    df_teams_final['possession'] = df_teams_final['passes'] / (df_teams_final['passes'] + df_teams_final['oppPasses']) 
    # matches played
    df_teams_final['90s'] = len(matches_data) * 2 / 28
    # calc number of touches each team gets per 90 min
    df_teams_final['teamTouches90'] = df_teams_final['touches'] / df_teams_final['90s']
    
    # apply import functions to create single df for each new stat (goals, shots, assisted xA, etc.)
    shots = calc_xg(events_df)
    pass_xA = calc_xa(events_df)
    goals = shots[shots['isGoal'] == 1]
    passes_carries = xThreat(events_df)
    prog_passes = passes_carries[passes_carries['progressivePass'] == True]
    prog_carries = passes_carries[passes_carries['progressiveCarry'] == True]
    box_passes = passes_carries[passes_carries['passIntoBox'] == True]
    box_carries = passes_carries[passes_carries['carryIntoBox'] == True]

    # 
    keyPassList = ['keyPassLong', 'keyPassShort', 'keyPassCross', 'keyPassCorner', 'keyPassThroughball',
                'keyPassFreekick', 'keyPassThrowin', 'keyPassOther', 'assistCross', 'assistCorner',
                'assistThroughball', 'assistFreekick', 'assistOther', 'assist']

    # filter df to include only key passes, not including throwIns
    key_passes = events_df[(events_df['throwIn'] != True) & ((events_df['type'] == 'Pass'))]

    # Create a boolean mask to select rows where at least one column is True
    mask = key_passes[keyPassList].any(axis=1)

    key_passes = key_passes[mask]

    # create shots df and concat
    #shots = calc_xg(events_df)

    shots_key_passes = pd.concat([key_passes, shots])

    # Group the dataframe by matchId and store each group in a list
    shots_kp_list = [group for _, group in shots_key_passes.groupby('matchId')]

    # sort each dataframe for each match by cumulative minutes
    for match in shots_kp_list:

        match.sort_values(by='cumulative_mins', ascending=True, inplace=True)
        match.reset_index(drop=True, inplace=True)
        
        # Create a new column to store xG values for passes
        match['assistedxA'] = np.nan

        # Iterate over each row in the dataframe
        for i, row in match.iterrows():
            if row['type'] == 'Pass':
                # Find the next shot event
                next_shot = match[(match['cumulative_mins'] > row['cumulative_mins']) & (match['type'] != 'Pass')].head(1)
                if not next_shot.empty:
                    # Assign the xG value of the shot to the corresponding pass
                    match.at[i, 'assistedxA'] = next_shot['xG'].values[0]
                    
    player_kp = pd.concat(shots_kp_list)
    player_kp = player_kp[player_kp['type'] == 'Pass']

    # merge xA with events_df
    xA_full = pass_xA['xA']
    merged_xA = pd.merge(xA_full, events_df, how='outer', left_index=True, right_index=True)

    # Group the dataframe by matchId and store each group in a list
    xA_received_list = [group for _, group in merged_xA.groupby('matchId')]

    # loop through the new list and create a new column, xA received, that assigns the xA value of each pass to the receiver
    for match in xA_received_list:

        match.sort_values(by='cumulative_mins', ascending=True, inplace=True)
        match.reset_index(drop=True, inplace=True)
        
        # Create a new column to store receivedxA values for pass recipients
        match['receivedxA'] = np.nan

        # Iterate over each row in the dataframe
        for i, row in match.iterrows():
            if row['type'] == 'Pass':
                # Find the next pass event
                next_pass = match[(match['cumulative_mins'] > row['cumulative_mins']) & (match['type'] == 'Pass')].head(1)
                if not next_pass.empty:
                    # Assign the xA value of the pass to the corresponding recipient
                    match.at[next_pass.index[0], 'receivedxA'] = row['xA']

    # concatenate list into one sinle df
    player_xA_received = pd.concat(xA_received_list)

    # the next few lines of code aggregate each stat and groups by playername
    passes_carries = passes_carries[(passes_carries['type'].isin(['Carry', 'Pass'])) &
                                (passes_carries['outcomeType'] == 'Successful')]
    passes = passes_carries[passes_carries['type'] == 'Pass']
    carries = passes_carries[passes_carries['type'] == 'Carry']

    xG = shots.groupby(['playerName']).agg({'xG': ['sum']})
    xG.columns = ['xG']
    xG = xG.reset_index()

    xA = pass_xA.groupby(['playerName']).agg({'xA': ['sum']})
    xA.columns = ['xA']
    xA = xA.reset_index()

    receivedxA = player_xA_received.groupby(['playerName']).agg({'receivedxA': ['sum']})
    receivedxA.columns = ['receivedxA']
    receivedxA = receivedxA.reset_index()

    assistedxA = player_kp.groupby(['playerName']).agg({'assistedxA': ['sum']})
    assistedxA.columns = ['assistedxA']
    assistedxA = assistedxA.reset_index()

    np_shots = shots[shots['isPenalty'] == 0]
    npxG = np_shots.groupby(['playerName']).agg({'xG': ['sum']})
    npxG.columns = ['npxG']
    npxG = npxG.reset_index()

    goal_stats = goals.groupby(['playerName']).agg({'isGoal': ['sum']})
    goal_stats.columns = ['G']
    goal_stats = goal_stats.reset_index()

    np_goal_stats = goals[goals['isPenalty'] == 0]
    np_goal_stats = np_goal_stats.groupby(['playerName']).agg({'isGoal': ['sum']})
    np_goal_stats.columns = ['npG']
    np_goal_stats = np_goal_stats.reset_index()

    xT = passes_carries.groupby(['playerName']).agg({'xT': ['sum']})
    xT.columns = ['xT']
    xT = xT.reset_index()

    xT_gen = passes_carries.groupby(['playerName']).agg({'xT_gen': ['sum']})
    xT_gen.columns = ['xT_gen']
    xT_gen = xT_gen.reset_index()

    xT_carries = carries.groupby(['playerName']).agg({'xT': ['sum']})
    xT_carries.columns = ['xTCarries']
    xT_carries = xT_carries.reset_index()

    xT_carries_gen = carries.groupby(['playerName']).agg({'xT_gen': ['sum']})
    xT_carries_gen.columns = ['xTCarriesGen']
    xT_carries_gen = xT_carries_gen.reset_index()

    xT_passes = passes.groupby(['playerName']).agg({'xT': ['sum']})
    xT_passes.columns = ['xTPasses']
    xT_passes = xT_passes.reset_index()

    xT_passes_gen = passes.groupby(['playerName']).agg({'xT_gen': ['sum']})
    xT_passes_gen.columns = ['xTPassesGen']
    xT_passes_gen = xT_passes_gen.reset_index()

    prog_pass_stats = prog_passes.groupby('playerName').size().reset_index(name='progressivePasses')
    carries_stats = carries.groupby('playerName').size().reset_index(name='carries')
    prog_carry_stats = prog_carries.groupby('playerName').size().reset_index(name='progressiveCarries')
    box_pass_stats = box_passes.groupby('playerName').size().reset_index(name='passesIntoBox')
    box_carry_stats = box_carries.groupby('playerName').size().reset_index(name='carriesIntoBox')

    # merge all the dataframes on "playerName" using outer join
    player_adv_stats = pd.merge(xG, npxG, on='playerName', how='outer')
    player_adv_stats = pd.merge(player_adv_stats, xA, on='playerName', how='outer')
    player_adv_stats = pd.merge(player_adv_stats, assistedxA, on='playerName', how='outer')
    player_adv_stats = pd.merge(player_adv_stats, receivedxA, on='playerName', how='outer')
    player_adv_stats = pd.merge(player_adv_stats, goal_stats, on='playerName', how='outer')
    player_adv_stats = pd.merge(player_adv_stats, np_goal_stats, on='playerName', how='outer')
    player_adv_stats = pd.merge(player_adv_stats, xT, on='playerName', how='outer')
    player_adv_stats = pd.merge(player_adv_stats, xT_gen, on='playerName', how='outer')
    player_adv_stats = pd.merge(player_adv_stats, xT_carries, on='playerName', how='outer')
    player_adv_stats = pd.merge(player_adv_stats, xT_carries_gen, on='playerName', how='outer')
    player_adv_stats = pd.merge(player_adv_stats, xT_passes, on='playerName', how='outer')
    player_adv_stats = pd.merge(player_adv_stats, xT_passes_gen, on='playerName', how='outer')
    player_adv_stats = pd.merge(player_adv_stats, prog_pass_stats, on='playerName', how='outer')
    player_adv_stats = pd.merge(player_adv_stats, prog_carry_stats, on='playerName', how='outer')
    player_adv_stats = pd.merge(player_adv_stats, box_pass_stats, on='playerName', how='outer')
    player_adv_stats = pd.merge(player_adv_stats, box_carry_stats, on='playerName', how='outer')
    player_adv_stats = pd.merge(player_adv_stats, carries_stats, on='playerName', how='outer')
    player_adv_stats = player_adv_stats.fillna(0)
    player_stats_df = pd.merge(df, player_adv_stats, on='playerName', how='outer')
    player_stats_df = player_stats_df.fillna(0)

    # create expected offensive value added column
    player_stats_df['xOVA'] = (player_stats_df['xG'] + player_stats_df['xA']) - player_stats_df['receivedxA']

    # create a column for 90s played and select only necessary columns
    player_stats_df['90s'] = player_stats_df['timePlayed'] / 90
    player_stats_df = player_stats_df[['playerName', 'playerPos', 'playerId', 'teamName', 'teamId', 'timePlayed', 
                                       'passes', 'touches', 'carries', 'passesComp', 'shots', 'shotsOnTarget', 'interceptions', 
                                       'keyPasses', 'tackles', 'tacklesSuccessful', 'shotsBlocked', 'clearances', 
                                       'xG', 'xA', 'assistedxA', 'receivedxA', 'xOVA', 'aerials', 'aerialsWon', 'foulsCommitted', 'dribblesWon', 'dribblesAttempted', 
                                       'dribblesLost', 'dispossessed', 'npxG', 'G', 'npG', 'xT', 'xT_gen', 'xTCarries', 
                                       'xTCarriesGen', 'xTPasses', 'xTPassesGen', 'progressivePasses', 'progressiveCarries',
                                       'passesIntoBox', 'carriesIntoBox', '90s']]
    # round advanced stats to 2 decimals and adjust all stats that reflect count of events to integer
    player_stats_df[['playerId', 'teamId', 'carries', 'timePlayed', 'shots', 'shotsOnTarget', 'interceptions', 'keyPasses', 'tackles', 'tacklesSuccessful', 'shotsBlocked', 'clearances', 'aerials', 'aerialsWon', 'foulsCommitted', 'dribblesWon', 'dribblesAttempted', 'dribblesLost', 'dispossessed', 'G', 'npG', 'passes', 'touches', 'passesComp', 'progressivePasses', 'progressiveCarries', 'passesIntoBox', 'carriesIntoBox']] = player_stats_df[['playerId', 'teamId', 'carries', 'timePlayed', 'shots', 'shotsOnTarget', 'interceptions', 'keyPasses', 'tackles', 'tacklesSuccessful', 'shotsBlocked', 'clearances', 'aerials', 'aerialsWon', 'foulsCommitted', 'dribblesWon', 'dribblesAttempted', 'dribblesLost', 'dispossessed', 'G', 'npG', 'passes', 'touches', 'passesComp', 'progressivePasses', 'progressiveCarries', 'passesIntoBox', 'carriesIntoBox']].astype(int)
    player_stats_df[['xG', 'xA', 'assistedxA', 'receivedxA', 'xOVA', 'npxG', 'xT', 'xT_gen', 'xTCarries', 'xTCarriesGen', 'xTPasses', 'xTPassesGen', '90s']] = player_stats_df[['xG', 'xA', 'assistedxA', 'receivedxA', 'xOVA', 'npxG', 'xT', 'xT_gen', 'xTCarries', 'xTCarriesGen', 'xTPasses', 'xTPassesGen', '90s']].applymap('{:.2f}'.format).astype(float)

    player_stats_df_90s = player_stats_df.copy()
    # duplicate each column that reflects and stat, divide by number of 90s played, and rename each new duplicate to include "per90"
    for i in range(6,40):
        player_stats_df_90s.iloc[:,i] = player_stats_df_90s.iloc[:,i]/player_stats_df_90s['90s']
    player_stats_df_90s = player_stats_df_90s.iloc[:,6:].add_suffix('Per90').applymap('{:.2f}'.format).astype(float)
    # join 90s df and total
    df_new = player_stats_df.join(player_stats_df_90s)
    df_new = df_new.drop(columns=['90sPer90'])
    df_new = df_new.dropna()

    # touch and possession adjustments
    df_new['AvgTeamPoss'] = float(0.0)
    df_new['OppTouches'] = int(1)
    df_new['TeamMins'] = int(1)
    df_new['TeamTouches90'] = float(0.0)

    player_list = list(df_new['playerName'])

    for i in range(len(player_list)):
        team_name = df_new[df_new['playerName']==player_list[i]]['teamName'].values[0]
        team_poss = df_teams_final[df_teams_final['teamName']==team_name]['possession'].values[0]
        opp_touch = df_teams_final[df_teams_final['teamName']==team_name]['oppTouches'].values[0]
        team_mins = df_teams_final[df_teams_final['teamName']==team_name]['timePlayed'].values[0]
        team_touches = df_teams_final[df_teams_final['teamName']==team_name]['teamTouches90'].values[0]
        df_new.at[i, 'AvgTeamPoss'] = team_poss
        df_new.at[i, 'OppTouches'] = opp_touch
        df_new.at[i, 'TeamMins'] = team_mins
        df_new.at[i, 'TeamTouches90'] = team_touches
        
    # All of these are the possession-adjusted columns. A couple touch-adjusted ones at the bottom
    df_new['pAdjTklPer90'] = (df_new['tacklesPer90']/(100-df_new['AvgTeamPoss']))*50
    df_new['pAdjIntPer90'] = (df_new['interceptionsPer90']/(100-df_new['AvgTeamPoss']))*50
    df_new['pAdjClrPer90'] = (df_new['clearancesPer90']/(100-df_new['AvgTeamPoss']))*50
    df_new['pAdjShBlocksPer90'] = (df_new['shotsBlockedPer90']/(100-df_new['AvgTeamPoss']))*50
    #df_new['pAdjPassBlocksPer90'] = (df_new['PassBlocksPer90']/(100-df_new['AvgTeamPoss']))*50
    #df_new['pAdjDrbTklPer90'] = (df_new['DrbTklPer90']/(100-df_new['AvgTeamPoss']))*50
    #df_new['pAdjTklWinPossPer90'] = (df_new['DrbTklPer90']/(100-df_new['AvgTeamPoss']))*50
    #df_new['pAdjDrbPastPer90'] = (df_new['DrbPastPer90']/(100-df_new['AvgTeamPoss']))*50
    df_new['pAdjAerialWinsPer90'] = (df_new['aerialsWonPer90']/(100-df_new['AvgTeamPoss']))*50
    #df_new['pAdjAerialLossPer90'] = (df_new['AerialLossPer90']/(100-df_new['AvgTeamPoss']))*50
    #df_new['pAdjDrbPastAttPer90'] = (df_new['DrbPastAttPer90']/(100-df_new['AvgTeamPoss']))*50
    df_new['TouchCentrality'] = (df_new['touchesPer90']/df_new['TeamTouches90'])*100
    # df['pAdj#OPAPer90'] =(df['#OPAPer90']/(100-df['AvgTeamPoss']))*50
    #df_new['TklPer600OppTouch'] = df_new['tackles'] /(df_new['OppTouches']*(df_new['Min']/df_new['TeamMins']))*600
    #df_new['IntPer600OppTouch'] = df_new['interceptions'] /(df_new['OppTouches']*(df_new['Min']/df_new['TeamMins']))*600
    df_new['pAdjTouchesPer90'] = (df_new['touchesPer90']/(df_new['AvgTeamPoss']))*50
    df_new['CarriesPer50Touches'] = df_new['carries'] / df_new['touches'] * 50
    df_new['ProgCarriesPer50Touches'] = df_new['progressiveCarries'] / df_new['touches'] * 50
    df_new['ProgPassesPer50CmpPasses'] = df_new['progressivePasses'] / df_new['passesComp'] * 50

    df_new[['AvgTeamPoss', 'TeamTouches90', 'pAdjTklPer90', 'pAdjIntPer90', 'pAdjClrPer90', 'pAdjShBlocksPer90', 'pAdjAerialWinsPer90', 'TouchCentrality', 'pAdjTouchesPer90', 'CarriesPer50Touches', 'ProgCarriesPer50Touches', 'ProgPassesPer50CmpPasses']] = df_new[['AvgTeamPoss', 'TeamTouches90', 'pAdjTklPer90', 'pAdjIntPer90', 'pAdjClrPer90', 'pAdjShBlocksPer90', 'pAdjAerialWinsPer90', 'TouchCentrality', 'pAdjTouchesPer90', 'CarriesPer50Touches', 'ProgCarriesPer50Touches', 'ProgPassesPer50CmpPasses']].applymap('{:.2f}'.format).astype(float)

    # save the data in the folder for each respective comp, depening on matches data

    # drop any duplicates and save to folder
    df_new = df_new.drop_duplicates(subset='playerName')
    df_new.to_csv(f'Data/{comp}/{season[5:]}/player-stats/player-advanced-stats.csv')

    return df_new

def player_rankings(df, date, background, fontcolor, facecolor, minutes, league, season):

    Rubik = FontManager('https://github.com/googlefonts/rubik/blob/main/old/version-2/fonts/ttf/Rubik-Bold.ttf')
    Rubik_italic = FontManager('https://github.com/googlefonts/rubik/blob/main/old/version-2/fonts/ttf/Rubik-BoldItalic.ttf')
    background = background
    fontcolor = fontcolor
    
    # some minor data cleaning and creation of new varaibales
    df = df.loc[(df['playerPos'].isin(['AML', 'AMC', 'AMR']))] # Filter by role. we want to exclude players in less attacking roles such as CDM/CM
    df['teamName'] = np.where(df.teamName.isin(['Borussia Mönchengladbach']),'Mönchengladbach', df['teamName']) # table spacing purposes
    df['teamName'] = np.where(df.teamName.isin(['Brighton and Hove Albion']),'Brighton', df['teamName']) # table spacing purposes
    df = df[df['timePlayed'] > minutes]

    # create a copy of the list of dfs and drop all irrelevant columnns to create a model df
    model_league = df.copy()
    model_league = model_league[['playerName', 'xAPer90', 'npxGPer90', 'xOVAPer90', 'xTPer90']] 

    # next steps are to standardize, normalize, and scale the data to 100 (per Liam Henshaw method)
    norm_df = model_league

    # zscore function used to calc position of each raw score in terms of standard deviations from mean 
    # mean being 0 and standard deviations +/- from the mean
    norm_df = (norm_df.select_dtypes(exclude='object')
               .apply(zscore) 
               .join(norm_df.select_dtypes(include='object')))  

    # norm.cdf normalizes the zscores to be between 0 and 1 (mean as 0.50)
    norm_df = (norm_df.select_dtypes(exclude='object')
               .apply(norm.cdf)
               .join(norm_df.select_dtypes(include='object')))

    # this step sclaes each metric to 100 for visualization and rating purposes
    norm_df[['xAPer90', 'npxGPer90', 'xOVAPer90', 'xTPer90']] = round(norm_df[['xAPer90', 'npxGPer90', 'xOVAPer90', 'xTPer90']] *100,1)

    # copy df and name as final data for rankings
    ranked_df = norm_df.copy()

    # rename columns as will be shown in table
    ranked_df = ranked_df.rename({'xAPer90': 'Play Making',
                                  'npxGPer90': 'Goal Scoring',
                                  'xOVAPer90': 'Contribution Added',
                                  'xTPer90': 'Threat Creation'}, axis=1)

    # now that we have ratings for each metric we can create and overal rating for each player and overall rank
    # overall ratings uses a weighted average approach; currently equally weighting each attribute
    ranked_df['Rating'] = ranked_df['Play Making']*0.25 + ranked_df['Goal Scoring']*0.25 + ranked_df['Contribution Added']*0.25 + ranked_df['Threat Creation']*0.25
    ranked_df = ranked_df.sort_values(by=['Rating'], ascending=True)
    ranked_df['Rank'] = list(range(1, len(ranked_df)+1))[::-1]

    # final step is to create a new variable matching each player with respective team
    teams = df.copy()
    teams = teams[['playerName', 'playerPos', 'playerId', 'teamName', 'teamId', 'xAPer90', 'npxGPer90', 'xOVAPer90', 'xTPer90']] 
    ranked_df = pd.merge(ranked_df, teams, on='playerName', how='outer')
    ranked_df = ranked_df[['playerName', 'teamName', 'Rank', 'Rating', 'Play Making', 'Goal Scoring', 'Contribution Added', 'Threat Creation']]

    # export full df
    ranked_df.to_csv(f'Data/{comp}/{season[5:]}/player-stats/rankings-{date}.csv')
    ranked_df = ranked_df.tail(10)
    
    # create a df for titles of each league; order in same order as index league key
    league_title = league

    ranks = ranked_df.copy()
    ranks_10 = ranks.tail(10)
    ranks_10 = ranks_10.copy()
    ranks_10['playerName'] = np.where(ranks_10['playerName'].isin(['Christoph Baumgartner']),'Baumgartner', ranks_10['playerName']) # table spacing purposes

    # formatting used to save output as of current date
    date = f'{datetime.datetime.now().strftime("%m%d%Y")}'
    minutes = minutes

    # set variables for each annotate
    twitter_handle = '@egudi_analysis' # adjust with your handle
    TITLE = f'{league} - Player Rankings'
    TITLE2 = f'Full {season} season | attacking midfielders* with {minutes}+ minutes played'
    CREDIT = f'Viz by {twitter_handle} | Table design by @sonofacorner | Data via Soccerment'
    NOTE = 'Rankings determined using the @HenshawAnalysis rating model for players classified as midfielders\nFinal rating is equally weighting the above atttributes'
    NOTE2 = '*excludes players with CDM/CM roles'
    METRIC = 'PLAY MAKING: expected assists (xA) per 90'
    METRIC2 = 'GOAL SCORING: expected goals (xG) per 90'
    METRIC4 = 'CONTRIBUTION ADDED: expected offensive value added (xOVA) per 90'
    METRIC5 = 'THREAT CREATION: expected threat (xT) per 90'

    # create your figure
    fig = plt.figure(figsize=(8, 2.5), dpi = 200)
    fig.set_facecolor(facecolor)
    ax = plt.subplot(111, facecolor = facecolor)

    ncols = 9 
    nrows = ranks_10.shape[0]

    ax.set_xlim(0, ncols)
    ax.set_ylim(0, nrows)

    # -- The CMAP fot the gradient
    cmap = cm.get_cmap(gradient)

    for y in range(nrows):
        # - player name
        x = 0.1
        player = ranks_10['playerName'].iloc[y]
        label_ = f'{player}'
        ax.annotate(
            xy=(x,y+0.35),
            text=label_,
            ha='left',
            va='center',
            color='white',
            fontproperties=Rubik_italic.prop,
            fontsize=7
        )   
        # - team name
        x = 2
        team = ranks_10['teamName'].iloc[y]
        label_ = f'{team}'
        ax.annotate(
            xy=(x,y+0.35),
            text=label_,
            ha='left',
            va='center',
            color='white',
            fontproperties=Rubik_italic.prop,
            fontsize=7
        )
        
        # -- rank
        rank = ranks_10['Rank'].iloc[y]
        label_r = f'{rank}'
        x = 4
        ax.annotate(
            xy=(x,y+0.35),
            text=label_r,
            ha='left',
            va='center',
            color='white',
            fontproperties=Rubik_italic.prop,
            fontsize=7
        )
        
        # -- rating
        rating = ranks_10['Rating'].iloc[y]
        rating_change = 0
        label_rt = f'{int(rating)}'
        x = 4.65
        ax_text(
            x=x,y=y+0.35,
            s=label_rt,
            ha='center',
            va='center',
            size=7,
            ax=ax,
            color='w',
            fontproperties=Rubik.prop
        )
        
        # -- play making
        play_making = ranks_10['Play Making'].iloc[y]
        play_making_change = 0
        x = 5.5
        if play_making_change < 0:
            high_color='#ffa07a'
            label_p = f'{int(play_making)} <({(play_making_change):.1f})>'
        elif play_making_change == 0:
            high_color='gray'
            label_p = f'{int(play_making)}   <({(play_making_change):.1f})>'
        elif play_making_change > 0:
            high_color='#00ff00'
            label_p = f'{int(play_making)} <(+{(play_making_change):.1f})>'
        else:
            high_color='#1d2849'
            label_p = f'{int(play_making)}  <          >'
        ax_text(
            x=x,y=y+0.35,
            s=label_p,
            ha='center',
            va='center',
            highlight_textprops=[{'size':5,'color':high_color}],
            size=7,
            ax=ax,
            color='w',
            fontproperties=Rubik.prop
        )
        
        # -- Adding the colors
        # -- We subtract and add .5 beacause the width of our colum in 1, 1/2 = .5
        x = 5.5
        ax.fill_between(
            x=[(x - .5), (x + .5)],
            y1=y,
            y2=y + 1,
            color=cmap(play_making/100),
            zorder=2,
            ec="None",
            alpha=0.75
        )
        
        # -- goal_scoring
        goal_scoring = ranks_10['Goal Scoring'].iloc[y]
        goal_scoring_change = 0
        x = 6.5
        if goal_scoring_change < 0:
            high_color='#ffa07a'
            label_g = f'{int(goal_scoring)} <({(goal_scoring_change):.1f})>'
        elif goal_scoring_change == 0:
            high_color='gray'
            label_g = f'{int(goal_scoring)}   <({(goal_scoring_change):.1f})>'
        elif goal_scoring_change > 0:
            high_color='#00ff00'
            label_g = f'{int(goal_scoring)} <(+{(goal_scoring_change):.1f})>'
        else:
            high_color='#1d2849'
            label_g = f'{int(goal_scoring)}  <          >'
        ax_text(
            x=x,y=y+0.35,
            s=label_g,
            ha='center',
            va='center',
            highlight_textprops=[{'size':5,'color':high_color}],
            size=7,
            ax=ax,
            color='w',
            fontproperties=Rubik.prop
        )
        
        # -- Adding the colors
        x = 6.5
        ax.fill_between(
            x=[(x - .5), (x + .5)],
            y1=y,
            y2=y + 1,
            color=cmap(goal_scoring/100),
            zorder=2,
            ec="None",
            alpha=0.75
        )
        
        # -- contribution added
        contribution_added = ranks_10['Contribution Added'].iloc[y]
        contribution_added_change = 0
        x = 8.5
        if contribution_added_change < 0:
            high_color='#ffa07a'
            label_c = f'{int(contribution_added)} <({(contribution_added_change):.1f})>'
        elif contribution_added_change == 0:
            high_color='gray'
            label_c = f'{int(contribution_added)}   <({(contribution_added_change):.1f})>'
        elif contribution_added_change > 0:
            high_color='#00ff00'
            label_c = f'{int(contribution_added)} <(+{(contribution_added_change):.1f})>'
        else:
            high_color='#1d2849'
            label_c = f'{int(contribution_added)}  <          >'
        ax_text(
            x=x,y=y+0.35,
            s=label_c,
            ha='center',
            va='center',
            highlight_textprops=[{'size':5,'color':high_color}],
            size=7,
            ax=ax,
            color='w',
            fontproperties=Rubik.prop
        )
        
        # -- Adding the colors
        x = 8.5
        ax.fill_between(
            x=[(x - .5), (x + .5)],
            y1=y,
            y2=y + 1,
            color=cmap(contribution_added/100),
            zorder=2,
            ec="None",
            alpha=0.75
        )
        
        # -- Threat Creation
        threat_creation = ranks_10['Threat Creation'].iloc[y]
        threat_creation_change = 0
        x = 9.5
        if threat_creation_change < 0:
            high_color='#ffa07a'
            label_t = f'{int(threat_creation)} <({(threat_creation_change):.1f})>'
        elif threat_creation_change == 0:
            high_color='gray'
            label_t = f'{int(threat_creation)}   <({(threat_creation_change):.1f})>'
        elif threat_creation_change > 0:
            high_color='#00ff00'
            label_t = f'{int(threat_creation)} <(+{(threat_creation_change):.1f})>'
        else:
            high_color='#1d2849'
            label_t = f'{int(threat_creation)}  <          >'
        ax_text(
            x=x,y=y+0.35,
            s=label_t,
            ha='center',
            va='center',
            highlight_textprops=[{'size':5,'color':high_color}],
            size=7,
            ax=ax,
            color='w',
            fontproperties=Rubik.prop
        )

        # -- Adding the colors
        x = 9.5
        ax.fill_between(
            x=[(x - .5), (x + .5)],
            y1=y,
            y2=y + 1,
            color=cmap(threat_creation/100),
            zorder=2,
            ec="None",
            alpha=0.75
        )
        
    # Table borders
    ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [nrows, nrows], lw = 1.5, color = 'white', marker = '', zorder = 4)
    ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [0, 0], lw = 1.5, color = 'white', marker = '', zorder = 4)
    for x in range(nrows):
        if x == 0:
            continue
        ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [x, x], lw = 1.15, color = 'gray', ls = ':', zorder = 3 , marker = '')
        
    # ----------------------------------------------------------------
    # - Column titles
    ax.set_axis_off()

    fig_text(
        x = 0.125, y = 1.175, 
        s = f'{TITLE}',
        va = "bottom", ha = "left",
        fontsize = 16, color = "w", fontproperties=Rubik.prop
    )

    fig_text(
        x = 0.125, y = 1.112, 
        s = f'{TITLE2}',
        va = "bottom", ha = "left",
        fontsize = 10, color = "w", fontproperties=Rubik.prop
    )

    fig_text(
        x = 0.125, y = 1.05, 
        s = f'{CREDIT}',
        va = "bottom", ha = "left",
        fontsize = 7, color = "w", fontproperties=Rubik.prop
    );

    fig_text(
        x = 0.1325, y = 0.9, 
        s = f'Player',
        va = "bottom", ha = "left",
        fontsize = 7, color = "w", fontproperties=Rubik.prop
    );

    fig_text(
        x = 0.28, y = 0.9, 
        s = f'Team',
        va = "bottom", ha = "left",
        fontsize = 7, color = "w", fontproperties=Rubik.prop
    );

    fig_text(
        x = 0.42, y = 0.9, 
        s = f'Rank',
        va = "bottom", ha = "left",
        fontsize = 7, color = "w", fontproperties=Rubik.prop
    );

    fig_text(
        x = 0.4635, y = 0.9, 
        s = f'Rating',
        va = "bottom", ha = "left",
        fontsize = 7, color = "w", fontproperties=Rubik.prop
    );

    fig_text(
        x = 0.515, y = 0.9, 
        s = f'       Play\n    Making',
        va = "bottom", ha = "left",
        fontsize = 7, color = "w", fontproperties=Rubik.prop
    );

    fig_text(
        x = 0.6025, y = 0.9, 
        s = '   Goal\nScoring',
        va = "bottom", ha = "left",
        fontsize = 7, color = "w", fontproperties=Rubik.prop
    );

    fig_text(
        x = 0.7415, y = 0.9, 
        s = 'Contribution\n        Added',
        va = "bottom", ha = "left",
        fontsize = 7, color = "w", fontproperties=Rubik.prop
    );

    fig_text(
        x = 0.83275, y = 0.9, 
        s = '  Threat\nCreation',
        va = "bottom", ha = "left",
        fontsize = 7, color = "w", fontproperties=Rubik.prop
    );

    fig_text(
        x = 0.465, y = 0.05, 
        s = f'{NOTE}\n{NOTE2}',
        va = "center", ha = "left",
        fontsize = 5, color = "w", fontproperties=Rubik.prop
    );

    fig_text(
        x = 0.125, y = 0.0, 
        s = f'{METRIC}\n{METRIC2}\n{METRIC4}\n{METRIC5}',
        va = "center", ha = "left",
        fontsize = 5, color = "w", fontproperties=Rubik.prop
    );

def team_match_logs(events_df, comp):

    non_passes = events_df[events_df['type'] != 'Pass']
    non_passes = calc_xg(non_passes)

    passes = events_df[events_df['type'] == 'Pass']
    passes = xThreat(passes)

    team_events_df = pd.concat([passes, non_passes], axis=0).reset_index(drop=True)

    # Step 1: Split DataFrame into multiple DataFrames based on unique match Ids
    match_ids = team_events_df['matchId'].unique()

    matches = []
    for match_id in match_ids:
        match_df = team_events_df[team_events_df['matchId'] == match_id]
        matches.append(match_df)

    match_stats = []

    for match in matches:
        
        team_stats = match.groupby(['teamId', 'matchId', 'startDate', 'h_a']).agg({'xG': ['sum'],
                                                                                   'xT': ['sum']})
        team_stats.columns = ['xG', 'xT']
        team_stats = team_stats.reset_index()
        team_stats_rev = team_stats.copy(deep=True)
        
        # Find the two unique teamIds
        unique_team_ids = team_stats_rev['teamId'].unique()
        # Create a dictionary to map each teamId to its corresponding opposite teamId
        team_id_mapping = {unique_team_ids[0]: unique_team_ids[1], unique_team_ids[1]: unique_team_ids[0]}
        # Replace the values in the 'teamId' column with their corresponding opposite teamIds
        team_stats_rev['teamId'] = team_stats_rev['teamId'].map(team_id_mapping)
        
        team_stats_rev = team_stats_rev.rename(columns={'xG': 'xGA',
                                                    'xT': 'xTA'})
        team_stats_rev = team_stats_rev[['teamId', 'xGA', 'xTA']]

        team_stats = pd.merge(team_stats, team_stats_rev, left_on='teamId', right_on='teamId', suffixes=('', '_rev'))
        
        match_stats.append(team_stats)

        matches_summary = pd.concat(match_stats).reset_index(drop=True)
    
    matches_summary = pd.concat(match_stats).reset_index(drop=True)
    matches_summary['startDate'] = matches_summary['startDate']

    # Convert the column to pandas datetime data type
    matches_summary['startDate'] = pd.to_datetime(matches_summary['startDate'])
    # Extract only the date portion and overwrite the column with the date values
    matches_summary['startDate'] = matches_summary['startDate'].dt.date

    # getting players dataframe
    match_teams_df = pd.DataFrame()
    team_Ids = []
    team_names = []

    for match in matches_data:
        
        home_df = match['home']
        home_Id = home_df['teamId']
        home_team = home_df['name']
        
        team_Ids.append(home_Id)
        team_names.append(home_team)
        
    match_teams_df['teamId'] = team_Ids  
    match_teams_df['teamName'] = team_names

    match_teams_df.drop_duplicates(subset=['teamName'], inplace=True)

    matches_summary = pd.merge(matches_summary, match_teams_df, on='teamId', how='left')

    matches_summary['xGD'] = matches_summary['xG'] - matches_summary['xGA']
    matches_summary['xTD'] = matches_summary['xT'] - matches_summary['xTA']

    # Create a self-join of the DataFrame on 'matchId'
    opponents = matches_summary.merge(matches_summary, on='matchId', suffixes=('', '_opponent'))

    # Keep only the rows where the 'teamName' and 'teamName_opponent' are different
    opponents = opponents[opponents['teamName'] != opponents['teamName_opponent']]

    # Select the relevant columns and rename the 'teamName_opponent' column to 'opponent'
    opponents = opponents[['teamName_opponent']].rename(columns={'teamName_opponent': 'opponent'})

    # Reset the index of the 'opponents' DataFrame to ensure consistent indices
    opponents.reset_index(drop=True, inplace=True)

    # Concatenate the original DataFrame and the merged DataFrame to get the final DataFrame
    matches_summary['opponent'] = opponents
    matches_summary = matches_summary[['teamName', 'teamId', 'opponent', 'matchId', 'startDate', 'h_a', 'xG', 'xT', 'xGA', 'xTA', 'xGD', 'xTD']]

    team_names = matches_summary.teamName.unique()

    match_logs = []

    for team in team_names:
        
        team_data = matches_summary[matches_summary['teamName'] == team].reset_index(drop=True)
        team = team_data['teamName'].unique()[0]
        
        team_data.to_csv(f'Data/{comp}/{season[5:]}/match-logs/{team}-match-logs.csv')    
        match_logs.append(team_data)

def add_team_abreviations():

    # Replace values in 'column_name' based on condition
    team_abbreviations =  {'Arsenal': 'ARS',
                           'Aston Villa': 'AVL',
                           'Brentford': 'BRE',
                           'Brighton': 'BRI',
                           'Luton': 'LUT',
                           'Chelsea': 'CHE',
                           'Fulham': 'FUL',
                           'Everton': 'EVE',
                           'Sheff Utd': 'SHU',
                           'Burnley': 'BUR',
                           'Liverpool': 'LIV',
                           'Man City': 'MCI',
                           'Man Utd': 'MUN',
                           'Newcastle': 'NEW',
                           'Tottenham': 'TOT',
                           'West Ham': 'WHU',
                           'Wolves': 'WOL',
                           'Nottingham Forest': 'NFO',
                           'Crystal Palace': 'CRY',
                           'Bournemouth': 'BOU'}

    # Add a new column 'team_abbreviation' using the mapping
    final_df['team_abbreviation'] = final_df['teamName'].map(team_abbreviations)

    # Replace values in 'column_name' based on condition
    team_abbreviations =  {'Deportivo Alaves': 'ALV',
                           'Almeria': 'ALM',
                           'Athletic Bilbao': 'ATB',
                           'Atletico': 'ATM',
                           'Barcelona': 'FCB',
                           'Cadiz': 'CCF',
                           'Celta Vigo': 'CLV',
                           'Getafe': 'GET',
                           'Girona': 'GIR',
                           'Granada': 'GCF',
                           'Las Palmas': 'LAP',
                           'Osasuna': 'OSA',
                           'Rayo Vallecano': 'RAY',
                           'Mallorca': 'RCD',
                           'Real Betis': 'BET',
                           'Real Madrid': 'MAD',
                           'Real Sociedad': 'SOC',
                           'Sevilla': 'SEV',
                           'Valencia': 'VAL',
                           'Villarreal': 'VIL'}
        