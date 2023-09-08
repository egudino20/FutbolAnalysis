# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 14:38:46 2020
@author: aliha
@twitter: rockingAli5 
"""

# import necessary packages

# embedded packages
import main

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
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.image as mpimg
import matplotlib.ticker as ticker
import matplotlib.patheffects as path_effects
from mplsoccer import PyPizza, add_image, FontManager
from matplotlib.colors import Normalize
import matplotlib.font_manager as fm
import matplotlib.colors as mcolors

# web scraping
from selenium import webdriver

# text and annotation
from itertools import combinations
from highlight_text import fig_text, ax_text
from highlight_text import fig_text, ax_text, HighlightText
from adjustText import adjust_text

# machine learning / statiscal analysis
from scipy import stats
from scipy.stats import poisson
from scipy.interpolate import interp1d, make_interp_spline
from scipy.interpolate import interp2d
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp2d
import pickle
from math import pi

# file handling and parsing
import os
import json
import glob

# Images
from PIL import Image
import requests
from io import BytesIO

# other
import datetime

Rubik = FontManager('https://github.com/googlefonts/rubik/blob/main/old/version-2/fonts/ttf/Rubik-Bold.ttf')
Rubik_italic = FontManager('https://github.com/googlefonts/rubik/blob/main/old/version-2/fonts/ttf/Rubik-BoldItalic.ttf')
#Rubkik_bold = FontManager('https://github.com/googlefonts/rubik/blob/main/fonts/ttf/Rubik-ExtraBold.ttf?raw=true')

def xThreat_old(df):

    colnames = ['col1', 'col2', 'col3', 'col4',
                'col5', 'col6', 'col7', 'col8',
                'col9', 'col10','col11','col12']

    # import xT grid
    xT = pd.read_csv('xT_Grid.csv', names=colnames, header=None)

    # convert xT df into an np array
    xT = np.array(xT)

    # calculate numer of rows and colums
    xT_rows, xT_cols = xT.shape

    # create bins to determine which section of the action started and ended
    df['x1_bin'] = pd.cut(df['x'], bins = xT_cols, labels=False)
    df['y1_bin'] = pd.cut(df['y'], bins = xT_rows, labels=False)
    df['x2_bin'] = pd.cut(df['endX'], bins = xT_cols, labels=False)
    df['y2_bin'] = pd.cut(df['endY'], bins = xT_rows, labels=False)

    # match xT value from grid to its corresponding bin by creating 2 new cols
    df['start_zone_value'] = df[['x1_bin', 'y1_bin']].apply(lambda x: xT[x[1]][x[0]], axis=1)
    df['end_zone_value'] = df[['x2_bin', 'y2_bin']].apply(lambda x: xT[x[1]][x[0]], axis=1)

    # calc xT by subtractinc start minus end zone probability
    df['xT'] = df['end_zone_value'] - df['start_zone_value']

def xThreat(events_df, interpolate=True, pitch_length=100, pitch_width=100):
    """ Add expected threat metric to whoscored-style events dataframe
    Function to apply Karun Singh's expected threat model to all successful pass and carry events within a
    whoscored-style events dataframe. This imposes a 12x8 grid of expected threat values on a standard pitch. An
    interpolate parameter can be passed to impose a continous set of expected threat values on the pitch.
    Args:
        events_df (pandas.DataFrame): whoscored-style dataframe of event data. Events can be from multiple matches.
        interpolate (bool, optional): selection of whether to impose a continous set of xT values. True by default.
        pitch_length (float, optional): extent of pitch x coordinate (based on event data). 100 by default.
        pitch_width (float, optional): extent of pitch y coordinate (based on event data). 100 by default.
    Returns:
        pandas.DataFrame: whoscored-style dataframe of events, including expected threat
    """

    # Define function to get cell in which an x, y value falls
    def get_cell_indexes(x_series, y_series, cell_cnt_l, cell_cnt_w, field_length, field_width):
        xi = x_series.divide(field_length).multiply(cell_cnt_l)
        yj = y_series.divide(field_width).multiply(cell_cnt_w)
        xi = xi.astype('int64').clip(0, cell_cnt_l - 1)
        yj = yj.astype('int64').clip(0, cell_cnt_w - 1)
        return xi, yj

    # Initialise output
    events_out = pd.DataFrame()

    # Get Karun Singh expected threat grid
    path = "https://karun.in/blog/data/open_xt_12x8_v1.json"
    xt_grid = pd.read_json(path)
    init_cell_count_w, init_cell_count_l = xt_grid.shape

    # Isolate actions that involve successfully moving the ball (successful carries and passes)
    move_actions = events_df[(events_df['type'].isin(['Carry', 'Pass'])) &
                             (events_df['outcomeType'] == 'Successful')]

    # Set-up bilinear interpolator if user chooses to
    if interpolate:
        cell_length = pitch_length / init_cell_count_l
        cell_width = pitch_width / init_cell_count_w
        x = np.arange(0.0, pitch_length, cell_length) + 0.5 * cell_length
        y = np.arange(0.0, pitch_width, cell_width) + 0.5 * cell_width
        interpolator = interp2d(x=x, y=y, z=xt_grid.values, kind='linear', bounds_error=False)
        interp_cell_count_l = int(pitch_length * 10)
        interp_cell_count_w = int(pitch_width * 10)
        xs = np.linspace(0, pitch_length, interp_cell_count_l)
        ys = np.linspace(0, pitch_width, interp_cell_count_w)
        grid = interpolator(xs, ys)
    else:
        grid = xt_grid.values

    # Set cell counts based on use of interpolator
    if interpolate:
        cell_count_l = interp_cell_count_l
        cell_count_w = interp_cell_count_w
    else:
        cell_count_l = init_cell_count_l
        cell_count_w = init_cell_count_w

    # For each match, apply expected threat grid (we go by match to avoid issues with identical event indicies)
    for match_id in move_actions['matchId'].unique():
        match_move_actions = move_actions[move_actions['matchId'] == match_id]

        # Get cell indices of start location of event
        startxc, startyc = get_cell_indexes(match_move_actions['x'], match_move_actions['y'], cell_count_l,
                                            cell_count_w, pitch_length, pitch_width)
        endxc, endyc = get_cell_indexes(match_move_actions['endX'], match_move_actions['endY'], cell_count_l,
                                        cell_count_w, pitch_length, pitch_width)

        # Calculate xt at start and end of events
        xt_start = grid[startyc.rsub(cell_count_w - 1), startxc]
        xt_end = grid[endyc.rsub(cell_count_w - 1), endxc]

        # Build dataframe of event index and net xt
        ratings = pd.DataFrame(data=xt_end-xt_start, index=match_move_actions.index, columns=['xT'])

        # Merge ratings dataframe to all match events
        match_events_and_ratings = pd.merge(left=events_df[events_df['matchId'] == match_id], right=ratings,
                                            how="left", left_index=True, right_index=True)
        events_out = pd.concat([events_out, match_events_and_ratings], ignore_index=True, sort=False)
        events_out['xT_gen'] = events_out['xT'].apply(lambda xt: xt if (xt > 0 or xt != xt) else 0)

    return events_out

def xThreatReverse(df):

    colnames = ['col1', 'col2', 'col3', 'col4',
                'col5', 'col6', 'col7', 'col8',
                'col9', 'col10', 'col11', 'col12']

    # import xT grid
    xT = pd.read_csv('xT_Grid.csv', names=colnames, header=None)
    xT = xT[xT.columns[::-1]]

    # convert xT df into an np array
    xT = np.array(xT)

    # calculate numer of rows and colums
    xT_rows, xT_cols = xT.shape

    # create bins to determine which section of the action started and ended
    df['x1_bin'] = pd.cut(df['x'], bins = xT_cols, labels=False)
    df['y1_bin'] = pd.cut(df['y'], bins = xT_rows, labels=False)
    df['x2_bin'] = pd.cut(df['endX'], bins = xT_cols, labels=False)
    df['y2_bin'] = pd.cut(df['endY'], bins = xT_rows, labels=False)

    # match xT value from grid to its corresponding bin by creating 2 new cols
    df['start_zone_value'] = df[['x1_bin', 'y1_bin']].apply(lambda x: xT[x[1]][x[0]], axis=1)
    df['end_zone_value'] = df[['x2_bin', 'y2_bin']].apply(lambda x: xT[x[1]][x[0]], axis=1)

    # calc xT by subtractinc start minus end zone probability
    df['xT'] = df['end_zone_value'] - df['start_zone_value']

# Custom function to create features in Opta Event data for all possible Event Types, Qualifier Types, and Satisfied Event Types, as well as multifeature attributes
def data_preparation(df):
        
    ## Create individual columns for Event Types, Qualifier Types, and Satisfied Event Types
    for i in range(1,229+1):   # Qualifier Types: 1-229
        df['isQualifierType_'+str(i)] = np.nan
        df['isQualifierType_'+str(i)] = np.where(df['qualifiers'].str.contains(f"value\': {i},", na=False), 1, 0)

    for i in range(0,219+1):   # Satisified Event Types: 0-219
        df['isSatisfiedEventType_'+str(i)] = np.nan
        df['isSatisfiedEventType_'+str(i)] = np.where(df['satisfiedEventsTypes'].str.contains(f", {i},", na=False), 1, 0)

    
    ## Create Dictionaries from reference data to rename the newly created features
    
    ### Create pandas DataFrames from the reference data
    df_event_types_ref = pd.read_csv('Data/opta_event_types.csv')
    df_qualifier_types_ref = pd.read_csv('Data/opta_qualifier_types.csv')
    

    ### Rename Events Types features

    ### Remove Null values
    df_event_types_ref = df_event_types_ref[df_event_types_ref['eventTypeId'].notna()]

    ### Convert data types
    df_event_types_ref['eventTypeId'] = df_event_types_ref['eventTypeId'].astype(int)

    ### 
    df_event_types_ref['eventTypeName'] = df_event_types_ref['eventTypeName'].str.title().str.replace(' ', '').str.replace('/', '').str.replace('-', '')
    df_event_types_ref['eventTypeName'] = 'is' + df_event_types_ref['eventTypeName'].astype(str)
    
    ### Rename Qualifier Types features

    ####
    df_qualifier_types_ref = df_qualifier_types_ref[df_qualifier_types_ref['qualifierTypeId'].notna()]

    ####
    df_qualifier_types_ref['qualifierTypeId'] = df_qualifier_types_ref['qualifierTypeId'].astype(int)
    df_qualifier_types_ref['qualifierTypeId'] = 'isQualifierType_' + df_qualifier_types_ref['qualifierTypeId'].astype(str)

    ####
    df_qualifier_types_ref['qualifierTypeName'] = df_qualifier_types_ref['qualifierTypeName'].str.title().str.replace(' ', '').str.replace('/', '').str.replace('-', '')
    df_qualifier_types_ref['qualifierTypeName'] = 'is' + df_qualifier_types_ref['qualifierTypeName'].astype(str)


    ### Create individual dictionaries for Event Types, Qualifier Types, and Satisfied Event Types features, to be mapped to the DataFrame
    dict_event_types = dict(zip(df_event_types_ref['eventTypeId'], df_event_types_ref['eventTypeName']))
    dict_qualifier_types = dict(zip(df_qualifier_types_ref['qualifierTypeId'], df_qualifier_types_ref['qualifierTypeName']))
    
    ## Create multifeature attributes - some logic not yet finished
    df['isLeftFooted'] = np.where( (df['shotRightFoot'] == False) &
                                  ((df['shotLeftFoot'] == True)
                                ) 
                               , 1, 0
                               )
    
    df['isRightFooted'] = np.where( (df['shotRightFoot'] == True) &
                                  ((df['shotLeftFoot'] == False)
                                ) 
                               , 1, 0
                               )
    
    df['isHead'] = np.where(df['shotBodyType'] == 'Head' 
                           , 1, 0
                           )
    
    df['isOtherBodyType'] = np.where(df['shotBodyType'] == 'OtherBodyPart' 
                           , 1, 0
                           )
    
    df['isRegularPlay'] = np.where(df['situation'] == 'OpenPlay' 
                           , 1, 0
                                  )
    
    df['isThrowIn'] = np.where(df['throwIn'] == True 
                           , 1, 0
                                  )
    
    df['isDirectFree'] = np.where(df['situation'] == 'DirectFreekick' 
                           , 1, 0
                                  )
    
    df['isFromCorner'] = np.where(df['situation'] == 'FromCorner' 
                           , 1, 0
                                  )
    
    df['isSetPiece'] = np.where(df['situation'] == 'SetPiece' 
                           , 1, 0
                                  )
    
    df['isOwnGoal'] = np.where(df['goalOwn'] == True 
                           , 1, 0
                                  )
                            
    df['isPenalty'] = np.where( (df['penaltyMissed'] == False) &
                               ((df['penaltyScored'] == False)
                                ) 
                               , 0, 1
                               )
    
    df['isGoal'] = np.where(df['isGoal'] == True 
                               ,1, 0
                               )

# Custom function to create features in Opta Event data for all possible Event Types, Qualifier Types, and Satisfied Event Types, as well as multifeature attributes
def data_preparation_xA(df):
        
    ## Create individual columns for Event Types, Qualifier Types, and Satisfied Event Types
    for i in range(1,229+1):   # Qualifier Types: 1-229
        df['isQualifierType_'+str(i)] = np.nan
        df['isQualifierType_'+str(i)] = np.where(df['qualifiers'].str.contains(f"value\': {i},", na=False), 1, 0)

    for i in range(0,219+1):   # Satisified Event Types: 0-219
        df['isSatisfiedEventType_'+str(i)] = np.nan
        df['isSatisfiedEventType_'+str(i)] = np.where(df['satisfiedEventsTypes'].str.contains(f", {i},", na=False), 1, 0)

    
    ## Create Dictionaries from reference data to rename the newly created features
    
    ### Create pandas DataFrames from the reference data
    df_event_types_ref = pd.read_csv('Data/opta_event_types.csv')
    df_qualifier_types_ref = pd.read_csv('Data/opta_qualifier_types.csv')
    

    ### Rename Events Types features

    ### Remove Null values
    df_event_types_ref = df_event_types_ref[df_event_types_ref['eventTypeId'].notna()]

    ### Convert data types
    df_event_types_ref['eventTypeId'] = df_event_types_ref['eventTypeId'].astype(int)

    ### 
    df_event_types_ref['eventTypeName'] = df_event_types_ref['eventTypeName'].str.title().str.replace(' ', '').str.replace('/', '').str.replace('-', '')
    df_event_types_ref['eventTypeName'] = 'is' + df_event_types_ref['eventTypeName'].astype(str)
    
    ### Rename Qualifier Types features

    ####
    df_qualifier_types_ref = df_qualifier_types_ref[df_qualifier_types_ref['qualifierTypeId'].notna()]

    ####
    df_qualifier_types_ref['qualifierTypeId'] = df_qualifier_types_ref['qualifierTypeId'].astype(int)
    df_qualifier_types_ref['qualifierTypeId'] = 'isQualifierType_' + df_qualifier_types_ref['qualifierTypeId'].astype(str)

    ####
    df_qualifier_types_ref['qualifierTypeName'] = df_qualifier_types_ref['qualifierTypeName'].str.title().str.replace(' ', '').str.replace('/', '').str.replace('-', '')
    df_qualifier_types_ref['qualifierTypeName'] = 'is' + df_qualifier_types_ref['qualifierTypeName'].astype(str)


    ### Create individual dictionaries for Event Types, Qualifier Types, and Satisfied Event Types features, to be mapped to the DataFrame
    dict_event_types = dict(zip(df_event_types_ref['eventTypeId'], df_event_types_ref['eventTypeName']))
    dict_qualifier_types = dict(zip(df_qualifier_types_ref['qualifierTypeId'], df_qualifier_types_ref['qualifierTypeName']))
    
    # column isCross determines if pass is cross or not
    df['isCross'] = np.where( (df['passCrossAccurate'] == True
                                ) 
                               , 1, 0
                               )
    
    # column isHead determines if pass is header or not
    df['isHead'] = np.where( (df['passHead'] == True
                                ) 
                               , 1, 0
                               )
     
    # column isThroughBall determines if pass is throughball or not
    df['isThroughBall'] = np.where( (df['passThroughBallAccurate'] == True
                                ) 
                               , 1, 0
                               )
            
    # column isRightFoot determines if pass is right foot or not
    df['isRightFoot'] = np.where( (df['passRight'] == True
                                ) 
                               , 1, 0
                               )
    
    # column isRightFoot determines if pass is left foot or not
    df['isLeftFoot'] = np.where( (df['passLeft'] == True
                                ) 
                               , 1, 0
                               )
            
    # column isFreeKick determines if pass is FK or not
    df['isFreeKick'] = np.where( (df['passFreekickAccurate'] == True
                                ) 
                               , 1, 0
                               )
            
    # column isThrowIn determines if pass is throw in or not
    df['isThrowIn'] = np.where( (df['throwIn'] == True
                                ) 
                               , 1, 0
                               )
            
    # column isCorner determines if pass is corner or not
    df['isCorner'] = np.where( (df['passCornerAccurate'] == True
                                ) 
                               , 1, 0
                               )
            
    # column isOpenPlay determines if pass is open play or not          
    df['isOpenPlay'] = np.where( (df['passCornerAccurate'] == False) &
                                  ((df['passFreekickAccurate'] == False) &
                                   ((df['throwIn'] == False)
                                   )
                                ) 
                               , 1, 0
                               )
    
    # column isCross determines if pass is corner or not
    df['isAssist'] = np.where( (df['assist'] == True
                                ) 
                               , 1, 0
                               )
    
    return df

def createShotmap(match_data, events_df, team, opponent, pitchcolor, shotcolor, goalcolor, 
                  home_color, away_color, titlecolor, legendcolor, gamma, ax):

    customcmap = LinearSegmentedColormap.from_list('custom cmap', [home_color,'#1d2849', away_color], gamma=gamma)

    # load the model from disk
    loaded_model_op = pickle.load(open('Models/expected_goals_model_lr.sav', 'rb'))
    loaded_model_non_op = pickle.load(open('Models/expected_goals_model_lr_v2.sav', 'rb'))

    # getting home team id and venue
    h_teamId = match_data['home']['teamId']
    h_venue = 'home'
    
    # getting away team id and venue
    a_teamId = match_data['away']['teamId']
    a_venue = 'away'
        
    total_shots = events_df.loc[events_df['isShot']==True].reset_index(drop=True)
    total_shots = total_shots[total_shots['period'] != 'PenaltyShootout']
    data_preparation(total_shots)
    total_shots['distance_to_goal'] = np.sqrt(((100 - total_shots['x'])**2) + ((total_shots['y'] - (100/2))**2) )
    total_shots['distance_to_center'] = abs(total_shots['y'] - 100/2)
    total_shots['angle'] = np.absolute(np.degrees(np.arctan((abs((100/2) - total_shots['y'])) / (100 - total_shots['x']))))
    total_shots['isFoot'] = np.where(((total_shots['isLeftFooted'] == 1) | (total_shots['isRightFooted'] == 1)) &
                                              (total_shots['isHead'] == 0)
                                            , 1, 0
                                            )

    features_op = ['distance_to_goal',
           #'distance_to_centerM',    # commented out as elected to go with 'angle' instead
            'angle',
            'isFoot',
            'isHead'
           ]
    
    features_non_op = ['distance_to_goal',
           #'distance_to_centerM',    # commented out as elected to go with 'angle' instead
            'angle',
            'isFoot',
            'isHead',
            'isDirectFree',
            'isSetPiece',
            'isFromCorner'
           ]

    # split model to only open play shots to apply open play model
    # calc xG for open play shots only
    total_shots_op = total_shots.loc[total_shots['isRegularPlay'] == 1]
    shots_df_op = total_shots_op[features_op]
    xG = loaded_model_op.predict_proba(shots_df_op)[:,1]
    shots_df_op['xG'] = xG
    xg = shots_df_op['xG']
    total_shots_op = pd.merge(total_shots_op, xg, left_index=True, right_index=True)
    
    # split dataframe to only set pieces to apply set piece model
    # calc xG for set pieces only (not including penalties)
    total_shots_non_op = total_shots.loc[total_shots['isRegularPlay'] == 0]
    total_shots_non_op = total_shots_non_op.loc[total_shots_non_op['isPenalty'] == 0]
    shots_df_non_op = total_shots_non_op[features_non_op]
    xG = loaded_model_non_op.predict_proba(shots_df_non_op)[:,1]
    shots_df_non_op['xG'] = xG
    xg = shots_df_non_op['xG']
    total_shots_non_op = pd.merge(total_shots_non_op, xg, left_index=True, right_index=True)    
    
    # split dataframe to only penalties and set xG for penalties to 0.79
    total_shots_pk = total_shots.loc[total_shots['isPenalty'] == 1]
    total_shots_pk['xG'] = 0.79
    
    # combine all three dataframes 
    total_shots = pd.concat([total_shots_op, total_shots_non_op, total_shots_pk], axis=0).reset_index(drop=True)
    
    # combine all columns with se

    h_team_shots = total_shots.loc[total_shots['teamId'] == h_teamId].reset_index(drop=True)
    h_mask_goal = h_team_shots.isGoal == True
    
    a_team_shots = total_shots.loc[total_shots['teamId'] == a_teamId].reset_index(drop=True)
    a_mask_goal = a_team_shots.isGoal == True

    # Setup the pitch
    # orientation='vertical'
    pitch = Pitch(pitch_type='opta', pitch_color='#1d2849', line_color='w',
                  half=False, pad_top=2)
    pitch.draw(ax=ax, tight_layout=True, constrained_layout=True)


    # Plot the home goals
    pitch.scatter(100-h_team_shots[h_mask_goal].x, 100-h_team_shots[h_mask_goal].y, s=(h_team_shots[h_mask_goal].xG * 1900) + 100,
                  zorder=3, label='goal', marker='football', alpha=0.6, ax=ax)
    pitch.scatter(100-h_team_shots[~h_mask_goal].x, 100-h_team_shots[~h_mask_goal].y,
                  edgecolors='w', linewidths=2.5, c=home_color, s=(h_team_shots[~h_mask_goal].xG * 1900) + 100, zorder=2,
                  label='shot', ax=ax)
    pitch.arrows(42, 101.25, 8, 101.25,
                     color=home_color, width=3, lw=3, zorder=6, ax=ax)
    
    # Plot the away goals
    pitch.scatter(a_team_shots[a_mask_goal].x, a_team_shots[a_mask_goal].y, s=(a_team_shots[a_mask_goal].xG * 1900) + 100,
                  zorder=3, marker='football', alpha=0.6, ax=ax)
    pitch.scatter(a_team_shots[~a_mask_goal].x, a_team_shots[~a_mask_goal].y,
                  edgecolors='w', linewidths=2.5, c=away_color, s=(a_team_shots[~a_mask_goal].xG * 1900) + 100, zorder=2,
                  ax=ax)
    pitch.arrows(58, 101.25, 92, 101.25,
                     color=away_color, width=3, lw=3, zorder=6, ax=ax)

    #draw a circle
    h_team_shots_non_pk = h_team_shots[h_team_shots['isPenalty'] != 1]
    h_shot_dist_to_goal = h_team_shots_non_pk.distance_to_goal.mean()
    u_h=0.     #x-position of the center home team
    v=50    #y-position of the center home/away team
    b_h=h_shot_dist_to_goal*1.5    #radius on the y-axis home
    a_h=h_shot_dist_to_goal    #radius on the x-axis home

    a_team_shots_non_pk = a_team_shots[a_team_shots['isPenalty'] != 1]
    a_shot_dist_to_goal = a_team_shots_non_pk.distance_to_goal.mean()
    u_a=100 #x-position of the center away team
    b_a=a_shot_dist_to_goal*1.5    #radius on the y-axis
    a_a=a_shot_dist_to_goal    #radius on the x-axis

    t = np.linspace(0, 2*pi/4, 100)
    t2 = np.linspace(0, -2*pi/4, 100)
    ax.plot( u_h+a_h*np.cos(t) , v+b_h*np.sin(t), linestyle='--', color='darkgray', zorder=5, linewidth=3)
    ax.plot( u_h+a_h*np.cos(t2) , v+b_h*np.sin(t2), linestyle='--', color='darkgray', zorder=5, linewidth=3)

    ax.plot( u_a-a_a*np.cos(t) , v+b_a*np.sin(t), linestyle='--', color='darkgray', zorder=5, linewidth=3)
    ax.plot( u_a-a_a*np.cos(t2) , v+b_a*np.sin(t2), linestyle='--', color='darkgray', zorder=5, linewidth=3)

    # Set the title
    ax.set_title(f'Zone Dominance & Shot Map\n', 
                 fontsize=32, color=titlecolor, 
                 fontproperties=Rubik.prop)

    home_op_shots = h_team_shots[h_team_shots['isRegularPlay'] == 1]
    home_non_op_shots = h_team_shots[h_team_shots['isRegularPlay'] == 0]
    home_pk_shots = h_team_shots[h_team_shots['isPenalty'] == 1]

    away_op_shots = a_team_shots[a_team_shots['isRegularPlay'] == 1]
    away_non_op_shots = a_team_shots[a_team_shots['isRegularPlay'] == 0]
    away_pk_shots = a_team_shots[a_team_shots['isPenalty'] == 1]

    home_goals = len(h_team_shots[h_mask_goal])
    home_shots = len(h_team_shots[~h_mask_goal]) + home_goals
    home_sot = len(h_team_shots[h_team_shots['shotOnTarget'] == True])
    home_xg = round(h_team_shots.xG.sum(),2)
    home_xg_op = round(home_op_shots.xG.sum(), 2)
    home_xg_sp = round(home_non_op_shots.xG.sum() - home_pk_shots.xG.sum(), 2)
    home_xg_pk = round(home_pk_shots.xG.sum(), 2)
    home_xg_shot = round(home_xg / home_shots,2)

    away_goals = len(a_team_shots[a_mask_goal])
    away_shots = len(a_team_shots[~a_mask_goal]) + away_goals
    away_sot = len(a_team_shots[a_team_shots['shotOnTarget'] == True])
    away_xg = round(a_team_shots.xG.sum(),2)
    away_xg_op = round(away_op_shots.xG.sum(), 2)
    away_xg_sp = round(away_non_op_shots.xG.sum() - away_pk_shots.xG.sum(), 2)
    away_xg_pk = round(away_pk_shots.xG.sum(), 2)
    away_xg_shot = round(away_xg / away_shots,2)
    
    annotation_string = (f'<{format(team).upper()}>')
    ax_text(25, 106, annotation_string, ha='center', va='center', fontsize=15,
        ax=ax, color='white', fontproperties=Rubik.prop,
        highlight_textprops=[{'color': home_color}])
    ax_text(25, 103, '<Attacking Direction>', ha='center', va='center', fontsize=10,
        ax=ax, color='white', fontproperties=Rubik.prop,
        highlight_textprops=[{'color': home_color}])
    
    annotation_string2 = (f'<{format(opponent).upper()}>')
    ax_text(75, 106, annotation_string2, ha='center', va='center', fontsize=15,
        ax=ax, color='white', fontproperties=Rubik.prop,
        highlight_textprops=[{'color': away_color}])
    ax_text(75, 103, '<Attacking Direction>', ha='center', va='center', fontsize=10,
        ax=ax, color='white', fontproperties=Rubik.prop,
        highlight_textprops=[{'color': away_color}])
    
    #annotation_string3 = (f'{home_goals}  -  {away_goals}')
    #ax_text(50, 112, annotation_string3, ha='center', va='center', fontsize=20,
    #    ax=ax, color='white', fontproperties=Rubik.prop)
    
    annotation_string4 = (f'Total Shots')
    ax_text(50, 95, annotation_string4, ha='center', va='center', fontsize=15,
        ax=ax, color='white', fontproperties=Rubik.prop)

    annotation_string4 = (f'Shots on Target')
    ax_text(50, 90, annotation_string4, ha='center', va='center', fontsize=15,
        ax=ax, color='white', fontproperties=Rubik.prop)
    
    annotation_string5 = (f'Expected Goals  (xG)')
    ax_text(50, 85, annotation_string5, ha='center', va='center', fontsize=15,
        ax=ax, color='white', fontproperties=Rubik.prop)

    annotation_string5 = (f'xG Open Play')
    ax_text(50, 80, annotation_string5, ha='center', va='center', fontsize=15,
        ax=ax, color='white', fontproperties=Rubik.prop)

    annotation_string5 = (f'xG Set Play')
    ax_text(50, 75, annotation_string5, ha='center', va='center', fontsize=15,
        ax=ax, color='white', fontproperties=Rubik.prop)

    annotation_string5 = (f'xG Penalty')
    ax_text(50, 70, annotation_string5, ha='center', va='center', fontsize=15,
        ax=ax, color='white', fontproperties=Rubik.prop)
    
    annotation_string6 = (f'xG per Shot')
    ax_text(50, 65, annotation_string6, ha='center', va='center', fontsize=15,
        ax=ax, color='white', fontproperties=Rubik.prop)

    annotation_string6 = (f'Mean Shot Distance (m)')
    ax_text(50, 60, annotation_string6, ha='center', va='center', fontsize=15,
        ax=ax, color='white', fontproperties=Rubik.prop)
    annotation_string6 = (f'---')
    ax_text(50, 57, annotation_string6, ha='center', va='center', fontsize=20,
        ax=ax, color='darkgray', fontproperties=Rubik.prop)

    # home stats

    annotation_string4 = (f'<{home_shots}>')
    ax_text(35, 95, annotation_string4, ha='center', va='center', fontsize=15,
        ax=ax, color='white', fontproperties=Rubkik_bold.prop,
        highlight_textprops=[{'color': home_color}])

    annotation_string4 = (f'<{home_sot}>')
    ax_text(35, 90, annotation_string4, ha='center', va='center', fontsize=15,
        ax=ax, color='white', fontproperties=Rubkik_bold.prop,
        highlight_textprops=[{'color': home_color}])
    
    annotation_string5 = (f'<{home_xg}>')
    ax_text(35, 85, annotation_string5, ha='center', va='center', fontsize=15,
        ax=ax, color='white', fontproperties=Rubkik_bold.prop,
        highlight_textprops=[{'color': home_color}])

    annotation_string5 = (f'<{home_xg_op}>')
    ax_text(35, 80, annotation_string5, ha='center', va='center', fontsize=15,
        ax=ax, color='white', fontproperties=Rubkik_bold.prop,
        highlight_textprops=[{'color': home_color}])

    annotation_string5 = (f'<{home_xg_sp}>')
    ax_text(35, 75, annotation_string5, ha='center', va='center', fontsize=15,
        ax=ax, color='white', fontproperties=Rubkik_bold.prop,
        highlight_textprops=[{'color': home_color}])

    annotation_string5 = (f'<{home_xg_pk}>')
    ax_text(35, 70, annotation_string5, ha='center', va='center', fontsize=15,
        ax=ax, color='white', fontproperties=Rubkik_bold.prop,
        highlight_textprops=[{'color': home_color}])
    
    annotation_string6 = (f'<{home_xg_shot}>')
    ax_text(35, 65, annotation_string6, ha='center', va='center', fontsize=15,
        ax=ax, color='white', fontproperties=Rubkik_bold.prop,
        highlight_textprops=[{'color': home_color}])

    annotation_string6 = (f'<{round(h_shot_dist_to_goal,2)}>')
    ax_text(35, 60, annotation_string6, ha='center', va='center', fontsize=15,
        ax=ax, color='white', fontproperties=Rubkik_bold.prop,
        highlight_textprops=[{'color': home_color}])

    # away stats

    annotation_string4 = (f'<{away_shots}>')
    ax_text(65, 95, annotation_string4, ha='center', va='center', fontsize=15,
        ax=ax, color='white', fontproperties=Rubkik_bold.prop,
        highlight_textprops=[{'color': away_color}])

    annotation_string4 = (f'<{away_sot}>')
    ax_text(65, 90, annotation_string4, ha='center', va='center', fontsize=15,
        ax=ax, color='white', fontproperties=Rubkik_bold.prop,
        highlight_textprops=[{'color': away_color}])
    
    annotation_string5 = (f'<{away_xg}>')
    ax_text(65, 85, annotation_string5, ha='center', va='center', fontsize=15,
        ax=ax, color='white', fontproperties=Rubkik_bold.prop,
        highlight_textprops=[{'color': away_color}])

    annotation_string5 = (f'<{away_xg_op}>')
    ax_text(65, 80, annotation_string5, ha='center', va='center', fontsize=15,
        ax=ax, color='white', fontproperties=Rubkik_bold.prop,
        highlight_textprops=[{'color': away_color}])

    annotation_string5 = (f'<{away_xg_sp}>')
    ax_text(65, 75, annotation_string5, ha='center', va='center', fontsize=15,
        ax=ax, color='white', fontproperties=Rubkik_bold.prop,
        highlight_textprops=[{'color': away_color}])

    annotation_string5 = (f'<{away_xg_pk}>')
    ax_text(65, 70, annotation_string5, ha='center', va='center', fontsize=15,
        ax=ax, color='white', fontproperties=Rubkik_bold.prop,
        highlight_textprops=[{'color': away_color}])
    
    annotation_string6 = (f'<{away_xg_shot}>')
    ax_text(65, 65, annotation_string6, ha='center', va='center', fontsize=15,
        ax=ax, color='white', fontproperties=Rubkik_bold.prop,
        highlight_textprops=[{'color': away_color}])

    annotation_string6 = (f'<{round(a_shot_dist_to_goal,2)}>')
    ax_text(65, 60, annotation_string6, ha='center', va='center', fontsize=15,
        ax=ax, color='white', fontproperties=Rubkik_bold.prop,
        highlight_textprops=[{'color': away_color}])

    ax.axvspan(39, 61, ymin=0.57,ymax=0.97, facecolor='#1d2849', zorder=1, alpha=0.7)

    ax.axvspan(45, 55, ymin=0.51,ymax=0.56, facecolor='#1d2849', zorder=1, alpha=0.7)
    ax_text(50, 53, 'xG Value', ha='center', va='center', fontsize=15,
        ax=ax, color='white', fontproperties=Rubik.prop)
    
#     ax_text(62, -2, annotation_string2, ha='center', va='center', fontsize=20,
#             ax=ax, color='white', fontproperties=Rubik.prop)

    pitch.scatter(47, 47.5, s=150, c='gray',
                  zorder=4, ax=ax, edgecolor='w', linewidths=2.5)
    pitch.scatter(50, 47.5, s=350, c='gray',
                  zorder=4, ax=ax, edgecolor='w', linewidths=2.5)
    pitch.scatter(53.5, 47.5, s=500, c='gray', 
                  zorder=4, ax=ax, edgecolor='w', linewidths=2.5)

    passes = events_df[events_df['type'] == 'Pass']
    passes_att_third = passes[passes['x'] > 66]

    h_passes_att_third = passes_att_third[passes_att_third['teamId'] == h_teamId]
    a_passes_att_third = passes_att_third[passes_att_third['teamId'] == a_teamId]

    field_tilt_home = round((len(h_passes_att_third) / len(passes_att_third)*100), 2)
    field_tilt_away = round((len(a_passes_att_third) / len(passes_att_third)*100), 2)

    total_passes_carries = events_df[events_df['type'].isin(['Pass', 'Carry'])]
    total_passes_carries = total_passes_carries[total_passes_carries['outcomeType'] == 'Successful']
    total_passes_carries['isOpenPlay'] = np.where( (total_passes_carries['passFreekick'] == False) &
                                                ((total_passes_carries['passCorner'] == False)
                                                ) 
                                               , 1, 0
                                               )

    # home
    h_total_passes_carries = total_passes_carries[total_passes_carries['teamId'] == h_teamId]

    h_total_passes_carries = xThreat(h_total_passes_carries)
    h_total_passes_carries['x'] = 100-h_total_passes_carries['x']
    h_total_passes_carries['y'] = 100-h_total_passes_carries['y']
    h_total_passes_carries['xT'] = -h_total_passes_carries['xT']

    # away
    a_total_passes_carries = total_passes_carries[total_passes_carries['teamId'] == a_teamId]

    a_total_passes_carries = xThreat(a_total_passes_carries)

    total_pass_carry = pd.concat([h_total_passes_carries, a_total_passes_carries], axis=0)

    vmax = np.max(total_pass_carry['xT'].max())
    vmin = np.min(total_pass_carry['xT'].min())
    norm = Normalize(vmin=vmin, vmax=vmax)
    midpoint = 0

    bin_statistic = pitch.bin_statistic(total_pass_carry.x, total_pass_carry.y, total_pass_carry.xT, statistic='sum', bins=(24, 16))
    bin_statistic['statistic'] = gaussian_filter(bin_statistic['statistic'], 1)
    pcm1 = pitch.heatmap(bin_statistic, ax=ax, cmap=customcmap, edgecolors='#1a223d', zorder=0, alpha=1)

    home_xT_op = h_total_passes_carries[h_total_passes_carries['isOpenPlay'] == 1]
    home_xT_non_op = h_total_passes_carries[h_total_passes_carries['isOpenPlay'] == 0]
    home_pass_xt = h_total_passes_carries[h_total_passes_carries['type'] == 'Pass']
    home_carry_xt = h_total_passes_carries[h_total_passes_carries['type'] == 'Carry']

    away_xT_op = a_total_passes_carries[a_total_passes_carries['isOpenPlay'] == 1]
    away_xT_non_op = a_total_passes_carries[a_total_passes_carries['isOpenPlay'] == 0]
    away_pass_xt = a_total_passes_carries[a_total_passes_carries['type'] == 'Pass']
    away_carry_xt = a_total_passes_carries[a_total_passes_carries['type'] == 'Carry']

    home_cum_xT = round(-h_total_passes_carries.xT.sum(),2)
    home_xt_cum_op = round(-home_xT_op.xT.sum(), 2)
    home_cum_non_op = round(-home_xT_non_op.xT.sum(), 2)
    home_xt_passes = round(-home_pass_xt.xT.sum(), 2)
    home_xt_carries = round(-home_carry_xt.xT.sum(),2)

    away_cum_xT = round(a_total_passes_carries.xT.sum(),2)
    away_xt_cum_op = round(away_xT_op.xT.sum(), 2)
    away_cum_non_op = round(away_xT_non_op.xT.sum(), 2)
    away_xt_passes = round(away_pass_xt.xT.sum(), 2)
    away_xt_carries = round(away_carry_xt.xT.sum(),2)
    
    annotation_string4 = (f'Expected Threat (xT)')
    ax_text(50, 35, annotation_string4, ha='center', va='center', fontsize=15,
        ax=ax, color='white', fontproperties=Rubik.prop)

    annotation_string4 = (f'xT Open Play')
    ax_text(50, 30, annotation_string4, ha='center', va='center', fontsize=15,
        ax=ax, color='white', fontproperties=Rubik.prop)
    
    annotation_string5 = (f'xT Set Play')
    ax_text(50, 25, annotation_string5, ha='center', va='center', fontsize=15,
        ax=ax, color='white', fontproperties=Rubik.prop)

    annotation_string5 = (f'xT Passes')
    ax_text(50, 20, annotation_string5, ha='center', va='center', fontsize=15,
        ax=ax, color='white', fontproperties=Rubik.prop)

    annotation_string5 = (f'xT Carries')
    ax_text(50, 15, annotation_string5, ha='center', va='center', fontsize=15,
        ax=ax, color='white', fontproperties=Rubik.prop)

    annotation_string5 = (f'Field Tilt')
    ax_text(50, 10, annotation_string5, ha='center', va='center', fontsize=15,
        ax=ax, color='white', fontproperties=Rubik.prop)

    # home stats

    annotation_string4 = (f'<{home_cum_xT}>')
    ax_text(35, 35, annotation_string4, ha='center', va='center', fontsize=15,
        ax=ax, color='white', fontproperties=Rubkik_bold.prop,
        highlight_textprops=[{'color': home_color}])

    annotation_string4 = (f'<{home_xt_cum_op}>')
    ax_text(35, 30, annotation_string4, ha='center', va='center', fontsize=15,
        ax=ax, color='white', fontproperties=Rubkik_bold.prop,
        highlight_textprops=[{'color': home_color}])
    
    annotation_string5 = (f'<{home_cum_non_op}>')
    ax_text(35, 25, annotation_string5, ha='center', va='center', fontsize=15,
        ax=ax, color='white', fontproperties=Rubkik_bold.prop,
        highlight_textprops=[{'color': home_color}])

    annotation_string5 = (f'<{home_xt_passes}>')
    ax_text(35, 20, annotation_string5, ha='center', va='center', fontsize=15,
        ax=ax, color='white', fontproperties=Rubkik_bold.prop,
        highlight_textprops=[{'color': home_color}])

    annotation_string5 = (f'<{home_xt_carries}>')
    ax_text(35, 15, annotation_string5, ha='center', va='center', fontsize=15,
        ax=ax, color='white', fontproperties=Rubkik_bold.prop,
        highlight_textprops=[{'color': home_color}])

    annotation_string5 = (f'<{field_tilt_home}>')
    ax_text(35, 10, annotation_string5, ha='center', va='center', fontsize=15,
        ax=ax, color='white', fontproperties=Rubkik_bold.prop,
        highlight_textprops=[{'color': home_color}])

    # away stats

    annotation_string4 = (f'<{away_cum_xT}>')
    ax_text(65, 35, annotation_string4, ha='center', va='center', fontsize=15,
        ax=ax, color='white', fontproperties=Rubkik_bold.prop,
        highlight_textprops=[{'color': away_color}])

    annotation_string4 = (f'<{away_xt_cum_op}>')
    ax_text(65, 30, annotation_string4, ha='center', va='center', fontsize=15,
        ax=ax, color='white', fontproperties=Rubkik_bold.prop,
        highlight_textprops=[{'color': away_color}])
    
    annotation_string5 = (f'<{away_cum_non_op}>')
    ax_text(65, 25, annotation_string5, ha='center', va='center', fontsize=15,
        ax=ax, color='white', fontproperties=Rubkik_bold.prop,
        highlight_textprops=[{'color': away_color}])

    annotation_string5 = (f'<{away_xt_passes}>')
    ax_text(65, 20, annotation_string5, ha='center', va='center', fontsize=15,
        ax=ax, color='white', fontproperties=Rubkik_bold.prop,
        highlight_textprops=[{'color': away_color}])

    annotation_string5 = (f'<{away_xt_carries}>')
    ax_text(65, 15, annotation_string5, ha='center', va='center', fontsize=15,
        ax=ax, color='white', fontproperties=Rubkik_bold.prop,
        highlight_textprops=[{'color': away_color}])

    annotation_string5 = (f'<{field_tilt_away}>')
    ax_text(65, 10, annotation_string5, ha='center', va='center', fontsize=15,
        ax=ax, color='white', fontproperties=Rubkik_bold.prop,
        highlight_textprops=[{'color': away_color}])

    ax.axvspan(40, 60, ymin=0.1,ymax=0.38, facecolor='#1d2849', zorder=1, alpha=0.7)
    
    # Set the figure facecolor
    #fig.set_facecolor(pitchcolor)    

def createPassNetworks(match_data, events_df, matchId, team, max_line_width, label_position,
                       edgewidth, arrow_width, dh_arrow_width, marker_color, touch_lim,
                       team_color, marker_edge_color, shrink, ax, kit_no_size=20):

    if match_data['home']['name'] == team:
        teamId = match_data['home']['teamId']
        venue = 'home'
    else:
        teamId = match_data['away']['teamId']
        venue = 'away'


    # getting opponent   
    if venue == 'home':
        opponent = match_data['away']['name']
    else:
        opponent = match_data['home']['name']


    # getting player dictionary
    team_players_dict = {}
    for player in match_data[venue]['players']:
        team_players_dict[player['playerId']] = player['name']

    #for i in events_df.index:
    #    if events_df.loc[i, 'type'] == 'SubstitutionOn' and events_df.loc[i, 'teamId'] == teamId:
    #        sub_minute = int(events_df.loc[i, 'minute'])
    #        sub_minute = str(sub_minute)
    #        break

    # getting players dataframe
    match_players_df = pd.DataFrame()
    player_names = []
    player_ids = []
    player_pos = []
    player_kit_number = []


    for player in match_data[venue]['players']:
        player_names.append(player['name'])
        player_ids.append(player['playerId'])
        player_pos.append(player['position'])
        player_kit_number.append(player['shirtNo'])

    match_players_df['playerId'] = player_ids
    match_players_df['playerName'] = player_names
    match_players_df['playerPos'] = player_pos
    match_players_df['playerKitNumber'] = player_kit_number

    # extracting passes
    passes_df = events_df.loc[events_df['teamId'] == teamId].reset_index().drop('index', axis=1)
    passes_df['playerId'] = passes_df['playerId'].astype('float').astype('Int64')
    if 'playerName' in passes_df.columns:
        passes_df = passes_df.drop(columns='playerName')
        passes_df.dropna(subset=["playerId"], inplace=True)
        passes_df.insert(27, column='playerName', value=[team_players_dict[i] for i in list(passes_df['playerId'])])
    if 'passRecipientId' in passes_df.columns:
        passes_df = passes_df.drop(columns='passRecipientId')
        passes_df = passes_df.drop(columns='passRecipientName')
    passes_df.insert(28, column='passRecipientId', value=passes_df['playerId'].shift(-1))  
    passes_df.insert(29, column='passRecipientName', value=passes_df['playerName'].shift(-1))  
    passes_df.dropna(subset=["passRecipientName"], inplace=True)
    passes_df = passes_df.loc[passes_df['type'] == 'Pass'].reset_index(drop=True)
    passes_df = passes_df.loc[passes_df['outcomeType'] == 'Successful'].reset_index(drop=True)
    #passes_df = passes_df.loc[passes_df['passFreekick'] == False, :].reset_index(drop=True)
    #passes_df = passes_df.loc[passes_df['passCorner'] == False, :].reset_index(drop=True)
    #passes_df = passes_df.loc[passes_df['throwIn'] == False, :].reset_index(drop=True)
    # calc xT of each pass
    passes_df = xThreat(passes_df)
    index_names = passes_df.loc[passes_df['playerName']==passes_df['passRecipientName']].index
    passes_df.drop(index_names, inplace=True)
    passes_df = passes_df.merge(match_players_df, on=['playerId', 'playerName'], how='left', validate='m:1')
    passes_df = passes_df.merge(match_players_df.rename({'playerId': 'passRecipientId', 'playerName':'passRecipientName'},
                                                        axis='columns'), on=['passRecipientId', 'passRecipientName'],
                                                        how='left', validate='m:1', suffixes=['', 'Receipt'])

    # extracting all ball touches
    touches = events_df.loc[events_df['teamId'] == teamId].reset_index().drop('index', axis=1)
    touches['playerId'] = touches['playerId'].astype('float').astype('Int64')
    if 'playerName' in touches.columns:
        touches = touches.drop(columns='playerName')
        touches.dropna(subset=["playerId"], inplace=True)
    touches.dropna(subset=["playerId"], inplace=True)
    touches.insert(27, column='playerName', value=[team_players_dict[i] for i in list(touches['playerId'])])
    touches = touches.loc[touches['touches'] == True, :].reset_index(drop=True)
    touches = touches.merge(match_players_df, on=['playerId', 'playerName'], how='left', validate='m:1')

    # getting number of touches; node size = number of touches
    touch_count = touches.groupby(['playerKitNumber']).agg({'playerKitNumber': ['count']})
    touch_count.columns = ['touch_counts']
    quantiles = pd.qcut(touch_count.touch_counts, np.linspace(0,1,5), labels=np.linspace(0.1,1,4))
    touch_count = touch_count.assign(Quartile=quantiles.values)
    touch_count['touch_count'] = touch_count['Quartile'].astype('float')
    touch_count['touch_count'] = np.where(touch_count['touch_count'] > 0.999 
                               , 0.9, touch_count['touch_count']
                               )
    touch_count['touch_count'] = np.where(touch_count['touch_count'] < 0.4 
                               , 0.4, touch_count['touch_count']
                               )

    # drop players who are subs and played had less than 30 touches
    positions = match_players_df[['playerKitNumber', 'playerPos']]
    positions = positions.set_index('playerKitNumber')
    touch_count = pd.merge(touch_count, positions, left_index=True, right_index=True)
    touch_count = touch_count.loc[~((touch_count['touch_counts'] < touch_lim) & (touch_count['playerPos'] == 'Sub'))]

    position_omits = touch_count.index.to_list()
    passes_df = passes_df[passes_df['playerKitNumber'].isin(position_omits)]

    # getting team formation
    formation = match_data[venue]['formations'][0]['formationName']
    formation = '-'.join(formation)

    # getting player average locations
    location_formation = passes_df[['playerKitNumber', 'x', 'y']]
    average_locs_and_count = location_formation.groupby('playerKitNumber').agg({'x': ['mean'], 'y': ['mean', 'count']})
    average_locs_and_count.columns = ['x', 'y', 'count']
    average_locs_and_count = pd.merge(average_locs_and_count, touch_count, left_index=True, right_index=True)

    # getting separate dataframe for selected columns 
    passes_formation = passes_df[['id', 'playerKitNumber', 'playerKitNumberReceipt']].copy()
    passes_formation['EPV'] = passes_df['EPV']
    passes_formation['xT'] = passes_df['xT']

    # getting dataframe for passes between players
    passes_between = passes_formation.groupby(['playerKitNumber', 'playerKitNumberReceipt']).agg({ 'id' : 'count', 'EPV' : 'sum'}).reset_index()        
    passes_between.rename({'id': 'pass_count'}, axis='columns', inplace=True)
    passes_between = passes_between.merge(average_locs_and_count, left_on='playerKitNumberReceipt', right_index=True)
    passes_between = passes_between.merge(average_locs_and_count, left_on='playerKitNumber', right_index=True,
                                          suffixes=['', '_end'])

    # getting total EPV from passes from each player
    passes_epv = passes_formation.groupby(['playerKitNumber']).agg({'EPV' : 'sum'})        
    passes_epv.rename({'id': 'pass_count'}, axis='columns', inplace=True)
    passes_epv['EPV'] = pd.qcut(passes_epv.EPV, np.linspace(0,1,11), labels=np.linspace(0.1,1,10))
    passes_epv['EPV'] = passes_epv.EPV.astype('float')
    average_locs_and_count = pd.merge(average_locs_and_count, passes_epv, left_index=True, right_index=True)

    # getting total xT from passes from each player; will be used to get alpha of each node
    passes_xt = passes_formation.groupby(['playerKitNumber']).agg({'xT' : 'sum'})        
    passes_xt.rename({'id': 'sum_xT'}, axis='columns', inplace=True)
    passes_xt['xT'] = pd.qcut(passes_xt.xT.rank(method='first'), np.linspace(0,1,11), labels=np.linspace(0.1,1,10))
    passes_xt['xT'] = passes_xt.xT.astype('float')
    passes_xt[passes_xt['xT'] == 0] = 0.08
    #passes_xt[passes_xt['xT'] > 0.9] = 0.9
    average_locs_and_count = pd.merge(average_locs_and_count, passes_xt, left_index=True, right_index=True)

    # filtering passes
    pass_filter = int(passes_between['pass_count'].mean())
    passes_between = passes_between.loc[passes_between['pass_count'] > pass_filter]

    # calculating the line width; line width = number of passes 
    passes_between['width'] = passes_between.pass_count / passes_between.pass_count.max() * 3
    passes_between = passes_between.reset_index(drop=True)

    # setting color to make the lines more transparent when fewer passes are made
    min_transparency = 0.3
    color = np.array(to_rgba('white'))
    color = np.tile(color, (len(passes_between), 1))
    c_transparency = passes_between.pass_count / passes_between.pass_count.max()
    c_transparency = (c_transparency * (1 - min_transparency)) + min_transparency
    color[:, 3] = c_transparency
    passes_between['alpha'] = color.tolist()

    # separating paired passes from normal passes
    passes_between_threshold = 50
    filtered_pair_df = []
    pair_list = [comb for comb in combinations(passes_between['playerKitNumber'].unique(), 2)]
    for pair in pair_list:
        df = passes_between[((passes_between['playerKitNumber']==pair[0]) & (passes_between['playerKitNumberReceipt']==pair[1])) | 
                            ((passes_between['playerKitNumber']==pair[1]) & (passes_between['playerKitNumberReceipt']==pair[0]))]
        if df.shape[0] == 2:
            if (np.array(df.pass_count)[0] >= passes_between_threshold) and (np.array(df.pass_count)[1] >= passes_between_threshold):
                filtered_pair_df.append(df)
                passes_between.drop(df.index, inplace=True)
    if len(filtered_pair_df) > 0:
        filtered_pair_df = pd.concat(filtered_pair_df).reset_index(drop=True)
        passes_between = passes_between.reset_index(drop=True)

    # plotting
    pitch = VerticalPitch(pitch_type='opta', pitch_color='#1d2849', line_color='w',
                          goal_type='box')
    pitch.draw(ax=ax, constrained_layout=True, tight_layout=True)
    average_locs_and_count['zorder'] = list(np.linspace(1,5,len(average_locs_and_count)))
    
    # split into subs and starting 11
    average_locs_and_count_11 = average_locs_and_count[average_locs_and_count['playerPos'] != 'Sub']
    average_locs_and_count_sub = average_locs_and_count[average_locs_and_count['playerPos'] == 'Sub']

    # plot
    for i in average_locs_and_count_11.index:
        pitch.scatter(average_locs_and_count_11.loc[i, 'x'], average_locs_and_count_11.loc[i, 'y'], s=average_locs_and_count_11.loc[i, 'touch_count']*1800,
                      color=marker_color, edgecolors=marker_edge_color, linewidth=edgewidth, 
                      alpha=average_locs_and_count_11.loc[i, 'xT'], zorder=average_locs_and_count_11.loc[i, 'zorder']*3, ax=ax)

    for i in average_locs_and_count_sub.index:
        pitch.scatter(average_locs_and_count_sub.loc[i, 'x'], average_locs_and_count_sub.loc[i, 'y'], s=average_locs_and_count_sub.loc[i, 'touch_count']*1800,
                      color=marker_color, edgecolors=marker_edge_color, linewidth=edgewidth, marker='X', 
                      alpha=average_locs_and_count_sub.loc[i, 'xT'], zorder=average_locs_and_count_sub.loc[i, 'zorder']*3, ax=ax)

    for i in passes_between.index:
        y = passes_between.loc[i, 'x']
        x = passes_between.loc[i, 'y']
        endY = passes_between.loc[i, 'x_end']
        endX = passes_between.loc[i, 'y_end']
        coordsA = "data"
        coordsB = "data"
        con = ConnectionPatch([endX, endY], [x, y],
                              coordsA, coordsB,
                              arrowstyle="simple", shrinkA=shrink, shrinkB=shrink,
                              mutation_scale=arrow_width, color=passes_between.loc[i, 'alpha'])
        ax.add_artist(con)

    if len(filtered_pair_df) > 0:
        for i in filtered_pair_df.index:
            y = filtered_pair_df.loc[i, 'x']
            x = filtered_pair_df.loc[i, 'y']
            endY = filtered_pair_df.loc[i, 'x_end']
            endX = filtered_pair_df.loc[i, 'y_end']
            coordsA = "data"
            coordsB = "data"
            con = ConnectionPatch([endX, endY], [x, y],
                                  coordsA, coordsB,
                                  arrowstyle="<|-|>", shrinkA=shrink, shrinkB=shrink,
                                  mutation_scale=dh_arrow_width, lw=arrow_width, 
                                  color=filtered_pair_df.loc[i, 'alpha'])
            ax.add_artist(con)

    for i in average_locs_and_count_11.index:
        pitch.annotate(i, xy=(average_locs_and_count_11.loc[i, 'x'], average_locs_and_count_11.loc[i, 'y']), 
                       family='DejaVu Sans', c='white', 
                       va='center', ha='center', zorder=average_locs_and_count_11.loc[i, 'zorder']*4, size=kit_no_size, weight='bold', ax=ax)
    for i in average_locs_and_count_sub.index:
        pitch.annotate(i, xy=(average_locs_and_count_sub.loc[i, 'x'], average_locs_and_count_sub.loc[i, 'y']), 
                       family='DejaVu Sans', c='white', 
                       va='center', ha='center', zorder=average_locs_and_count_sub.loc[i, 'zorder']*4, size=kit_no_size*0.55, weight='bold', ax=ax)

    # create a df for defensive actions and get avg distance from goal (x)
    defensive_actions = ['Foul', 'Tackle', 'Interception']
    defensive_events = events_df[events_df['type'].isin(defensive_actions)]
    defensive_events = defensive_events.loc[defensive_events['teamId'] == teamId].reset_index().drop('index', axis=1)
    dist_from_goal = defensive_events.x.mean()

    # plot defensive line height
    ax.axhline(y=dist_from_goal, xmin=0.04, xmax=0.96, color='gray', linestyle='--', zorder=5)

    #ax.text(50, 104, "{} (Mins 1-{})".format(team, int(sub_minute)).upper(), size=14, ha='center',
    #       va='center', color=team_color, fontproperties=Rubik.prop);
    ax.text(50, 104, "{} (Full Time)".format(team).upper(), size=14, ha='center',
           va='center', color=team_color, fontproperties=Rubik.prop);
    ax.text(50, 96, '{}'.format(formation), size=10, c=team_color, ha='center', va='center', fontproperties=Rubik.prop);
    ax.text(label_position, dist_from_goal-2, 'Def Line\nHeight:\n\n{}m'.format(round(dist_from_goal,2)), 
            size=12, c='w', ha='left', va='center', fontproperties=Rubik.prop);
    #ax.text(50, 88, f'Arrow Brightnes = Number of Passes\nNode Size = Number of Touches\nNode Brightnes = xT from Passes', size=9, c='w', ha='center', va='center', fontproperties=Rubik.prop);    
    ax.text(50, 88, f'Arrow Brightnes = Number of Passes\nNode Size = Number of Touches\nNode Brightnes = xT from Passes\n\nSub: X ({touch_lim}+ touches)', size=9, c='w', ha='center', va='center', fontproperties=Rubik.prop);
    
def createAttPassNetworks(match_data, events_df, matchId, team, max_line_width, 
                      marker_size, edgewidth, dh_arrow_width, marker_color, 
                      marker_edge_color, shrink, ax, kit_no_size = 20):
    
    # getting team id and venue
    if match_data['home']['name'] == team:
        teamId = match_data['home']['teamId']
        venue = 'home'
    else:
        teamId = match_data['away']['teamId']
        venue = 'away'
    
    
    # getting opponent   
    if venue == 'home':
        opponent = match_data['away']['name']
    else:
        opponent = match_data['home']['name']
    
    
    # getting player dictionary
    team_players_dict = {}
    for player in match_data[venue]['players']:
        team_players_dict[player['playerId']] = player['name']
    
    
    # getting minute of first substitution
    for i in events_df.index:
        if events_df.loc[i, 'type'] == 'SubstitutionOn' and events_df.loc[i, 'teamId'] == teamId:
            sub_minute = str(events_df.loc[i, 'minute'])
            break
    
    
    # getting players dataframe
    match_players_df = pd.DataFrame()
    player_names = []
    player_ids = []
    player_pos = []
    player_kit_number = []


    for player in match_data[venue]['players']:
        player_names.append(player['name'])
        player_ids.append(player['playerId'])
        player_pos.append(player['position'])
        player_kit_number.append(player['shirtNo'])

    match_players_df['playerId'] = player_ids
    match_players_df['playerName'] = player_names
    match_players_df['playerPos'] = player_pos
    match_players_df['playerKitNumber'] = player_kit_number
    
    
    # extracting passes
    passes_df = events_df.loc[events_df['teamId'] == teamId].reset_index().drop('index', axis=1)
    passes_df['playerId'] = passes_df['playerId'].astype('float').astype('Int64')
    if 'playerName' in passes_df.columns:
        passes_df = passes_df.drop(columns='playerName')
    passes_df.dropna(subset=["playerId"], inplace=True)
    passes_df.insert(27, column='playerName', value=[team_players_dict[i] for i in list(passes_df['playerId'])])
    if 'passRecipientId' in passes_df.columns:
        passes_df = passes_df.drop(columns='passRecipientId')
        passes_df = passes_df.drop(columns='passRecipientName')
    passes_df.insert(28, column='passRecipientId', value=passes_df['playerId'].shift(-1))  
    passes_df.insert(29, column='passRecipientName', value=passes_df['playerName'].shift(-1))  
    passes_df.dropna(subset=["passRecipientName"], inplace=True)
    passes_df = passes_df.loc[events_df['type'] == 'Pass', :].reset_index(drop=True)
    passes_df = passes_df.loc[events_df['outcomeType'] == 'Successful', :].reset_index(drop=True)
    index_names = passes_df.loc[passes_df['playerName']==passes_df['passRecipientName']].index
    passes_df.drop(index_names, inplace=True)
    passes_df = passes_df.merge(match_players_df, on=['playerId', 'playerName'], how='left', validate='m:1')
    passes_df = passes_df.merge(match_players_df.rename({'playerId': 'passRecipientId', 'playerName':'passRecipientName'},
                                                        axis='columns'), on=['passRecipientId', 'passRecipientName'],
                                                        how='left', validate='m:1', suffixes=['', 'Receipt'])
    passes_df = passes_df[passes_df['playerPos'] != 'Sub']
    
    
    # getting team formation
    formation = match_data[venue]['formations'][0]['formationName']
    formation = '-'.join(formation)
    
    
    # getting player average locations
    location_formation = passes_df[['playerKitNumber', 'x', 'y']]
    average_locs_and_count = location_formation.groupby('playerKitNumber').agg({'x': ['mean'], 'y': ['mean', 'count']})
    average_locs_and_count.columns = ['x', 'y', 'count']
    
    
    # filtering progressive passes 
    passes_df = passes_df.loc[passes_df['EPV'] > 0]

    
    # getting separate dataframe for selected columns 
    passes_formation = passes_df[['id', 'playerKitNumber', 'playerKitNumberReceipt']].copy()
    passes_formation['EPV'] = passes_df['EPV']


    # getting dataframe for passes between players
    passes_between = passes_formation.groupby(['playerKitNumber', 'playerKitNumberReceipt']).agg({ 'id' : 'count', 'EPV' : 'sum'}).reset_index()
    passes_between.rename({'id': 'pass_count'}, axis='columns', inplace=True)
    passes_between = passes_between.merge(average_locs_and_count, left_on='playerKitNumberReceipt', right_index=True)
    passes_between = passes_between.merge(average_locs_and_count, left_on='playerKitNumber', right_index=True,
                                          suffixes=['', '_end'])
    
    
    # filtering passes
    pass_filter = int(passes_between['pass_count'].mean())
    passes_between = passes_between.loc[passes_between['pass_count'] > pass_filter*2]
    
    
    # calculating the line width and marker sizes relative to the largest counts
    passes_between['width'] = passes_between.pass_count / passes_between.pass_count.max() * max_line_width
    passes_between = passes_between.reset_index(drop=True)
    
    
    # setting color to make the lines more transparent when fewer passes are made
    min_transparency = 0.3
    color = np.array(to_rgba('white'))
    color = np.tile(color, (len(passes_between), 1))
    c_transparency = passes_between.EPV / passes_between.EPV.max()
    c_transparency = (c_transparency * (1 - min_transparency)) + min_transparency
    color[:, 3] = c_transparency
    passes_between['alpha'] = color.tolist()
    
    
    # separating paired passes from normal passes
    passes_between_threshold = 20
    filtered_pair_df = []
    pair_list = [comb for comb in combinations(passes_between['playerKitNumber'].unique(), 2)]
    for pair in pair_list:
        df = passes_between[((passes_between['playerKitNumber']==pair[0]) & (passes_between['playerKitNumberReceipt']==pair[1])) | 
                            ((passes_between['playerKitNumber']==pair[1]) & (passes_between['playerKitNumberReceipt']==pair[0]))]
        if df.shape[0] == 2:
            if np.array(df.pass_count)[0]+np.array(df.pass_count)[1] >= passes_between_threshold:
                filtered_pair_df.append(df)
                passes_between.drop(df.index, inplace=True)
    if len(filtered_pair_df) > 0:
        filtered_pair_df = pd.concat(filtered_pair_df).reset_index(drop=True)
        passes_between = passes_between.reset_index(drop=True)
    
    
    # plotting
    pitch = Pitch(pitch_type='opta', pitch_color='#171717', line_color='#5c5c5c',
                  goal_type='box')
    pitch.draw(ax=ax, constrained_layout=True, tight_layout=True)
    
    average_locs_and_count['zorder'] = list(np.linspace(1,5,11))
    for i in average_locs_and_count.index:
        pitch.scatter(average_locs_and_count.loc[i, 'x'], average_locs_and_count.loc[i, 'y'], s=marker_size,
                      color=marker_color, edgecolors=marker_edge_color, linewidth=edgewidth, 
                      alpha=1, zorder=average_locs_and_count.loc[i, 'zorder'], ax=ax)
    
    for i in passes_between.index:
        x = passes_between.loc[i, 'x']
        y = passes_between.loc[i, 'y']
        endX = passes_between.loc[i, 'x_end']
        endY = passes_between.loc[i, 'y_end']
        coordsA = "data"
        coordsB = "data"
        con = ConnectionPatch([endX, endY], [x, y],
                              coordsA, coordsB,
                              arrowstyle="simple", shrinkA=shrink, shrinkB=shrink,
                              mutation_scale=passes_between.loc[i, 'width']*max_line_width, color=passes_between.loc[i, 'alpha'])
        ax.add_artist(con)
    
    if len(filtered_pair_df) > 0:
        for i in filtered_pair_df.index:
            x = filtered_pair_df.loc[i, 'x']
            y = filtered_pair_df.loc[i, 'y']
            endX = filtered_pair_df.loc[i, 'x_end']
            endY = filtered_pair_df.loc[i, 'y_end']
            coordsA = "data"
            coordsB = "data"
            con = ConnectionPatch([endX, endY], [x, y],
                                  coordsA, coordsB,
                                  arrowstyle="<|-|>", shrinkA=shrink, shrinkB=shrink,
                                  mutation_scale=dh_arrow_width, lw=filtered_pair_df.loc[i, 'width']*max_line_width/5, 
                                  color=filtered_pair_df.loc[i, 'alpha'])
            ax.add_artist(con)
    
    for i in average_locs_and_count.index:
        pitch.annotate(i, xy=(average_locs_and_count.loc[i, 'x'], average_locs_and_count.loc[i, 'y']), 
                       family='DejaVu Sans', c='white', 
                       va='center', ha='center', zorder=average_locs_and_count.loc[i, 'zorder'], size=kit_no_size, weight='bold', ax=ax)
    ax.text(50, 104, "{} (Mins 1-{}\n\n\n{})".format(team, sub_minute,formation).upper(), size=10, fontweight='bold', ha='center',
           va='center')
    ax.text(50, 95, '{}'.format(formation), size=9, c='grey', va='center', ha='center')


def getTeamSuccessfulBoxPasses(events_df, teamId, team, pitch_color, cmap):
    """
    Parameters
    ----------
    events_df : DataFrame of all events.
    
    teamId : ID of the team, the passes of which are required.
    
    team : Name of the team, the passes of which are required.
    
    pitch_color : color of the pitch.
    
    cmap : color design of the pass lines. 
           You can select more cmaps here: 
               https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
    Returns
    -------
    Pitch Plot.
    """
    
    # Get Total Passes
    passes_df = events_df.loc[events_df['type']=='Pass'].reset_index(drop=True)
    
    # Get Team Passes
    team_passes = passes_df.loc[passes_df['teamId'] == teamId]
        
    # Extracting Box Passes from Total Passes
    box_passes = team_passes.copy()
    for i,pas in box_passes.iterrows():
        X = pas["x"]/100*120
        Xend = pas["endX"]/100*120
        Y = pas["y"]/100*80
        Yend = pas["endY"]/100*80
        if Xend >= 102 and Yend >= 18 and Yend <= 62:
            if X >=102 and Y >= 18 and Y <= 62:
                box_passes = box_passes.drop([i])
            else:
                pass
        else:
            box_passes = box_passes.drop([i])
            
    
    successful_box_passes = box_passes.loc[box_passes['outcomeType']=='Successful'].reset_index(drop=True)
    
        
    # orientation='vertical'
    pitch = VerticalPitch(pitch_type='statsbomb', pitch_color=pitch_color, line_color='#c7d5cc',
                          figsize=(16, 11), half=True, pad_top=2)
    fig, ax = pitch.draw(tight_layout=True)
    
    # Plot the completed passes
    pitch.lines(successful_box_passes.x/100*120, 80-successful_box_passes.y/100*80,
                successful_box_passes.endX/100*120, 80-successful_box_passes.endY/100*80,
                lw=5, cmap=cmap, opp_comet=True, opp_transparent=True,
                label='Successful Passes', ax=ax)
    
    pitch.scatter(successful_box_passes.x/100*120, 80-successful_box_passes.y/100*80,
                  edgecolors='white', c='white', s=50, zorder=2,
                  ax=ax)
    
    # Set the title
    fig.suptitle(f'Completed Box Passes - {team}', y=.95, fontsize=15)
    
    # Set the subtitle
    ax.set_title('Data : Whoscored/Opta', fontsize=8, loc='right', fontstyle='italic', fontweight='bold')
    
    # set legend
    #ax.legend(facecolor='#22312b', edgecolor='None', fontsize=8, loc='lower center', handlelength=4)
    
    # Set the figure facecolor
    fig.set_facecolor(pitch_color) 








def getTeamTotalPasses(events_df, teamId, team, opponent, pitch_color):
    """
    
    Parameters
    ----------
    events_df : DataFrame of all events.
    
    teamId : ID of the team, the passes of which are required.
    
    team : Name of the team, the passes of which are required.
    
    opponent : Name of opponent team.
    
    pitch_color : color of the pitch.
    Returns
    -------
    Pitch Plot.
    """
    
    # Get Total Passes
    passes_df = events_df.loc[events_df['type']=='Pass'].reset_index(drop=True)
    
    # Get Team Passes
    team_passes = passes_df.loc[passes_df['teamId'] == teamId]
        
    successful_passes = team_passes.loc[team_passes['outcomeType']=='Successful'].reset_index(drop=True)
    unsuccessful_passes = team_passes.loc[team_passes['outcomeType']=='Unsuccessful'].reset_index(drop=True)
            
    # Setup the pitch
    pitch = Pitch(pitch_type='statsbomb', pitch_color=pitch_color, line_color='#c7d5cc', figsize=(16, 11))
    fig, ax = pitch.draw(constrained_layout=True, tight_layout=False)
    
    # Plot the completed passes
    pitch.arrows(successful_passes.x/100*120, 80-successful_passes.y/100*80,
                 successful_passes.endX/100*120, 80-successful_passes.endY/100*80, width=1,
                 headwidth=10, headlength=10, color='#ad993c', ax=ax, label='Completed')
    
    # Plot the other passes
    pitch.arrows(unsuccessful_passes.x/100*120, 80-unsuccessful_passes.y/100*80,
                 unsuccessful_passes.endX/100*120, 80-unsuccessful_passes.endY/100*80, width=1,
                 headwidth=6, headlength=5, headaxislength=12, color='#ba4f45', ax=ax, label='Blocked')
    
    # setup the legend
    ax.legend(facecolor=pitch_color, handlelength=5, edgecolor='None', fontsize=8, loc='upper left', shadow=True)
    
    # Set the title
    fig.suptitle(f'{team} Passes vs {opponent}', y=1, fontsize=15)
    
    
    # Set the subtitle
    ax.set_title('Data : Whoscored/Opta', fontsize=8, loc='right', fontstyle='italic', fontweight='bold')
    
    
    # Set the figure facecolor
    
    fig.set_facecolor(pitch_color)
    
    
    
    
    

def normalize(values, bounds):
    return [bounds['desired']['lower'] + (x - bounds['actual']['lower']) * (bounds['desired']['upper'] 
            - bounds['desired']['lower']) / (bounds['actual']['upper'] - bounds['actual']['lower']) for x in values]




    
def createPVFormationMap(match_data, events_df, team, color_palette,
                        markerstyle, markersize, markeredgewidth, labelsize, labelcolor, ax):
    
    # getting team id and venue
    if match_data['home']['name'] == team:
        teamId = match_data['home']['teamId']
        venue = 'home'
    else:
        teamId = match_data['away']['teamId']
        venue = 'away'


    # getting opponent   
    if venue == 'home':
        opponent = match_data['away']['name']
    else:
        opponent = match_data['home']['name']


    # getting player dictionary
    team_players_dict = {}
    for player in match_data[venue]['players']:
        team_players_dict[player['playerId']] = player['name']


    # getting minute of first substitution
    for i,row in events_df.iterrows():
        if row['type'] == 'SubstitutionOn' and row['teamId'] == teamId:
            sub_minute = str(row['minute'])
            break
    

    # getting players dataframe
    match_players_df = pd.DataFrame()
    player_names = []
    player_ids = []
    player_pos = []
    player_kit_number = []

    for player in match_data[venue]['players']:
        player_names.append(player['name'])
        player_ids.append(player['playerId'])
        player_pos.append(player['position'])
        player_kit_number.append(player['shirtNo'])

    match_players_df['playerId'] = player_ids
    match_players_df['playerName'] = player_names
    match_players_df['playerPos'] = player_pos
    match_players_df['playerKitNumber'] = player_kit_number


    # extracting passes
    passes_df = events_df.loc[events_df['teamId'] == teamId].reset_index().drop('index', axis=1)
    passes_df['playerId'] = passes_df['playerId'].astype('float').astype('Int64')
    if 'playerName' in passes_df.columns:
        passes_df = passes_df.drop(columns='playerName')
    passes_df.dropna(subset=["playerId"], inplace=True)
    passes_df.insert(27, column='playerName', value=[team_players_dict[i] for i in list(passes_df['playerId'])])
    if 'passRecipientId' in passes_df.columns:
        passes_df = passes_df.drop(columns='passRecipientId')
        passes_df = passes_df.drop(columns='passRecipientName')
    passes_df.insert(28, column='passRecipientId', value=passes_df['playerId'].shift(-1))  
    passes_df.insert(29, column='passRecipientName', value=passes_df['playerName'].shift(-1))  
    passes_df.dropna(subset=["passRecipientName"], inplace=True)
    passes_df = passes_df.loc[events_df['type'] == 'Pass', :].reset_index(drop=True)
    passes_df = passes_df.loc[events_df['outcomeType'] == 'Successful', :].reset_index(drop=True)
    index_names = passes_df.loc[passes_df['playerName']==passes_df['passRecipientName']].index
    passes_df.drop(index_names, inplace=True)
    passes_df = passes_df.merge(match_players_df, on=['playerId', 'playerName'], how='left', validate='m:1')
    passes_df = passes_df.merge(match_players_df.rename({'playerId': 'passRecipientId', 'playerName':'passRecipientName'},
                                                        axis='columns'), on=['passRecipientId', 'passRecipientName'],
                                                        how='left', validate='m:1', suffixes=['', 'Receipt'])
    # passes_df = passes_df[passes_df['playerPos'] != 'Sub']
    
    
    # Getting net possesion value for passes
    netPVPassed = passes_df.groupby(['playerId', 'playerName'])['EPV'].sum().reset_index()
    netPVReceived = passes_df.groupby(['passRecipientId', 'passRecipientName'])['EPV'].sum().reset_index()
    

    
    # Getting formation and player ids for first 11
    formation = match_data[venue]['formations'][0]['formationName']
    formation_positions = match_data[venue]['formations'][0]['formationPositions']
    playerIds = match_data[venue]['formations'][0]['playerIds'][:11]

    
    # Getting all data in a dataframe
    formation_data = []
    for playerId, pos in zip(playerIds, formation_positions):
        pl_dict = {'playerId': playerId}
        pl_dict.update(pos)
        formation_data.append(pl_dict)
    formation_data = pd.DataFrame(formation_data)
    formation_data['vertical'] = normalize(formation_data['vertical'], 
                                           {'actual': {'lower': 0, 'upper': 10}, 'desired': {'lower': 10, 'upper': 110}})
    formation_data['horizontal'] = normalize(formation_data['horizontal'],
                                             {'actual': {'lower': 0, 'upper': 10}, 'desired': {'lower': 80, 'upper': 0}})
    formation_data = netPVPassed.join(formation_data.set_index('playerId'), on='playerId', how='inner').reset_index(drop=True)
    formation_data = formation_data.rename(columns={"EPV": "PV"})


    # Plotting
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#171717', line_color='#5c5c5c',
                  goal_type='box')
    pitch.draw(ax=ax, constrained_layout=True, tight_layout=True)
    
    sns.scatterplot(x='vertical', y='horizontal', data=formation_data, hue='PV', s=markersize, marker=markerstyle, legend=False, 
                    palette=color_palette, linewidth=markeredgewidth, ax=ax)
    
    ax.text(2, 78, '{}'.format('-'.join(formation)), size=20, c='grey')
    
    for index, row in formation_data.iterrows():
        pitch.annotate(str(round(row.PV*100,2))+'%', xy=(row.vertical, row.horizontal), c=labelcolor, va='center',
                       ha='center', size=labelsize, zorder=2, weight='bold', ax=ax)
        pitch.annotate(row.playerName, xy=(row.vertical, row.horizontal+5), c=labelcolor, va='center',
                       ha='center', size=labelsize, zorder=2, weight='bold', ax=ax)


def plot_team_lineup(match_data, events_df, team, ax):

    if match_data['home']['name'] == team:
        teamId = match_data['home']['teamId']
        venue = 'home'
    else:
        teamId = match_data['away']['teamId']
        venue = 'away'


    # getting opponent   
    if venue == 'home':
        opponent = match_data['away']['name']
    else:
        opponent = match_data['home']['name']


    # getting player dictionary
    team_players_dict = {}
    for player in match_data[venue]['players']:
        team_players_dict[player['playerId']] = player['name']


    # getting minute of first substitution
    #for i in events_df.index:
    #    if events_df.loc[i, 'type'] == 'SubstitutionOn' and events_df.loc[i, 'teamId'] == teamId:
    #        sub_minute = str(events_df.loc[i, 'minute'])
    #        break
    
    # getting players dataframe
    match_players_df = pd.DataFrame()
    player_names = []
    player_ids = []
    player_pos = []
    player_kit_number = []


    for player in match_data[venue]['players']:
        player_names.append(player['name'])
        player_ids.append(player['playerId'])
        player_pos.append(player['position'])
        player_kit_number.append(player['shirtNo'])

    match_players_df['playerId'] = player_ids
    match_players_df['playerName'] = player_names
    match_players_df['playerPos'] = player_pos
    match_players_df['playerKitNumber'] = player_kit_number

    ### Comment out with sub issue ###
    #match_players_df = match_players_df.loc[match_players_df['playerPos'] != 'Sub']
    
   # extracting passes & carries
    passes_and_carries_df = events_df.loc[events_df['teamId'] == teamId].reset_index().drop('index', axis=1)
    passes_and_carries_df['playerId'] = passes_and_carries_df['playerId'].astype('float').astype('Int64')
    if 'playerName' in passes_and_carries_df.columns:
        passes_and_carries_df = passes_and_carries_df.drop(columns='playerName')
        passes_and_carries_df.dropna(subset=["playerId"], inplace=True)
        passes_and_carries_df.insert(27, column='playerName', value=[team_players_dict[i] for i in list(passes_and_carries_df['playerId'])])
    if 'passRecipientId' in passes_and_carries_df.columns:
        passes_and_carries_df = passes_and_carries_df.drop(columns='passRecipientId')
        passes_and_carries_df = passes_and_carries_df.drop(columns='passRecipientName')
    passes_and_carries_df.insert(28, column='passRecipientId', value=passes_and_carries_df['playerId'].shift(-1))  
    passes_and_carries_df.insert(29, column='passRecipientName', value=passes_and_carries_df['playerName'].shift(-1))  
    passes_and_carries_df.dropna(subset=["passRecipientName"], inplace=True)
    passes_and_carries_df = passes_and_carries_df.loc[passes_and_carries_df['type'].isin(['Pass', 'Carry'])].reset_index(drop=True)
    passes_and_carries_df = passes_and_carries_df.loc[passes_and_carries_df['outcomeType'] == 'Successful'].reset_index(drop=True)
    #passes_df = passes_df.loc[passes_df['passFreekick'] == False, :].reset_index(drop=True)
    #passes_df = passes_df.loc[passes_df['passCorner'] == False, :].reset_index(drop=True)
    #passes_df = passes_df.loc[passes_df['throwIn'] == False, :].reset_index(drop=True)
    # calc xT of each pass
    passes_and_carries_df = xThreat(passes_and_carries_df)
    passes_df = passes_and_carries_df[passes_and_carries_df['type'] == 'Pass']
    carries_df = passes_and_carries_df[passes_and_carries_df['type'] == 'Carry']
    carries_df = carries_df.merge(match_players_df, on=['playerId', 'playerName'], how='left', validate='m:1')
    index_names = passes_df.loc[passes_df['playerName']==passes_df['passRecipientName']].index
    passes_df.drop(index_names, inplace=True)
    passes_df = passes_df.merge(match_players_df, on=['playerId', 'playerName'], how='left', validate='m:1')
    passes_df = passes_df.merge(match_players_df.rename({'playerId': 'passRecipientId', 'playerName':'passRecipientName'},
                                                        axis='columns'), on=['passRecipientId', 'passRecipientName'],
                                                        how='left', validate='m:1', suffixes=['', 'Receipt'])


    # getting player average locations and pass counts
    location_formation = passes_df[['playerKitNumber', 'x', 'y', 'xT_gen']]
    average_locs_and_count = location_formation.groupby('playerKitNumber').agg({'x': ['mean'], 'y': ['mean', 'count'], 'xT_gen': ['sum']})
    average_locs_and_count.columns = ['x', 'y', 'count', 'xT_pass']
    average_locs_and_count.reset_index(inplace=True)
    average_locs_and_count['playerKitNumber'] = average_locs_and_count['playerKitNumber'].astype('int')

    # getting player xT from carries
    location_formation_carry = carries_df[['playerKitNumber', 'x', 'y', 'xT_gen']]
    average_locs_and_count_carry = location_formation_carry.groupby('playerKitNumber').agg({'x': ['mean'], 'y': ['mean', 'count'], 'xT_gen': ['sum']})
    average_locs_and_count_carry.columns = ['x_carry', 'y_carry', 'count_carry', 'xT_carry']
    average_locs_and_count_carry.reset_index(inplace=True)
    average_locs_and_count_carry['playerKitNumber'] = average_locs_and_count_carry['playerKitNumber'].astype('int')

    # merge carries and passes
    average_locs_and_count = pd.merge(average_locs_and_count_carry, average_locs_and_count, on='playerKitNumber', how='outer')

    # merge to create lineup df
    match_players_df = pd.merge(match_players_df, average_locs_and_count, on='playerKitNumber', how='outer')
    match_players_df = match_players_df.rename({'playerKitNumber': '#',
                                                'count': 'Qty',
                                                'playerName': 'Player',
                                                'xT_pass': 'xT_Pass',
                                                'xT_carry': 'xT_Carry'}, axis=1)
    match_players_df = match_players_df[['#', 'Player', 'playerPos', 'Qty', 'xT_Pass', 'xT_Carry']]
    
    sub_off = events_df[events_df['type'] == 'SubstitutionOff']
    sub_off = sub_off[['playerName', 'type', 'minute']]
    sub_off = sub_off.rename({'playerName': 'Player'}, axis=1)

    sub_on = events_df[events_df['type'] == 'SubstitutionOn']
    sub_on = sub_on[['playerName', 'type', 'minute']]
    sub_on = sub_on.rename({'playerName': 'Player'}, axis=1)

    subs = pd.concat([sub_on, sub_off], axis=0)

    match_players_df = pd.merge(subs, match_players_df, on='Player', how='outer')
    match_players_df = match_players_df.dropna(subset=['playerPos'])
    match_players_df = match_players_df[match_players_df['Qty'] > 0]

    match_players_df = match_players_df.sort_values(by='xT_Pass', ascending=True)

    match_players_df['#'] = match_players_df['#'].fillna(0).astype(int)
    match_players_df['Qty'] = match_players_df['Qty'].fillna(0).astype(int)
    match_players_df['xT_Pass'] = match_players_df['xT_Pass'].fillna(0).astype(float)
    match_players_df['xT_Carry'] = match_players_df['xT_Carry'].fillna(0).astype(float)

    ncols = 4 
    nrows = len(match_players_df)

    ax.set_xlim(0.5, ncols-0.5)
    ax.set_ylim(0, nrows)

    for y in range(nrows):
        # - player kit number
        x = 0.645
        kitNumber = match_players_df['#'].iloc[y]
        #kitNumber = kitNumber.fillna(0, inplace=True)
        label_ = f'{int(kitNumber)}'
        ax.annotate(
            xy=(x,y+0.38),
            text=label_,
            ha='left',
            va='center',
            color='white',
            fontproperties=Rubik.prop,
            fontsize=10
        )   
        # - player name
        x = 0.8
        playerName = match_players_df['Player'].iloc[y]
        subbed = match_players_df['type'].iloc[y]
        minute = match_players_df['minute'].iloc[y].astype('int')
        player_pos = match_players_df['playerPos'].iloc[y]

        if subbed == 'SubstitutionOff':
            text_color = '#ff6961'
            label_ = f"<{minute}'>  {playerName}   ({player_pos})"
            arrow = '->'
        elif subbed == 'SubstitutionOn':
            text_color = '#98fb98'
            label_ = f"<{minute}'>  {playerName}   ({player_pos})"
            arrow = '<-'
        else:
            text_color = '#1d2849'
            label_ = f"<50'>  {playerName}   ({player_pos})"
            arrow = '->'
        HighlightText(x=0.9,y=y+0.38,
                       s=label_,
                       ha='left',
                       va='center',
                       color='w',
                       fontproperties=Rubik.prop,
                       fontsize=10,
                       highlight_textprops=[{'size':10,'color':text_color}],
                       annotationbbox_kw={'frameon': False, 'pad': 1,
                                         'arrowprops': dict(arrowstyle=arrow,
                                                            color=text_color),
                                         'xybox': (1.075, y+0.38)
                                          },
                       ax=ax)

        # -- pass count
        x = 2.25
        count = match_players_df['Qty'].iloc[y]
        label_ = f'{int(count)}'
        ax.annotate(
            xy=(x,y+0.38),
            text=label_,
            ha='left',
            va='center',
            color='white',
            fontproperties=Rubik.prop,
            fontsize=10
        )

         # -- xt pass
        x = 2.685
        xt_pass = match_players_df['xT_Pass'].iloc[y]
        label_ = f'{round(xt_pass,2)}'
        ax.annotate(
            xy=(x,y+0.33),
            text=label_,
            ha='left',
            va='center',
            color='white',
            fontproperties=Rubik.prop,
            fontsize=10
        )

         # -- xt carry
        x = 3.185
        xt_carry = match_players_df['xT_Carry'].iloc[y]
        label_ = f'{round(xt_carry,2)}'
        ax.annotate(
            xy=(x,y+0.33),
            text=label_,
            ha='left',
            va='center',
            color='white',
            fontproperties=Rubik.prop,
            fontsize=10
        )
        
        # Table borders
    ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [nrows, nrows], lw = 1.5, color = 'white', marker = '', zorder = 4)
    ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [0, 0], lw = 0.5, color = 'white', marker = '', zorder = 4)
    for x in range(nrows):
        if x == 0:
            continue
        ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [x, x], lw = 1.15, color = 'gray', ls = ':', zorder = 3 , marker = '')

    # ----------------------------------------------------------------
    # - Column titles
    ax.set_axis_off()
    
    fig_text(
        x = 0.044, y = 0.25, 
        s = f'#',
        va = "bottom", ha = "left",
        fontsize = 10, color = "w", fontproperties=Rubik.prop
    );

    fig_text(
        x = 0.1310, y = 0.25, 
        s = f'Player',
        va = "bottom", ha = "left",
        fontsize = 10, color = "w", fontproperties=Rubik.prop
    );

    fig_text(
        x = 0.275, y = 0.25,
        s = f'Pass Count',
        va = "bottom", ha = "left",
        fontsize = 10, color = "w", fontproperties=Rubik.prop
    );

    fig_text(
        x = 0.355, y = 0.25,
        s = f'Pass xT',
        va = "bottom", ha = "left",
        fontsize = 10, color = "w", fontproperties=Rubik.prop
    );

    fig_text(
        x = 0.435, y = 0.25,
        s = f'Carry xT',
        va = "bottom", ha = "left",
        fontsize = 10, color = "w", fontproperties=Rubik.prop
    );
    
    fig_text(
        x = 0.544, y = 0.25,
        s = f'#',
        va = "bottom", ha = "left",
        fontsize = 10, color = "w", fontproperties=Rubik.prop
    );

    fig_text(
        x = 0.630, y = 0.25, 
        s = f'Player',
        va = "bottom", ha = "left",
        fontsize = 10, color = "w", fontproperties=Rubik.prop
    );

    fig_text(
        x = 0.775, y = 0.25, 
        s = f'Pass Count',
        va = "bottom", ha = "left",
        fontsize = 10, color = "w", fontproperties=Rubik.prop
    );

    fig_text(
        x = 0.855, y = 0.25, 
        s = f'Pass xT',
        va = "bottom", ha = "left",
        fontsize = 10, color = "w", fontproperties=Rubik.prop
    );

    fig_text(
        x = 0.935, y = 0.25, 
        s = f'Carry xT',
        va = "bottom", ha = "left",
        fontsize = 10, color = "w", fontproperties=Rubik.prop
    );
    

def createExpectedThreatFlowMap(match_data, events_df, matchId, 
                                team, opponent, home_color, away_color, ax):

    # create xT df per min for home team
    
    match_data['home']['name'] == team
    teamId_home = match_data['home']['teamId']
    venue_home = 'home'

    # getting player dictionary
    team_players_dict_home = {}
    for player in match_data[venue_home]['players']:
        team_players_dict_home[player['playerId']] = player['name']


    # getting minute of first substitution
    for i in events_df.index:
        if events_df.loc[i, 'type'] == 'SubstitutionOn' and events_df.loc[i, 'teamId'] == teamId_home:
            sub_minute = str(events_df.loc[i, 'minute'])
            break


    # getting players dataframe
    match_players_df_home = pd.DataFrame()
    player_names = []
    player_ids = []
    player_pos = []
    player_kit_number = []


    for player in match_data[venue_home]['players']:
        player_names.append(player['name'])
        player_ids.append(player['playerId'])
        player_pos.append(player['position'])
        player_kit_number.append(player['shirtNo'])

    match_players_df_home['playerId'] = player_ids
    match_players_df_home['playerName'] = player_names
    match_players_df_home['playerPos'] = player_pos
    match_players_df_home['playerKitNumber'] = player_kit_number

    # get goals
    home_goals_df = events_df[events_df['teamId'] == teamId_home].reset_index().drop('index', axis=1)
    home_goals_df = home_goals_df[home_goals_df['isGoal'] == True]
    home_goals_df = home_goals_df[home_goals_df['period'] != 'PenaltyShootout']

    # extracting passes & carries
    passes_carries_df_home = events_df.loc[events_df['teamId'] == teamId_home].reset_index().drop('index', axis=1)
    passes_carries_df_home['playerId'] = passes_carries_df_home['playerId'].astype('float').astype('Int64')
    if 'playerName' in passes_carries_df_home.columns:
        passes_carries_df_home = passes_carries_df_home.drop(columns='playerName')
        passes_carries_df_home.dropna(subset=["playerId"], inplace=True)
        passes_carries_df_home.insert(27, column='playerName', value=[team_players_dict_home[i] for i in list(passes_carries_df_home['playerId'])])
    if 'passRecipientId' in passes_carries_df_home.columns:
        passes_carries_df_home = passes_carries_df_home.drop(columns='passRecipientId')
        passes_carries_df_home = passes_carries_df_home.drop(columns='passRecipientName')
    passes_carries_df_home.insert(28, column='passRecipientId', value=passes_carries_df_home['playerId'].shift(-1))  
    passes_carries_df_home.insert(29, column='passRecipientName', value=passes_carries_df_home['playerName'].shift(-1))  
    passes_carries_df_home.dropna(subset=["passRecipientName"], inplace=True)
    passes_carries_df_home = passes_carries_df_home[passes_carries_df_home['type'].isin(['Pass', 'Carry'])].reset_index(drop=True)
    passes_carries_df_home = passes_carries_df_home.loc[passes_carries_df_home['outcomeType'] == 'Successful'].reset_index(drop=True)
    #passes_df = passes_df.loc[passes_df['passFreekick'] == False, :].reset_index(drop=True)
    #passes_df = passes_df.loc[passes_df['passCorner'] == False, :].reset_index(drop=True)
    #passes_df = passes_df.loc[passes_df['throwIn'] == False, :].reset_index(drop=True)
    # calc xT of each pass
    passes_carries_df_home = (passes_carries_df_home)
    #index_names = passes_carries_df_home.loc[passes_carries_df_home['playerName']==passes_carries_df_home['passRecipientName']].index
    #passes_df_home.drop(index_names, inplace=True)
    passes_carries_df_home = passes_carries_df_home.merge(match_players_df_home, on=['playerId', 'playerName'], how='left', validate='m:1')
    #passes_carries_df_home = passes_carries_df_home.merge(match_players_df_home.rename({'playerId': 'passRecipientId', 'playerName':'passRecipientName'},
    #                                                    axis='columns'), on=['passRecipientId', 'passRecipientName'],
    #                                                    how='left', validate='m:1', suffixes=['', 'Receipt'])
    
    xT_min_home = passes_carries_df_home.groupby('minute').agg({'xT': ['sum']})
    xT_min_home.columns = ['xT_min_h']
    xT_min_home = xT_min_home.sort_values(['minute'], ascending=True)


    # create xt per min for away team
    match_data['away']['name'] == opponent
    teamId_away = match_data['away']['teamId']
    venue_away = 'away'

    # getting player dictionary
    team_players_dict_away = {}
    for player in match_data[venue_away]['players']:
        team_players_dict_away[player['playerId']] = player['name']


    # getting minute of first substitution
    for i in events_df.index:
        if events_df.loc[i, 'type'] == 'SubstitutionOn' and events_df.loc[i, 'teamId'] == teamId_away:
            sub_minute2 = str(events_df.loc[i, 'minute'])
            break


    # getting players dataframe
    match_players_df_away = pd.DataFrame()
    player_names = []
    player_ids = []
    player_pos = []
    player_kit_number = []


    for player in match_data[venue_away]['players']:
        player_names.append(player['name'])
        player_ids.append(player['playerId'])
        player_pos.append(player['position'])
        player_kit_number.append(player['shirtNo'])

    match_players_df_away['playerId'] = player_ids
    match_players_df_away['playerName'] = player_names
    match_players_df_away['playerPos'] = player_pos
    match_players_df_away['playerKitNumber'] = player_kit_number


    # extracting away goals
    away_goals_df = events_df[events_df['teamId'] == teamId_away].reset_index().drop('index', axis=1)
    away_goals_df = away_goals_df[away_goals_df['isGoal'] == True]
    away_goals_df = away_goals_df[away_goals_df['period'] != 'PenaltyShootout']

    # extracting passes and carries
    passes_carries_df_away = events_df.loc[events_df['teamId'] == teamId_away].reset_index().drop('index', axis=1)
    passes_carries_df_away['playerId'] = passes_carries_df_away['playerId'].astype('float').astype('Int64')
    if 'playerName' in passes_carries_df_away.columns:
        passes_carries_df_away = passes_carries_df_away.drop(columns='playerName')
        passes_carries_df_away.dropna(subset=["playerId"], inplace=True)
        passes_carries_df_away.insert(27, column='playerName', value=[team_players_dict_away[i] for i in list(passes_carries_df_away['playerId'])])
    if 'passRecipientId' in passes_carries_df_away.columns:
        passes_carries_df_away = passes_carries_df_away.drop(columns='passRecipientId')
        passes_carries_df_away = passes_carries_df_away.drop(columns='passRecipientName')
    passes_carries_df_away.insert(28, column='passRecipientId', value=passes_carries_df_away['playerId'].shift(-1))  
    passes_carries_df_away.insert(29, column='passRecipientName', value=passes_carries_df_away['playerName'].shift(-1))  
    passes_carries_df_away.dropna(subset=["passRecipientName"], inplace=True)
    passes_carries_df_away = passes_carries_df_away.loc[passes_carries_df_away['type'] == 'Pass'].reset_index(drop=True)
    passes_carries_df_away = passes_carries_df_away.loc[passes_carries_df_away['outcomeType'] == 'Successful'].reset_index(drop=True)
    #passes_df = passes_df.loc[passes_df['passFreekick'] == False, :].reset_index(drop=True)
    #passes_df = passes_df.loc[passes_df['passCorner'] == False, :].reset_index(drop=True)
    #passes_df = passes_df.loc[passes_df['throwIn'] == False, :].reset_index(drop=True)
    # calc xaway each pass
    passes_carries_df_away = xThreat(passes_carries_df_away)
    #index_names = passes_df_away.loc[passes_df_away['playerName']==passes_df_away['passRecipientName']].index
    #passes_df_away.drop(index_names, inplace=True)
    passes_carries_df_away = passes_carries_df_away.merge(match_players_df_away, on=['playerId', 'playerName'], how='left', validate='m:1')
    #passes_df_away = passes_df_away.merge(match_players_df_away.rename({'playerId': 'passRecipientId', 'playerName':'passRecipientName'},
    #                                                    axis='columns'), on=['passRecipientId', 'passRecipientName'],
    #                                                    how='left', validate='m:1', suffixes=['', 'Receipt'])
    
    xT_min_away = passes_carries_df_away.groupby('minute').agg({'xT': ['sum']})
    xT_min_away.columns = ['xT_min_a']
    xT_min_away = xT_min_away.sort_values(['minute'], ascending=True)

    # merge dfs to get difference
    xT_min = pd.merge(xT_min_home, xT_min_away, on='minute', how='outer')
    xT_min['xT_min_h'] = pd.to_numeric(xT_min['xT_min_h'], errors='coerce').fillna(0)
    xT_min['xT_min_a'] = pd.to_numeric(xT_min['xT_min_a'], errors='coerce').fillna(0)
    xT_min = xT_min.sort_values(['minute'], ascending=True)
    
    xT_min['xTSMA_h'] = xT_min['xT_min_h'].rolling(window=5).mean()
    xT_min['xTSMA_a'] = xT_min['xT_min_a'].rolling(window=5).mean()
    xT_min['xT_difference'] = xT_min['xTSMA_h'] - xT_min['xTSMA_a']

    xT_min.reset_index(level=0, inplace=True)
    
    xT_min = xT_min.dropna()
    x = xT_min['minute']
    y = xT_min['xT_difference']

    X_Y_Spline = make_interp_spline(x, y)
    X_ = np.linspace(x.min(), x.max(), 500)
    Y_ = X_Y_Spline(X_)
    
    ax.plot(X_, Y_, c='w', zorder=0)
    ax.set_ylim(Y_.min(), Y_.max()+0.05)
    ax.fill_between(X_, Y_, 0, where=Y_>0, color=home_color, alpha=0.8, interpolate=True, zorder=0)
    ax.fill_between(X_, Y_, 0, where=Y_<0, color=away_color, alpha=0.4, interpolate=True, zorder=0)
    
    ax.set_xticks([0,15,30,45,60,75,90,105,120])
    ax.set_xlim([0,x.max()+2])
    ax.tick_params(axis='x', colors='w')    
    ax.tick_params(axis='y', colors='w')
    for label in ax.get_xticklabels():
        label.set_fontproperties(Rubik.prop)
        label.set_color('w')
    for label in ax.get_yticklabels():
        label.set_fontproperties(Rubik.prop)
        label.set_color('w')

    ticks =  ax.get_yticks()

    # set labels to absolute values and with integer representation
    ax.set_yticklabels([format(abs(tick),'.2f') for tick in ticks])

    goals = pd.concat([home_goals_df, away_goals_df], axis=0)

    goals['teamId'] = np.where( (goals['isOwnGoal'] == True) & (goals['teamId'] == teamId_home), 
                                teamId_away,
                                np.where( (goals['isOwnGoal'] == True) & (goals['teamId'] == teamId_away), teamId_home, 
                                    goals['teamId']
                                    )
                                )

    #goals['teamId'] = np.where((goals['isOwnGoal'] == True) & (goals['teamId'] == teamId_away), 
    #                            teamId_homr, goals['teamId'])

    home_goals_df = goals[goals['teamId'] == teamId_home]
    away_goals_df = goals[goals['teamId'] == teamId_away]

    home_min = list(home_goals_df.minute)
    away_min = list(away_goals_df.minute)
    for min in home_min:
        ax.axvline(x=min, color=home_color, linestyle='--', zorder=2)
    for min in away_min:    
        ax.axvline(x=min, color=away_color, linestyle='--', zorder=2)

    spines = ['top', 'right', 'bottom', 'left']
    for s in spines:
        if s in ['top', 'right']:
            ax.spines[s].set_visible(False)
        else:
            ax.spines[s].set_color('w')
            ax.spines[s].set_linewidth(2)
    ax.set_facecolor('#1d2849')
    #ax.text(18, y.max()+0.03, "xT Flow", size=16, ha='left',
    #       va='center', color='w', fontproperties=Rubik.prop);
    #ax.text(15, 0, "15", size=15, ha='center', zorder=1, 
    #       va='center', color='w', fontproperties=Rubik.prop);
    #ax.text(30, 0, "30", size=15, ha='center', zorder=1,
    #       va='center', color='w', fontproperties=Rubik.prop);
    #ax.text(45, 0, "45", size=15, ha='center', zorder=1,
    #       va='center', color='w', fontproperties=Rubik.prop);
    #ax.text(60, 0, "60", size=15, ha='center', zorder=1,
    #       va='center', color='w', fontproperties=Rubik.prop);
    #ax.text(75, 0, "75", size=15, ha='center', zorder=1,
    #       va='center', color='w', fontproperties=Rubik.prop);
    #ax.text(90, 0, "90", size=15, ha='center', zorder=1,
    #       va='center', color='w', fontproperties=Rubik.prop);

def nums_cumulative_sum(nums_list):
    return [sum(nums_list[:i+1]) for i in range(len(nums_list))]

def createStepPlots(match_data, events_df, ax, team, opponent, home_color, away_color):
    # load the model from disk
    loaded_model_op = pickle.load(open('Models/expected_goals_model_lr.sav', 'rb'))
    loaded_model_non_op = pickle.load(open('Models/expected_goals_model_lr_v2.sav', 'rb'))

    # getting home team id and venue
    h_teamId = match_data['home']['teamId']
    h_venue = 'home'

    # getting away team id and venue
    a_teamId = match_data['away']['teamId']
    a_venue = 'away'

    total_shots = events_df.loc[events_df['isShot']==True].reset_index(drop=True)

    total_shots['teamId'] = np.where((total_shots['isOwnGoal'] == True) & (total_shots['teamId'] == h_teamId), 
                                        a_teamId, 
                                        np.where((total_shots['isOwnGoal'] == True) & (total_shots['teamId'] == a_teamId), 
                                        h_teamId, total_shots['teamId'])) 

    #total_shots['teamId'] = np.where((total_shots['isOwnGoal'] == True) & (total_shots['teamId'] == a_teamId), 
    #                                    h_teamId, total_shots['teamId']) 
                                       
    #total_shots = total_shots.loc[total_shots['period']!='PenaltyShootout'].reset_index(drop=True)
    data_preparation(total_shots)
    total_shots['distance_to_goal'] = np.sqrt(((100 - total_shots['x'])**2) + ((total_shots['y'] - (100/2))**2) )
    total_shots['distance_to_center'] = abs(total_shots['y'] - 100/2)
    total_shots['angle'] = np.absolute(np.degrees(np.arctan((abs((100/2) - total_shots['y'])) / (100 - total_shots['x']))))
    total_shots['isFoot'] = np.where(((total_shots['isLeftFooted'] == 1) | (total_shots['isRightFooted'] == 1)) &
                                              (total_shots['isHead'] == 0)
                                            , 1, 0
                                            )
    

    features_op = ['distance_to_goal',
           #'distance_to_centerM',    # commented out as elected to go with 'angle' instead
            'angle',
            'isFoot',
            'isHead'
           ]
    
    features_non_op = ['distance_to_goal',
           #'distance_to_centerM',    # commented out as elected to go with 'angle' instead
            'angle',
            'isFoot',
            'isHead',
            'isDirectFree',
            'isSetPiece',
            'isFromCorner'
           ]

    # split model to only open play shots to apply open play model
    # calc xG for open play shots only
    total_shots_op = total_shots.loc[total_shots['isRegularPlay'] == 1]
    shots_df_op = total_shots_op[features_op]
    xG = loaded_model_op.predict_proba(shots_df_op)[:,1]
    shots_df_op['xG'] = xG
    xg = shots_df_op['xG']
    total_shots_op = pd.merge(total_shots_op, xg, left_index=True, right_index=True)
    
    # split dataframe to only set pieces to apply set piece model
    # calc xG for set pieces only (not including penalties)
    total_shots_non_op = total_shots.loc[total_shots['isRegularPlay'] == 0]
    total_shots_non_op = total_shots_non_op.loc[total_shots_non_op['isPenalty'] == 0]
    shots_df_non_op = total_shots_non_op[features_non_op]
    xG = loaded_model_non_op.predict_proba(shots_df_non_op)[:,1]
    shots_df_non_op['xG'] = xG
    xg = shots_df_non_op['xG']
    total_shots_non_op = pd.merge(total_shots_non_op, xg, left_index=True, right_index=True)    
    
    # split dataframe to only penalties and set xG for penalties to 0.79
    total_shots_pk = total_shots.loc[total_shots['isPenalty'] == 1]
    total_shots_pk['xG'] = 0.79
    
    # combine all three dataframes 
    total_shots = pd.concat([total_shots_op, total_shots_non_op, total_shots_pk], axis=0).reset_index(drop=True)
    total_shots = total_shots.sort_values(by='minute', ascending=True)

    h_team_shots = total_shots.loc[total_shots['teamId'] == h_teamId].reset_index(drop=True)
    h_mask_goal = h_team_shots.isGoal == True

    a_team_shots = total_shots.loc[total_shots['teamId'] == a_teamId].reset_index(drop=True)
    a_mask_goal = a_team_shots.isGoal == True

    # create a list of tuples of including the confidence interval of each xG value
    home_array = np.array(h_team_shots.xG)
    home_conf_intervals = []
    for xG in home_array:
        home_sigma = np.std(home_array)
        home_conf_int = stats.norm.interval(0.90, loc=xG, 
        scale=home_sigma)
        home_conf_intervals.append(home_conf_int)

    away_array = np.array(a_team_shots.xG)
    away_conf_intervals = []
    for xG in away_array:
        away_sigma = np.std(away_array)
        away_conf_int = stats.norm.interval(0.90, loc=xG, 
        scale=away_sigma)
        away_conf_intervals.append(away_conf_int)

    h_xG = [0]
    a_xG = [0]
    h_min = [0]
    a_min = [0]

    for index, row in h_team_shots.iterrows():
        h_xG.append(row['xG'])
        h_min.append(row['minute'])

    for index, row in a_team_shots.iterrows():
        a_xG.append(row['xG'])
        a_min.append(row['minute'])

    home_cumulative = nums_cumulative_sum(h_xG)
    home_cum = home_cumulative[1:]
    away_cumulative = nums_cumulative_sum(a_xG)
    away_cum = away_cumulative[1:]

    h_team_shots['cum_xG'] = home_cum
    a_team_shots['cum_xG'] = away_cum

    all_xg = pd.concat([a_team_shots, h_team_shots])

    home_goal = h_team_shots.loc[h_team_shots['isGoal'] == 1]
    ax.scatter(home_goal.minute, home_goal.cum_xG, marker='o', color=home_color, s=400, zorder=3, alpha=0.4)
    for index, row in home_goal.iterrows():
        ax.annotate(f'GOAL', xy=(row.minute, row.cum_xG), fontproperties=Rubik.prop, size=12, c='w', ha='center', va='center')
    for index, row in home_goal.iterrows():
        if row.h_a == 'a':
            ax.annotate(f'\n\n{row.playerName} (OG)', xy=(row.minute, row.cum_xG), fontproperties=Rubik.prop, size=12, c='w', ha='center', va='center')
        else:
            ax.annotate(f'\n\n{row.playerName}', xy=(row.minute, row.cum_xG), fontproperties=Rubik.prop, size=12, c='w', ha='center', va='center')


    away_goal = a_team_shots.loc[a_team_shots['isGoal'] == 1]
    ax.scatter(away_goal.minute, away_goal.cum_xG, marker='o', color=away_color, s=400, zorder=3, alpha=0.4)
    for index, row in away_goal.iterrows():
        ax.annotate(f'GOAL', xy=(row.minute, row.cum_xG), fontproperties=Rubik.prop, size=12, c='w', ha='center', va='center')
    for index, row in away_goal.iterrows():
        if row.h_a == 'h':
            ax.annotate(f'\n\n{row.playerName} (OG)', xy=(row.minute, row.cum_xG), fontproperties=Rubik.prop, size=12, c='w', ha='center', va='center')    
        else:
            ax.annotate(f'\n\n{row.playerName}', xy=(row.minute, row.cum_xG), fontproperties=Rubik.prop, size=12, c='w', ha='center', va='center')

    ax.step(x=h_min, y=home_cumulative, where='post', label=team, c=home_color, zorder=2)
    ax.step(x=a_min, y=away_cumulative, where='post', label=opponent, c=away_color, zorder=2)
    ax.axvline(x=45, color='gray', linestyle='--', zorder=2)
    ax.axvline(x=90, color='gray', linestyle='--', zorder=2)
    #ax.set_xlabel('Minute', fontproperties=Rubik.prop, color='white', fontsize=12)
    #ax.set_ylabel('Expected Goals', fontproperties=Rubik.prop, color='white', fontsize=12)
    #ax.grid(lw="0.75", color='lightgrey', zorder=1, alpha=0.35, axis='y')
    ax.set_xticks([0,15,30,45,60,75,90,105,120])
    #ax.set_xlim([0,all_xg.minute.max()+5])
    #mpl.rcParams['xtick.color'] = 'white'
    #mpl.rcParams['ytick.color'] = 'white'
    ax.xaxis.label.set_color('w')
    ax.yaxis.label.set_color('w')

    h_team_shots['difference'] = h_team_shots['cum_xG'] - h_team_shots['xG']
    lower_bound = [x[0] for x in home_conf_intervals]
    h_team_shots['lower_bound'] = lower_bound
    h_team_shots['lower_bound'] = h_team_shots['lower_bound'] + h_team_shots['difference']
    upper_bound = [x[1] for x in home_conf_intervals]
    h_team_shots['upper_bound'] = upper_bound
    h_team_shots['upper_bound'] = h_team_shots['upper_bound'] + h_team_shots['difference']
    home_percent_inc = h_team_shots.upper_bound.max()/h_team_shots.cum_xG.max() - 1
    home_percent_dec = h_team_shots.cum_xG.max() /h_team_shots.lower_bound.max() - 1
    h_team_shots['lower_int'] = h_team_shots.cum_xG * (1 - home_percent_dec)
    h_team_shots['upper_int'] = h_team_shots.cum_xG *(1 + home_percent_inc)


    ax.fill_between(
        h_min[1:], 
        h_team_shots['lower_int'],
        h_team_shots['upper_int'], 
        interpolate = True,
        alpha = 0.1,
        zorder = 0, 
        color=home_color,
        step='post'
    )

    a_team_shots['difference'] = a_team_shots['cum_xG'] - a_team_shots['xG']
    lower_bound = [x[0] for x in away_conf_intervals]
    a_team_shots['lower_bound'] = lower_bound
    a_team_shots['lower_bound'] = a_team_shots['lower_bound'] + a_team_shots['difference']
    upper_bound = [x[1] for x in away_conf_intervals]
    a_team_shots['upper_bound'] = upper_bound
    a_team_shots['upper_bound'] = a_team_shots['upper_bound'] + a_team_shots['difference']
    away_percent_inc = a_team_shots.upper_bound.max()/a_team_shots.cum_xG.max() - 1
    away_percent_dec = a_team_shots.cum_xG.max() /a_team_shots.lower_bound.max() - 1
    a_team_shots['lower_int'] = a_team_shots.cum_xG * (1 - away_percent_dec)
    a_team_shots['upper_int'] = a_team_shots.cum_xG * (1 + away_percent_inc)

    ax.fill_between(
        a_min[1:], 
        a_team_shots['lower_int'],
        a_team_shots['upper_int'], 
        interpolate = True,
        alpha = 0.25,
        zorder = 0, 
        color=away_color,
        step='post'
    )

    all_xg = pd.concat([a_team_shots, h_team_shots])
    ax.set_ylim([0,all_xg.upper_int.max()])
    ax.set_yticks(np.arange(0, all_xg.upper_bound.max(), 0.25))
    ax.tick_params(axis='x', colors='w')    
    ax.tick_params(axis='y', colors='w')
    for label in ax.get_xticklabels():
        label.set_fontproperties(Rubik.prop)
        label.set_color('w')
    for label in ax.get_yticklabels():
        label.set_fontproperties(Rubik.prop)
        label.set_color('w')

    spines = ['top', 'right', 'bottom', 'left']
    for s in spines:
        if s in ['top', 'right']:
            ax.spines[s].set_visible(False)
        else:
            ax.spines[s].set_color('w')
            ax.spines[s].set_linewidth(2)

def plot_exp_threat(match_data, events_df, ax, background, home_color, away_color, 
                    team, opponent, score, text_color, home_ha, home_va, away_ha, away_va):

        # getting home team id and venue
    h_teamId = match_data['home']['teamId']
    h_venue = 'home'

    # getting away team id and venue
    a_teamId = match_data['away']['teamId']
    a_venue = 'away'

    # extracting passes & carries
    passes_and_carries_df = events_df
    passes_and_carries_df['playerId'] = passes_and_carries_df['playerId'].astype('float').astype('Int64')
    #passes_and_carries_df.insert(28, column='passRecipientId', value=passes_and_carries_df['playerId'].shift(-1))  
    #passes_and_carries_df.insert(29, column='passRecipientName', value=passes_and_carries_df['playerName'].shift(-1))  
    #passes_and_carries_df.dropna(subset=["passRecipientName"], inplace=True)
    passes_and_carries_df = passes_and_carries_df.loc[passes_and_carries_df['type'].isin(['Pass', 'Carry'])].reset_index(drop=True)
    passes_and_carries_df = passes_and_carries_df.loc[passes_and_carries_df['outcomeType'] == 'Successful'].reset_index(drop=True)

    # home data set
    h_passes_and_carries_df = passes_and_carries_df[passes_and_carries_df['teamId'] == h_teamId]
    # away data set
    a_passes_and_carries_df = passes_and_carries_df[passes_and_carries_df['teamId'] == a_teamId]

    # calc xT of each pass and carry
    h_passes_and_carries_df = xThreat(h_passes_and_carries_df)
    a_passes_and_carries_df = xThreat(a_passes_and_carries_df)

    h_passes = h_passes_and_carries_df[h_passes_and_carries_df['type'] == 'Pass']
    h_carries = h_passes_and_carries_df[h_passes_and_carries_df['type'] == 'Carry']

    h_ex_threat_passes = h_passes.groupby('playerName').agg({'xT_gen': ['sum']})
    h_ex_threat_passes.columns = ['passes_xT']
    h_ex_threat_passes.reset_index(inplace=True)

    h_ex_threat_carries = h_carries.groupby('playerName').agg({'xT_gen': ['sum']})
    h_ex_threat_carries.columns = ['carries_xT']
    h_ex_threat_carries.reset_index(inplace=True)

    h_ex_threat_df = pd.merge(h_ex_threat_carries, h_ex_threat_passes, on='playerName', how='outer')
    h_ex_threat_df.fillna(0, inplace=True)

    h_ex_threat_df = pd.merge(h_ex_threat_carries, h_ex_threat_passes, on='playerName', how='outer')
    h_ex_threat_df.fillna(0, inplace=True)
    h_ex_threat_df['xT'] = h_ex_threat_df['carries_xT'] + h_ex_threat_df['passes_xT']
    h_ex_threat_df['h_a'] = 'h'

    a_passes = a_passes_and_carries_df[a_passes_and_carries_df['type'] == 'Pass']
    a_carries = a_passes_and_carries_df[a_passes_and_carries_df['type'] == 'Carry']

    a_ex_threat_passes = a_passes.groupby('playerName').agg({'xT_gen': ['sum']})
    a_ex_threat_passes.columns = ['passes_xT']
    a_ex_threat_passes.reset_index(inplace=True)

    a_ex_threat_carries = a_carries.groupby('playerName').agg({'xT_gen': ['sum']})
    a_ex_threat_carries.columns = ['carries_xT']
    a_ex_threat_carries.reset_index(inplace=True)

    a_ex_threat_df = pd.merge(a_ex_threat_carries, a_ex_threat_passes, on='playerName', how='outer')
    a_ex_threat_df.fillna(0, inplace=True)
    a_ex_threat_df['xT'] = a_ex_threat_df['carries_xT'] + a_ex_threat_df['passes_xT'] 
    a_ex_threat_df['h_a'] = 'a'

    all_ex_threat = pd.concat([a_ex_threat_df, h_ex_threat_df], axis=0)
    all_ex_threat['fontsize'] = all_ex_threat.xT.rank()//2.5
    all_ex_threat['fontsize'] = np.where(all_ex_threat['fontsize'] < 1 
                                         , 1, all_ex_threat['fontsize']
                                       )

    h_ex_threat_df = all_ex_threat[all_ex_threat['h_a'] == 'h']
    #h_ex_threat_df = h_ex_threat_df[h_ex_threat_df['carries_xT'] > -0.01]
    a_ex_threat_df = all_ex_threat[all_ex_threat['h_a'] == 'a']
    #a_ex_threat_df = a_ex_threat_df[a_ex_threat_df['carries_xT'] > -0.01]

    # add grid
    #ax.grid(ls='dotted', lw="0.5", color='lightgrey', zorder=1)

    y = h_ex_threat_df.passes_xT
    x = h_ex_threat_df.carries_xT
    z = h_ex_threat_df.xT

    y2 = a_ex_threat_df.passes_xT
    x2 = a_ex_threat_df.carries_xT
    z2 = a_ex_threat_df.xT
    
    ax.scatter(x,y, edgecolors=background, alpha=0.6, s=z*1500, 
               lw=0.5, zorder=3, color=home_color)
    ax.scatter(x2,y2, edgecolors=background, alpha=0.6, s=z2*1500,
               lw=0.5, zorder=3, color=away_color)

    ax.axvline(x=all_ex_threat.carries_xT.median(), linestyle='--', color='gray', alpha=0.8, zorder=2, label='Median')
    #ax.legend(loc='upper right')
    ax.axhline(y=all_ex_threat.passes_xT.median(), linestyle='--', color='gray', alpha=0.8, zorder=2)

    x = all_ex_threat.carries_xT.min() - 0.025
    y = all_ex_threat.passes_xT.max() + 0.025

    fig_text(0.1,0.97, "Goal Probability Added (xT)", 
             fontsize=16, color=text_color, fontproperties=Rubik.prop)

    annotation_string = (f'<{format(team).upper()}>  {score}  <{format(opponent).upper()}>')
    fig_text(0.1, 0.93, annotation_string, fontweight='regular', fontsize=14, 
             color='white', fontproperties=Rubik.prop,
             highlight_textprops=[{'color': home_color},
                                  {'color': away_color}])

    # add x and y labels
    ax.set_xlabel("Goal Probability Added via Carry", fontsize=12, color='white', fontproperties=Rubik.prop)
    ax.set_ylabel('Goal Probability Added via Pass', fontsize=12, color='white', fontproperties=Rubik.prop)
    
    for index, row in h_ex_threat_df.iterrows():
        ax.annotate(
            xy=(row.carries_xT, row.passes_xT),
            text=row.playerName,
            ha=home_ha,
            va=home_va,
            color='w',
            fontproperties=Rubik.prop,
            fontsize=row.fontsize
        )
    
    for index, row in a_ex_threat_df.iterrows():
        ax.annotate(
            xy=(row.carries_xT, row.passes_xT),
            text=row.playerName,
            ha=away_ha,
            va=away_va,
            color='w',
            fontproperties=Rubik.prop,
            fontsize=row.fontsize
        )

    ax.xaxis.label.set_color('w')
    ax.yaxis.label.set_color('w')

    ax.tick_params(axis='x', colors='w')    
    ax.tick_params(axis='y', colors='w')
    for label in ax.get_xticklabels():
        label.set_fontproperties(Rubik.prop)
        label.set_color('w')
    for label in ax.get_yticklabels():
        label.set_fontproperties(Rubik.prop)
        label.set_color('w')

    spines = ['top', 'right', 'bottom', 'left']
    for s in spines:
        if s in ['top', 'right']:
            ax.spines[s].set_visible(False)
        else:
            ax.spines[s].set_color('w')
            ax.spines[s].set_linewidth(2)

    fig_text(0.1, 0.045, "Negative threat passes and carries are assigned zero threat\nViz by @egudi_analysis | Data via Opta",
             fontsize=9, fontproperties=Rubik.prop, color=text_color)

def calc_xg(events_df):

    # load the models from disk to get xG values
    loaded_model_op = pickle.load(open('Models/expected_goals_model_lr.sav', 'rb'))
    loaded_model_non_op = pickle.load(open('Models/expected_goals_model_lr_v2.sav', 'rb'))

    # split up df into shots data
    shots_df = events_df[events_df['isShot'] == True]
    shots_df = shots_df[shots_df['period'] != 'PenaltyShootout']
    # prepare data to calc xG
    data_preparation(shots_df)
    shots_df['distance_to_goal'] = np.sqrt(((100 - shots_df['x'])**2) + ((shots_df['y'] - (100/2))**2) )
    shots_df['distance_to_center'] = abs(shots_df['y'] - 100/2)
    shots_df['angle'] = np.absolute(np.degrees(np.arctan((abs((100/2) - shots_df['y'])) / (100 - shots_df['x']))))
    shots_df['isFoot'] = np.where(((shots_df['isLeftFooted'] == 1) | (shots_df['isRightFooted'] == 1)) &
                                              (shots_df['isHead'] == 0)
                                  , 1,0
                                 )

    features_op = ['distance_to_goal',
           #'distance_to_centerM',    # commented out as elected to go with 'angle' instead
            'angle',
            'isFoot',
            'isHead'
           ]

    features_non_op = ['distance_to_goal',
           #'distance_to_centerM',    # commented out as elected to go with 'angle' instead
            'angle',
            'isFoot',
            'isHead',
            'isDirectFree',
            'isSetPiece',
            'isFromCorner'
           ]

    # split model to only open play shots to apply open play model
    # calc xG for open play shots only
    total_shots_op = shots_df.loc[shots_df['isRegularPlay'] == 1]
    shots_df_op = total_shots_op[features_op]
    xG = loaded_model_op.predict_proba(shots_df_op)[:,1]
    shots_df_op['xG'] = xG
    xg = shots_df_op['xG']
    total_shots_op = pd.merge(total_shots_op, xg, left_index=True, right_index=True)

    # split dataframe to only set pieces to apply set piece model
    # calc xG for set pieces only (not including penalties)
    total_shots_non_op = shots_df.loc[shots_df['isRegularPlay'] == 0]
    total_shots_non_op = total_shots_non_op.loc[total_shots_non_op['isPenalty'] == 0]
    shots_df_non_op = total_shots_non_op[features_non_op]
    xG = loaded_model_non_op.predict_proba(shots_df_non_op)[:,1]
    shots_df_non_op['xG'] = xG
    xg = shots_df_non_op['xG']
    total_shots_non_op = pd.merge(total_shots_non_op, xg, left_index=True, right_index=True)    

    # split dataframe to only penalties and set xG for penalties to 0.79
    total_shots_pk = shots_df.loc[shots_df['isPenalty'] == 1]
    total_shots_pk['xG'] = 0.79

    # combine all three dataframes 
    shots_df_adv = pd.concat([total_shots_op, total_shots_non_op, total_shots_pk], axis=0).reset_index(drop=True)

    return shots_df_adv

def calc_xa(events_df):
    
    # load the models from disk to get xG values
    loaded_model = pickle.load(open('Models/expected_assists_model_lr.sav', 'rb'))

    # split up df into pass data
    model_df = events_df[events_df['type'] == 'Pass']
    model_df = model_df[model_df['outcomeType'] == 'Successful']
    
    # prepare data to calc xG
    data_preparation_xA(model_df)
    pass_start_x = model_df.x
    pass_start_y = model_df.y
    pass_end_x = model_df.endX
    pass_end_y = model_df.endY
    
    model_df['distance_of_pass'] = np.sqrt(((pass_end_x - pass_start_x)**2) + ((pass_end_y - pass_start_y)**2) )
    
    model_df['isFoot'] = np.where(((model_df['isLeftFoot'] == 1) | (model_df['isRightFoot'] == 1)) &
                                              (model_df['isHead'] == 0)
                                            , 1, 0
                                            )

    features = ['distance_of_pass',
                'isOpenPlay',
                'isFoot',
                'isHead',
                'isFreeKick',
                'isCorner',
                'isThroughBall'
                ]

    # calc xA
    model_df_xA = model_df[features]
    xA = loaded_model.predict_proba(model_df_xA)[:,1]
    model_df_xA['xA'] = xA
    xa = model_df_xA['xA']
    model_df_final = pd.merge(model_df, xa, left_index=True, right_index=True)
    
    return model_df_final

def match_stats(events_df, match_data):
    
    "compiles all the match stats into one single data frame and saves in each team folder"
    
    # get match Id
    matchId = match_data['matchId']

    # get home team name
    home_team_name = match_data['home']['name']
    home_team_id = match_data['home']['teamId']

    # get away team name
    away_team_name = match_data['away']['name']
    away_team_id = match_data['away']['teamId']

    # getting home team id and venue
    h_teamId = match_data['home']['teamId']
    h_venue = 'home'

    # getting away team id and venue
    a_teamId = match_data['away']['teamId']
    a_venue = 'away'
    
    def possession():
        
        "Calculates ball possession based on number of passes made"
        
        passes = events_df[events_df['type'] == 'Pass']
        
        h_passes = passes[passes['teamId'] == h_teamId]
        a_passes = passes[passes['teamId'] == a_teamId]
        
        poss_home = round((len(h_passes) / len(passes)*100), 2)
        poss_away = round((len(a_passes) / len(passes)*100), 2)
        
        home_poss_df = pd.DataFrame({'team': [home_team_name],
                                     'teamId': [home_team_id],
                                     'matchId': [matchId],
                                     'metric': 'Possession',
                                     'stat': [poss_home]})

        away_poss_df = pd.DataFrame({'team': [away_team_name],
                                     'teamId': [away_team_id],
                                     'matchId': [matchId],
                                     'metric': 'Possession',
                                     'stat': [poss_away]})
        
        match_poss = pd.concat([home_poss_df, away_poss_df], axis=0).reset_index(drop=True)

        return match_poss

    def field_tilt():

        "Calculates the field tilt based on the share of each team's passes in the attacking third"

        passes = events_df[events_df['type'] == 'Pass']
        passes_att_third = passes[passes['x'] > 66]

        h_passes_att_third = passes_att_third[passes_att_third['teamId'] == h_teamId]
        a_passes_att_third = passes_att_third[passes_att_third['teamId'] == a_teamId]

        field_tilt_home = round((len(h_passes_att_third) / len(passes_att_third)*100), 2)
        field_tilt_away = round((len(a_passes_att_third) / len(passes_att_third)*100), 2)

        home_tilt_df = pd.DataFrame({'team': [home_team_name],
                                     'teamId': [home_team_id],
                                     'matchId': [matchId],
                                     'metric': 'FieldTilt',
                                     'stat': [field_tilt_home]})

        away_tilt_df = pd.DataFrame({'team': [away_team_name],
                                     'teamId': [away_team_id],
                                     'matchId': [matchId],
                                     'metric': 'FieldTilt',
                                     'stat': [field_tilt_away]})
        
        match_tilt = pd.concat([home_tilt_df, away_tilt_df], axis=0).reset_index(drop=True)

        return match_tilt
    
    def shots():
        
        "Calculates xG from a match"
        
        loaded_model_op = pickle.load(open('Models/expected_goals_model_lr.sav', 'rb'))
        loaded_model_non_op = pickle.load(open('Models/expected_goals_model_lr_v2.sav', 'rb'))
        
        total_shots = events_df.loc[events_df['isShot']==True].reset_index(drop=True)
        total_shots = total_shots[total_shots['period'] != 'PenaltyShootout']
        data_preparation(total_shots)
        total_shots['distance_to_goal'] = np.sqrt(((100 - total_shots['x'])**2) + ((total_shots['y'] - (100/2))**2) )
        total_shots['distance_to_center'] = abs(total_shots['y'] - 100/2)
        total_shots['angle'] = np.absolute(np.degrees(np.arctan((abs((100/2) - total_shots['y'])) / (100 - total_shots['x']))))
        total_shots['isFoot'] = np.where(((total_shots['isLeftFooted'] == 1) | (total_shots['isRightFooted'] == 1)) &
                                                  (total_shots['isHead'] == 0)
                                                , 1, 0
                                                )

        features_op = ['distance_to_goal',
               #'distance_to_centerM',    # commented out as elected to go with 'angle' instead
                'angle',
                'isFoot',
                'isHead'
               ]

        features_non_op = ['distance_to_goal',
               #'distance_to_centerM',    # commented out as elected to go with 'angle' instead
                'angle',
                'isFoot',
                'isHead',
                'isDirectFree',
                'isSetPiece',
                'isFromCorner'
               ]

        # split model to only open play shots to apply open play model
        # calc xG for open play shots only
        total_shots_op = total_shots.loc[total_shots['isRegularPlay'] == 1]
        shots_df_op = total_shots_op[features_op]
        xG = loaded_model_op.predict_proba(shots_df_op)[:,1]
        shots_df_op['xG'] = xG
        xg = shots_df_op['xG']
        total_shots_op = pd.merge(total_shots_op, xg, left_index=True, right_index=True)

        # split dataframe to only set pieces to apply set piece model
        # calc xG for set pieces only (not including penalties)
        total_shots_non_op = total_shots.loc[total_shots['isRegularPlay'] == 0]
        total_shots_non_op = total_shots_non_op.loc[total_shots_non_op['isPenalty'] == 0]
        shots_df_non_op = total_shots_non_op[features_non_op]
        xG = loaded_model_non_op.predict_proba(shots_df_non_op)[:,1]
        shots_df_non_op['xG'] = xG
        xg = shots_df_non_op['xG']
        total_shots_non_op = pd.merge(total_shots_non_op, xg, left_index=True, right_index=True)    

        # split dataframe to only penalties and set xG for penalties to 0.79
        total_shots_pk = total_shots.loc[total_shots['isPenalty'] == 1]
        total_shots_pk['xG'] = 0.79

        # combine all three dataframes 
        total_shots = pd.concat([total_shots_op, total_shots_non_op, total_shots_pk], axis=0).reset_index(drop=True)
        
        h_team_shots = total_shots.loc[total_shots['teamId'] == h_teamId].reset_index(drop=True)
        h_mask_goal = h_team_shots.isGoal == True

        a_team_shots = total_shots.loc[total_shots['teamId'] == a_teamId].reset_index(drop=True)
        a_mask_goal = a_team_shots.isGoal == True

        home_op_shots = h_team_shots[h_team_shots['isRegularPlay'] == 1]
        home_non_op_shots = h_team_shots[h_team_shots['isRegularPlay'] == 0]
        home_pk_shots = h_team_shots[h_team_shots['isPenalty'] == 1]

        away_op_shots = a_team_shots[a_team_shots['isRegularPlay'] == 1]
        away_non_op_shots = a_team_shots[a_team_shots['isRegularPlay'] == 0]
        away_pk_shots = a_team_shots[a_team_shots['isPenalty'] == 1]

        home_goals = len(h_team_shots[h_mask_goal])
        home_shots = len(h_team_shots[~h_mask_goal]) + home_goals
        home_sot = len(h_team_shots[h_team_shots['shotOnTarget'] == True])
        home_xg = round(h_team_shots.xG.sum(),2)
        home_xg_op = round(home_op_shots.xG.sum(), 2)
        home_xg_sp = round(home_non_op_shots.xG.sum() - home_pk_shots.xG.sum(), 2)
        home_xg_pk = round(home_pk_shots.xG.sum(), 2)
        home_xg_shot = round(home_xg / home_shots,2)

        away_goals = len(a_team_shots[a_mask_goal])
        away_shots = len(a_team_shots[~a_mask_goal]) + away_goals
        away_sot = len(a_team_shots[a_team_shots['shotOnTarget'] == True])
        away_xg = round(a_team_shots.xG.sum(),2)
        away_xg_op = round(away_op_shots.xG.sum(), 2)
        away_xg_sp = round(away_non_op_shots.xG.sum() - away_pk_shots.xG.sum(), 2)
        away_xg_pk = round(away_pk_shots.xG.sum(), 2)
        away_xg_shot = round(away_xg / away_shots,2)
        
        metrics = ['Shots', 'ShotsConceded', 'xG', 'xGperShot', 'xGA', 'xGperShotA' ]
        
        home_shots_df = pd.DataFrame({'team': [home_team_name]*6,
                                     'teamId': [home_team_id]*6,
                                     'matchId': [matchId]*6,
                                     'metric': ['Shots', 'ShotsConceded', 'xG', 'xGperShot', 'xGA', 'xGperShotA' ],
                                     'stat': [home_shots, away_shots, home_xg, home_xg_shot, away_xg, away_xg_shot]})

        away_shots_df = pd.DataFrame({'team': [away_team_name]*6,
                                     'teamId': [away_team_id]*6,
                                     'matchId': [matchId]*6,
                                     'metric': ['Shots', 'ShotsConceded', 'xG', 'xGperShot', 'xGA', 'xGperShotA' ],
                                     'stat': [away_shots, home_shots, away_xg, away_xg_shot, home_xg, home_xg_shot]})
        
        match_shots = pd.concat([home_shots_df, away_shots_df], axis=0).reset_index(drop=True)

        return match_shots
    
    def ppda():
        
        "Calculates passes allowed per defensive action in final 60% of the pitch"
        "Defensive actions include tackles, clearances, interceptions, and clearances"
        
        events_60 = events_df[events_df['x'] > 60]
        
        defensive_actions = ['Tackle', 'Foul', 'Clearance', 'Interception']
        def_actions = events_60[events_60['type'].isin(defensive_actions)]
        h_def_actions = def_actions[def_actions['teamId'] == h_teamId]
        a_def_actions = def_actions[def_actions['teamId'] == a_teamId]
        
        passes = events_60[events_60['type'] == 'Pass']
        h_passes_ag = passes[passes['teamId'] == a_teamId]
        a_passes_ag = passes[passes['teamId'] == h_teamId]
        
        home_ppda = round((len(h_passes_ag) / len(h_def_actions)),2)
        away_ppda = round((len(a_passes_ag) / len(a_def_actions)),2)
        
        home_ppda_df = pd.DataFrame({'team': [home_team_name],
                                     'teamId': [home_team_id],
                                     'matchId': [matchId],
                                     'metric': 'PPDA',
                                     'stat': [home_ppda]})

        away_ppda_df = pd.DataFrame({'team': [away_team_name],
                                     'teamId': [away_team_id],
                                     'matchId': [matchId],
                                     'metric': 'PPDA',
                                     'stat': [away_ppda]})
        
        match_ppda = pd.concat([home_ppda_df, away_ppda_df], axis=0).reset_index(drop=True)

        return match_ppda
    
    def def_line_height():
        
        "Calculates the mean distance from own goal where a team attempts a defensive action"
        
        defensive_actions = ['Foul', 'Tackle', 'Interception']
        defensive_events = events_df[events_df['type'].isin(defensive_actions)]
        
        h_def_actions = defensive_events[defensive_events['teamId'] == h_teamId]
        a_def_actions = defensive_events[defensive_events['teamId'] == a_teamId]
        
        h_dist_from_goal = round(h_def_actions.x.mean(),2)
        a_dist_from_goal = round(a_def_actions.x.mean(),2)
        
        home_line_df = pd.DataFrame({'team': [home_team_name],
                                     'teamId': [home_team_id],
                                     'matchId': [matchId],
                                     'metric': 'DefLineHeight',
                                     'stat': [h_dist_from_goal]})

        away_line_df = pd.DataFrame({'team': [away_team_name],
                                     'teamId': [away_team_id],
                                     'matchId': [matchId],
                                     'metric': 'DefLineHeight',
                                     'stat': [a_dist_from_goal]})
        
        match_def_height = pd.concat([home_line_df, away_line_df], axis=0).reset_index(drop=True)

        return match_def_height
    
    def team_xT():
        
        "Calculates cumulative xT for each team per match"
        
        total_passes_carries = events_df[events_df['type'].isin(['Pass', 'Carry'])]
        total_passes_carries = total_passes_carries[total_passes_carries['outcomeType'] == 'Successful']

        # home
        h_total_passes_carries = total_passes_carries[total_passes_carries['teamId'] == h_teamId]
        h_total_passes_carries = xThreat(h_total_passes_carries)

        # away
        a_total_passes_carries = total_passes_carries[total_passes_carries['teamId'] == a_teamId]
        a_total_passes_carries = xThreat(a_total_passes_carries)

        home_cum_xT = round(h_total_passes_carries.xT.sum(),2)

        away_cum_xT = round(a_total_passes_carries.xT.sum(),2)
        
        home_xT = pd.DataFrame({'team': [home_team_name]*2,
                                'teamId': [home_team_id]*2,
                                'matchId': [matchId]*2,
                                'metric': ['xT', 'xTA'],
                                'stat': [home_cum_xT, away_cum_xT]})

        away_xT = pd.DataFrame({'team': [away_team_name]*2,
                                'teamId': [away_team_id]*2,
                                'matchId': [matchId]*2,
                                'metric': ['xT', 'xTA'],
                                'stat': [away_cum_xT, home_cum_xT]})
        
        match_xT = pd.concat([home_xT, away_xT], axis=0).reset_index(drop=True)

        return match_xT
    
    match_tilt = field_tilt()
    match_poss = possession()
    match_shots = shots()
    match_ppda = ppda()
    match_height = def_line_height()
    match_xT = team_xT()
    
    match_stats = pd.concat([match_tilt, match_poss, match_shots, match_ppda, match_height, match_xT], axis=0).reset_index(drop=True)
    
    match_stats.to_csv(f"Data/liga-profesional/match-stats/{home_team_name}/{home_team_name}-{matchId}.csv")
    match_stats.to_csv(f"Data/liga-profesional/match-stats/{away_team_name}/{away_team_name}-{matchId}.csv")
    
    return match_stats

def get_match_stats(match_data):

    teams = ['Lanus', 'Huracan', 'Defensa y Justicia', 'Talleres', 'San Lorenzo',
             'River Plate', 'Boca Juniors', 'Tigre', 'Racing Club', 'Godoy Cruz',
             'Velez Sarsfield', 'Sarmiento', 'Argentinos Juniors', 'Banfield', 'Instituto',
             'Gimnasia LP', 'Rosario Central', 'Barracas Central', 'Estudiantes', "Newell's Old Boys",
             'Belgrano', 'Colon', 'Union', 'Club Atletico Platense', 'Independiente', 'Arsenal Sarandi',
             'Central Cordoba de Santiago', 'Atletico Tucuman']

    # file path and title positioning conditions
    if match_data['region'] == 'Argentina':
        comp = 'liga-profesional'
    elif match_data['region'] == 'Spain':
        comp = 'la-liga'
    elif match_data['region'] == 'England':
        comp = 'premier-league'
    else:
        comp = 'serie-a'

    matchStats = []

    for team in teams:

        # folder to loop through
        teamName = team

        # Set the directory path for the folder of CSV files
        folder_path = f"Data/{comp}/match-stats/{teamName}"

        # Loop through each file in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith(".csv"):
                # Read the CSV file into a dataframe using pd.read_csv
                filepath = os.path.join(folder_path, filename)
                df = pd.read_csv(filepath).reset_index(drop=True)
                # Append the dataframe to the list
                matchStats.append(df)

    # loop through each DataFrame in the list and reorganize using pd.pivot_table
    for i, df in enumerate(matchStats):

        # create pivot table where attribute values become columns and value is column value
        pivot_table = pd.pivot_table(df, index=['team', 'teamId', 'matchId'], columns='metric', values='stat')

        matchStats[i] = pivot_table

    allMatchStats = pd.concat(matchStats, axis=0).reset_index()
    allMatchStats = allMatchStats.drop_duplicates().reset_index(drop=True)
    
    toDateLeagueStats = allMatchStats.groupby(['team', 'teamId']).agg({'DefLineHeight': ['mean'],
                                                                   'FieldTilt': ['mean'],
                                                                   'PPDA': ['mean'],
                                                                   'Possession': ['mean'],
                                                                   'Shots': ['sum'],
                                                                   'ShotsConceded': ['sum'],
                                                                   'xG': ['sum'],
                                                                   'xGA': ['sum'],
                                                                   'xT': ['sum'],
                                                                   'xTA': ['sum']})

    toDateLeagueStats.columns = ['DefLineHeight', 'FieldTilt', 'PPDA', 'Possession',
                                 'Shots', 'ShotsConceded', 'xG', 'xGA', 'xT', 'xTA']


    toDateLeagueStats = toDateLeagueStats.reset_index()
    
    toDateLeagueStats['xGD'] = toDateLeagueStats['xG'] - toDateLeagueStats['xGA']
    toDateLeagueStats['xTD'] = toDateLeagueStats['xT'] - toDateLeagueStats['xTA']
    toDateLeagueStats['xGperShot'] = toDateLeagueStats['xG'] / toDateLeagueStats['Shots']
    toDateLeagueStats['xGperShotAg'] = toDateLeagueStats['xGA'] / toDateLeagueStats['ShotsConceded']
    
    today = datetime.date.today()
    today_str = today.strftime("%Y-%m-%d")
    today_str

    toDateLeagueStats.to_csv(f"Data/{comp}/league-stats/{match_data['league']}-{match_data['season']}-{today_str}.csv")


def plot_team_styles(match_data): 

    # get match_stats by pulling today's dataframe
    get_match_stats(match_data)
    
     # file path and title positioning conditions
    if match_data['region'] == 'Argentina':
        comp = 'liga-profesional'
    elif match_data['region'] == 'Spain':
        comp = 'la-liga'
    elif match_data['region'] == 'England':
        comp = 'premier-league'
    else:
        comp = 'serie-a'
        
    today = datetime.date.today()
    today_str = today.strftime("%Y-%m-%d")
    today_str

    # set the fontproperties to use
    Rubik = FontManager('https://github.com/googlefonts/rubik/blob/main/old/version-2/fonts/ttf/Rubik-Bold.ttf')
    Rubik_italic = FontManager('https://github.com/googlefonts/rubik/blob/main/old/version-2/fonts/ttf/Rubik-BoldItalic.ttf')
    #Rubkik_bold = FontManager('https://github.com/googlefonts/rubik/blob/main/fonts/ttf/Rubik-ExtraBold.ttf?raw=true')

    df = pd.read_csv(f"Data/{comp}/league-stats/{match_data['league']}-{match_data['season']}-{today_str}.csv")
    df = df.loc[:, ~df.columns.str.match('Unnamed')]

    df = df.drop(columns=['xGD', 'xTD'])
    
    # round values of all but two columns to 2 digits
    cols_to_round = [col for col in df.columns if col not in ['team', 'teamId']]
    df[cols_to_round] = df[cols_to_round].round(2)

    # rank all columns except column team, teamID, xGA, PPDA, xTA, xGpSA, Shots Ag
    ranked = df.loc[:, ~df.columns.isin(['team', 'teamId', 'PPDA', 'ShotsConceded', 'xGA', 'xTA', 'xGperShotAg'])].rank(axis=0, ascending=False)

    ranked_reversed = df.loc[:, df.columns.isin(['PPDA', 'ShotsConceded', 'xGA', 'xTA', 'xGperShotAg'])].rank(axis=0, ascending=True)

    # combine the ranked columns with the original dataframe
    ranked = pd.concat([df[['team', 'teamId']], ranked, ranked_reversed], axis=1)

    # convert all columns except column A and column D to integers
    converted = ranked.loc[:, ~ranked.columns.isin(['team', 'teamId'])].applymap(int)

    # combine the converted columns with the original dataframe
    ranked = pd.concat([ranked[['team', 'teamId']], converted], axis=1)
    ranked = ranked[['team', 'teamId', 'DefLineHeight', 'FieldTilt', 'PPDA', 'Possession', 'Shots', 'ShotsConceded', 'xG', 'xGA', 'xT', 'xTA', 'xGperShot', 'xGperShotAg']]

    ranked = pd.merge(df, ranked, on='team', how='outer')
    
    for index, row in ranked.iterrows():

        # get each row of the dataframe and turn into list
        row_list = row.tolist()

        # parameter list
        params = ["Defensive\nLine Height" + f"\n{row_list[2]}m",
                  "Field Tilt" + f"\n{row_list[3]}%",
                  "PPDA" + f"\n{row_list[4]}",
                  "Avg Possession" + f"\n{row_list[5]}%",
                  "Shots" + f"\n{int(row_list[6])}", 
                  "Shots Conceded" + f"\n{int(row_list[7])}", 
                  "xG" + f"\n{row_list[8]}", 
                  "xGA" + f"\n{row_list[9]}", 
                  "xT" + f"\n{row_list[10]}", 
                  "xTA" + f"\n{row_list[11]}", 
                  "xG per Shot" + f"\n{row_list[12]}",
                  "xG per\nShot Against" + f"\n{row_list[13]}"]
        
        row_ranked_list = row.tolist()

        # split by team name and values
        teamName = row_ranked_list[0]
        teamId = row_ranked_list[1]

        ranked_values = row_ranked_list[15:]
        values = row_list[2:14]

        # color for the slices and text
        slice_colors = ["#D70232"] * 12
        text_colors = ["#F2F2F2"] * 12

        min_range = [30]*12
        max_range = [0,0,0,0,0,0,0,0,0,0,0,0]

        # instantiate PyPizza class
        baker = PyPizza(
            params=params,                  # list of parameters
            min_range=min_range,
            max_range=max_range,
            background_color="#1d2849",     # background color
            straight_line_color="#000000",  # color for straight lines
            straight_line_lw=1,             # linewidth for straight lines
            last_circle_color="#000000",    # color for last line
            last_circle_lw=1,               # linewidth of last circle
            other_circle_lw=1,              # linewidth for other circles
            inner_circle_size=20            # size of inner circle
        )

        # plot pizza
        fig, ax = baker.make_pizza(
            ranked_values,                          # list of values
            figsize=(8, 8.5),                # adjust the figsize according to your need
            color_blank_space="same",        # use the same color to fill blank space
            slice_colors=slice_colors,       # color for individual slices
            value_colors=text_colors,        # color for the value-text
            value_bck_colors=slice_colors,   # color for the blank spaces
            blank_alpha=0.4,                 # alpha for blank-space colors
            kwargs_slices=dict(
                edgecolor="#000000", zorder=2, linewidth=1, 
            ),                               # values to be used when plotting slices
            kwargs_params=dict(
                color="#F2F2F2", fontsize=11,
                fontproperties=Rubik.prop, va="center"
            ),                               # values to be used when adding parameter labels
            kwargs_values=dict(
                color="#F2F2F2", fontsize=11,
                fontproperties=Rubik.prop, zorder=3,
                bbox=dict(
                    edgecolor="#000000", facecolor="cornflowerblue",
                    boxstyle="round,pad=0.2", lw=1
                )
            )                                # values to be used when adding parameter-values labels
        )

        # add title
        fig.text(
            0.515, 0.975, f"{teamName}", size=16,
            ha="center", fontproperties=Rubik.prop, color="#F2F2F2"
        )

        # add subtitle
        fig.text(
            0.515, 0.945,
            f"Team metrics and rankings in the {match_data['season']} {match_data['league']}",
            size=13,
            ha="center", fontproperties=Rubik.prop, color="#F2F2F2"
        )

        # add credits
        CREDIT_1 = "Data via Opta | viz by @egudi_analysis"
        CREDIT_2 = "inspired by: @Worville, @FootballSlices, @somazerofc, @BeGriffis, & @Soumyaj15209314"

        fig.text(
            0.99, 0.02, f"{CREDIT_1}\n{CREDIT_2}", size=9,
            fontproperties=Rubik.prop, color="#F2F2F2",
            ha="right"
        )

        # team
        path = f'Logos/{comp}/{teamName}.png'
        ax_team = fig.add_axes([.45,.4425,0.12,0.12])
        ax_team.axis('off')
        im = plt.imread(path)
        ax_team.imshow(im);

        # league logo
        path = f'Logos/{comp}/{comp}.png'
        ax_team = fig.add_axes([0.86,0.88,0.12,0.12])
        ax_team.axis('off')
        im = plt.imread(path)
        ax_team.imshow(im);

        # save figure
        fig.savefig(f'Output/{comp}/team-styles/{today_str}/{teamName}-style-{today_str}', dpi=None, bbox_inches="tight")

def full_season_shot_heatmaps(events_df, matches_data, fig, axes):

    "Function that plots shot and xT maps comparing where they take shots and create chances"
    "Function takes events_df"
    
    # load the models from disk to get xG values
    loaded_model_op = pickle.load(open('Models/expected_goals_model_lr.sav', 'rb'))
    loaded_model_non_op = pickle.load(open('Models/expected_goals_model_lr_v2.sav', 'rb'))

    # split up df into shots data
    shots_df = events_df[events_df['isShot'] == True]
    shots_df = shots_df[shots_df['period'] != 'PenaltyShootout']
    # prepare data to calc xG
    data_preparation(shots_df)
    shots_df['distance_to_goal'] = np.sqrt(((100 - shots_df['x'])**2) + ((shots_df['y'] - (100/2))**2) )
    shots_df['distance_to_center'] = abs(shots_df['y'] - 100/2)
    shots_df['angle'] = np.absolute(np.degrees(np.arctan((abs((100/2) - shots_df['y'])) / (100 - shots_df['x']))))
    shots_df['isFoot'] = np.where(((shots_df['isLeftFooted'] == 1) | (shots_df['isRightFooted'] == 1)) &
                                              (shots_df['isHead'] == 0)
                                  , 1,0
                                 )

    features_op = ['distance_to_goal',
           #'distance_to_centerM',    # commented out as elected to go with 'angle' instead
            'angle',
            'isFoot',
            'isHead'
           ]

    features_non_op = ['distance_to_goal',
           #'distance_to_centerM',    # commented out as elected to go with 'angle' instead
            'angle',
            'isFoot',
            'isHead',
            'isDirectFree',
            'isSetPiece',
            'isFromCorner'
           ]

    # split model to only open play shots to apply open play model
    # calc xG for open play shots only
    total_shots_op = shots_df.loc[shots_df['isRegularPlay'] == 1]
    shots_df_op = total_shots_op[features_op]
    xG = loaded_model_op.predict_proba(shots_df_op)[:,1]
    shots_df_op['xG'] = xG
    xg = shots_df_op['xG']
    total_shots_op = pd.merge(total_shots_op, xg, left_index=True, right_index=True)

    # split dataframe to only set pieces to apply set piece model
    # calc xG for set pieces only (not including penalties)
    total_shots_non_op = shots_df.loc[shots_df['isRegularPlay'] == 0]
    total_shots_non_op = total_shots_non_op.loc[total_shots_non_op['isPenalty'] == 0]
    shots_df_non_op = total_shots_non_op[features_non_op]
    xG = loaded_model_non_op.predict_proba(shots_df_non_op)[:,1]
    shots_df_non_op['xG'] = xG
    xg = shots_df_non_op['xG']
    total_shots_non_op = pd.merge(total_shots_non_op, xg, left_index=True, right_index=True)    

    # split dataframe to only penalties and set xG for penalties to 0.79
    total_shots_pk = shots_df.loc[shots_df['isPenalty'] == 1]
    total_shots_pk['xG'] = 0.79

    # combine all three dataframes 
    shots_df_adv = pd.concat([total_shots_op, total_shots_non_op, total_shots_pk], axis=0).reset_index(drop=True)

        # group the original dataframe by the unique values in matchId column
    grouped = events_df.groupby('matchId')

    # create an empty list to store the individual dataframes
    matches_list = []

    # iterate over each group, create a new dataframe and append it to the list
    for matchId, group_df in grouped:
        matches_list.append(group_df)

    match_points = []

    for match_df in matches_list:

        ids = match_df.teamId.unique()
        teamId_home = ids[0]
        teamId_away = ids[1]

        match_df['teamId'] = np.where( (match_df['isOwnGoal'] == True) & (match_df['teamId'] == teamId_home), 
                                        teamId_away,
                                        np.where( (match_df['isOwnGoal'] == True) & (match_df['teamId'] == teamId_away), teamId_home, 
                                            match_df['teamId']
                                            )
                                        )

        #match_df['isGoal'] = match_df['isGoal'].astype(int)

        score = match_df.groupby('teamId').agg({'isGoal': ['sum']})
        score.columns = ['goals']
        score = score.reset_index()

        team1_goals = score['goals'][0]
        team2_goals = score['goals'][1]

        points = []

        if team1_goals > team2_goals:
            points.append(3)
            points.append(0)
        elif team1_goals < team2_goals:
            points.append(0)
            points.append(3)
        else:
            points.append(1)
            points.append(1)

        score['points'] = points
        score['goals_ag'] = pd.Series(score['goals'].iloc[::-1].values)

        match_points.append(score)

    # concat the above into a league table to use later on
    league_table = pd.concat(match_points)
    league_table = league_table.groupby('teamId').agg({'points': ['sum'],
                                                      'goals': ['sum'],
                                                      'goals_ag': ['sum']})

    league_table.columns = ['points', 'goals_scored', 'goals_conceded']
    league_table = league_table.reset_index()
    league_table = league_table.sort_values(by='points', ascending=False).reset_index(drop=True)
    league_table['position'] = league_table['points'].rank(ascending=False).astype(int)

    # create an empty list and loop through match data df to get each team name and append to list
    teamNames = []

    for match in matches_data:

        teamHomeName = match['home']['name']
        teamAwayName = match['away']['name']

        teamNames.append(teamHomeName)
        teamNames.append(teamAwayName)

    # create an empty list and loop through match data df to get each team id and append to list
    teamIds = []

    for match in matches_data:

        teamHomeId = match['home']['teamId']
        teamAwayId = match['away']['teamId']

        teamIds.append(teamHomeId)
        teamIds.append(teamAwayId)

    # create a df of each name and their respective id
    teams = pd.DataFrame({'teamName': teamNames,
                          'teamId': teamIds})

    # drop duplicates
    teams = teams.drop_duplicates().reset_index(drop=True)

    # create a list of colors for each team
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', 
              '#7f7f7f', '#bcbd22', '#17becf', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#1f77b4', 
              '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    teams['teamColors'] = colors

    # merge with the league table
    teams = pd.merge(league_table, teams, on='teamId', how='outer')
    teams['gd'] = teams['goals_scored'] - teams['goals_conceded']
    teams = teams.sort_values(by=['points', 'gd', 'goals_scored'], ascending=[False, False, False])
    teams = teams[['position', 'teamName', 'teamId', 'points', 'goals_scored', 'goals_conceded', 'gd', 'teamColors']].reset_index(drop=True)

    # We need to invert our coordinates
    shots_df_adv.rename(columns = {"x":"y", "y":"x"}, inplace = True)

    # We define the cuts for our data (same as our pitch divisions)
    # Only difference is we need to add the edges

    y_bins = [100] + [100 - 5.75*x for x in range(1,10)] + [50]
    x_bins = [100] + [100 - 10*x for x in range(1,10)] + [0]

    x_bins.sort()
    y_bins.sort()

    #Assign a team to each team id in the events data
    team_info_dict = dict(zip(teams['teamId'], teams['teamName']))

    for index, row in shots_df_adv.iterrows():
        team_id = row['teamId']
        team_name = team_info_dict.get(team_id)
        shots_df_adv.at[index, 'teamName'] = team_name

    shots_df_adv["bins_x"] = pd.cut(shots_df_adv["x"], bins = x_bins)

    shots_df_adv["bins_y"] = pd.cut(shots_df_adv["y"], bins = y_bins)

    #Group and sum xG by side and location
    df_teams = (
        shots_df_adv.groupby(
            ["bins_x", "bins_y", "teamId", 'teamName'], 
            observed = True
        )["xG"].sum()
        .reset_index()
    )

    # And we sort it based on the bins_y and bins_x columns
    df_teams = (
        df_teams.
        sort_values(by = ["bins_y", "bins_x"]).
        reset_index(drop = True)
    )

    # now we can plot the data

    # create league identifiers
    league = f"{matches_data[0]['region']} {matches_data[0]['league']} {matches_data[0]['season']}"

    # todays date for saving
    date = f'{datetime.datetime.now().strftime("%m/%d/%Y")}'

    # file path and title positioning conditions
    if matches_data[0]['region'] == 'Argentina':
        comp = 'liga-profesional'
    elif matches_data[0]['region'] == 'Spain':
        comp = 'la-liga'
    elif matches_data[0]['region'] == 'England':
        comp = 'premier-league'
    else:
        comp = 'serie-a'

    title = f'{league} - Where do teams generate threat and take shots from?'
    subtitle = f'Shaded areas represent threat generated from passes and carries. Negative threat events excluded.'
    subtitle2 = f'Shots exclude penalty kicks. Dotted curve represents median shot distance\nCorrect as of {date} | Data via Opta. Viz by @egudi_analysis.'
    footnote = f'Expected goals (xG) trained on ~10k shots from the 2021/2022 EPL season.'
    footnote2 = f'Expected threat (xT) model by Karun Singh.'

    fig.text(0.125,0.925, title, fontsize=35, fontproperties=Rubik.prop, color='w')
    fig.text(0.125, 0.895, f'{subtitle}\n{subtitle2}', fontsize=25, fontproperties=Rubik.prop, color='w')
    fig.text(0.125, 0.115, f'{footnote} {footnote2}', fontsize=25, fontproperties=Rubik.prop, color='w')

    # league logo
    path = f'Logos/{comp}/{comp}.png'
    ax_team = fig.add_axes([0.83,0.875,0.075,0.075])
    ax_team.axis('off')
    im = plt.imread(path)
    ax_team.imshow(im);

    pitch = VerticalPitch(pitch_type='opta', pitch_color='#1d2849', line_color='w',
                          half=True, pad_top=28)

    teamIds = teams['teamId']
    teamNames = teams['teamName']

    def modify_team_name(name):
        if name == "Central Cordoba de Santiago":
            return "Central Cordoba"
        elif name == "Argentinos Juniors":
            return "Argentinos Jrs"
        elif name == "Club Atletico Platense":
            return "Club Atl. Platense"
        else:
            return name
        
    total_passes_carries = events_df[events_df['type'].isin(['Pass', 'Carry'])]
    total_passes_carries = total_passes_carries[total_passes_carries['outcomeType'] == 'Successful']
    total_passes_carries['isOpenPlay'] = np.where( (total_passes_carries['passFreekick'] == False) &
                                                ((total_passes_carries['passCorner'] == False)
                                                ) 
                                               , 1, 0
                                               )

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', 
              '#7f7f7f', '#bcbd22', '#17becf', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#1f77b4', 
              '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    total_passes_carries = xThreat(total_passes_carries)

    for i, ax in enumerate(axes):
        
        threat = total_passes_carries[total_passes_carries['teamId'] == teamIds[i]]
        vmax = np.max(threat['xT'].max())
        vmin = np.min(threat['xT'].min())
        norm = Normalize(vmin=vmin, vmax=vmax)
        midpoint = threat.xT.median()
        shot_data = shots_df_adv[shots_df_adv['teamId'] == teamIds[i]]
        nineties = len(events_df.matchId.unique())/14
        xG = round(shot_data.xG.sum(),2)
        xT = round(threat.xT.sum(),2)
        shots = len(shot_data)
        goals = len(shot_data[shot_data['isGoal'] == True])
        xG_shot = xG/shots
        
        logo_paths = f'Logos/{comp}/{teamNames[i]}.png'
        
        customcmap = LinearSegmentedColormap.from_list('custom cmap',['#1d2849', '#db2824', '#67000d'])
        
        pitch.draw(ax=ax, tight_layout=True, constrained_layout=True)

        bin_statistic = pitch.bin_statistic(threat.x, threat.y, threat.xT, statistic='sum', bins=(24, 16))
        bin_statistic['statistic'] = gaussian_filter(bin_statistic['statistic'], 1)
        pcm1 = pitch.heatmap(bin_statistic, ax=ax, cmap=customcmap, edgecolors='#1a223d', zorder=0, alpha=1,
                             norm=norm, vmin=0, vmax=vmax)
        
        pitch.scatter(shot_data.y, shot_data.x, edgecolors='w', linewidths=2.5, c=colors[i],
                      s=(shot_data.xG * 1200) + 100, zorder=2, ax=ax, alpha=0.4)
        
        #draw a circle
        shot_data = shot_data[shot_data['isPenalty'] != 1]
        shot_dist_to_goal = shot_data.distance_to_goal.mean()
        u=50.     #x-position of the center 
        v=100    #y-position of the center 
        b=shot_dist_to_goal    #radius on the y-axis 
        a=shot_dist_to_goal*1.5    #radius on the x-axis 

        t = np.linspace(0, -4*pi/4, 100)
        ax.plot( u+a*np.cos(t) , v+b*np.sin(t), linestyle='--', color='darkgray', zorder=5, linewidth=4, alpha=0.85)
        
        logo = plt.imread(logo_paths)  # read the logo image
        ax.imshow(logo, extent=[22, 2, 102, 118], interpolation='nearest', aspect="auto", zorder=10)  # assign the logo to the axis

        teamNames2 = [modify_team_name(name) for name in teamNames]
        ax.text(100, 115, f'{teamNames2[i]}', ha='left', va='center', color='w', 
                fontsize=30, fontproperties=Rubik.prop)
        
        xG_text = f'xG: {(xG):.2f}'
        xT_text = f'xT90: {(xT/nineties):.2f}'
        ax.text(100, 102.25, f'{xG_text} ({shots} shots / {goals} goals)\nxG/shot: {xG_shot:.2f}\nxT: {xT:.2f}', fontsize=15,
                color='w', fontproperties=Rubik.prop)
        
        ax.text(88, 58, 'xG Value', ha='center', va='center', fontsize=15,
                color='white', fontproperties=Rubik.prop)
        
        pitch.scatter(52, 95.5, s=150, c='gray',
                      zorder=4, ax=ax, edgecolor='w', linewidths=2.5)
        pitch.scatter(52, 89, s=350, c='gray',
                      zorder=4, ax=ax, edgecolor='w', linewidths=2.5)
        pitch.scatter(52, 80, s=500, c='gray', 
                      zorder=4, ax=ax, edgecolor='w', linewidths=2.5)
        
    plt.subplots_adjust(wspace=0.1)

def full_season_shot_heatmaps_ag(events_df, matches_data, fig, axes):

    "Function that plots shot and xT maps comparing where they concede shots and chances"
    "Function takes events_df"
    
    # load the models from disk to get xG values
    loaded_model_op = pickle.load(open('Models/expected_goals_model_lr.sav', 'rb'))
    loaded_model_non_op = pickle.load(open('Models/expected_goals_model_lr_v2.sav', 'rb'))

    # split up df into shots data
    shots_df = events_df[events_df['isShot'] == True]
    shots_df = shots_df[shots_df['period'] != 'PenaltyShootout']
    # prepare data to calc xG
    data_preparation(shots_df)
    shots_df['distance_to_goal'] = np.sqrt(((100 - shots_df['x'])**2) + ((shots_df['y'] - (100/2))**2) )
    shots_df['distance_to_center'] = abs(shots_df['y'] - 100/2)
    shots_df['angle'] = np.absolute(np.degrees(np.arctan((abs((100/2) - shots_df['y'])) / (100 - shots_df['x']))))
    shots_df['isFoot'] = np.where(((shots_df['isLeftFooted'] == 1) | (shots_df['isRightFooted'] == 1)) &
                                              (shots_df['isHead'] == 0)
                                  , 1,0
                                 )

    features_op = ['distance_to_goal',
           #'distance_to_centerM',    # commented out as elected to go with 'angle' instead
            'angle',
            'isFoot',
            'isHead'
           ]

    features_non_op = ['distance_to_goal',
           #'distance_to_centerM',    # commented out as elected to go with 'angle' instead
            'angle',
            'isFoot',
            'isHead',
            'isDirectFree',
            'isSetPiece',
            'isFromCorner'
           ]

    # split model to only open play shots to apply open play model
    # calc xG for open play shots only
    total_shots_op = shots_df.loc[shots_df['isRegularPlay'] == 1]
    shots_df_op = total_shots_op[features_op]
    xG = loaded_model_op.predict_proba(shots_df_op)[:,1]
    shots_df_op['xG'] = xG
    xg = shots_df_op['xG']
    total_shots_op = pd.merge(total_shots_op, xg, left_index=True, right_index=True)

    # split dataframe to only set pieces to apply set piece model
    # calc xG for set pieces only (not including penalties)
    total_shots_non_op = shots_df.loc[shots_df['isRegularPlay'] == 0]
    total_shots_non_op = total_shots_non_op.loc[total_shots_non_op['isPenalty'] == 0]
    shots_df_non_op = total_shots_non_op[features_non_op]
    xG = loaded_model_non_op.predict_proba(shots_df_non_op)[:,1]
    shots_df_non_op['xG'] = xG
    xg = shots_df_non_op['xG']
    total_shots_non_op = pd.merge(total_shots_non_op, xg, left_index=True, right_index=True)    

    # split dataframe to only penalties and set xG for penalties to 0.79
    total_shots_pk = shots_df.loc[shots_df['isPenalty'] == 1]
    total_shots_pk['xG'] = 0.79

    # combine all three dataframes 
    shots_df_adv = pd.concat([total_shots_op, total_shots_non_op, total_shots_pk], axis=0).reset_index(drop=True)

    # group the original dataframe by the unique values in matchId column
    grouped = events_df.groupby('matchId')

    # create an empty list to store the individual dataframes
    matches_list = []

    # iterate over each group, create a new dataframe and append it to the list
    for matchId, group_df in grouped:
        matches_list.append(group_df)
        
    match_points = []

    for match_df in matches_list:
        
        ids = match_df.teamId.unique()
        teamId_home = ids[0]
        teamId_away = ids[1]
        
        match_df['teamId'] = np.where( (match_df['isOwnGoal'] == True) & (match_df['teamId'] == teamId_home), 
                                        teamId_away,
                                        np.where( (match_df['isOwnGoal'] == True) & (match_df['teamId'] == teamId_away), teamId_home, 
                                            match_df['teamId']
                                            )
                                        )
        
        #match_df['isGoal'] = match_df['isGoal'].astype(int)
        
        score = match_df.groupby('teamId').agg({'isGoal': ['sum']})
        score.columns = ['goals']
        score = score.reset_index()
        
        team1_goals = score['goals'][0]
        team2_goals = score['goals'][1]
        
        points = []
        
        if team1_goals > team2_goals:
            points.append(3)
            points.append(0)
        elif team1_goals < team2_goals:
            points.append(0)
            points.append(3)
        else:
            points.append(1)
            points.append(1)
            
        score['points'] = points

        match_points.append(score)

    # concat the above into a league table to use later on
    league_table = pd.concat(match_points)
    league_table = league_table.groupby('teamId').agg({'points': ['sum']})
    league_table.columns = ['points']
    league_table = league_table.reset_index()
    league_table = league_table.sort_values(by='points', ascending=False).reset_index(drop=True)
    league_table['position'] = league_table['points'].rank(ascending=False).astype(int)

    # create an empty list and loop through match data df to get each team name and append to list
    teamNames = []

    for match in matches_data:
        
        teamHomeName = match['home']['name']
        teamAwayName = match['away']['name']
        
        teamNames.append(teamHomeName)
        teamNames.append(teamAwayName)
        
    # create an empty list and loop through match data df to get each team id and append to list
    teamIds = []

    for match in matches_data:
        
        teamHomeId = match['home']['teamId']
        teamAwayId = match['away']['teamId']
        
        teamIds.append(teamHomeId)
        teamIds.append(teamAwayId)
        
    # create a df of each name and their respective id
    teams = pd.DataFrame({'teamName': teamNames,
                          'teamId': teamIds})

    # drop duplicates
    teams = teams.drop_duplicates().reset_index(drop=True)

    # create a list of colors for each team
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', 
              '#7f7f7f', '#bcbd22', '#17becf', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#1f77b4', 
              '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    teams['teamColors'] = colors

    # merge with the league table
    teams = pd.merge(league_table, teams, on='teamId', how='outer')
    teams = teams[['position', 'teamName', 'teamId', 'points', 'teamColors']]

    # We need to invert our coordinates
    shots_df_adv.rename(columns = {"x":"y", "y":"x"}, inplace = True)

    # We define the cuts for our data (same as our pitch divisions)
    # Only difference is we need to add the edges

    y_bins = [100] + [100 - 5.75*x for x in range(1,10)] + [50]
    x_bins = [100] + [100 - 10*x for x in range(1,10)] + [0]

    x_bins.sort()
    y_bins.sort()

    #Assign a team to each team id in the events data
    team_info_dict = dict(zip(teams['teamId'], teams['teamName']))

    for index, row in shots_df_adv.iterrows():
        team_id = row['teamId']
        team_name = team_info_dict.get(team_id)
        shots_df_adv.at[index, 'teamName'] = team_name

    shots_df_adv["bins_x"] = pd.cut(shots_df_adv["x"], bins = x_bins)

    shots_df_adv["bins_y"] = pd.cut(shots_df_adv["y"], bins = y_bins)

    #Group and sum xG by side and location
    df_teams = (
        shots_df_adv.groupby(
            ["bins_x", "bins_y", "teamId", 'teamName'], 
            observed = True
        )["xG"].sum()
        .reset_index()
    )

    # And we sort it based on the bins_y and bins_x columns
    df_teams = (
        df_teams.
        sort_values(by = ["bins_y", "bins_x"]).
        reset_index(drop = True)
    )

    # now we can plot the data

    # create league identifiers
    league = f"{matches_data[0]['region']} {matches_data[0]['league']} {matches_data[0]['season']}"

    # todays date for saving
    date = f'{datetime.datetime.now().strftime("%m/%d/%Y")}'

    # file path and title positioning conditions
    if matches_data[0]['region'] == 'Argentina':
        comp = 'liga-profesional'
    elif matches_data[0]['region'] == 'Spain':
        comp = 'la-liga'
    elif matches_data[0]['region'] == 'England':
        comp = 'premier-league'
    else:
        comp = 'serie-a'

    title = f'{league} - Where do teams give up threat and concede shots from?'
    subtitle = f'Shaded areas represent open play passes and carries. Negative threat events excluded.'
    subtitle2 = f'Shots exclude penalty kicks. Dotted curve represents median shot distance\nCorrect as of {date} | Data via Opta. Viz by @egudi_analysis.'
    footnote = f'Expected goals (xG) trained on ~10k shots from the 2021/2022 EPL season.'
    footnote2 = f'Expected threat (xT) model by Karun Singh.'

    fig.text(0.125,0.925, title, fontsize=35, fontproperties=Rubik.prop, color='w')
    fig.text(0.125, 0.895, f'{subtitle}\n{subtitle2}', fontsize=25, fontproperties=Rubik.prop, color='w')
    fig.text(0.125, 0.115, f'{footnote} {footnote2}', fontsize=25, fontproperties=Rubik.prop, color='w')

    # league logo
    path = f'Logos/{comp}/{comp}.png'
    ax_team = fig.add_axes([0.83,0.875,0.075,0.075])
    ax_team.axis('off')
    im = plt.imread(path)
    ax_team.imshow(im);

    pitch = VerticalPitch(pitch_type='opta', pitch_color='#1d2849', line_color='w',
                          half=True, pad_top=28)

    teamIds = teams['teamId']
    teamNames = teams['teamName']

    def modify_team_name(name):
        if name == "Central Cordoba de Santiago":
            return "Central Cordoba"
        elif name == "Argentinos Juniors":
            return "Argentinos Jrs"
        elif name == "Club Atletico Platense":
            return "Club Atl. Platense"
        else:
            return name
        
    total_passes_carries = events_df[events_df['type'].isin(['Pass', 'Carry'])]
    total_passes_carries = total_passes_carries[total_passes_carries['outcomeType'] == 'Successful']
    total_passes_carries['isOpenPlay'] = np.where( (total_passes_carries['passFreekick'] == False) &
                                                ((total_passes_carries['passCorner'] == False)
                                                ) 
                                               , 1, 0
                                               )

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', 
              '#7f7f7f', '#bcbd22', '#17becf', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#1f77b4', 
              '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    total_passes_carries = xThreat(total_passes_carries)

    for i, ax in enumerate(axes):
        
        threat = total_passes_carries[total_passes_carries['teamId'] == teamIds[i]]
        vmax = np.max(threat['xT'].max())
        vmin = np.min(threat['xT'].min())
        norm = Normalize(vmin=vmin, vmax=vmax)
        midpoint = threat.xT.median()
        shot_data = shots_df_adv[shots_df_adv['teamId'] == teamIds[i]]
        nineties = len(events_df.matchId.unique())/14
        xG = round(shot_data.xG.sum(),2)
        xT = round(threat.xT.sum(),2)
        shots = len(shot_data)
        goals = len(shot_data[shot_data['isGoal'] == True])
        xG_shot = xG/shots
        
        logo_paths = f'Logos/{comp}/{teamNames[i]}.png'
        
        customcmap = LinearSegmentedColormap.from_list('custom cmap',['#1d2849', '#db2824', '#67000d'])
        
        pitch.draw(ax=ax, tight_layout=True, constrained_layout=True)

        bin_statistic = pitch.bin_statistic(threat.x, threat.y, threat.xT, statistic='sum', bins=(24, 16))
        bin_statistic['statistic'] = gaussian_filter(bin_statistic['statistic'], 1)
        pcm1 = pitch.heatmap(bin_statistic, ax=ax, cmap=customcmap, edgecolors='#1a223d', zorder=0, alpha=1,
                             norm=norm, vmin=0, vmax=vmax)
        
        pitch.scatter(shot_data.y, shot_data.x, edgecolors='w', linewidths=2.5, c=colors[i],
                      s=(shot_data.xG * 1200) + 100, zorder=2, ax=ax, alpha=0.4)
        
        #draw a circle
        shot_data = shot_data[shot_data['isPenalty'] != 1]
        shot_dist_to_goal = shot_data.distance_to_goal.mean()
        u=50.     #x-position of the center 
        v=100    #y-position of the center 
        b=shot_dist_to_goal    #radius on the y-axis 
        a=shot_dist_to_goal*1.5    #radius on the x-axis 

        t = np.linspace(0, -4*pi/4, 100)
        ax.plot( u+a*np.cos(t) , v+b*np.sin(t), linestyle='--', color='darkgray', zorder=5, linewidth=4, alpha=0.85)
        
        logo = plt.imread(logo_paths)  # read the logo image
        ax.imshow(logo, extent=[22, 2, 102, 118], interpolation='nearest', aspect="auto", zorder=10)  # assign the logo to the axis

        teamNames2 = [modify_team_name(name) for name in teamNames]
        ax.text(100, 115, f'{teamNames2[i]}', ha='left', va='center', color='w', 
                fontsize=30, fontproperties=Rubik.prop)
        
        xG_text = f'xGA: {(xG):.2f}'
        xT_text = f'xT90: {(xT/nineties):.2f}'
        ax.text(100, 102.25, f'{xG_text} ({shots} shots conceded / {goals} goals conceded.)\nxG/shot against: {xG_shot:.2f}\nxT against: {xT:.2f}', fontsize=15,
                color='w', fontproperties=Rubik.prop)
        
        ax.text(88, 58, 'xG Value', ha='center', va='center', fontsize=15,
                color='white', fontproperties=Rubik.prop)
        
        pitch.scatter(52, 95.5, s=150, c='gray',
                      zorder=4, ax=ax, edgecolor='w', linewidths=2.5)
        pitch.scatter(52, 89, s=350, c='gray',
                      zorder=4, ax=ax, edgecolor='w', linewidths=2.5)
        pitch.scatter(52, 80, s=500, c='gray', 
                      zorder=4, ax=ax, edgecolor='w', linewidths=2.5)
        
    plt.subplots_adjust(wspace=0.1)

def soc_pitch_divisions(ax, grids = False):
    '''
    This function returns a vertical football pitch
    divided in specific locations.

    Args:
        ax (obj): a matplotlib axes.
        grids (bool): should we draw the grid lines?
    '''

    # Notice the extra parameters passed to the object
    pitch = VerticalPitch(
        pitch_type = "opta",
        half = True,
        goal_type='box',
        linewidth = 2,
        line_color='white',
        pad_top=28
    )

    pitch.draw(ax = ax)

    # Where we'll draw the lines
    if grids:
        y_lines = [100 - 5.75*x for x in range(1,10)]
        x_lines = [100 - 10*x for x in range(1,10)]

        for i in x_lines:
            ax.plot(
                [i, i], [50, 100], 
                color = "lightgray", 
                ls = "--",
                lw = 0.75,
                zorder = -1
            )
        for j in y_lines:
            ax.plot(
                [100, 0], [j, j],
                color = "lightgray", 
                ls = "--",
                lw = 0.75,
                zorder = -1
            )

def soc_xG_plot(ax, grids, teamId, color, data):
    '''
    This plots our shot heat map based on the grids defined
    by the soc_pitch_divisions function.

    Args:
        ax (obj): a matplotlib Axes object.
        grids (bool): whether or not to plot the grids.
        teamId (int): the teamId of the side we wish to plot.
        color (str): color of the tiles
        data (pd.DataFrame): the data
    '''

    df = data.copy()
    df = data[data["teamId"] == teamId]
    total_xG = df["xG"].sum()

    df = (
        df
        .assign(xG_share = lambda x: x.xG/total_xG)
    )
    df = (
        df
        .assign(xG_scaled = lambda x: x.xG_share/x.xG_share.max())
    )

    soc_pitch_divisions(ax, grids = grids)

    counter = 0
    for X, Y in zip(df["bins_x"], df["bins_y"]):
        
        ax.fill_between(
            x = [X.left, X.right],
            y1 = Y.left,
            y2 = Y.right,
            color = color,
            alpha = df["xG_scaled"].iloc[counter],
            zorder = -1,
            lw = 0
        )

        if df['xG_share'].iloc[counter] > .02:
            text_ = ax.annotate(
                xy = (X.right - (X.right - X.left)/2, Y.right - (Y.right - Y.left)/2),
                text = f"{df['xG_share'].iloc[counter]:.0%}",
                ha = "center",
                va = "center",
                color = "white",
                size = 14,
                zorder = 3,
                fontproperties=Rubik.prop
            )

            text_.set_path_effects(
                [path_effects.Stroke(linewidth=1.5, foreground="white"), path_effects.Normal()]
            )

        counter += 1
    
    return ax


def progressive_pass(single_event, inplay=True, successful_only=True):
    """ Identify progressive pass from WhoScored-style pass event.
    Function to identify progressive passes. A pass is considered progressive if the distance between the
    starting point and the next touch is: (i) at least 30 meters closer to the opponent’s goal if the starting and
    finishing points are within a team’s own half, (ii) at least 15 meters closer to the opponent’s goal if the
    starting and finishing points are in different halves, (iii) at least 10 meters closer to the opponent’s goal if
    the starting and finishing points are in the opponent’s half. The function takes in a single event and returns a
    boolean (True = successful progressive pass.) This function is best used with the dataframe apply method.
    Args:
        single_event (pandas.Series): series corresponding to a single event (row) from WhoScored-style event dataframe.
        inplay (bool, optional): selection of whether to include 'in-play' events only. True by default.
        successful_only (bool, optional): selection of whether to only include successful passes. True by default
    Returns:
        bool: True = progressive pass, nan = non-progressive pass, unsuccessful progressive pass or not a pass
    """

    # Determine if event is pass
    if single_event['type'] == 'Pass':
        
        not_open_play = ['passFreekickAccurate', 'passFreekickInaccurate', 'keyPassFreekick', 'passFreekick', 'throwIn', 'keyPassThrowin', 'passCorner', 'passCornerInaccurate']

        # Check success (if successful_only = True)
        if successful_only:
            check_success = single_event['outcomeType'] == 'Successful'
        else:
            check_success = True

        # Check pass made in-play (if inplay = True)
        if inplay:
            check_inplay = not any(item in single_event['satisfiedEventsTypes'] for item in not_open_play)
        else:
            check_inplay = True

        # Determine pass start and end position in yards (assuming standard pitch), and determine whether progressive
        x_startpos = single_event['x']
        y_startpos = single_event['y']
        x_endpos = single_event['endX']
        y_endpos = single_event['endY']
        delta_goal_dist = (np.sqrt((x_startpos) ** 2 + (y_startpos) ** 2) -
                           np.sqrt((x_endpos) ** 2 + (y_endpos) ** 2))

        # At least 30m closer to the opponent’s goal if the starting and finishing points are within a team’s own half
        if (check_success and check_inplay) and (x_startpos < 50 and x_endpos < 50) and delta_goal_dist >= 30:
            return True

        # At least 15m closer to the opponent’s goal if the starting and finishing points are in different halves
        elif (check_success and check_inplay) and (x_startpos < 50 and x_endpos >= 50) and delta_goal_dist >= 15:
            return True

        # At least 10m closer to the opponent’s goal if the starting and finishing points are in the opponent’s half
        elif (check_success and check_inplay) and (x_startpos >= 50 and x_endpos >= 50) and delta_goal_dist >= 10:
            return True
        else:
            return float('nan')

    else:
        return float('nan')


def progressive_carry(single_event, successful_only=True):
    """ Identify progressive carry from WhoScored-style carry event.
    Function to identify progressive carries. A carry is considered progressive if the distance between the
    starting point and the end position is: (i) at least 30 meters closer to the opponent’s goal if the starting and
    finishing points are within a team’s own half, (ii) at least 15 meters closer to the opponent’s goal if the
    starting and finishing points are in different halves, (iii) at least 10 meters closer to the opponent’s goal if
    the starting and finishing points are in the opponent’s half. The function takes in a single event and returns a
    boolean (True = successful progressive carry.) This function is best used with the dataframe apply method.
    Args:
        single_event (pandas.Series): series corresponding to a single event (row) from WhoScored-style event dataframe.
        successful_only (bool, optional): selection of whether to only include successful carries. True by default
    Returns:
        bool: True = progressive carry, nan = non-progressive carry, unsuccessful progressive carry or not a carry
    """

    # Determine if event is pass
    if single_event['type'] == 'Carry':

        # Check success (if successful_only = True)
        if successful_only:
            check_success = single_event['outcomeType'] == 'Successful'
        else:
            check_success = True

        # Determine pass start and end position in yards (assuming standard pitch), and determine whether progressive
        x_startpos = single_event['x']
        y_startpos = single_event['y']
        x_endpos = single_event['endX']
        y_endpos = single_event['endY']
        delta_goal_dist = (np.sqrt((x_startpos) ** 2 + (y_startpos) ** 2) -
                           np.sqrt((x_endpos) ** 2 + (y_endpos) ** 2))

        # At least 30m closer to the opponent’s goal if the starting and finishing points are within a team’s own half
        if check_success and (x_startpos < 50 and x_endpos < 50) and delta_goal_dist >= 30:
            return True

        # At least 15m closer to the opponent’s goal if the starting and finishing points are in different halves
        elif check_success and (x_startpos < 50 and x_endpos >= 50) and delta_goal_dist >= 15:
            return True

        # At least 10m closer to the opponent’s goal if the starting and finishing points are in the opponent’s half
        elif check_success and (x_startpos >= 50 and x_endpos >= 50) and delta_goal_dist >= 10:
            return True
        else:
            return float('nan')

    else:
        return float('nan')


def pass_into_box(single_event, inplay=True, successful_only=True):
    """ Identify successful pass into box from whoscored-style pass event.
    Function to identify successful passes that end up in the opposition box. The function takes in a single event,
    and returns a boolean (True = successful pass into the box.) This function is best used with the dataframe apply
    method.
    Args:
        single_event (pandas.Series): series corresponding to a single event (row) from whoscored-style event dataframe.
        inplay (bool, optional): selection of whether to include 'in-play' events only. True by default.
        successful_only (bool, optional): selection of whether to only include successful passes. True by default
    Returns:
        bool: True = successful pass into the box, nan = not box pass, unsuccessful pass or not a pass.
    """

    # Determine if event is pass and check pass success
    if single_event['type'] == 'Pass':
        
        not_open_play = ['passFreekickAccurate', 'passFreekickInaccurate', 'keyPassFreekick', 'passFreekick',
                         'throwIn', 'keyPassThrowin', 'passCorner', 'passCornerInaccurate']

        # Check success (if successful_only = True)
        if successful_only:
            check_success = single_event['outcomeType'] == 'Successful'
        else:
            check_success = True

        # Check pass made in-play (if inplay = True)
        if inplay:
            check_inplay = not any(item in single_event['satisfiedEventsTypes'] for item in not_open_play)
        else:
            check_inplay = True

        # Determine pass end position, and whether it's a successful pass into box
        x_position = single_event['endX']
        y_position = single_event['endY']
        if (check_success and check_inplay) and (x_position >= 102) and (18 <= y_position <= 62):
            return True
        else:
            return float('nan')

    else:
        return float('nan')


def carry_into_box(single_event, successful_only=True):
    """ Identify successful carry into box from whoscored-style pass event.
    Function to identify successful carries that end up in the opposition box. The function takes in a single event,
    and returns a boolean (True = successful carry into the box.) This function is best used with the dataframe apply
    method.
    Args:
        single_event (pandas.Series): series corresponding to a single event (row) from whoscored-style event dataframe.
        successful_only (bool, optional): selection of whether to only include successful carries. True by default
    Returns:
        bool: True = successful carry into the box, nan = not box carry, unsuccessful carry or not a carry.
    """

    # Determine if event is pass and check pass success
    if single_event['type'] == 'Carry':

        # Check success (if successful_only = True)
        if successful_only:
            check_success = single_event['outcomeType'] == 'Successful'
        else:
            check_success = True

        # Determine pass end position, and whether it's a successful pass into box
        x_position = 120 * single_event['endX'] / 100
        y_position = 80 * single_event['endY'] / 100
        if check_success and (x_position >= 102) and (18 <= y_position <= 62):
            return True
        else:
            return float('nan')

    else:
        return float('nan')

def calc_assisted_xa(events_df):

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

    return player_kp

def team_performance(team, comp, season, league, metric_1, metric_2, web_app=False):

    # Assuming the "Roboto" font is installed on your system, you can specify it as the default font family.
    plt.rcParams['font.family'] = 'Roboto'

    title_font = 'Roboto'
    body_font = 'Roboto'

    main_folder = r"analysis_tools"

    if web_app == True:

        df = pd.read_csv(f'{main_folder}/Data/{comp}/{season[5:]}/match-logs/{team}-match-logs.csv', index_col=0)

    else:

        df = pd.read_csv(f'Data/{comp}/{season[5:]}/match-logs/{team}-match-logs.csv', index_col=0)
        
    df['yrAvg'] = df[metric_1].rolling(window=10).mean()
    df['zrAvg'] = df[metric_2].rolling(window=10).mean()

    background = '#1d2849'
    text_color='w'
    text_color_2='gray'
    mpl.rcParams['xtick.color'] = text_color
    mpl.rcParams['ytick.color'] = text_color

    filler = 'grey'
    primary = 'red'

    # create figure and axes
    fig, ax = plt.subplots(figsize=(12,6))
    fig.set_facecolor(background)
    ax.patch.set_facecolor(background)

    # add grid
    # ax.grid(ls='dotted', lw="0.5", color='lightgrey', zorder=1)

    x = df.startDate
    x = range(1,39)
    y = df.yrAvg
    z = df.zrAvg

    ax.plot(x, y, color='#4B9CD3', alpha=0.9, lw=1, zorder=2,
            label=f'Rolling 10 game {metric_1}', marker='o')
    ax.plot(x, z, color='#39ff14', alpha=0.9, lw=1, zorder=3,
           label=f'Rolling 10 game {metric_2}', marker='o')

    # Add a dotted horizontal line at y=0
    ax.axhline(y=0, color='gray', lw=1, linestyle='--', zorder=0)

    # Add a legend to the plot with custom font properties and white text color
    legend_font = fm.FontProperties(family='Roboto', weight='bold')
    legend = plt.legend(prop=legend_font, loc='upper left', frameon=False)
    plt.setp(legend.texts, color='white')  # Set legend text color to white

    # add title and subtitle

    df['startDate'] = df['startDate'].astype(str)
    start_date = df['startDate'].unique()[0]
    end_date = df['startDate'].unique()[37]

    fig.text(0.12,1.115, "{}".format(team), 
             fontsize=20, color=text_color, fontfamily=title_font, fontweight='bold')
    fig.text(0.12,1.065, f"{league}", fontsize=14, 
             color='white', fontfamily=title_font, fontweight='bold')
    fig.text(0.12,1.015, f"All matches, {start_date} to {end_date}", fontsize=14, 
             color='white', fontfamily=title_font, fontweight='bold')

    ax.tick_params(axis='both', length=0)

    spines = ['top', 'right', 'bottom', 'left']
    for s in spines:
        if s in ['top', 'right']:
            ax.spines[s].set_visible(False)
        else:
            ax.spines[s].set_color(text_color)
            
    # Set the x-axis ticks to increment by 2
    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=2)) 
    for label in ax.get_xticklabels():
        label.set_fontfamily(body_font)
        label.set_fontproperties(fm.FontProperties(weight='bold'))
    for label in ax.get_yticklabels():
        label.set_fontfamily(body_font)
        label.set_fontproperties(fm.FontProperties(weight='bold'))
            
    ax2 = fig.add_axes([0.05,0.99,0.08,0.08])
    ax2.axis('off')

    path = f'{main_folder}/Logos/{comp}/{team}.png'
    ax_team = fig.add_axes([-0.02,0.99,0.175,0.175])
    ax_team.axis('off')
    im = plt.imread(path)
    ax_team.imshow(im);

    fig.text(0.05, -0.025, "Viz by @egudi_analysis | Data via Opta", fontsize=9,
             fontfamily=body_font, fontweight='bold', color=text_color)

    fig.text(0.05, -0.05, "Expected goals model trained on ~10k shots from the 2021/2022 EPL season.", fontsize=9,
             fontfamily=body_font, fontweight='bold', color=text_color)

    fig.text(0.05, -0.075, "Expected threat model by Karun Singh", fontsize=9,
             fontfamily=body_font, fontweight='bold', color=text_color)

    plt.tight_layout()

    return fig

    # save figure
    #fig.savefig(f'{main_folder}/Output/{comp}/{season[5:]}/{team}-{start_date}-{end_date}', dpi=None, bbox_inches="tight")

import pandas as pd

def process_and_export_match_data(events_df, matches_data, comp, season):

    teamNames = []

    for match in matches_data:

        teamHomeName = match['home']['name']
        teamAwayName = match['away']['name']

        teamNames.append(teamHomeName)
        teamNames.append(teamAwayName)
        
    teamIds = []
        
    for match in matches_data:

        teamHomeId = match['home']['teamId']
        teamAwayId = match['away']['teamId']

        teamIds.append(teamHomeId)
        teamIds.append(teamAwayId)
        
    teams = pd.DataFrame({'teamId': teamIds,
                          'teamName': teamNames})

    teams = teams.drop_duplicates().reset_index(drop=True)

    passes = events_df[events_df['type'] == 'Pass']
    passes = xThreat(passes)  # Assuming xThreat is a function you have defined
    
    carries = events_df[events_df['type'] == 'Carry']
    carries = xThreat(carries)  # Assuming xThreat is a function you have defined
    
    df = pd.concat([passes, carries], axis=0)
    df = pd.merge(df, teams, on='teamId', how='left')
    
    teamNames = teams['teamName']
    
    matchIds = list(df.matchId.unique())
    
    df['startDate'] = pd.to_datetime(df['startDate'])
    df['startDate'] = df['startDate'].dt.date
    
    dfs = []
    
    for matchId in matchIds:
        team_df = df[df['matchId'] == matchId]
        home = team_df.teamName.unique()[0]
        away = team_df.teamName.unique()[1]
        date = team_df.startDate.unique()[0]
        team_df.to_csv(f'Data/{comp}/{season[5:]}/raw-season-data/{home}-{away}-{date}-passes-carries.csv')
        dfs.append(df)
    
    return dfs

def load_individual_match_team_dfs(comp, season, pass_filter='passes-carries'):

    # Enter a team for pass_filter

    path_to_folder = f'Data/{comp}/{season[5:]}/raw-season-data/'
    team_csv_files = glob.glob(os.path.join(path_to_folder, f'*{pass_filter}*.csv'))
    team_dataframes = []
    
    for csv_file in team_csv_files:
        match_data = pd.read_csv(csv_file)
        match_data = match_data.drop(match_data.columns[match_data.columns.str.contains('Unnamed', case=False)], axis=1)
        team_dataframes.append(match_data)
    
    return team_dataframes

def generate_match_week_zone_control_viz(team_dataframes, match_week, league, comp, season, off_week=True):

    # todays date for saving
    output = f'{comp}-{match_week}'

    final_dfs = []

    for match in team_dataframes:   
        
        match = match[match['outcomeType'] == 'Successful']
        match['isOpenPlay'] = np.where((match['passFreekick'] == False) &
                                      ((match['passCorner'] == False)
                                                    ) 
                                                   , 1, 0
                                                   )
        match = match[match['isOpenPlay'] == 1]
        
        match = match[['playerName', 'teamName', 'teamId', 'matchId', 'startDate', 'type', 'x', 'y', 'endX', 'endY', 'xT', 'h_a']]

        match['xT'] = match['xT'].fillna(0)

        # Create a boolean mask to identify rows where "teamName" is not equal to the home team
        home_team_df = match[match['h_a'] == 'h']
        home_team = home_team_df.teamName.unique()[0]
        mask = match['teamName'] != home_team

        # Multiply the values in "xT" column by -1 where the mask is True
        match.loc[mask, 'xT'] *= -1

        # Flip the "x" and "y" coordinates by subtracting them from 100 where the mask is True
        match.loc[mask, ['x', 'y', 'endX', 'endY']] = 100 - match.loc[mask, ['x', 'y', 'endX', 'endY']]
        
        final_dfs.append(match)
        
    final_df = pd.concat(final_dfs, axis=0)

    final_df = final_df.sort_values(by='startDate')

    if comp == 'la-liga':
        
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

    else:
        
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

    # Assuming the "Roboto" font is installed on your system, you can specify it as the default font family.
    plt.rcParams['font.family'] = 'Roboto'

    title_font = 'Roboto'
    body_font = 'Roboto'

    # Define the custom colormap with colors for negative, zero, and positive values
    negative_color = '#ff4500'   # Red for negative values
    zero_color = '#1d2849'        # Dark blue for zero values
    positive_color = '#39ff14'    # Green for positive values

    colors = [(negative_color), (zero_color), (positive_color)]
    n_bins = 100  # Number of color bins
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=n_bins)

    # List of matchIds
    match_ids_list = list(final_df.matchId.unique())

    # Define the number of matches per gameweek (adjust as needed)
    matches_per_gameweek = 10

    # Calculate the index range for the desired gameweek
    start_index = (match_week - 1) * matches_per_gameweek

    if off_week == False:
        start_index = start_index - 1
    else:
        pass  # Do nothing
        

    end_index = (match_week * matches_per_gameweek)

    # Adjust the match_ids_list using the calculated index range
    match_ids_list = match_ids_list[start_index:end_index]

    # Setup the pitch
    pitch = VerticalPitch(pitch_type='opta', pitch_color='#1d2849', line_color='w', line_zorder=5,
                          half=False, pad_top=2, axis=False, 
                          positional=True, positional_color='#eadddd', positional_zorder=5)

    # Create the subplot grid using mplsoccer
    fig, axs = pitch.grid(nrows=3, ncols=4, figheight=30,
                          endnote_height=0.01, endnote_space=0.01,
                          axis=False, space=0.1,
                          title_height=0.04, grid_height=0.84)
    fig.set_facecolor('#1d2849')

    # Set the title
    title = f"Match Week {match_week} - Zone Control"
    fig.text(0.0315,1.02,title, fontsize=40,
             fontfamily=body_font, fontweight='bold', color='w')

    if comp == 'la-liga':
        x_coor = 0.11
    else:
        x_coor = 0.165

    ax_text(x_coor, 96, f"{league}", fontsize=28, ha='center',
            fontfamily=body_font, fontweight='bold', color='w')

    ax_text(0, 94, f"<Green> shaded zones represent zones controlled by the home team", 
            fontsize=24,
            fontfamily=body_font, fontweight='bold', color='w',
            highlight_textprops=[{'color': positive_color}])

    ax_text(0, 92.5, "<Red> shaded zones represent zones controlled by the away team", 
            fontsize=24,
            fontfamily=body_font, fontweight='bold', color='w',
            highlight_textprops=[{'color': negative_color}])

    ax_text(0, 91, "Blue shaded zones represent neutral zones, not controlled by the home team or the away team", 
            fontsize=24,
            fontfamily=body_font, fontweight='bold', color='w')

    # Set the footnote
    footnote = "Zone Control is the difference of expected threat (xT) generated (+) and conceded (-)\nby the home team in each zone based on the start location of open play passes and carries."
    footnote2 = "Expected threat model by Karun Singh."
    footnote3 = 'Data via Opta | Created by @egudi_analysis'
    ax_text(0.73, 5.8, f"{footnote}\n{footnote2} {footnote3}", fontsize=17, ha='center',
             fontfamily=body_font, fontweight='bold', color='w')

    # Calculate the title height for positioning the logos
    title_height = 1  # Adjust as needed

    # Cycle through the grid axes and plot the heatmaps for each match
    for idx, ax in enumerate(axs['pitch'].flat):
        if idx < len(match_ids_list):
            match_id = match_ids_list[idx]
            match_test = final_df[final_df['matchId'] == match_id]
            home_team_df = match_test[match_test['h_a'] == 'h']
            home_team = home_team_df.teamName.unique()[0]
            away_team_df = match_test[match_test['h_a'] == 'a']
            away_team = away_team_df.teamName.unique()[0]
            home_abrev = home_team_df.team_abbreviation.unique()[0]
            away_abrev = away_team_df.team_abbreviation.unique()[0]
            
            # Calculate the sum total of 'xT' in each bin
            bin_statistic = pitch.bin_statistic_positional(match_test.x, match_test.y, match_test.xT, statistic='sum', positional='full',
                                                           normalize=True)
           
            # Use the colormap to create the heatmap using mplsoccer
            pitch.heatmap_positional(bin_statistic, ax=ax, edgecolors='#1a223d', cmap=cmap, vmin=-1, vmax=1)
            
            #labels = pitch.label_heatmap(bin_statistic, color='#f4edf0', fontsize=18,
            #                             ax=ax, ha='center', va='center',
            #                             str_format='{:.0}')
            
            ax_text(75,107, f"<{home_abrev}> vs <{away_abrev}>", color='w', fontsize=26,
                    fontfamily=body_font, fontweight='bold', ax=ax,
                    highlight_textprops=[{'color': positive_color},
                                         {'color': negative_color}])
            
            # Load logo images
            logo_paths = f'Logos/{comp}/{away_team}.png'
            team_logo_path = f'Logos/{comp}/{home_team}.png'
            logo = plt.imread(logo_paths)
            team_logo = plt.imread(team_logo_path)

            # Position the logo next to the title
            logo_ax = fig.add_axes([ax.get_position().x0 + 0.17, ax.get_position().y1, 0.03, 0.03])
            logo_ax.imshow(logo)
            logo_ax.axis('off')
            
            # Position the logo next to the title
            logo_ax = fig.add_axes([ax.get_position().x0+0.0025, ax.get_position().y1, 0.03, 0.03])
            logo_ax.imshow(team_logo)
            logo_ax.axis('off')
            
            ax.axis('off')  # Turn off axis for a clean visualization
            
    # Now, remove the last few plots using a separate loop
    for ax in axs['pitch'].flat[len(match_ids_list):]:
        ax.remove()
        
    # league logo
    path = f'Logos/{comp}/{comp}.png'
    ax_team = fig.add_axes([0.88,0.94,0.105,0.105])
    ax_team.axis('off')
    im = plt.imread(path)
    ax_team.imshow(im);

    fig.savefig(f'Output/{comp}/{season[5:]}/{output}', dpi=None, bbox_inches="tight")

def generate_team_zone_control_viz(team_dataframes, team, league, comp, season, juego_de_po=True):

    final_dfs = []

    for match in team_dataframes:   
        
        match = match[match['outcomeType'] == 'Successful']
        match['isOpenPlay'] = np.where((match['passFreekick'] == False) &
                                    ((match['passCorner'] == False)
                                                    ) 
                                                , 1, 0
                                                )
        match = match[match['isOpenPlay'] == 1]
        
        match = match[['playerName', 'teamName', 'teamId', 'matchId', 'startDate', 'type', 'x', 'y', 'endX', 'endY', 'xT', 'h_a']]

        match['xT'] = match['xT'].fillna(0)

        # Create a boolean mask to identify rows where "teamName" is not equal to "Arsenal"
        mask = match['teamName'] != team

        # Multiply the values in "xT" column by -1 where the mask is True
        match.loc[mask, 'xT'] *= -1

        # Flip the "x" and "y" coordinates by subtracting them from 100 where the mask is True
        match.loc[mask, ['x', 'y', 'endX', 'endY']] = 100 - match.loc[mask, ['x', 'y', 'endX', 'endY']]
        
        final_dfs.append(match)
        
    final_df = pd.concat(final_dfs, axis=0)

    final_df = final_df.sort_values(by='startDate')

    if comp == 'la-liga':

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
                            'Mallorcoca': 'RCD',
                            'Real Betis': 'BET',
                            'Real Madrid': 'MAD',
                            'Real Sociedad': 'SOC',
                            'Sevilla': 'SEV',
                            'Valencia': 'VAL',
                            'Villarreal': 'VIL'}

    else:

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

    # Assuming the "Roboto" font is installed on your system, you can specify it as the default font family.
    plt.rcParams['font.family'] = 'Roboto'

    title_font = 'Roboto'
    body_font = 'Roboto'

    if juego_de_po == True:

        # Define the custom colormap with colors for negative, zero, and positive values
        negative_color = '#ff4500'   # Red for negative values
        zero_color = '#1d2849'        # Dark blue for zero values
        positive_color = '#39ff14'    # Green for positive values

        colors = [(negative_color), (zero_color), (positive_color)]
        n_bins = 100  # Number of color bins
        cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=n_bins)
        
    else:

        # Set up the custom colormap with different shades for negative, zero, and positive values
        negative_color = '#ff4500'
        central_color = '#1d2849'
        positive_color = '#39ff14'

        n_bins = 256  # Number of color bins

        # Create a smooth transition colormap
        cmap = LinearSegmentedColormap.from_list('custom_cmap', [positive_color, central_color, negative_color], N=n_bins)

        # Define the range for colormap values
        vmax_global = None  # Global max value for all matches
        vmin_global = None  # Global min value for all matches

    # List of matchIds
    match_ids_list = list(final_df.matchId.unique())

    if juego_de_po == True:

        # Setup the pitch
        pitch = VerticalPitch(pitch_type='opta', pitch_color='#1d2849', line_color='w', line_zorder=5,
                            half=False, pad_top=2, axis=False, 
                            positional=True, positional_color='#eadddd', positional_zorder=5)

    else:    
        
        # Setup the pitch
        pitch = VerticalPitch(pitch_type='opta', pitch_color='#1d2849', line_color='w', line_zorder=5,
                            half=False, pad_top=2, axis=False, 
                            positional=True, positional_color='#eadddd', positional_zorder=5)
        
    if len(match_ids_list) < 6:
        rows = 1
    elif 6 <= len(match_ids_list) < 12:
        rows = 2
    elif 12 <= len(match_ids_list) < 18:
        rows = 3
    elif 18 <= len(match_ids_list) < 24:
        rows = 4
    elif 24 <= len(match_ids_list) < 30:
        rows = 5
    elif 30 <= len(match_ids_list) < 36:
        rows = 6
    else:
        rows = 7
        
    cols = len(match_ids_list)

    # Create the subplot grid using mplsoccer
    fig, axs = pitch.grid(nrows=rows, ncols=cols, figheight=30,
                        endnote_height=0.01, endnote_space=0.01,
                        axis=False, space=0.2,
                        title_height=0.04, grid_height=0.84)
    fig.set_facecolor('#1d2849')

    # Calculate dynamic text size and position based on the number of rows
    title_fontsize = 100 - 2 * rows
    subtitle_fontsize = 60 - (rows - 1)
    title_y = 1.17  # Adjust this value as needed
    subtitle_y = 109 # Adjust this value for spacing

    # Calculate dynamic x-coordinate for subtitle based on the number of columns
    subtitle_x = 0  # Center the subtitle

    # Set the title
    title = f"{team} Zone Control per Match"
    fig.text(0.036, title_y, title, fontsize=title_fontsize,
            fontfamily=body_font, fontweight='bold', color='w')

    # Set the individual subtitles with explicit vertical positioning
    ax_text(subtitle_x, subtitle_y, f"{league}", fontsize=subtitle_fontsize,
            fontfamily=body_font, fontweight='bold', color='w')

    ax_text(subtitle_x, subtitle_y - 4, f"<Green> shaded zones represent zones controlled by the home team",
            fontsize=subtitle_fontsize, fontfamily=body_font, fontweight='bold', color='w',
            highlight_textprops=[{'color': positive_color}])

    ax_text(subtitle_x, subtitle_y - 7, "<Red> shaded zones represent zones controlled by the away tean",
            fontsize=subtitle_fontsize, fontfamily=body_font, fontweight='bold', color='w',
            highlight_textprops=[{'color': negative_color}])

    ax_text(subtitle_x, subtitle_y -10, "Blue shaded zones represent neutral zones, not controlled by the home or away team",
            fontsize=subtitle_fontsize, fontfamily=body_font, fontweight='bold', color='w')

    # Calculate dynamic text size and position for footnotes
    footnote_fontsize = 35 - (rows - 1)
    footnote_y = 0.01  # Adjust this value for vertical position

    # Set the footnotes
    footnote = "Zone Control is defined as the sum of expected threat (xT) generated (+) and conceded (-) in each zone by\nLiverpool from the start location of open play passes and carries. Expected threat model by Karun Singh."
    footnote2 = 'Data via Opta | Created by @egudi_analysis'
    ax_text(0.275, footnote_y, f"{footnote}\n{footnote2}", fontsize=footnote_fontsize, ha='center',
            fontfamily=body_font, fontweight='bold', color='w')

    # Calculate the title height for positioning the logos
    title_height = 1  # Adjust as needed

    # Calculate logo positioning and size based on the number of rows and columns
    logo_width = 0.1  # Adjust this value as needed
    logo_height = 0.1  # Adjust this value as needed
    logo_spacing_x = 0.05  # Adjust this value as needed
    logo_spacing_y = 0.0105  # Adjust this value as needed

    # Cycle through the grid axes and plot the heatmaps for each match
    for idx, ax in enumerate(axs['pitch'].flat):
        if idx < len(match_ids_list):
            match_id = match_ids_list[idx]
            match_test = final_df[final_df['matchId'] == match_id]
            home_team_df = match_test[match_test['h_a'] == 'h']
            home_team = home_team_df.teamName.unique()[0]
            away_team_df = match_test[match_test['h_a'] == 'a']
            away_team = away_team_df.teamName.unique()[0]
            home_abrev = home_team_df.team_abbreviation.unique()[0]
            away_abrev = away_team_df.team_abbreviation.unique()[0]
            
            if juego_de_po == True:
            
                # Calculate the sum total of 'xT' in each bin
                bin_statistic = pitch.bin_statistic_positional(match_test.x, match_test.y, match_test.xT, statistic='sum', positional='full',
                                                            normalize=True)
            
                # Use the colormap to create the heatmap using mplsoccer
                pitch.heatmap_positional(bin_statistic, ax=ax, edgecolors='#1a223d', cmap=cmap, vmin=-1, vmax=1)
            
            else:
            
                # Calculate the sum total of 'xT' in each bin
                bin_statistic = pitch.bin_statistic(match_test.x, match_test.y, match_test.xT, statistic='sum', bins=(12, 6))
                bin_statistic['statistic'] = gaussian_filter(bin_statistic['statistic'], 1)

                # Update the global max and min values for colormap normalization
                if vmax_global is None or np.max(bin_statistic['statistic']) > vmax_global:
                    vmax_global = np.max(bin_statistic['statistic'])
                if vmin_global is None or np.min(bin_statistic['statistic']) < vmin_global:
                    vmin_global = np.min(bin_statistic['statistic'])

                # Normalize the sum total of 'xT' to colormap indices
                norm = plt.Normalize(vmin_global, vmax_global)

                # Map the sum total of 'xT' to colormap indices
                bin_statistic['color_index'] = norm(bin_statistic['statistic'])

                # Use the colormap to create the heatmap using mplsoccer
                pitch.heatmap(bin_statistic, ax=ax, edgecolors='#1a223d', cmap=cmap)
            
            ax_text(69,107, f"<{home_abrev}> vs <{away_abrev}>", color='w', fontsize=subtitle_fontsize+(subtitle_fontsize*0.25),
                    fontfamily=body_font, fontweight='bold', ax=ax,
                    highlight_textprops=[{'color': positive_color},
                                        {'color': negative_color}])
            
            # Load logo images
            logo_paths = f'Logos/{comp}/{away_team}.png'
            team_logo_path = f'Logos/{comp}/{home_team}.png'
            logo = plt.imread(logo_paths)
            team_logo = plt.imread(team_logo_path)

            # Calculate logo positions
            logo_x_opponent = ax.get_position().x0 + logo_spacing_x + 0.14
            logo_y = ax.get_position().y1 - logo_spacing_y
            
            logo_x_team = ax.get_position().x0 - logo_spacing_x - logo_width +0.15
            
            # Add opponent logo
            logo_ax_opponent = fig.add_axes([logo_x_opponent, logo_y, logo_width, logo_height])
            logo_ax_opponent.imshow(logo)
            logo_ax_opponent.axis('off')
            
            # Add team logo
            logo_ax_team = fig.add_axes([logo_x_team, logo_y, logo_width, logo_height])
            logo_ax_team.imshow(team_logo)
            logo_ax_team.axis('off')
            
            ax.axis('off')  # Turn off axis for a clean visualization
            
    # Now, remove the last few plots using a separate loop
    for ax in axs['pitch'].flat[len(match_ids_list):]:
        ax.remove()
        
    # league logo
    path = f'Logos/{comp}/{comp}.png'
    ax_team = fig.add_axes([0.825,1.01,0.2,0.2])
    ax_team.axis('off')
    im = plt.imread(path)
    ax_team.imshow(im);

    fig.savefig(f'Output/{comp}/{season[5:]}/{team}-control-{season[5:]}', dpi=None, bbox_inches="tight")

def process_and_export_season_match_data(comp, season, matches_data):
    try:
        teamNames = []

        for match in matches_data:
            teamHomeName = match['home']['name']
            teamAwayName = match['away']['name']

            teamNames.append(teamHomeName)
            teamNames.append(teamAwayName)

        teamIds = []

        for match in matches_data:
            teamHomeId = match['home']['teamId']
            teamAwayId = match['away']['teamId']

            teamIds.append(teamHomeId)
            teamIds.append(teamAwayId)

        teams = pd.DataFrame({'teamId': teamIds,
                              'teamName': teamNames})

        teams = teams.drop_duplicates().reset_index(drop=True)

        path_to_folder = f'Data/{comp}/{season[5:]}/raw-season-data/'

        clubs = list(teams.teamName.unique())

        for club in clubs:
            team_csv_files = glob.glob(os.path.join(path_to_folder, f'*{club}*.csv'))

            team_dataframes = []

            for csv_file in team_csv_files:
                match_data = pd.read_csv(csv_file)
                match_data = match_data.drop(match_data.columns[match_data.columns.str.contains('Unnamed', case=False)], axis=1)
                team_dataframes.append(match_data)

            club_dataframes = pd.concat(team_dataframes, axis=0)
            club_dataframes.to_csv(f'Data/{comp}/{season[5:]}/team-files/{club}-{season[5:]}.csv')

    except Exception as e:
        print(f"An error occurred: {e}")

def load_season_match_team_dfs(matches_data, comp, season):
    
    teamNames = []

    for match in matches_data:
        teamHomeName = match['home']['name']
        teamAwayName = match['away']['name']

        teamNames.append(teamHomeName)
        teamNames.append(teamAwayName)

    teamIds = []

    for match in matches_data:
        teamHomeId = match['home']['teamId']
        teamAwayId = match['away']['teamId']

        teamIds.append(teamHomeId)
        teamIds.append(teamAwayId)

    teams = pd.DataFrame({'teamId': teamIds,
                            'teamName': teamNames})

    teams = teams.drop_duplicates().reset_index(drop=True)

    clubs = list(teams.teamName.unique())

    # Replace 'path_to_folder' with the path to your folder containing the CSV files
    path_to_folder = f'Data/{comp}/{season[5:]}/team-files/'

        # Define the data types for each column in your CSV files
    dtypes = {
        'playerName': 'str',
        'teamName': 'str',
        'teamId': 'int',
        'matchId': 'int',
        'startDate': 'str',
        'type': 'str',
        'x': 'float',
        'y': 'float',
        'endX': 'float',
        'endY': 'float',
        'xT': 'float'
    }

    final_club_dfs = []

    for club in clubs:

        club_df = pd.read_csv(f'{path_to_folder}{club}-{season[5:]}.csv')

        club_df = club_df[club_df['outcomeType'] == 'Successful']
        club_df['isOpenPlay'] = np.where((club_df['passFreekick'] == False) &
                                      ((club_df['passCorner'] == False)
                                                    ) 
                                                   , 1, 0
                                                   )
        club_df = club_df[club_df['isOpenPlay'] == 1]

        club_df = club_df[['playerName', 'teamName', 'teamId', 'matchId', 'startDate', 'type', 'x', 'y', 'endX', 'endY', 'xT']]

        club_df['xT'] = club_df['xT'].fillna(0)

        # Create a boolean mask to identify rows where "teamName" is not equal to "Arsenal"
        mask = club_df['teamName'] != club

        # Multiply the values in "xT" column by -1 where the mask is True
        club_df.loc[mask, 'xT'] *= -1

        # Flip the "x" and "y" coordinates by subtracting them from 100 where the mask is True
        club_df.loc[mask, ['x', 'y', 'endX', 'endY']] = 100 - club_df.loc[mask, ['x', 'y', 'endX', 'endY']]

        final_club_dfs.append(club_df)

    return final_club_dfs

def generate_all_teams_zone_control(matches_data, league, final_club_dfs, comp, season, juego_de_po=True):

    teamNames = []

    for match in matches_data:
        teamHomeName = match['home']['name']
        teamAwayName = match['away']['name']

        teamNames.append(teamHomeName)
        teamNames.append(teamAwayName)

    teamIds = []

    for match in matches_data:
        teamHomeId = match['home']['teamId']
        teamAwayId = match['away']['teamId']

        teamIds.append(teamHomeId)
        teamIds.append(teamAwayId)

    teams = pd.DataFrame({'teamId': teamIds,
                            'teamName': teamNames})

    teams = teams.drop_duplicates().reset_index(drop=True)

    clubs = list(teams.teamName.unique())

    # Assuming the "Roboto" font is installed on your system, you can specify it as the default font family.
    plt.rcParams['font.family'] = 'Roboto'

    title_font = 'Roboto'
    body_font = 'Roboto'

    if juego_de_po == True:

        # Define the custom colormap with colors for negative, zero, and positive values
        negative_color = '#ff4500'   # Red for negative values
        zero_color = '#1d2849'        # Dark blue for zero values
        positive_color = '#39ff14'    # Green for positive values

        colors = [(negative_color), (zero_color), (positive_color)]
        n_bins = 100  # Number of color bins
        cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=n_bins)

    else:

        # Set up the custom colormap with different shades for negative, zero, and positive values
        negative_color = '#ff4500'
        central_color = '#1d2849'
        positive_color = '#39ff14'

        n_bins = 256  # Number of color bins

        # Create a smooth transition colormap
        cmap = LinearSegmentedColormap.from_list('custom_cmap', [negative_color, central_color, positive_color], N=n_bins)

        # Define the range for colormap values
        vmax_global = None  # Global max value for all matches
        vmin_global = None  # Global min value for all matches

    if juego_de_po == True:

        # Setup the pitch
        pitch = VerticalPitch(pitch_type='opta', pitch_color='#1d2849', line_color='w', line_zorder=5,
                            half=False, pad_top=2, axis=False, 
                            positional=True, positional_color='#eadddd', positional_zorder=5)
    
    else:
    
        # Setup the pitch
        pitch = VerticalPitch(pitch_type='opta', pitch_color='#1d2849', line_color='w', line_zorder=5,
                            half=False, pad_top=2, axis=False, 
                            positional=True, positional_color='#eadddd', positional_zorder=5)

    # Create the subplot grid using mplsoccer
    fig, axs = pitch.grid(nrows=4, ncols=5, figheight=30,
                        endnote_height=0.01, endnote_space=0.01,
                        axis=False, space=0.1,
                        title_height=0.08, grid_height=0.84)
    fig.set_facecolor('#1d2849')

    # Set the title
    title = f"Match Zone Control"
    fig.text(0.032,1.0,title, fontsize=40,
            fontfamily=body_font, fontweight='bold', color='w')

    ax_text(0.245, 97, f"{league} - All Matches", fontsize=28, ha='center',
            fontfamily=body_font, fontweight='bold', color='w')

    ax_text(0, 94, f"<Green> shaded zones represent zones controlled by the team represented in each map", 
            fontsize=24, fontfamily=body_font, fontweight='bold', color='w',
            highlight_textprops=[{'color': positive_color}])

    ax_text(0, 92.5, "<Red> shaded zones represent zones controlled by the opposition", 
            fontsize=24, fontfamily=body_font, fontweight='bold', color='w',
            highlight_textprops=[{'color': negative_color}])

    ax_text(0, 91, "Blue shaded zones represent neutral zones, not domindated by the represented team and opposition", 
            fontsize=24, fontfamily=body_font, fontweight='bold', color='w')

    # Set the footnote
    footnote = "Zone Control is defined as the difference of expected threat (xT) generated (+) and conceded (-) in each zone\nfrom the start location of open play passes and carries. Expected threat model by Karun Singh."
    footnote2 = 'Data via Opta | Created by @egudi_analysis'
    ax_text(0.39, 0, f"{footnote}\n{footnote2}", fontsize=20, ha='center',
            fontfamily=body_font, fontweight='bold', color='w')


    # Calculate the title height for positioning the logos
    title_height = 1  # Adjust as needed

    # Cycle through the grid axes and plot the heatmaps for each match
    for idx, (ax, df) in enumerate(zip(axs['pitch'].flat, final_club_dfs)):
        if idx < len(clubs):
        # Assuming final_club_dfs is your list of dataframes
            highest_counts = df.teamName.value_counts()
            team_with_highest_count = highest_counts.idxmax()
            team_name_string = str(team_with_highest_count)

            if juego_de_po == True:
            
                # Calculate the sum total of 'xT' in each bin
                bin_statistic = pitch.bin_statistic_positional(df.x, df.y, df.xT, statistic='sum', positional='full',
                                                               normalize=True)
            
                # Use the colormap to create the heatmap using mplsoccer
                pitch.heatmap_positional(bin_statistic, ax=ax, edgecolors='#1a223d', cmap=cmap, vmin=-1, vmax=1)

            else:

                # Calculate the sum total of 'xT' in each bin
                bin_statistic = pitch.bin_statistic(df.x, df.y, df.xT, statistic='sum', bins=(24, 12))
                bin_statistic['statistic'] = gaussian_filter(bin_statistic['statistic'], 1)
                
                # Update the global max and min values for colormap normalization
                if vmax_global is None or np.max(bin_statistic['statistic']) > vmax_global:
                    vmax_global = np.max(bin_statistic['statistic'])
                if vmin_global is None or np.min(bin_statistic['statistic']) < vmin_global:
                    vmin_global = np.min(bin_statistic['statistic'])
                
                # Normalize the sum total of 'xT' to colormap indices
                norm = plt.Normalize(vmin_global, vmax_global)
                
                # Map the sum total of 'xT' to colormap indices
                bin_statistic['color_index'] = norm(bin_statistic['statistic'])
                
                # Use the colormap to create the heatmap using mplsoccer
                pitch.heatmap(bin_statistic, ax=ax, edgecolors='#1a223d', cmap=cmap)
            
            ax_text(83,110, f"<{team_name_string}>", color='w', fontsize=25,
                    fontfamily=body_font, fontweight='bold', ax=ax)
            
            # Load logo images
            team_logo_path = f'Logos/{comp}/{team_name_string}.png'
            team_logo = plt.imread(team_logo_path)
            
            # Position the logo next to the title
            logo_ax = fig.add_axes([ax.get_position().x0, ax.get_position().y1, 0.03, 0.03])
            logo_ax.imshow(team_logo)
            logo_ax.axis('off')
            
            ax.axis('off')  # Turn off axis for a clean visualization
        
    # league logo
    path = f'Logos/{comp}/{comp}.png'
    ax_team = fig.add_axes([0.87,0.91,0.12,0.12])
    ax_team.axis('off')
    im = plt.imread(path)
    ax_team.imshow(im);

    fig.savefig(f'Output/{comp}/{season[5:]}/control-full-{season[5:]}', dpi=None, bbox_inches="tight")
