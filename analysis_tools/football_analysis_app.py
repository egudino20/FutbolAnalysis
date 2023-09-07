# Import your module here
from visuals import team_performance, process_and_export_match_data, load_individual_match_team_dfs
from visuals import load_individual_match_team_dfs, generate_match_week_zone_control_viz, generate_team_zone_control_viz
from visuals import generate_all_teams_zone_control, process_and_export_season_match_data, load_season_match_team_dfs

import os
import json
import pandas as pd
import streamlit as st

def main():
    print("Current working directory:", os.getcwd())
    st.title('Football Analysis')

    # Add sidebar for league
    country = st.sidebar.selectbox('Choose a league', ['Argentina', 'Spain', 'England', 'France', 'Germany', 'Italy'])

    # Add sidebar for season
    year = st.sidebar.selectbox('Choose a season', ['2023', '2024'])

    # Add sidebar for date
   # date = st.sidebar.date_input('Choose a date')

    # Convert date to string in the format "mm.dd.yyyy"
    #date_str = date.strftime("%m.%d.%Y")

    # Map league to competition
    league_to_comp = {
        'Argentina': 'liga-profesional',
        'Spain': 'la-liga',
        'England': 'premier-league',
        'France': 'ligue-1',
        'Germany': 'bundesliga',
        'Italy': 'serie-a'
    }
    league_folder = league_to_comp[country]

    main_folder = r"C:\Users\Enrique\PythonProjects\futbol-analysis\analysis_tools"
    filename = "matches_data.json"
    #filepath = os.path.join(main_folder, "Data", league_folder, year, "match-data", filename)
    filepath = "https://storage.cloud.google.com/matches-data/matches_data.json"

    with open(filepath, "r") as f:
        matches_data = json.load(f)

    # league identifiers
    league = f"{matches_data[0]['region']} {matches_data[0]['league']} {matches_data[0]['season']}"

    season = matches_data[0]['season'] 

    #events_df = pd.read_csv(f'{main_folder}/Data/{league_folder}/{season[5:]}/raw-season-data/{league_folder}-{date_str}.csv', low_memory=False)
    #events_df.drop('Unnamed: 0', axis=1, inplace=True)

    options = ['Team Performance', 'Match Data', 'Individual Match Team Data', 
           'Match Week Zone Control Visualization', 'Team Zone Control Visualization', 
           'All Teams Zone Control', 'Season Match Data', 'Season Match Team Data']

    option = st.sidebar.selectbox('Choose an option', options)

    if option == 'Team Performance':
        metrics = ['xG', 'xGA', 'xGD', 'xT', 'xTA', 'xTD']

        if country == 'England':
            premier_league_teams = ['Arsenal', 'Aston Villa', 'Brentford', 'Brighton & Hove Albion', 
                                    'Burnley', 'Chelsea', 'Crystal Palace', 'Everton', 'Leeds United', 
                                    'Leicester City', 'Liverpool', 'Manchester City', 'Manchester United',
                                    'Newcastle United', 'Norwich City', 'Southampton', 'Tottenham Hotspur', 
                                    'Watford', 'West Ham United', 'Wolverhampton Wanderers']        
            team = st.sidebar.selectbox('Select a team', premier_league_teams)
        else:
            team = st.sidebar.text_input('Enter a team name')
        selected_metric_1 = st.sidebar.selectbox('Select metric 1', metrics)
        selected_metric_2 = st.sidebar.selectbox('Select metric 2', metrics)
        try:
            result = team_performance(team, league_folder, season, league, selected_metric_1, selected_metric_2, web_app=True)
            st.pyplot(result)  # Display the plot
        except:
            st.write("No Data")

        # Load the underlying data
        data_path = f'{main_folder}/Data/{league_folder}/{season[5:]}/match-logs/{team}-match-logs.csv'
        data_df = pd.read_csv(data_path)
        data_df = data_df[['teamName', 'opponent', 'matchId', 'startDate'] + metrics]

        # Check if data_df is empty
        if len(data_df) == 0:
            st.write("No Data")
        else:
            # Display the data as a table
            st.write(f"{team} Match Logs")
            st.table(data_df)

            # Add a button to export the data as a CSV file
            csv_export_button = st.download_button(
                label="Export Data as CSV",
                data=data_df.to_csv(index=False),
                file_name="team_performance_data.csv",
                mime="text/csv"
            )
    
    #elif option == 'Match Data':
        # Call your process_and_export_match_data method here
        # result = my_module.process_and_export_match_data()
        # st.write(result)

    #elif option == 'Individual Match Team Data':
        # Call your load_individual_match_team_dfs method here
        # result = my_module.load_individual_match_team_dfs()
        # st.write(result)

    #elif option == 'Match Week Zone Control Visualization':
        # Call your generate_match_week_zone_control_viz method here
        # result = my_module.generate_match_week_zone_control_viz()
        # st.write(result)

    #elif option == 'Team Zone Control Visualization':
        # Call your generate_team_zone_control_viz method here
        # result = my_module.generate_team_zone_control_viz()
        # st.write(result)

    #elif option == 'All Teams Zone Control':
        # Call your generate_all_teams_zone_control method here
        # result = my_module.generate_all_teams_zone_control()
        # st.write(result)

    #elif option == 'Season Match Data':
        # Call your process_and_export_season_match_data method here
        # result = my_module.process_and_export_season_match_data()
        # st.write(result)

    #elif option == 'Season Match Team Data':
        # Call your load_season_match_team_dfs method here
        # result = my_module.load_season_match_team_dfs()
        # st.write(result)

if __name__ == "__main__":
    main()