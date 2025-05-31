import pandas as pd
import numpy as np
import requests
import re
from nba_api.stats.static import players
from nba_api.stats.endpoints import commonplayerinfo
from nba_api.stats.endpoints import playbyplayv3
from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.endpoints import gamerotation
import streamlit as st

headers  = {
    'Connection': 'keep-alive',
    'Accept': 'application/json, text/plain, */*',
    'x-nba-stats-token': 'true',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36',
    'x-nba-stats-origin': 'stats',
    'Sec-Fetch-Site': 'same-origin',
    'Sec-Fetch-Mode': 'cors',
    'Referer': 'https://stats.nba.com/',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'en-US,en;q=0.9',
}

gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable='2024-25', league_id_nullable='00', season_type_nullable='Playoffs')
games = gamefinder.get_data_frames()[0]

knicks_pacers = ['NYK @ IND', 'NYK vs. IND']
games = games[games['MATCHUP'].isin(knicks_pacers)]
game_ids = games['GAME_ID'].unique().tolist()

games = []
for id in game_ids:
    pbp = playbyplayv3.PlayByPlayV3(id)
    pbp = pbp.get_data_frames()[0]
    games.append(pbp)
pbp = pd.concat(games, ignore_index=True)

def parse_iso_time(duration_str):
    match = re.match(r"PT(\d+)M([\d.]+)S", duration_str)
    if match:
        minutes = int(match.group(1))
        seconds = float(match.group(2))
        return minutes * 60 + seconds
    else:
        return None

clutch = pbp[pbp['period'] >= 4].copy()
clutch['clock_secs'] = clutch['clock'].apply(parse_iso_time)
clutch['scoreHome'] = pd.to_numeric(clutch['scoreHome'], errors='coerce')
clutch['scoreAway'] = pd.to_numeric(clutch['scoreAway'], errors='coerce')
clutch['point_dif'] = clutch.scoreHome - clutch.scoreAway
clutch['point_dif'] = clutch['point_dif'].ffill()
clutch = clutch[(clutch['clock_secs'] <= 300) & (clutch['point_dif'].abs() <= 5)]

clutch['FGA'] = clutch['actionType'].str.contains('Shot')
clutch['FGM'] = clutch['actionType'].str.contains('Made Shot')
clutch['shotValue'] = clutch['shotValue'].astype(str)
clutch['3PA'] = clutch['shotValue'].str.contains('3')
clutch['3PM'] = clutch['shotValue'].str.contains('3') & clutch['actionType'].str.contains('Made Shot')
clutch['TO'] = clutch['actionType'].str.contains('Turnover')
clutch['REB'] = clutch['actionType'].str.contains('Rebound')
clutch['FTA'] = clutch['actionType'].str.contains('Free Throw')
clutch['FTM'] = clutch.apply(lambda row: False if 'MISS' in row['description'] and row['actionType']=='Free Throw'  else True if 'Free Throw' in row['description'] else False, axis=1)
clutch['PTS'] = clutch.apply(lambda row: int(row['shotValue']) if 'Made' in row['shotResult'] else 1 if row['FTM'] else 0, axis=1)

clutch_stats = clutch.groupby(['gameId', 'playerName']).agg({
    'PTS': 'sum',
    'FGA': 'sum',
    'FGM': 'sum',
    '3PA': 'sum',
    '3PM': 'sum',
    'TO': 'sum',
    'REB': 'sum',
    'FTA': 'sum',
    'FTM': 'sum'
}).reset_index()
clutch_stats = clutch_stats.groupby('playerName').sum(numeric_only=True)
clutch_stats['FG%'] = clutch_stats['FGM'].astype(int)/clutch_stats['FGA'].astype(int)
clutch_stats['3P%'] = clutch_stats['3PM'].astype(int)/clutch_stats['3PA'].astype(int)
clutch_stats['USG'] = clutch_stats['FGA'].astype(int) + clutch_stats['TO'].astype(int) + clutch_stats['FTA'].astype(int)
clutch_stats = clutch_stats.iloc[1:]

clutch_stats = clutch_stats.reset_index()

# rotation = gamerotation.GameRotation(game_id='0042400305')
# rotation = rotation.get_data_frames()[1]
# rotation['IN_TIME_REAL'] = rotation['IN_TIME_REAL'].astype(int)/10
# rotation['OUT_TIME_REAL'] = rotation['OUT_TIME_REAL'].astype(int)/10
# rotation['playerName'] = rotation['PLAYER_FIRST'] + ' ' + rotation['PLAYER_LAST']
# # print(rotation)

def players_on_the_court(rotation, game_time_end, game_time_start=0):
    return rotation[
        (rotation['IN_TIME_REAL'] <= game_time_start) &
        (rotation['OUT_TIME_REAL'] >= game_time_end)][['playerName', 'TEAM_ID', 'IN_TIME_REAL', 'OUT_TIME_REAL']]

def when_player_is_in(rotation, playerName):
    return rotation[rotation['playerName'] == playerName]

def stats_at_a_moment(pbp, game_time_end, game_time_start=0):
    pbp['clock_secs'] = (pbp['clock'].apply(parse_iso_time))
    pbp['real_time'] = None
    pbp.loc[pbp['period'] <= 4, 'real_time'] = ((pbp['period'] - 1) * 720 + (720 - pbp['clock_secs']))
    pbp.loc[pbp['period'] > 4, 'real_time'] = (2880 + 300 - pbp['clock_secs'])
    pbp['scoreHome'] = pd.to_numeric(pbp['scoreHome'], errors='coerce')
    pbp['scoreAway'] = pd.to_numeric(pbp['scoreAway'], errors='coerce')
    pbp['point_dif'] = pbp.scoreHome - pbp.scoreAway
    pbp['point_dif'] = pbp['point_dif'].ffill()
    pbp = pbp[(pbp['real_time'] >= game_time_start) & (pbp['real_time'] <= game_time_end)].copy()
    clutch = pbp

    clutch['FGA'] = clutch['actionType'].str.contains('Shot')
    clutch['FGM'] = clutch['actionType'].str.contains('Made Shot')
    clutch['shotValue'] = clutch['shotValue'].astype(str)
    clutch['3PA'] = clutch['shotValue'].str.contains('3')
    clutch['3PM'] = clutch['shotValue'].str.contains('3') & clutch['actionType'].str.contains('Made Shot')
    clutch['TO'] = clutch['actionType'].str.contains('Turnover')
    clutch['REB'] = clutch['actionType'].str.contains('Rebound')
    clutch['FTA'] = clutch['actionType'].str.contains('Free Throw')
    clutch['FTM'] = clutch.apply(lambda row: False if 'MISS' in row['description'] and row[
        'actionType'] == 'Free Throw' else True if 'Free Throw' in row['description'] else False, axis=1)
    clutch['PTS'] = clutch.apply(
        lambda row: int(row['shotValue']) if 'Made' in row['shotResult'] else 1 if row['FTM'] else 0, axis=1)

    clutch_team_stats = clutch.groupby(['gameId', 'teamTricode']).agg({
        'PTS': 'sum',
        'FGA': 'sum',
        'FGM': 'sum',
        '3PA': 'sum',
        '3PM': 'sum',
        'TO': 'sum',
        'REB': 'sum',
        'FTA': 'sum',
        'FTM': 'sum'
    }).reset_index()
    clutch_team_stats = clutch_team_stats.groupby('teamTricode').sum(numeric_only=True)
    clutch_team_stats['FG%'] = clutch_team_stats['FGM'].astype(int) / clutch_team_stats['FGA'].astype(int)
    clutch_team_stats['3P%'] = clutch_team_stats['3PM'].astype(int) / clutch_team_stats['3PA'].astype(int)
    clutch_team_stats = clutch_team_stats.iloc[1:]
    clutch_team_stats = clutch_team_stats.reset_index()

    clutch_stats = clutch.groupby(['gameId', 'playerName']).agg({
        'PTS': 'sum',
        'FGA': 'sum',
        'FGM': 'sum',
        '3PA': 'sum',
        '3PM': 'sum',
        'TO': 'sum',
        'REB': 'sum',
        'FTA': 'sum',
        'FTM': 'sum'
    }).reset_index()
    clutch_stats = clutch_stats.groupby('playerName').sum(numeric_only=True)
    clutch_stats['FG%'] = clutch_stats['FGM'].astype(int) / clutch_stats['FGA'].astype(int)
    clutch_stats['3P%'] = clutch_stats['3PM'].astype(int) / clutch_stats['3PA'].astype(int)
    clutch_stats['USG'] = clutch_stats['FGA'].astype(int) + clutch_stats['TO'].astype(int) + clutch_stats['FTA'].astype(
        int)
    clutch_stats = clutch_stats.iloc[1:]
    clutch_stats = clutch_stats.reset_index()
    return [clutch_team_stats, clutch_stats]

# st.title('Knicks vs Pacers ECF')
#
# stat_options = st.multiselect('Select stats to plot', options=clutch_stats.columns[1:], max_selections=2)
# if stat_options:
#     st.scatter_chart(clutch_stats, x=stat_options[0], y=stat_options[1], color='playerName')
# else:
#     st.warning('Select at least something')

start, end = st.select_slider('Select timing for rotations', options=np.linspace(0, 48, 49), value=[0,48])
chart_data = stats_at_a_moment(pbp, end * 60, start * 60)[0]
st.area_chart(chart_data, x='FG%', y='TO', color='teamTricode')