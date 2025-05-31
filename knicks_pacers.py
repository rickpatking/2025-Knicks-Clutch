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
from sklearn.preprocessing import MultiLabelBinarizer
import seaborn as sns
import matplotlib.pyplot as plt

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
clutch['clock_secs'] = (clutch['clock'].apply(parse_iso_time))
clutch['real_time'] = None
clutch.loc[clutch['period'] <= 4, 'real_time'] = ((clutch['period'] - 1) * 720 + (720 - clutch['clock_secs']))
clutch.loc[clutch['period'] > 4, 'real_time'] = (2880 + 300 - clutch['clock_secs'])
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
clutch['AST'] = clutch['description'].str.contains('AST')
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

rotations=[]
for id in game_ids:
    rotation = gamerotation.GameRotation(game_id=id)
    rotation = rotation.get_data_frames()
    for rot in rotation:
        rot['IN_TIME_REAL'] = rot['IN_TIME_REAL'].astype(int)/10
        rot['OUT_TIME_REAL'] = rot['OUT_TIME_REAL'].astype(int)/10
        rot['playerName'] = rot['PLAYER_FIRST'] + ' ' + rot['PLAYER_LAST']
        rotations.append(rot)
rotations = pd.concat(rotations, ignore_index=True)

def players_on_the_court(rotation, game_time_start=0):
    return rotation[
        (rotation['IN_TIME_REAL'] <= game_time_start) &
        (rotation['OUT_TIME_REAL'] >= game_time_start)][['playerName', 'TEAM_ID', 'IN_TIME_REAL', 'OUT_TIME_REAL', 'GAME_ID']]

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
    clutch['AST'] = clutch['description'].str.contains('AST')
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

clutch['players_on_court'] = clutch['real_time'].apply(lambda t: players_on_the_court(rotations, t)['playerName'].tolist())

records = []
for _, row in clutch.iterrows():
    if row['TO'] or row['AST']:
        for player in row['players_on_court']:
            records.append({
                'playerName': player,
                'event_type': 'TO' if row['TO'] else 'AST',
                'game_time': row['real_time']
            })

clutch_start = clutch['real_time'].min()
clutch_end = clutch['real_time'].max()
def get_clutch_minutes(row):
    in_time = max(row['IN_TIME_REAL'], clutch_start)
    out_time = min(row['OUT_TIME_REAL'], clutch_end)
    return max(0, out_time-in_time)/60
rotations['clutch_minutes'] = rotations.apply(get_clutch_minutes, axis=1)
clutch_minutes_df = rotations.groupby('playerName', as_index=False)['clutch_minutes'].sum()

player_event_df = pd.DataFrame(records)
player_impact = (player_event_df.groupby(['playerName', 'event_type']).size().unstack(fill_value=0).reset_index())
rotations['clutch_minutes'] = rotations['clutch_minutes']
player_minutes = rotations.groupby('playerName', as_index=False)['clutch_minutes'].sum()
player_impact = player_impact.merge(player_minutes, on='playerName')
player_impact['AST_per_min'] = player_impact['AST'] / player_impact['clutch_minutes']
player_impact['TO_per_min'] = player_impact['TO'] / player_impact['clutch_minutes']
player_impact['AST_per_5'] = player_impact['AST_per_min'] * 5
player_impact['TO_per_5'] = player_impact['TO_per_min'] * 5
player_team_map = rotations[['playerName', 'TEAM_NAME']].drop_duplicates()
player_impact = player_impact.merge(player_team_map[['playerName', 'TEAM_NAME']], on='playerName', how='left')
player_impact['AST_to_TO'] = player_impact['AST_per_5'] / player_impact['TO_per_5']
player_impact['eligible'] = player_impact['clutch_minutes'].astype(int) > 2
eligible_impact = player_impact[player_impact['eligible']]


# st.title('Knicks vs Pacers ECF')
#
# stat_options = st.multiselect('Select stats to plot', options=clutch_stats.columns[1:], max_selections=2)
# if stat_options:
#     st.scatter_chart(clutch_stats, x=stat_options[0], y=stat_options[1], color='playerName')
# else:
#     st.warning('Select at least something')

# start, end = st.select_slider('Select timing for rotations', options=np.linspace(0, 48, 49), value=[0,48])
# chart_data = stats_at_a_moment(pbp, end * 60, start * 60)[0]
# st.bar_chart(chart_data, x='teamTricode', y='TO')

# st.scatter_chart(player_impact, x='AST_per_36', y='TO_per_36', color='playerName')
# st.bar_chart(player_impact, x='TEAM_NAME', y='AST_to_TO', color='playerName', stack=False)

heatmap_data = eligible_impact.set_index('playerName')[['AST_per_5', 'TO_per_5']]
plt.figure(figsize=(10,6))
sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('AST and TO per 5 minutes per player')
plt.xlabel('Metric')
plt.ylabel('Player')
plt.tight_layout()
plt.show()