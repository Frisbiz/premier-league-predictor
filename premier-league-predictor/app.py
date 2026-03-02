from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import pickle
import os
from datetime import datetime
import json
from scipy.optimize import minimize
from scipy.special import gammaln
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ==================== SIMPLE POISSON MODEL ====================

from scipy.stats import poisson

class SimplePoissonModel:
    """Simple Poisson model with team strength estimation"""
    
    def __init__(self):
        self.team_attack = {}  # Goals scored per game at home/away
        self.team_defense = {}  # Goals conceded per game at home/away
        self.home_advantage = 0.1
        self.n_teams = 0
        self.global_home_avg = 0
        self.global_away_avg = 0
        
    def fit(self, df, teams):
        """Fit simple Poisson model based on historical averages"""
        self.n_teams = len(teams)
        
        # Store global averages
        self.global_home_avg = df['FTHG'].mean()
        self.global_away_avg = df['FTAG'].mean()
        
        # Calculate average goals scored/conceded for each team
        for team in teams:
            home_matches = df[df['HomeTeam'] == team]
            away_matches = df[df['AwayTeam'] == team]
            
            if len(home_matches) > 0:
                home_gs = home_matches['FTHG'].mean()  # Goals scored at home
                home_gc = home_matches['FTAG'].mean()  # Goals conceded at home
            else:
                home_gs = self.global_home_avg
                home_gc = self.global_home_avg
                
            if len(away_matches) > 0:
                away_gs = away_matches['FTAG'].mean()  # Goals scored away
                away_gc = away_matches['FTHG'].mean()  # Goals conceded away
            else:
                away_gs = self.global_away_avg
                away_gc = self.global_away_avg
            
            self.team_attack[team] = {'home': home_gs, 'away': away_gs}
            self.team_defense[team] = {'home': home_gc, 'away': away_gc}
        
        print(f"✓ Simple Poisson model fitted for {len(teams)} teams")
        print(f"  Global averages - Home: {self.global_home_avg:.2f}, Away: {self.global_away_avg:.2f}")
        
    def predict(self, home_team, away_team, exclude_draw=False):
        """Predict match using Poisson distribution"""
        if home_team not in self.team_attack or away_team not in self.team_attack:
            return None
        
        # Get raw averages
        home_goals_for = self.team_attack[home_team]['home']
        away_goals_for = self.team_attack[away_team]['away']
        home_goals_against = self.team_defense[home_team]['home']
        away_goals_against = self.team_defense[away_team]['away']
        
        # Expected goals = attack * opponent defense, normalized by global average
        lam = home_goals_for * away_goals_against / self.global_home_avg
        mu = away_goals_for * home_goals_against / self.global_away_avg
        
        # Apply home advantage (~0.35 goals)
        lam *= np.exp(0.35)
        
        # Keep within realistic bounds
        lam = max(0.5, min(lam, 3.0))
        mu = max(0.5, min(mu, 3.0))
        
        # Calculate ALL score probabilities
        score_probs = {}
        home_win_total = 0
        away_win_total = 0
        draw_total = 0
        best_prob = 0
        best_score = (1, 1)
        
        for h in range(7):
            for a in range(7):
                prob = poisson.pmf(h, lam) * poisson.pmf(a, mu)
                score_probs[(h, a)] = prob
                
                # Track outcome probabilities
                if h > a:
                    home_win_total += prob
                elif a > h:
                    away_win_total += prob
                else:
                    draw_total += prob
        
        # If exclude_draw is True, find best non-1-1 score
        if exclude_draw:
            # Remove 1-1 from consideration
            if (1, 1) in score_probs:
                del score_probs[(1, 1)]
        
        # Find most likely score (after potential exclusion)
        for (h, a), prob in score_probs.items():
            if prob > best_prob:
                best_prob = prob
                best_score = (h, a)
        
        # Confidence = probability of the most likely outcome
        confidence = best_prob
        
        return {
            'home_goals': best_score[0],
            'away_goals': best_score[1],
            'home_prob': home_win_total,
            'draw_prob': draw_total,
            'away_prob': away_win_total,
            'confidence': min(0.95, confidence * 2)  # Scale for readability
        }

# Premier League teams 2024-25 + some Championship teams
premier_league_teams = [
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
    "Chelsea", "Crystal Palace", "Everton", "Fulham", "Ipswich",
    "Leicester", "Liverpool", "Man City", "Man United", "Newcastle",
    "Nottingham Forest", "Southampton", "Tottenham", "West Ham", "Wolves",
    "Sunderland", "Leeds", "Burnley", "Sheffield United", "Norwich"
]

def fetch_extended_data():
    """Fetch 10+ seasons of Premier League data"""
    seasons = [
        ("1415", "2014-15"),
        ("1516", "2015-16"),
        ("1617", "2016-17"),
        ("1718", "2017-18"),
        ("1819", "2018-19"),
        ("1920", "2019-20"),
        ("2021", "2020-21"),
        ("2122", "2021-22"),
        ("2223", "2022-23"),
        ("2324", "2023-24"),
        ("2425", "2024-25"),
    ]
    
    all_data = []
    for season_code, season_name in seasons:
        url = f"https://www.football-data.co.uk/mmz4281/{season_code}/E0.csv"
        try:
            df = pd.read_csv(url)
            df['Season'] = season_name
            all_data.append(df)
            print(f"✓ Loaded {season_name}")
        except Exception as e:
            print(f"✗ Could not load {season_name}: {e}")
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        print(f"\nTotal matches loaded: {len(combined)}")
        return combined
    return None

def calculate_advanced_stats(df):
    """Calculate comprehensive team statistics"""
    df = df[df['FTR'].notna()].copy()
    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    
    team_stats = {}
    
    for team in teams:
        # All matches
        home_matches = df[df['HomeTeam'] == team].sort_values('Date', ascending=True)
        away_matches = df[df['AwayTeam'] == team].sort_values('Date', ascending=True)
        all_matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].sort_values('Date', ascending=True)
        
        if len(all_matches) < 5:
            continue
        
        # Basic stats
        home_gs = home_matches['FTHG'].mean() if len(home_matches) > 0 else 1.5
        home_gc = home_matches['FTAG'].mean() if len(home_matches) > 0 else 1.5
        away_gs = away_matches['FTAG'].mean() if len(away_matches) > 0 else 1.2
        away_gc = away_matches['FTHG'].mean() if len(away_matches) > 0 else 1.2
        
        # Win rates
        home_wins = len(home_matches[home_matches['FTR'] == 'H']) / max(len(home_matches), 1)
        away_wins = len(away_matches[away_matches['FTR'] == 'A']) / max(len(away_matches), 1)
        total_matches = len(all_matches)
        overall_win_rate = (len(home_matches[home_matches['FTR'] == 'H']) + 
                           len(away_matches[away_matches['FTR'] == 'A'])) / max(total_matches, 1)
        
        # Recent form (last 10 matches)
        recent = all_matches.tail(10)
        form_points = 0
        for _, match in recent.iterrows():
            if match['HomeTeam'] == team:
                if match['FTR'] == 'H':
                    form_points += 3
                elif match['FTR'] == 'D':
                    form_points += 1
            else:
                if match['FTR'] == 'A':
                    form_points += 3
                elif match['FTR'] == 'D':
                    form_points += 1
        
        # Goals in last 5
        recent5 = all_matches.tail(5)
        goals_last5 = 0
        for _, match in recent5.iterrows():
            if match['HomeTeam'] == team:
                goals_last5 += match['FTHG']
            else:
                goals_last5 += match['FTAG']
        
        # xG (expected goals) if available
        home_xg = home_matches['HxG'].mean() if 'HxG' in home_matches.columns and home_matches['HxG'].notna().any() else home_gs
        away_xg = away_matches['AxG'].mean() if 'AxG' in away_matches.columns and away_matches['AxG'].notna().any() else away_gs
        
        # Clean sheets
        home_cs = len(home_matches[home_matches['FTAG'] == 0]) / max(len(home_matches), 1)
        away_cs = len(away_matches[away_matches['FTHG'] == 0]) / max(len(away_matches), 1)
        
        team_stats[team] = {
            'home_gs': home_gs,
            'home_gc': home_gc,
            'away_gs': away_gs,
            'away_gc': away_gc,
            'home_win_rate': home_wins,
            'away_win_rate': away_wins,
            'overall_win_rate': overall_win_rate,
            'form_points': form_points,
            'goals_last5': goals_last5,
            'home_xg': home_xg,
            'away_xg': away_xg,
            'home_cs_rate': home_cs,
            'away_cs_rate': away_cs,
            'matches_played': total_matches
        }
    
    return team_stats

def get_head_to_head(df, team1, team2):
    """Get head-to-head statistics between two teams"""
    h2h = df[((df['HomeTeam'] == team1) & (df['AwayTeam'] == team2)) |
             ((df['HomeTeam'] == team2) & (df['AwayTeam'] == team1))]
    
    if len(h2h) == 0:
        return {'team1_wins': 0, 'team2_wins': 0, 'draws': 0, 'avg_goals': 2.5}
    
    team1_wins = 0
    team2_wins = 0
    draws = 0
    total_goals = 0
    
    for _, match in h2h.iterrows():
        if match['HomeTeam'] == team1:
            if match['FTR'] == 'H':
                team1_wins += 1
            elif match['FTR'] == 'A':
                team2_wins += 1
            else:
                draws += 1
        else:
            if match['FTR'] == 'A':
                team1_wins += 1
            elif match['FTR'] == 'H':
                team2_wins += 1
            else:
                draws += 1
        total_goals += match['FTHG'] + match['FTAG']
    
    return {
        'team1_wins': team1_wins,
        'team2_wins': team2_wins,
        'draws': draws,
        'avg_goals': total_goals / len(h2h)
    }

def build_features(df, team_stats):
    """Build feature matrix for training"""
    X = []
    y_home = []
    y_away = []
    
    for _, match in df.iterrows():
        home = match['HomeTeam']
        away = match['AwayTeam']
        
        if home not in team_stats or away not in team_stats:
            continue
        
        hs = team_stats[home]
        ast = team_stats[away]
        h2h = get_head_to_head(df, home, away)
        
        features = [
            hs['home_gs'], hs['home_gc'], hs['home_win_rate'],
            hs['form_points'] / 10.0,  # Normalize
            hs['goals_last5'] / 5.0,
            hs['home_xg'], hs['home_cs_rate'],
            ast['away_gs'], ast['away_gc'], ast['away_win_rate'],
            ast['form_points'] / 10.0,
            ast['goals_last5'] / 5.0,
            ast['away_xg'], ast['away_cs_rate'],
            h2h['team1_wins'] / max(h2h['team1_wins'] + h2h['team2_wins'] + h2h['draws'], 1),
            h2h['avg_goals'] / 5.0,
            hs['overall_win_rate'],
            ast['overall_win_rate']
        ]
        
        X.append(features)
        y_home.append(match['FTHG'])
        y_away.append(match['FTAG'])
    
    return np.array(X), np.array(y_home), np.array(y_away)

def train_models(df, team_stats):
    """Train prediction models"""
    print("Building feature matrix...")
    X, y_home, y_away = build_features(df, team_stats)
    
    print(f"Training on {len(X)} matches...")
    
    # Use Gradient Boosting for better accuracy
    model_home = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        random_state=42
    )
    
    model_away = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        random_state=42
    )
    
    model_home.fit(X, y_home)
    model_away.fit(X, y_away)
    
    # Calculate accuracy
    home_score = model_home.score(X, y_home)
    away_score = model_away.score(X, y_away)
    print(f"Model R² scores - Home: {home_score:.3f}, Away: {away_score:.3f}")
    
    return model_home, model_away

def predict_match(home, away, team_stats, models, df):
    """Predict match outcome"""
    if home not in team_stats or away not in team_stats:
        return None, None, 0
    
    hs = team_stats[home]
    ast = team_stats[away]
    h2h = get_head_to_head(df, home, away)
    
    features = np.array([[
        hs['home_gs'], hs['home_gc'], hs['home_win_rate'],
        hs['form_points'] / 10.0,
        hs['goals_last5'] / 5.0,
        hs['home_xg'], hs['home_cs_rate'],
        ast['away_gs'], ast['away_gc'], ast['away_win_rate'],
        ast['form_points'] / 10.0,
        ast['goals_last5'] / 5.0,
        ast['away_xg'], ast['away_cs_rate'],
        h2h['team1_wins'] / max(h2h['team1_wins'] + h2h['team2_wins'] + h2h['draws'], 1),
        h2h['avg_goals'] / 5.0,
        hs['overall_win_rate'],
        ast['overall_win_rate']
    ]])
    
    model_home, model_away = models # Unpack the models here
    
    home_pred = model_home.predict(features)[0]
    away_pred = model_away.predict(features)[0]
    
    # Round and ensure non-negative
    home_goals = max(0, round(home_pred))
    away_goals = max(0, round(away_pred))
    
    # Calculate confidence based on prediction variance
    home_preds_trees = np.array([tree[0].predict(features)[0] for tree in model_home.estimators_])
    away_preds_trees = np.array([tree[0].predict(features)[0] for tree in model_away.estimators_])
    
    home_std = np.std(home_preds_trees)
    away_std = np.std(away_preds_trees)
    avg_std = (home_std + away_std) / 2
    
    # Use std directly: map 0->80%, 0.1->65%, 0.3->35%, 0.5+->15%
    confidence = max(0.15, 0.80 - (avg_std * 1.5))
    
    return home_goals, away_goals, confidence

# Global storage
dc_model = None
df_global = None
last_updated = None

@app.before_request
def init_model():
    """Lazy load model on first request"""
    global dc_model, df_global, last_updated
    
    if dc_model is None:
        print("🔄 Initializing Dixon-Coles model...")
        df_global = fetch_extended_data()
        
        if df_global is not None and len(df_global) > 100:
            # Filter to only teams we have
            available_teams = [t for t in premier_league_teams if t in df_global['HomeTeam'].values or t in df_global['AwayTeam'].values]
            dc_model = SimplePoissonModel()
            dc_model.fit(df_global, available_teams)
            last_updated = datetime.now()
            print(f"✅ Dixon-Coles model ready! Trained on {len(df_global)} matches")
        else:
            print("⚠️ Could not load sufficient data")

@app.route('/')
def index():
    return render_template('index.html', teams=premier_league_teams)

@app.route('/api/teams')
def get_teams():
    return jsonify({'teams': premier_league_teams})

@app.route('/api/predict', methods=['POST'])
def predict():
    global dc_model, df_global
    
    data = request.json
    home = data.get('home_team')
    away = data.get('away_team')
    exclude_draw = data.get('exclude_draw', False)
    
    if not home or not away:
        return jsonify({'error': 'Please select both teams'}), 400
    
    if home == away:
        return jsonify({'error': 'Teams must be different'}), 400
    
    if dc_model is None:
        return jsonify({'error': 'Model not ready. Please try again in a moment.'}), 503
    
    try:
        result = dc_model.predict(home, away, exclude_draw=exclude_draw)
        
        if result is None:
            return jsonify({'error': 'Insufficient data for these teams'}), 400
        
        home_goals = result['home_goals']
        away_goals = result['away_goals']
        
        # Determine winner
        if home_goals > away_goals:
            winner = home
            result_type = "Home Win"
        elif away_goals > home_goals:
            winner = away
            result_type = "Away Win"
        else:
            winner = "Draw"
            result_type = "Draw"
        
        return jsonify({
            'home_team': home,
            'away_team': away,
            'predicted_home_goals': home_goals,
            'predicted_away_goals': away_goals,
            'predicted_score': f"{home_goals} - {away_goals}",
            'predicted_winner': winner,
            'result_type': result_type,
            'confidence': f"{result['confidence'] * 100:.0f}%",
            'home_prob': f"{result['home_prob'] * 100:.0f}%",
            'draw_prob': f"{result['draw_prob'] * 100:.0f}%",
            'away_prob': f"{result['away_prob'] * 100:.0f}%"
        })
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': 'Prediction failed. Please try again.'}), 500

@app.route('/api/stats/<team>')
def get_team_stats(team):
    if team_stats and team in team_stats:
        return jsonify(team_stats[team])
    return jsonify({'error': 'Team not found'}), 404

@app.route('/api/refresh', methods=['POST'])
def refresh_data():
    global dc_model, df_global, last_updated
    
    try:
        print("🔄 Refreshing data...")
        df_global = fetch_extended_data()
        
        if df_global is not None and len(df_global) > 100:
            available_teams = [t for t in premier_league_teams if t in df_global['HomeTeam'].values or t in df_global['AwayTeam'].values]
            dc_model = SimplePoissonModel()
            dc_model.fit(df_global, available_teams)
            last_updated = datetime.now()
            return jsonify({
                'success': True,
                'matches': len(df_global),
                'teams': dc_model.n_teams,
                'model_type': 'Simple Poisson',
                'last_updated': last_updated.strftime('%Y-%m-%d %H:%M')
            })
        else:
            return jsonify({'error': 'Could not load data'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status')
def get_status():
    return jsonify({
        'loaded': dc_model is not None,
        'model_type': 'Simple Poisson',
        'last_updated': last_updated.strftime('%Y-%m-%d %H:%M') if last_updated else None,
        'matches': len(df_global) if df_global is not None else 0,
        'teams': dc_model.n_teams if dc_model else 0
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
