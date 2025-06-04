from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import logging
import pandas as pd
import numpy as np
import os

# --- Logging Setup ---
os.makedirs('match_winner_output', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('match_winner_output/prediction.log')
    ]
)
logger = logging.getLogger(__name__)

# --- FastAPI App Setup ---
app = FastAPI(title="Cricket Prediction API")

# --- Load Models and Artifacts ---
try:
    match_winner_model = pickle.load(open('match_winner_output/match_winner_model.pkl', 'rb'))
    scaler = pickle.load(open('match_winner_output/scaler.pkl', 'rb'))
    selector = pickle.load(open('match_winner_output/selector.pkl', 'rb'))
    venue_map = pickle.load(open('match_winner_output/venue_mapping.pkl', 'rb'))
    player_agg = pd.read_csv('player_aggregate_stats.csv')
    team_agg = pd.read_csv('team_aggregate_stats.csv')
    team_venue = pd.read_csv('team_venue_aggregate_stats.csv')
    team_venue['venue'] = team_venue['venue'].str.strip()
    h2h_df = pd.read_csv('head_to_head_stats.csv', index_col=0)
    h2h = h2h_df.to_dict()
except Exception as e:
    logger.error(f"Error loading match winner prediction artifacts: {e}")
    raise Exception(f"Failed to load match winner prediction artifacts: {e}")

# --- Pydantic Model ---
class MatchInput(BaseModel):
    team: str
    opponent: str
    venue: str
    is_test_match: int
    toss_winner: int
    toss_decision_field: int
    is_tournament_match: int

# --- Helper Functions ---
def add_opponent_win_rate(df, team_df):
    opp = team_df[['team', 'win_rate']].rename(columns={'team': 'opponent', 'win_rate': 'opponent_win_rate'})
    df = df.merge(opp, on='opponent', how='left')
    df['team_win_rate'] = pd.to_numeric(df['team_win_rate'], errors='coerce')
    df['opponent_win_rate'] = pd.to_numeric(df['opponent_win_rate'], errors='coerce')
    df['team_win_rate_diff'] = df['team_win_rate'] - df['opponent_win_rate']
    return df

def build_player_stats(df):
    grouped = df.groupby('team')
    agg = grouped.agg({
        'runs_scored': 'mean',
        'strike_rate': 'mean',
        'wickets_taken': 'mean',
        'bowling_avg': lambda s: s.replace([np.inf, -np.inf], np.nan).mean(),
        'economy_rate': 'mean',
        'potm_rate': 'mean'
    }).rename(columns={
        'runs_scored': 'team_player_avg_runs',
        'strike_rate': 'team_player_avg_strike_rate',
        'wickets_taken': 'team_player_avg_wickets',
        'bowling_avg': 'team_player_avg_bowling_avg',
        'economy_rate': 'team_player_avg_economy_rate',
        'potm_rate': 'team_player_avg_potm_rate'
    }).reset_index()

    top_bats = (df.sort_values('runs_scored', ascending=False)
                .groupby('team').head(3)
                .groupby('team')['runs_scored'].mean()
                .rename('team_top_batsman_avg'))
    top_bwl = (df.sort_values('wickets_taken', ascending=False)
               .groupby('team').head(3)
               .groupby('team')['wickets_taken'].mean()
               .rename('team_top_bowler_wickets'))

    return agg.set_index('team').join(top_bats).join(top_bwl).reset_index()

# --- Prediction Endpoint ---
@app.post("/predict-match-winner")
async def predict_match_winner(input_data: MatchInput):
    df = pd.DataFrame([input_data.dict()])
    df['venue'] = df['venue'].str.strip()

    # Validate input columns
    for col in ['team', 'opponent', 'venue', 'is_test_match', 'toss_winner', 'toss_decision_field', 'is_tournament_match']:
        if col not in df:
            raise HTTPException(status_code=400, detail=f"Missing input field: {col}")

    # Merge features
    df = df.merge(team_agg.add_prefix('team_'), left_on='team', right_on='team_team', how='left').drop(columns='team_team')
    df = df.merge(
        team_venue[['team', 'venue', 'win_rate']].rename(columns={'win_rate': 'team_venue_win_rate'}),
        on=['team', 'venue'], how='left'
    )
    df = df.merge(build_player_stats(player_agg), on='team', how='left')
    df['head_to_head_wins'] = df.apply(lambda r: h2h.get(r['team'], {}).get(r['opponent'], 0), axis=1)
    df = add_opponent_win_rate(df, team_agg)
    df['is_close_match'] = (df['team_win_rate_diff'].abs() < 0.1).astype(int)
    df['recent_win_rate'] = df['team_win_rate'].fillna(0)

    # Impute & encode
    num_cols = df.select_dtypes(include='number').columns.difference([
        'is_test_match', 'toss_winner', 'toss_decision_field', 'is_tournament_match'
    ])
    for c in num_cols:
        df[c].fillna(df[c].median(), inplace=True)
    df['venue'] = df['venue'].map(venue_map).fillna(max(venue_map.values()) + 1)

    features = [
        'is_test_match', 'toss_winner', 'toss_decision_field', 'venue',
        'head_to_head_wins', 'is_tournament_match',
        'team_matches', 'team_wins', 'team_win_rate', 'team_avg_runs', 'team_avg_wickets',
        'team_toss_wins', 'team_toss_win_rate', 'team_field_first_rate',
        'team_player_avg_runs', 'team_player_avg_strike_rate', 'team_player_avg_wickets',
        'team_player_avg_bowling_avg', 'team_player_avg_economy_rate', 'team_player_avg_potm_rate',
        'team_top_batsman_avg', 'team_top_bowler_wickets', 'recent_win_rate',
        'team_venue_win_rate', 'team_win_rate_diff', 'is_close_match'
    ]
    X = df[features]
    try:
        X_scaled = scaler.transform(X)
        X_sel = selector.transform(X_scaled)
    except Exception as e:
        logger.error(f"Error transforming features: {e}")
        raise HTTPException(status_code=500, detail=f"Feature transformation error: {e}")

    try:
        pred_bin = match_winner_model.predict(X_sel)
        prob = match_winner_model.predict_proba(X_sel)[:, 1]
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    out = df[['team', 'opponent', 'venue']].copy()
    out['win_probability'] = prob
    out['predicted_winner'] = np.where(pred_bin == 1, out['team'], out['opponent'])
    inv_map = {v: k for k, v in venue_map.items()}
    out['venue'] = out['venue'].map(inv_map)

    return out.to_dict(orient='records')[0]

@app.get("/")
async def root():
    return {"message": "Cricket Prediction API is up!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)