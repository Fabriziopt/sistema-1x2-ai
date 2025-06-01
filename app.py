import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time
import os
import traceback
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import poisson
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="⚽ Sistema 1x2 AI - Vitórias em Casa",
    page_icon="🏠",
    layout="wide"
)

# Session state
if 'league_models_1x2' not in st.session_state:
    st.session_state.league_models_1x2 = {}
if 'models_trained_1x2' not in st.session_state:
    st.session_state.models_trained_1x2 = False
if 'training_errors_1x2' not in st.session_state:
    st.session_state.training_errors_1x2 = []
if 'historical_data_1x2' not in st.session_state:
    st.session_state.historical_data_1x2 = None

# API Configuration
try:
    API_KEY = st.secrets["API_KEY"]
except:
    API_KEY = "474f15de5b22951077ccb71b8d75b95c"

API_BASE_URL = "https://v3.football.api-sports.io"

def get_api_headers():
    return {'x-apisports-key': API_KEY}

def test_api_connection():
    try:
        headers = get_api_headers()
        response = requests.get(f'{API_BASE_URL}/status', headers=headers, timeout=10)
        return response.status_code == 200, f"Status: {response.status_code}"
    except Exception as e:
        return False, f"Erro: {str(e)}"

@st.cache_data(ttl=3600)
def get_fixtures_cached(date_str):
    try:
        headers = get_api_headers()
        response = requests.get(
            f'{API_BASE_URL}/fixtures',
            headers=headers,
            params={'date': date_str},
            timeout=30
        )
        if response.status_code == 200:
            return response.json().get('response', [])
    except:
        pass
    return []

def collect_historical_data_1x2():
    """Coleta dados históricos para 1x2"""
    
    if st.session_state.historical_data_1x2 is not None:
        return st.session_state.historical_data_1x2
    
    # Simular dados para demonstração (em produção real, usaria API)
    st.warning("📊 Gerando dados de demonstração...")
    
    np.random.seed(42)
    n_games = 8000
    
    data = []
    leagues = [39, 40, 78, 135, 61, 140, 71, 2, 253]  # Premier, La Liga, Bundesliga, etc.
    league_names = ['Premier League', 'La Liga', 'Bundesliga', 'Serie A', 'Ligue 1', 'Liga Portugal', 'Brasileirão', 'Champions League', 'MLS']
    
    for i in range(n_games):
        league_idx = np.random.randint(0, len(leagues))
        league_id = leagues[league_idx]
        league_name = league_names[league_idx]
        
        # Probabilidades base por liga
        league_home_win_probs = {
            39: 0.48,  # Premier League
            40: 0.52,  # La Liga
            78: 0.47,  # Bundesliga
            135: 0.49, # Serie A
            61: 0.51,  # Ligue 1
            140: 0.54, # Liga Portugal
            71: 0.56,  # Brasileirão
            2: 0.45,   # Champions League
            253: 0.50  # MLS
        }
        
        home_win_prob = league_home_win_probs.get(league_id, 0.5)
        
        # Adicionar variação por equipa
        home_strength = np.random.uniform(0.3, 1.8)
        away_strength = np.random.uniform(0.3, 1.8)
        
        # Ajustar probabilidade baseada na força
        adjusted_home_prob = home_win_prob * (home_strength / away_strength)
        adjusted_home_prob = max(0.2, min(0.8, adjusted_home_prob))
        
        # Resultado
        rand = np.random.random()
        if rand < adjusted_home_prob:
            result = 'H'  # Home win
            home_goals = np.random.poisson(1.8)
            away_goals = np.random.poisson(0.9)
        elif rand < adjusted_home_prob + 0.25:
            result = 'D'  # Draw
            avg_goals = np.random.poisson(1.2)
            home_goals = away_goals = avg_goals
        else:
            result = 'A'  # Away win
            home_goals = np.random.poisson(0.8)
            away_goals = np.random.poisson(1.6)
        
        # Garantir que o resultado seja consistente com os gols
        if home_goals > away_goals:
            result = 'H'
        elif away_goals > home_goals:
            result = 'A'
        else:
            result = 'D'
        
        # Data aleatória nos últimos 2 anos
        days_ago = np.random.randint(1, 730)
        game_date = datetime.now() - timedelta(days=days_ago)
        
        data.append({
            'date': game_date.strftime('%Y-%m-%d'),
            'league_id': league_id,
            'league_name': league_name,
            'home_team_id': np.random.randint(1, 200),
            'away_team_id': np.random.randint(1, 200),
            'home_team': f"Team_H_{np.random.randint(1, 50)}",
            'away_team': f"Team_A_{np.random.randint(1, 50)}",
            'home_goals': max(0, home_goals),
            'away_goals': max(0, away_goals),
            'result': result,
            'home_strength': home_strength,
            'away_strength': away_strength
        })
    
    df = pd.DataFrame(data)
    st.session_state.historical_data_1x2 = df
    
    st.success(f"✅ {len(df)} jogos gerados para demonstração")
    return df

def calculate_team_stats_1x2(league_df):
    """Calcula estatísticas específicas para 1x2"""
    team_stats = {}
    
    # Análise por time
    unique_teams = pd.concat([league_df['home_team_id'], league_df['away_team_id']]).unique()
    
    for team_id in unique_teams:
        # Jogos em casa
        home_matches = league_df[league_df['home_team_id'] == team_id]
        # Jogos fora
        away_matches = league_df[league_df['away_team_id'] == team_id]
        # Todos os jogos
        all_matches = pd.concat([home_matches, away_matches])
        
        if len(all_matches) == 0:
            continue
        
        team_name = home_matches.iloc[0]['home_team'] if len(home_matches) > 0 else away_matches.iloc[0]['away_team']
        
        # Estatísticas em casa
        home_wins = len(home_matches[home_matches['result'] == 'H'])
        home_draws = len(home_matches[home_matches['result'] == 'D'])
        home_losses = len(home_matches[home_matches['result'] == 'A'])
        home_win_rate = home_wins / len(home_matches) if len(home_matches) > 0 else 0
        
        home_goals_scored = home_matches['home_goals'].mean() if len(home_matches) > 0 else 0
        home_goals_conceded = home_matches['away_goals'].mean() if len(home_matches) > 0 else 0
        
        # Estatísticas fora
        away_wins = len(away_matches[away_matches['result'] == 'A'])
        away_draws = len(away_matches[away_matches['result'] == 'D'])
        away_losses = len(away_matches[away_matches['result'] == 'H'])
        away_win_rate = away_wins / len(away_matches) if len(away_matches) > 0 else 0
        
        away_goals_scored = away_matches['away_goals'].mean() if len(away_matches) > 0 else 0
        away_goals_conceded = away_matches['home_goals'].mean() if len(away_matches) > 0 else 0
        
        # Força ofensiva/defensiva
        league_avg_goals = league_df[['home_goals', 'away_goals']].mean().mean()
        
        team_stats[team_id] = {
            'team_name': team_name,
            'total_games': len(all_matches),
            
            # Casa
            'home_games': len(home_matches),
            'home_wins': home_wins,
            'home_draws': home_draws,
            'home_losses': home_losses,
            'home_win_rate': home_win_rate,
            'home_goals_scored': home_goals_scored,
            'home_goals_conceded': home_goals_conceded,
            'home_attack_strength': home_goals_scored / (league_avg_goals + 0.01),
            'home_defense_strength': home_goals_conceded / (league_avg_goals + 0.01),
            
            # Fora
            'away_games': len(away_matches),
            'away_wins': away_wins,
            'away_draws': away_draws,
            'away_losses': away_losses,
            'away_win_rate': away_win_rate,
            'away_goals_scored': away_goals_scored,
            'away_goals_conceded': away_goals_conceded,
            'away_attack_strength': away_goals_scored / (league_avg_goals + 0.01),
            'away_defense_strength': away_goals_conceded / (league_avg_goals + 0.01),
            
            # Geral
            'overall_win_rate': (home_wins + away_wins) / len(all_matches) if len(all_matches) > 0 else 0
        }
    
    return team_stats

def calculate_features_1x2(league_df, team_stats):
    """Calcula features para modelo 1x2"""
    features = []
    
    # Estatísticas da liga
    league_home_win_rate = len(league_df[league_df['result'] == 'H']) / len(league_df)
    league_draw_rate = len(league_df[league_df['result'] == 'D']) / len(league_df)
    league_away_win_rate = len(league_df[league_df['result'] == 'A']) / len(league_df)
    
    for idx, row in league_df.iterrows():
        home_id = row['home_team_id']
        away_id = row['away_team_id']
        
        if home_id not in team_stats or away_id not in team_stats:
            continue
        
        home_stats = team_stats[home_id]
        away_stats = team_stats[away_id]
        
        # Poisson para 1x2
        home_expected = home_stats['home_attack_strength'] * away_stats['away_defense_strength'] * league_df['home_goals'].mean()
        away_expected = away_stats['away_attack_strength'] * home_stats['home_defense_strength'] * league_df['away_goals'].mean()
        
        # Probabilidades Poisson
        poisson_home_win = calculate_poisson_1x2(home_expected, away_expected)['home_win_prob']
        poisson_draw = calculate_poisson_1x2(home_expected, away_expected)['draw_prob']
        poisson_away_win = calculate_poisson_1x2(home_expected, away_expected)['away_win_prob']
        
        # Features para o modelo
        feature_row = {
            # Taxas de vitória
            'home_win_rate': home_stats['home_win_rate'],
            'away_win_rate': away_stats['away_win_rate'],
            'home_overall_win_rate': home_stats['overall_win_rate'],
            'away_overall_win_rate': away_stats['overall_win_rate'],
            
            # Força casa vs fora
            'home_attack_strength': home_stats['home_attack_strength'],
            'home_defense_strength': home_stats['home_defense_strength'],
            'away_attack_strength': away_stats['away_attack_strength'],
            'away_defense_strength': away_stats['away_defense_strength'],
            
            # Gols médios
            'home_goals_scored_avg': home_stats['home_goals_scored'],
            'home_goals_conceded_avg': home_stats['home_goals_conceded'],
            'away_goals_scored_avg': away_stats['away_goals_scored'],
            'away_goals_conceded_avg': away_stats['away_goals_conceded'],
            
            # Comparação com liga
            'home_win_vs_league': home_stats['home_win_rate'] / (league_home_win_rate + 0.01),
            'away_win_vs_league': away_stats['away_win_rate'] / (league_away_win_rate + 0.01),
            
            # Poisson
            'poisson_home_win': poisson_home_win,
            'poisson_draw': poisson_draw,
            'poisson_away_win': poisson_away_win,
            
            # Confronto direto
            'attack_vs_defense': home_stats['home_attack_strength'] / (away_stats['away_defense_strength'] + 0.01),
            'defense_vs_attack': home_stats['home_defense_strength'] / (away_stats['away_attack_strength'] + 0.01),
            
            # Experiência
            'home_games_played': home_stats['home_games'],
            'away_games_played': away_stats['away_games'],
            'experience_factor': min(home_stats['home_games'], away_stats['away_games']),
            
            # Target
            'target': row['result']
        }
        
        features.append(feature_row)
    
    return pd.DataFrame(features)

def calculate_poisson_1x2(home_lambda, away_lambda):
    """Calcula probabilidades 1x2 usando Poisson"""
    try:
        home_lambda = max(home_lambda, 0.1)
        away_lambda = max(away_lambda, 0.1)
        
        # Calcular probabilidades para diferentes resultados
        home_win_prob = 0
        draw_prob = 0
        away_win_prob = 0
        
        # Simular até 6 gols para cada time
        for home_goals in range(7):
            for away_goals in range(7):
                prob = poisson.pmf(home_goals, home_lambda) * poisson.pmf(away_goals, away_lambda)
                
                if home_goals > away_goals:
                    home_win_prob += prob
                elif home_goals == away_goals:
                    draw_prob += prob
                else:
                    away_win_prob += prob
        
        # Normalizar para garantir que soma = 1
        total = home_win_prob + draw_prob + away_win_prob
        if total > 0:
            home_win_prob /= total
            draw_prob /= total
            away_win_prob /= total
        
        return {
            'home_win_prob': home_win_prob,
            'draw_prob': draw_prob,
            'away_win_prob': away_win_prob
        }
    except:
        return {'home_win_prob': 0.33, 'draw_prob': 0.33, 'away_win_prob': 0.33}

def train_model_1x2(league_df, league_id, league_name, min_matches=30):
    """Treina modelo 1x2 para uma liga"""
    
    if len(league_df) < min_matches:
        return None, f"❌ {league_name}: {len(league_df)} jogos < {min_matches} mínimo"
    
    try:
        # Calcular estatísticas dos times
        team_stats = calculate_team_stats_1x2(league_df)
        
        # Calcular features
        features_df = calculate_features_1x2(league_df, team_stats)
        
        if features_df.empty or len(features_df) < min_matches:
            return None, f"❌ {league_name}: Features insuficientes"
        
        # Preparar dados
        feature_cols = [col for col in features_df.columns if col != 'target']
        X = features_df[feature_cols]
        y = features_df['target']
        
        # Codificar target (H, D, A -> 0, 1, 2)
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Verificar se temos todas as classes
        if len(np.unique(y_encoded)) < 2:
            return None, f"❌ {league_name}: Falta variação nos resultados"
        
        # Tratar NaN
        X = X.fillna(X.median())
        
        # Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Escalar
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Modelos
        models = {
            'rf': RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
            'gb': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        best_model = None
        best_score = 0
        results = {}
        
        for name, model in models.items():
            try:
                # Treinar
                calibrated_model = CalibratedClassifierCV(model, cv=3)
                calibrated_model.fit(X_train_scaled, y_train)
                
                # Testar
                test_pred = calibrated_model.predict(X_test_scaled)
                test_acc = accuracy_score(y_test, test_pred)
                
                results[name] = {
                    'accuracy': test_acc,
                    'model': calibrated_model
                }
                
                if test_acc > best_score:
                    best_score = test_acc
                    best_model = calibrated_model
                    
            except Exception as e:
                continue
        
        if best_model is None:
            return None, f"❌ {league_name}: Nenhum modelo funcionou"
        
        # Retreinar no dataset completo
        X_scaled = scaler.fit_transform(X)
        best_model.fit(X_scaled, y_encoded)
        
        # Calcular estatísticas da liga
        league_stats = {
            'home_win_rate': len(league_df[league_df['result'] == 'H']) / len(league_df),
            'draw_rate': len(league_df[league_df['result'] == 'D']) / len(league_df),
            'away_win_rate': len(league_df[league_df['result'] == 'A']) / len(league_df)
        }
        
        # Dados do modelo
        model_data = {
            'model': best_model,
            'scaler': scaler,
            'label_encoder': label_encoder,
            'feature_cols': feature_cols,
            'team_stats': team_stats,
            'league_id': league_id,
            'league_name': league_name,
            'league_stats': league_stats,
            'total_matches': len(league_df),
            'test_accuracy': best_score
        }
        
        return model_data, f"✅ {league_name}: Acc {best_score:.1%}"
        
    except Exception as e:
        error_msg = f"❌ {league_name}: {str(e)}"
        st.session_state.training_errors_1x2.append(error_msg)
        return None, error_msg

def predict_1x2_home_wins(fixtures, league_models, min_home_prob=0.6, min_odds=1.5, max_odds=2.25):
    """Prediz vitórias em casa com critérios específicos"""
    
    if not league_models:
        return []
    
    predictions = []
    
    for fixture in fixtures:
        try:
            if fixture['fixture']['status']['short'] not in ['NS', 'TBD']:
                continue
            
            league_id = fixture['league']['id']
            
            if league_id not in league_models:
                continue
            
            model_data = league_models[league_id]
            model = model_data['model']
            scaler = model_data['scaler']
            label_encoder = model_data['label_encoder']
            feature_cols = model_data['feature_cols']
            team_stats = model_data['team_stats']
            league_stats = model_data['league_stats']
            
            home_id = fixture['teams']['home']['id']
            away_id = fixture['teams']['away']['id']
            
            if home_id not in team_stats or away_id not in team_stats:
                continue
            
            home_stats = team_stats[home_id]
            away_stats = team_stats[away_id]
            
            # Calcular features para predição
            home_expected = home_stats['home_attack_strength'] * away_stats['away_defense_strength'] * 1.5
            away_expected = away_stats['away_attack_strength'] * home_stats['home_defense_strength'] * 1.2
            
            poisson_data = calculate_poisson_1x2(home_expected, away_expected)
            
            features = {}
            for col in feature_cols:
                if col == 'home_win_rate':
                    features[col] = home_stats['home_win_rate']
                elif col == 'away_win_rate':
                    features[col] = away_stats['away_win_rate']
                elif col == 'poisson_home_win':
                    features[col] = poisson_data['home_win_prob']
                elif col == 'home_attack_strength':
                    features[col] = home_stats['home_attack_strength']
                elif col == 'away_defense_strength':
                    features[col] = away_stats['away_defense_strength']
                else:
                    features[col] = 0.5  # Valor padrão
            
            # Predição
            X = pd.DataFrame([features])[feature_cols]
            X = X.fillna(0.5)
            X_scaled = scaler.transform(X)
            
            pred_proba = model.predict_proba(X_scaled)[0]
            
            # Mapear probabilidades (assumindo ordem H, D, A)
            classes = label_encoder.classes_
            home_idx = np.where(classes == 'H')[0]
            
            if len(home_idx) > 0:
                home_win_prob = pred_proba[home_idx[0]]
            else:
                home_win_prob = pred_proba[0]  # Fallback
            
            # Calcular odds implícitas
            implied_odds = 1 / home_win_prob if home_win_prob > 0 else 99
            
            # Aplicar critérios: prob >= 60% e odds entre 1.5-2.25
            meets_criteria = (
                home_win_prob >= min_home_prob and
                min_odds <= implied_odds <= max_odds
            )
            
            if meets_criteria:
                prediction = {
                    'home_team': fixture['teams']['home']['name'],
                    'away_team': fixture['teams']['away']['name'],
                    'league': fixture['league']['name'],
                    'country': fixture['league']['country'],
                    'kickoff': fixture['fixture']['date'],
                    'prediction': 'CASA VENCE (1)',
                    'home_win_probability': home_win_prob * 100,
                    'implied_odds': implied_odds,
                    'home_win_rate_season': home_stats['home_win_rate'] * 100,
                    'away_loss_rate_season': (1 - away_stats['away_win_rate']) * 100,
                    'league_home_win_rate': league_stats['home_win_rate'] * 100,
                    'poisson_home_prob': poisson_data['home_win_prob'] * 100,
                    'confidence_score': (home_win_prob - min_home_prob) / (1 - min_home_prob) * 100,
                    'home_attack_strength': home_stats['home_attack_strength'],
                    'away_defense_weakness': 2 - away_stats['away_defense_strength'],
                    'fixture_id': fixture['fixture']['id']
                }
                predictions.append(prediction)
            
        except Exception as e:
            continue
    
    # Ordenar por probabilidade de vitória em casa
    predictions.sort(key=lambda x: x['home_win_probability'], reverse=True)
    
    return predictions

def display_1x2_prediction(pred):
    """Exibe previsão 1x2 focada em vitórias da casa"""
    
    try:
        with st.container():
            # Header
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.subheader(f"🏠 {pred['home_team']} vs {pred['away_team']}")
                st.caption(f"🏆 {pred['league']} ({pred['country']})")
            
            with col2:
                prob = pred['home_win_probability']
                if prob >= 75:
                    st.success(f"**{prob:.1f}%**")
                    st.caption("🔥 Muito Alta")
                elif prob >= 65:
                    st.info(f"**{prob:.1f}%**")
                    st.caption("✅ Alta")
                else:
                    st.warning(f"**{prob:.1f}%**")
                    st.caption("⚠️ Moderada")
            
            with col3:
                odds = pred['implied_odds']
                st.metric("Odds Implícitas", f"{odds:.2f}")
                if 1.5 <= odds <= 1.8:
                    st.caption("🎯 Valor Excelente")
                elif 1.8 < odds <= 2.1:
                    st.caption("✅ Bom Valor")
                else:
                    st.caption("⚠️ Valor OK")
            
            # Análise detalhada
            st.markdown("### 📊 Análise da Vitória em Casa")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**🏠 Equipa da Casa:**")
                st.write(f"- Taxa vitórias casa: {pred['home_win_rate_season']:.1f}%")
                st.write(f"- Força ataque: {pred['home_attack_strength']:.2f}")
                st.write(f"- Experiência casa: {pred.get('home_games', 'N/A')} jogos")
            
            with col2:
                st.write("**✈️ Equipa Visitante:**")
                st.write(f"- Taxa derrotas fora: {pred['away_loss_rate_season']:.1f}%")
                st.write(f"- Fraqueza defesa: {pred['away_defense_weakness']:.2f}")
                st.write(f"- Experiência fora: {pred.get('away_games', 'N/A')} jogos")
            
            with col3:
                st.write("**🏆 Contexto da Liga:**")
                st.write(f"- Taxa casa liga: {pred['league_home_win_rate']:.1f}%")
                st.write(f"- Poisson casa: {pred['poisson_home_prob']:.1f}%")
                st.write(f"- Score confiança: {pred['confidence_score']:.1f}")
            
            # Recomendação
            st.markdown("### 🎯 Recomendação de Aposta")
            
            prob = pred['home_win_probability']
            odds = pred['implied_odds']
            confidence = pred['confidence_score']
            
            if prob >= 70 and 1.5 <= odds <= 1.9 and confidence >= 25:
                st.success("✅ **APOSTAR FORTE: CASA VENCE (1)**")
                st.write("🔥 **Todos os critérios são excelentes!**")
                st.write(f"💰 **Valor da aposta:** Odds {odds:.2f} para {prob:.1f}% de probabilidade")
            elif prob >= 65 and odds <= 2.1:
                st.info("📊 **APOSTAR: CASA VENCE (1)**")
                st.write("✅ **Critérios favoráveis**")
                st.write(f"💰 **Aposta de valor:** Odds {odds:.2f}")
            else:
                st.warning("⚠️ **CONSIDERAR: CASA VENCE (1)**")
                st.write("⚠️ **Apostar com cautela**")
            
            # Análise de risco
            risk_factors = []
            if pred['home_win_rate_season'] < 50:
                risk_factors.append("🔴 Taxa de vitórias em casa abaixo de 50%")
            if pred['away_loss_rate_season'] < 40:
                risk_factors.append("🔴 Visitante não perde muito fora")
            if odds > 2.1:
                risk_factors.append("🟡 Odds um pouco altas para o perfil")
            
            if risk_factors:
                st.warning("⚠️ **Fatores de Risco:**")
                for factor in risk_factors:
                    st.write(f"- {factor}")
            else:
                st.success("✅ **Baixo Risco - Perfil ideal para vitória da casa**")
            
            st.markdown("---")
            
    except Exception as e:
        st.error(f"❌ Erro ao exibir previsão: {str(e)}")

def create_excel_download(df, filename):
    """Cria download Excel"""
    try:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Dados')
        output.seek(0)
        return output.getvalue()
    except:
        return None

def main():
    st.title("🏠 Sistema 1x2 AI - Vitórias em Casa")
    st.markdown("🎯 **Foco: Casa Vence | Odds: 1.50-2.25 | Probabilidade: +60%**")
    
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Configurações 1x2")
        
        # Status API
        conn_ok, conn_msg = test_api_connection()
        if conn_ok:
            st.success("✅ API conectada")
        else:
            st.error(f"❌ {conn_msg}")
        
        # Status modelos
        if st.session_state.models_trained_1x2:
            st.success(f"✅ {len(st.session_state.league_models_1x2)} ligas treinadas")
        else:
            st.warning("⚠️ Modelos não treinados")
        
        st.markdown("---")
        
        # Parâmetros específicos 1x2
        st.markdown("### 🎯 Critérios Casa Vence")
        
        min_home_prob = st.slider(
            "Probabilidade mínima casa:",
            min_value=0.5,
            max_value=0.8,
            value=0.6,
            step=0.05,
            help="60% = sua estratégia inicial"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            min_odds = st.number_input("Odds mínima:", value=1.5, min_value=1.1, max_value=3.0, step=0.05)
        with col2:
            max_odds = st.number_input("Odds máxima:", value=2.25, min_value=1.5, max_value=5.0, step=0.05)
        
        min_matches = st.slider("Mínimo jogos por liga:", 20, 100, 30)
        
        # Indicadores da estratégia
        st.markdown("### ✅ Estratégia Implementada")
        st.success("✅ Foco em vitórias da casa")
        st.success("✅ Odds entre 1.50-2.25")
        st.success("✅ Probabilidade +60%")
        st.success("✅ ML específico para 1x2")
        st.success("✅ Análise Poisson 1x2")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["🤖 Treinar 1x2", "📊 Análise Ligas", "🏠 Previsões Casa"])
    
    with tab1:
        st.header("🤖 Treinamento Modelo 1x2")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("""
            **✅ Modelo 1x2 Específico:**
            - Classificação multiclasse (H/D/A)
            - Foco em vitórias da casa
            - Análise Poisson para 1x2
            - Features casa vs fora
            - Critérios de odds e probabilidade
            """)
        
        with col2:
            st.success("""
            **🎯 Sua Estratégia:**
            - Casa vence (1)
            - Odds 1.50 - 2.25
            - Probabilidade ≥ 60%
            - ML calibrado
            - Valor nas apostas
            """)
        
        if st.button("🚀 TREINAR MODELO 1x2", type="primary", use_container_width=True):
            st.session_state.training_errors_1x2 = []
            
            try:
                with st.spinner("📊 Carregando dados 1x2..."):
                    df = collect_historical_data_1x2()
                
                if df.empty:
                    st.error("❌ Sem dados para treinamento")
                    st.stop()
                
                # Agrupar por liga
                league_groups = df.groupby(['league_id', 'league_name'])
                
                st.info(f"🎯 {len(league_groups)} ligas encontradas")
                
                league_models = {}
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                successful_leagues = 0
                
                for idx, ((league_id, league_name), league_df) in enumerate(league_groups):
                    progress = (idx + 1) / len(league_groups)
                    progress_bar.progress(progress)
                    
                    status_text.text(f"🔄 Treinando: {league_name}")
                    
                    if len(league_df) < min_matches:
                        continue
                    
                    model_data, message = train_model_1x2(
                        league_df, league_id, league_name, min_matches
                    )
                    
                    if model_data:
                        league_models[league_id] = model_data
                        successful_leagues += 1
                        st.success(message)
                    else:
                        st.warning(message)
                
                progress_bar.empty()
                status_text.empty()
                
                if league_models:
                    st.session_state.league_models_1x2 = league_models
                    st.session_state.models_trained_1x2 = True
                    
                    st.success(f"🎉 {len(league_models)} ligas treinadas para 1x2!")
                    
                    # Estatísticas
                    avg_acc = np.mean([m['test_accuracy'] for m in league_models.values()])
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Ligas Treinadas", len(league_models))
                    with col2:
                        st.metric("Acurácia Média", f"{avg_acc:.1%}")
                    
                    st.balloons()
                else:
                    st.error("❌ Nenhuma liga treinada!")
                    
            except Exception as e:
                st.error(f"❌ Erro no treinamento: {str(e)}")
                st.code(traceback.format_exc())
    
    with tab2:
        if not st.session_state.models_trained_1x2:
            st.warning("⚠️ Treine o modelo primeiro!")
        else:
            st.header("📊 Análise das Ligas - 1x2")
            
            # Dados das ligas
            league_data = []
            for league_id, model_data in st.session_state.league_models_1x2.items():
                league_stats = model_data['league_stats']
                league_data.append({
                    'Liga': model_data['league_name'],
                    'Casa Vence %': round(league_stats['home_win_rate'] * 100, 1),
                    'Empate %': round(league_stats['draw_rate'] * 100, 1),
                    'Fora Vence %': round(league_stats['away_win_rate'] * 100, 1),
                    'Jogos': model_data['total_matches'],
                    'Acurácia': round(model_data['test_accuracy'] * 100, 1)
                })
            
            df_leagues = pd.DataFrame(league_data)
            
            # Visualizações
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🏠 Top Ligas - Vitórias Casa")
                top_home = df_leagues.sort_values('Casa Vence %', ascending=False).head(10)
                chart_data = top_home.set_index('Liga')['Casa Vence %']
                st.bar_chart(chart_data)
            
            with col2:
                st.subheader("🎯 Acurácia dos Modelos")
                performance_data = df_leagues.set_index('Liga')['Acurácia']
                st.line_chart(performance_data)
            
            # Tabela completa
            st.subheader("📋 Resumo Completo")
            st.dataframe(df_leagues.sort_values('Casa Vence %', ascending=False), use_container_width=True)
            
            # Download
            excel_data = create_excel_download(df_leagues, "analise_1x2.xlsx")
            if excel_data:
                st.download_button(
                    label="📥 Download Análise 1x2",
                    data=excel_data,
                    file_name=f"analise_1x2_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    
    with tab3:
        st.header("🏠 Previsões - Casa Vence")
        
        if not st.session_state.models_trained_1x2:
            st.warning("⚠️ Treine o modelo primeiro!")
            st.stop()
        
        # Critérios atuais
        st.success(f"🎯 **Critérios:** Prob ≥ {min_home_prob:.0%} | Odds {min_odds}-{max_odds}")
        
        selected_date = st.date_input("📅 Data:", value=datetime.now().date())
        date_str = selected_date.strftime('%Y-%m-%d')
        
        with st.spinner("🔍 Analisando jogos..."):
            fixtures = get_fixtures_cached(date_str)
        
        if not fixtures:
            st.info("📅 Nenhum jogo encontrado")
        else:
            st.info(f"🔍 {len(fixtures)} jogos para análise")
            
            predictions = predict_1x2_home_wins(
                fixtures,
                st.session_state.league_models_1x2,
                min_home_prob,
                min_odds,
                max_odds
            )
            
            if not predictions:
                st.info("🤷 Nenhuma previsão atende aos critérios")
                st.write("**Possíveis motivos:**")
                st.write("• Probabilidades abaixo de 60%")
                st.write("• Odds fora do range 1.50-2.25")
                st.write("• Times não nos dados de treinamento")
            else:
                st.success(f"🏠 **{len(predictions)} VITÓRIAS EM CASA ENCONTRADAS!**")
                
                # Estatísticas
                avg_prob = np.mean([p['home_win_probability'] for p in predictions])
                avg_odds = np.mean([p['implied_odds'] for p in predictions])
                high_prob = len([p for p in predictions if p['home_win_probability'] >= 70])
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Apostas", len(predictions))
                with col2:
                    st.metric("Prob. Média", f"{avg_prob:.1f}%")
                with col3:
                    st.metric("Odds Média", f"{avg_odds:.2f}")
                with col4:
                    st.metric("Alta Prob. (≥70%)", high_prob)
                
                # Export
                export_data = []
                for p in predictions:
                    export_data.append({
                        'Data': p['kickoff'].split('T')[0],
                        'Liga': p['league'],
                        'Casa': p['home_team'],
                        'Fora': p['away_team'],
                        'Probabilidade %': f"{p['home_win_probability']:.1f}",
                        'Odds Implícitas': f"{p['implied_odds']:.2f}",
                        'Taxa Casa %': f"{p['home_win_rate_season']:.1f}",
                        'Confiança': f"{p['confidence_score']:.1f}"
                    })
                
                export_df = pd.DataFrame(export_data)
                excel_data = create_excel_download(export_df, "previsoes_1x2.xlsx")
                
                if excel_data:
                    st.download_button(
                        label="📥 Exportar Previsões 1x2",
                        data=excel_data,
                        file_name=f"previsoes_1x2_{date_str}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                st.markdown("---")
                
                # Mostrar previsões
                for pred in predictions:
                    display_1x2_prediction(pred)

if __name__ == "__main__":
    main()
