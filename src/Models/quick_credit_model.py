"""
CredTech Hackathon - Quick Credit Scoring Model
Build a prototype explainable credit scoring model
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns

class QuickCreditScorer:
    """Quick prototype credit scoring model with explainability"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.explainer = None
        self.feature_names = None
        
    def load_data(self, file_path='data/processed/companies_summary.csv'):
        """Load the collected data"""
        try:
            df = pd.read_csv(file_path)
            print(f"📊 Loaded data: {df.shape[0]} companies, {df.shape[1]} features")
            return df
        except FileNotFoundError:
            print("❌ Data file not found. Please run data collection first!")
            return None
    
    def engineer_features(self, df):
        """Create additional features for credit scoring"""
        df = df.copy()
        
        # Handle missing values
        df = df.fillna(0)
        
        # Financial strength indicators
        df['financial_strength'] = (
            (df['roa'] * 0.3) + 
            (df['profit_margin'] * 0.3) + 
            ((1 / (1 + df['debt_ratio'])) * 0.4)  # Lower debt is better
        )
        
        # Market stability (inverse of beta)
        df['market_stability'] = 1 / (1 + df['beta'])
        
        # News sentiment weighted by volume
        df['sentiment_weighted'] = df['news_sentiment'] * np.log1p(df['news_volume'])
        
        # PE ratio health (moderate PE is good)
        df['pe_health'] = np.where(
            (df['pe_ratio'] > 5) & (df['pe_ratio'] < 25), 
            1.0 - abs(df['pe_ratio'] - 15) / 15,  # Optimal around 15
            0.5 if df['pe_ratio'] == 0 else 0.3  # Very high or very low PE is risky
        )
        
        # Dividend reliability
        df['dividend_reliability'] = np.where(df['dividend_yield'] > 0, 1, 0)
        
        print("✅ Feature engineering completed")
        return df
    
    def create_credit_rating(self, df):
        """Create credit rating target variable"""
        df = df.copy()
        
        # Create composite credit score (0-1 scale)
        credit_score = (
            df['financial_strength'] * 0.35 +
            df['market_stability'] * 0.15 +
            df['sentiment_weighted'] * 0.15 +
            df['pe_health'] * 0.20 +
            df['dividend_reliability'] * 0.15
        )
        
        # Normalize to 0-1 range
        credit_score = (credit_score - credit_score.min()) / (credit_score.max() - credit_score.min())
        
        # Convert to credit rating categories
        df['credit_score_raw'] = credit_score
        df['credit_rating'] = pd.cut(
            credit_score,
            bins=[0, 0.25, 0.50, 0.75, 1.0],
            labels=['D', 'C', 'B', 'A'],
            include_lowest=True
        )
        
        print("✅ Credit ratings created:")
        print(df['credit_rating'].value_counts())
        
        return df
    
    def prepare_features(self, df):
        """Prepare features for model training"""
        feature_columns = [
            'market_cap', 'pe_ratio', 'roa', 'profit_margin', 'debt_ratio',
            'beta', 'dividend_yield', 'news_sentiment', 'news_volume',
            'financial_strength', 'market_stability', 'sentiment_weighted',
            'pe_health', 'dividend_reliability'
        ]
        
        X = df[feature_columns].copy()
        y = df['credit_rating']
        
        # Handle any remaining missing values
        X = X.fillna(0)
        
        # Log transform market cap for better distribution
        X['market_cap_log'] = np.log1p(X['market_cap'])
        X = X.drop('market_cap', axis=1)
        
        self.feature_names = X.columns.tolist()
        
        print(f"✅ Features prepared: {X.shape[1]} features")
        return X, y
    
    def train_model(self, X, y):
        """Train the credit scoring model"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Try multiple models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBClassifier(random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42)
        }
        
        best_score = 0
        best_model = None
        best_name = ""
        
        print("🔄 Training multiple models...")
        for name, model in models.items():
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=3)
            avg_score = cv_scores.mean()
            
            print(f"   {name}: {avg_score:.3f} (+/- {cv_scores.std() * 2:.3f})")
            
            if avg_score > best_score:
                best_score = avg_score
                best_model = model
                best_name = name
        
        # Train best model
        self.model = best_model
        self.model.fit(X_train_scaled, y_train)
        
        # Test performance
        test_score = self.model.score(X_test_scaled, y_test)
        predictions = self.model.predict(X_test_scaled)
        
        print(f"\\n✅ Best model: {best_name}")
        print(f"📊 Test accuracy: {test_score:.3f}")
        print(f"📊 Cross-val score: {best_score:.3f}")
        
        # Classification report
        print("\\n📋 Classification Report:")
        print(classification_report(y_test, predictions))
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def setup_explainability(self, X_sample):
        """Set up SHAP explainer"""
        if isinstance(self.model, (RandomForestClassifier, GradientBoostingClassifier)):
            self.explainer = shap.TreeExplainer(self.model)
        else:  # XGBoost
            self.explainer = shap.Explainer(self.model)
        
        # Calculate SHAP values for sample
        shap_values = self.explainer.shap_values(X_sample)
        
        print("✅ SHAP explainer set up")
        return shap_values
    
    def explain_predictions(self, X_sample, company_names=None):
        """Generate explanations for predictions"""
        if self.explainer is None:
            print("❌ Explainer not set up. Run setup_explainability first.")
            return
        
        # Get SHAP values
        shap_values = self.explainer.shap_values(X_sample)
        
        # If multi-class, use the values for the positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[-1]  # Use highest class
        
        # Feature importance
        feature_importance = np.abs(shap_values).mean(0)
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        print("🔍 Feature Importance Ranking:")
        for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
            print(f"   {i+1}. {row['feature']}: {row['importance']:.4f}")
        
        return shap_values, importance_df
    
    def predict_with_explanation(self, company_data):
        """Predict credit rating with explanation for a single company"""
        if self.model is None:
            print("❌ Model not trained yet!")
            return None
        
        # Prepare features
        features = np.array([company_data]).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        
        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]
        
        # Get explanation
        if self.explainer:
            shap_values = self.explainer.shap_values(features_scaled)
            if isinstance(shap_values, list):
                shap_values = shap_values[-1][0]  # Use highest class
            else:
                shap_values = shap_values[0]
            
            # Top contributing features
            feature_contributions = list(zip(self.feature_names, shap_values))
            feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        
        return {
            'predicted_rating': prediction,
            'probability_distribution': dict(zip(self.model.classes_, probability)),
            'top_factors': feature_contributions[:5] if self.explainer else []
        }

def main():
    """Main function to run the quick credit scoring model"""
    print("🚀 CredTech Hackathon - Quick Credit Scoring Model")
    print("=" * 55)
    
    # Initialize model
    scorer = QuickCreditScorer()
    
    # Load data
    df = scorer.load_data()
    if df is None:
        print("Please run data collection first: python src\\data_sources\\data_collector.py")
        return
    
    # Engineer features
    df_features = scorer.engineer_features(df)
    
    # Create credit ratings
    df_final = scorer.create_credit_rating(df_features)
    
    # Prepare features and train model
    X, y = scorer.prepare_features(df_final)
    X_train, X_test, y_train, y_test = scorer.train_model(X, y)
    
    # Set up explainability
    shap_values = scorer.setup_explainability(X_test[:5])  # Sample for explanation
    shap_values, importance_df = scorer.explain_predictions(X_test[:5])
    
    print("\\n🎉 Quick credit scoring model ready!")
    print("\\nExample company prediction:")
    
    # Example prediction for first company
    if len(df_final) > 0:
        example_company = df_final.iloc[0]
        company_features = X.iloc[0].values
        
        result = scorer.predict_with_explanation(company_features)
        
        print(f"\\n🏢 Company: {example_company['company_name']} ({example_company['symbol']})")
        print(f"📊 Predicted Rating: {result['predicted_rating']}")
        print(f"🎯 Confidence: {max(result['probability_distribution'].values()):.2f}")
        
        print("\\n🔍 Top Contributing Factors:")
        for i, (feature, contribution) in enumerate(result['top_factors']):
            direction = "📈" if contribution > 0 else "📉"
            print(f"   {i+1}. {direction} {feature}: {contribution:.4f}")
    
    print("\\n📁 Model training complete!")
    print("Next steps:")
    print("1. Build dashboard interface")
    print("2. Add real-time scoring capability")
    print("3. Enhance explainability visualizations")

if __name__ == "__main__":
    main()