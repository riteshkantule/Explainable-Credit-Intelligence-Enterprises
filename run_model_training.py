"""
CredTech Hackathon - Model Training Script (FIXED)
Simple wrapper to run model training
"""

import sys
import os
from pathlib import Path

# Add src to Python path BEFORE imports
project_root = Path(__file__).parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

def main():
    try:
        # Now import AFTER path is set
        from models.quick_credit_model import QuickCreditScorer
        
        print("🚀 Starting CredTech Hackathon Model Training")
        print("=" * 60)
        
        # Initialize model
        scorer = QuickCreditScorer()
        
        # Load data
        print("📊 Loading data...")
        df = scorer.load_data()
        if df is None:
            print("❌ No data found! Please run data collection first:")
            print("   python run_data_collection.py")
            return
        
        # Engineer features
        print("🔧 Engineering features...")
        df_features = scorer.engineer_features(df)
        
        # Create credit ratings
        print("📊 Creating credit ratings...")
        df_final = scorer.create_credit_rating(df_features)
        
        # Prepare features and train model
        print("🤖 Training model...")
        X, y = scorer.prepare_features(df_final)
        X_train, X_test, y_train, y_test = scorer.train_model(X, y)
        
        # Set up explainability
        print("🔍 Setting up explainability...")
        shap_values = scorer.setup_explainability(X_test[:5])
        shap_values, importance_df = scorer.explain_predictions(X_test[:5])
        
        print("\n✅ Model training completed successfully!")
        print(f"📊 Model performance metrics displayed above")
        print(f"🔍 SHAP explanations generated")
        
        print(f"\n🎯 Next steps:")
        print(f"   1. Launch dashboard: streamlit run src/dashboard/streamlit_app.py")
        print(f"   2. Start API: uvicorn src.api.fastapi_app:app --reload")
        print(f"   3. View dashboard at: http://localhost:8501")
        
        # Example prediction
        if len(df_final) > 0:
            print(f"\n🏢 Example prediction:")
            example_company = df_final.iloc[0]
            company_features = X.iloc[0].values
            
            result = scorer.predict_with_explanation(company_features)
            
            print(f"Company: {example_company.get('company_name', 'N/A')} ({example_company['symbol']})")
            print(f"Predicted Rating: {result['predicted_rating']}")
            print(f"Confidence: {max(result['probability_distribution'].values()):.2f}")
            
            print(f"\nTop Contributing Factors:")
            for i, (feature, contribution) in enumerate(result['top_factors'][:5]):
                direction = "📈" if contribution > 0 else "📉"
                print(f"   {i+1}. {direction} {feature}: {contribution:.4f}")
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print(f"\n🔧 Fix by ensuring these files exist:")
        print(f"   • src/__init__.py")
        print(f"   • src/models/__init__.py")
        print(f"   • src/models/quick_credit_model.py")
        print(f"   • data/processed/companies_summary.csv (run data collection first)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"\n🔧 Troubleshooting:")
        print(f"   • Run data collection first: python run_data_collection.py")
        print(f"   • Check all files are in correct locations")
        print(f"   • Verify requirements: pip install -r requirements.txt")

if __name__ == "__main__":
    main()