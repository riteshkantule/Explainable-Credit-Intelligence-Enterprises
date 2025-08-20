# Create a new file: enhanced_credit_rating.py
import torch
import pandas as pd
import numpy as np
from deepcross_model import CreditRatingDataProcessor, predict_credit_rating

class EnhancedCreditRating:
    def __init__(self, model_path='models/deepcross_model.pth'):
        self.model = None
        self.processor = None
        self.load_model(model_path)
    
    def load_model(self, model_path):
        try:
            checkpoint = torch.load(model_path)
            # We need to reconstruct the model architecture first
            # For now, we'll assume the model is trained and saved with the processor
            self.model = checkpoint['model']
            self.processor = checkpoint['processor']
            print("✅ DeepCross model loaded successfully")
        except:
            print("❌ Could not load model. Please train the model first.")
    
    def calculate_deepcross_credit_score(self, company_data):
        """Calculate credit score using the DeepCross model"""
        if self.model is None or self.processor is None:
            print("Model not loaded. Using fallback scoring.")
            return self.calculate_enhanced_credit_score(company_data)
        
        try:
            credit_score, rating, attention_weights = predict_credit_rating(
                self.model, self.processor, company_data
            )
            
            # Add explainability using attention weights
            explanation = self.generate_explanation(attention_weights, company_data)
            
            return {
                'credit_score_raw': credit_score,
                'credit_rating': rating,
                'explanation': explanation,
                'model': 'deepcross'
            }
        except Exception as e:
            print(f"Error in DeepCross model: {e}")
            return self.calculate_enhanced_credit_score(company_data)
    
    def generate_explanation(self, attention_weights, company_data):
        """Generate explanation based on attention weights"""
        # This is a simplified version - you would need to map attention weights to features
        top_features = []
        
        # For demonstration, we'll just return the top 3 features by attention
        if hasattr(attention_weights, 'cpu'):
            attention_weights = attention_weights.cpu().numpy()
        
        if len(attention_weights) > 0:
            # Get indices of top features
            top_indices = np.argsort(attention_weights[0])[-3:][::-1]
            
            # Map indices to feature names (this would need to be implemented based on your feature order)
            feature_names = list(company_data.keys())
            top_features = [feature_names[i] for i in top_indices if i < len(feature_names)]
        
        explanation = f"The rating was primarily influenced by: {', '.join(top_features)}" if top_features else "No specific features stood out as most influential."
        
        return explanation
    
    def calculate_enhanced_credit_score(self, data):
        """Your original scoring function as fallback"""
        # ... (your original calculate_enhanced_credit_score code here)
        pass

# Integrate with your existing MultiAPIDataCollector
def collect_enhanced_data_with_deepcross():
    """Main enhanced data collection function with DeepCross"""
    print("🏆 CredTech Hackathon - DEEPCROSS ENHANCED VERSION")
    print("🚀 Combining Alpha Vantage + Yahoo Finance + SEC EDGAR + DeepCross Model")
    print("=" * 80)
    
    collector = MultiAPIDataCollector()
    credit_rater = EnhancedCreditRating()
    
    # Target companies
    companies = [
        ('AAPL', 'Apple Inc'),
        ('MSFT', 'Microsoft Corporation'),
        ('GOOGL', 'Alphabet Inc')
    ]
    
    all_data = []
    
    for i, (symbol, name) in enumerate(companies):
        try:
            # Collect comprehensive data
            company_data = collector.collect_company_data(symbol, name)
            
            # Calculate credit score with DeepCross model
            credit_data = credit_rater.calculate_deepcross_credit_score(company_data)
            company_data.update(credit_data)
            
            all_data.append(company_data)
            
            print(f"\n🎯 DEEPCROSS RESULTS FOR {symbol}:")
            print(f"   🏢 Company: {company_data['company_name']}")
            print(f"   📊 Credit Rating: {company_data['credit_rating']}")
            print(f"   🎯 Credit Score: {company_data['credit_score_raw']:.3f}")
            print(f"   🤖 Model: {company_data.get('model', 'traditional')}")
            print(f"   📝 Explanation: {company_data.get('explanation', 'N/A')}")
            
            # Pause between companies
            if i < len(companies) - 1:
                print(f"   ⏳ Brief pause...")
                time.sleep(3)
                
        except Exception as e:
            print(f"   ❌ Error processing {symbol}: {e}")
    
    # Save enhanced data
    df = pd.DataFrame(all_data)
    
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save files
    raw_file = f'data/raw/deepcross_data_{timestamp}.json'
    csv_file = 'data/processed/companies_summary_deepcross.csv'
    
    with open(raw_file, 'w') as f:
        json.dump(all_data, f, indent=2, default=str)
    
    df.to_csv(csv_file, index=False)
    
    print(f"\n🎉 DEEPCROSS COLLECTION COMPLETED!")
    print(f"{'='*60}")
    print(f"📊 Companies processed: {len(df)}")
    print(f"📁 DeepCross data: {csv_file}")
    
    return df

# Update your main function
def main():
    """Main execution"""
    # First, train the model if needed
    try:
        df = pd.read_csv('data/processed/companies_summary.csv')
        from deepcross_model import train_deepcross_model
        model, processor = train_deepcross_model(df)
        
        # Save the trained model
        torch.save({
            'model': model,
            'processor': processor
        }, 'models/deepcross_model.pth')
        print("✅ DeepCross model trained and saved!")
    except Exception as e:
        print(f"❌ Could not train model: {e}")
    
    # Then collect data with DeepCross
    collect_enhanced_data_with_deepcross()
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"1. 📊 Review DeepCross data: data/processed/companies_summary_deepcross.csv")
    print(f"2. 🖥️  Launch dashboard: streamlit run src/dashboard/streamlit_app.py")
    print(f"3. � Now with explainable AI credit ratings!")

if __name__ == "__main__":
    main()