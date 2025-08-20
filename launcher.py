"""
CredTech Hackathon - Quick Setup & Launch Script
This script helps you set up and run the entire platform
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if all requirements are met"""
    print("🔍 Checking requirements...")
    
    issues = []
    
    # Check Python version
    if sys.version_info < (3, 8):
        issues.append("❌ Python 3.8+ required")
    else:
        print("✅ Python version OK")
    
    # Check .env file
    if not os.path.exists('.env'):
        if os.path.exists('.env.example'):
            issues.append("❌ Please copy .env.example to .env and add your API keys")
        else:
            issues.append("❌ .env file not found")
    else:
        print("✅ .env file found")
    
    # Check if virtual environment is active
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        issues.append("⚠️  Virtual environment not detected (recommended)")
    else:
        print("✅ Virtual environment active")
    
    # Check key Python packages
    try:
        import pandas, streamlit, fastapi, xgboost, shap
        print("✅ Key packages installed")
    except ImportError as e:
        issues.append(f"❌ Missing package: {e}")
    
    return issues

def setup_project_structure():
    """Create necessary directories"""
    print("📁 Setting up project structure...")
    
    directories = [
        'src', 'src/data_sources', 'src/models', 'src/dashboard', 'src/api', 'src/utils',
        'data', 'data/raw', 'data/processed', 'data/models',
        'tests', 'notebooks', 'scripts', 'config', 'docs'
    ]
    
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
    
    # Create __init__.py files
    init_files = [
        'src/__init__.py',
        'src/data_sources/__init__.py',
        'src/models/__init__.py', 
        'src/dashboard/__init__.py',
        'src/api/__init__.py',
        'src/utils/__init__.py'
    ]
    
    for init_file in init_files:
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write('# Python package initialization\\n')
    
    print("✅ Project structure created")

def install_requirements():
    """Install Python requirements"""
    if os.path.exists('requirements.txt'):
        print("📦 Installing requirements...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
            print("✅ Requirements installed")
        except subprocess.CalledProcessError:
            print("❌ Failed to install requirements")
            return False
    else:
        print("❌ requirements.txt not found")
        return False
    return True

def check_api_keys():
    """Check if API keys are configured"""
    print("🔑 Checking API keys...")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    alpha_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    market_token = os.getenv('MARKETAUX_API_TOKEN')
    
    if not alpha_key or alpha_key == 'your_alpha_vantage_key_here':
        print("❌ ALPHA_VANTAGE_API_KEY not configured")
        return False
    
    if not market_token or market_token == 'your_marketaux_token_here':
        print("❌ MARKETAUX_API_TOKEN not configured")
        return False
    
    print("✅ API keys configured")
    return True

def run_data_collection():
    """Run the data collection pipeline"""
    print("🚀 Starting data collection...")
    try:
        subprocess.run([sys.executable, 'run_data_collection.py'], check=True)
        return True
    except subprocess.CalledProcessError:
        print("❌ Data collection failed")
        return False

def run_model_training():
    """Run model training"""
    print("🤖 Starting model training...")
    try:
        subprocess.run([sys.executable, 'run_model_training.py'], check=True)
        return True
    except subprocess.CalledProcessError:
        print("❌ Model training failed")
        return False

def launch_dashboard():
    """Launch Streamlit dashboard"""
    print("🚀 Launching dashboard...")
    print("Dashboard will open at: http://localhost:8501")
    try:
        subprocess.run(['streamlit', 'run', 'src/dashboard/streamlit_app.py'])
    except KeyboardInterrupt:
        print("\\n👋 Dashboard stopped by user")
    except FileNotFoundError:
        print("❌ Streamlit not found. Install with: pip install streamlit")

def launch_api():
    """Launch FastAPI server"""
    print("🚀 Launching API server...")
    print("API will be available at: http://localhost:8000")
    print("API docs at: http://localhost:8000/docs")
    try:
        subprocess.run(['uvicorn', 'src.api.fastapi_app:app', '--host', '0.0.0.0', '--port', '8000', '--reload'])
    except KeyboardInterrupt:
        print("\\n👋 API server stopped by user")
    except FileNotFoundError:
        print("❌ Uvicorn not found. Install with: pip install uvicorn")

def main():
    print("🏆 CredTech Hackathon - Platform Setup & Launch")
    print("=" * 60)
    
    while True:
        print("\\nChoose an option:")
        print("1. 🔧 Check system requirements")
        print("2. 📦 Install requirements")
        print("3. 🔑 Setup API keys")
        print("4. 📊 Run data collection")
        print("5. 🤖 Train models")
        print("6. 🖥️  Launch dashboard")
        print("7. 🌐 Launch API server")
        print("8. 🚀 Full setup (steps 1-5)")
        print("9. ❌ Exit")
        
        choice = input("\\nEnter choice (1-9): ").strip()
        
        if choice == '1':
            issues = check_requirements()
            if issues:
                print("\\n⚠️  Issues found:")
                for issue in issues:
                    print(f"   {issue}")
            else:
                print("\\n✅ All requirements met!")
        
        elif choice == '2':
            setup_project_structure()
            install_requirements()
        
        elif choice == '3':
            if not os.path.exists('.env.example'):
                print("❌ .env.example not found")
            else:
                print("📝 Please follow these steps:")
                print("1. Copy .env.example to .env")
                print("2. Edit .env and add your API keys:")
                print("   - Get Alpha Vantage key: https://www.alphavantage.co/support/#api-key")
                print("   - Get MarketAux token: https://www.marketaux.com/")
                input("Press Enter when done...")
                
                if check_api_keys():
                    print("✅ API keys configured successfully!")
                else:
                    print("❌ API keys still not configured")
        
        elif choice == '4':
            if not check_api_keys():
                print("❌ Please configure API keys first (option 3)")
            else:
                success = run_data_collection()
                if success:
                    print("\\n✅ Data collection completed!")
                    print("📁 Check data/processed/companies_summary.csv")
        
        elif choice == '5':
            if not os.path.exists('data/processed/companies_summary.csv'):
                print("❌ No data found. Please run data collection first (option 4)")
            else:
                success = run_model_training()
                if success:
                    print("\\n✅ Model training completed!")
        
        elif choice == '6':
            launch_dashboard()
        
        elif choice == '7':
            launch_api()
        
        elif choice == '8':
            print("🚀 Running full setup...")
            
            # Check requirements
            issues = check_requirements()
            if issues:
                print("\\n⚠️  Please fix these issues first:")
                for issue in issues:
                    print(f"   {issue}")
                continue
            
            # Setup structure and install requirements
            setup_project_structure()
            if not install_requirements():
                continue
            
            # Check API keys
            if not check_api_keys():
                print("\\n❌ Please configure API keys and try again")
                continue
            
            # Run data collection
            if not run_data_collection():
                continue
            
            # Train models
            if not run_model_training():
                continue
            
            print("\\n🎉 Full setup completed successfully!")
            print("\\nYou can now:")
            print("• Launch dashboard: python launcher.py (option 6)")
            print("• Launch API: python launcher.py (option 7)")
            print("• View data: data/processed/companies_summary.csv")
        
        elif choice == '9':
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please enter 1-9.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\\n\\n👋 Setup interrupted by user")
    except Exception as e:
        print(f"\\n❌ Unexpected error: {e}")
        print("Please check your setup and try again.")