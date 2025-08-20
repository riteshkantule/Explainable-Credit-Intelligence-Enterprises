"""
CredTech Hackathon - Data Collection Script (FIXED)
Simple wrapper to run the data collection pipeline
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
        from data_sources.data_collector import IntegratedDataCollector
        
        print("🚀 Starting CredTech Hackathon Data Collection")
        print("=" * 60)
        
        # Initialize collector
        collector = IntegratedDataCollector()
        
        # Collect all data
        data = collector.collect_all_data()
        
        # Save data
        saved_files = collector.save_data()
        
        # Generate summary
        summary = collector.generate_summary_report()
        
        print("\n✅ Data collection completed successfully!")
        print(f"📁 Files saved:")
        for file_type, file_path in saved_files.items():
            print(f"   • {file_type}: {file_path}")
        
        print(f"\n🎯 Next steps:")
        print(f"   1. Review: data/processed/companies_summary.csv")
        print(f"   2. Train models: python run_model_training.py")
        print(f"   3. Launch dashboard: streamlit run src/dashboard/streamlit_app.py")
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print(f"\n🔧 Fix by ensuring these files exist:")
        print(f"   • src/__init__.py")
        print(f"   • src/data_sources/__init__.py")
        print(f"   • src/data_sources/data_collector.py")
        print(f"   • src/data_sources/alpha_vantage_client.py")
        print(f"   • src/data_sources/marketaux_client.py")
        print(f"   • src/data_sources/sec_edgar_client.py")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"\n🔧 Troubleshooting:")
        print(f"   • Check your .env file has API keys")
        print(f"   • Verify all files are in correct locations")
        print(f"   • Run: pip install -r requirements.txt")

if __name__ == "__main__":
    main()