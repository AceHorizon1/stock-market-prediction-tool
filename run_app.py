#!/usr/bin/env python3
"""
Stock Market Prediction Tool - Launcher Script
This script launches the Streamlit application with proper setup and error handling.
"""

import os
import sys
import subprocess
import importlib.util

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'pandas', 'numpy', 'yfinance', 'scikit-learn', 'streamlit',
        'plotly', 'matplotlib', 'seaborn', 'ta'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        spec = importlib.util.find_spec(package)
        if spec is None:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nPlease install missing packages using:")
        print("pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def check_files():
    """Check if all required files exist"""
    required_files = [
        'main.py',
        'data_collector.py',
        'feature_engineering.py',
        'models.py',
        'evaluation.py'
    ]
    
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    print("✅ All required files are present")
    return True

def run_streamlit():
    """Run the Streamlit application"""
    try:
        print("🚀 Starting Stock Market Prediction Tool...")
        print("📱 Opening web interface...")
        print("🌐 The app will open in your default browser")
        print("⏹️  Press Ctrl+C to stop the application")
        print("-" * 50)
        
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "main.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
        
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error running application: {str(e)}")

def run_example():
    """Run the basic example"""
    try:
        print("🧪 Running basic example...")
        subprocess.run([sys.executable, "examples/basic_usage.py"])
    except Exception as e:
        print(f"❌ Error running example: {str(e)}")

def main():
    """Main function"""
    print("📈 Stock Market Prediction Tool")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Check files
    if not check_files():
        return
    
    # Show menu
    print("\nWhat would you like to do?")
    print("1. Run the web application")
    print("2. Run basic example")
    print("3. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == '1':
                run_streamlit()
                break
            elif choice == '2':
                run_example()
                break
            elif choice == '3':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main() 