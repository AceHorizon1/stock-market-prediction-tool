#!/usr/bin/env python3
"""
Setup script for Stock Market Prediction Tool
This script helps users install the tool and its dependencies.
"""

import os
import sys
import subprocess
import platform

def print_banner():
    """Print the tool banner"""
    print("=" * 60)
    print("📈 Stock Market Prediction Tool - Setup")
    print("=" * 60)
    print("A comprehensive AI-powered tool for predicting stock market movements")
    print("=" * 60)

def check_python_version():
    """Check if Python version is compatible"""
    print("🔍 Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected")
        print("   This tool requires Python 3.8 or higher")
        print("   Please upgrade your Python installation")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    
    try:
        # Upgrade pip first
        print("   Upgrading pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install requirements
        print("   Installing requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        
        print("✅ Dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Make sure you have internet connection")
        print("2. Try running: pip install --upgrade pip")
        print("3. Try installing packages one by one")
        return False
    except FileNotFoundError:
        print("❌ requirements.txt not found")
        return False

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    directories = [
        "data",
        "models",
        "logs",
        "results"
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"   Created: {directory}/")
        else:
            print(f"   Exists: {directory}/")
    
    print("✅ Directories created")

def create_config_file():
    """Create a sample configuration file"""
    print("\n⚙️  Creating configuration file...")
    
    config_content = """# Stock Market Prediction Tool Configuration

# API Keys (optional - for enhanced data collection)
# Get free API keys from:
# Alpha Vantage: https://www.alphavantage.co/support/#api-key
# FRED: https://fred.stlouisfed.org/docs/api/api_key.html

ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
FRED_API_KEY=your_fred_key_here

# Model Settings
DEFAULT_MODEL_TYPE=ensemble
DEFAULT_TASK=regression
DEFAULT_HORIZON=1

# Data Settings
DEFAULT_PERIOD=5y
DEFAULT_INTERVAL=1d
INCLUDE_MARKET_DATA=true
INCLUDE_ECONOMIC_DATA=false

# Feature Engineering
CORRELATION_THRESHOLD=0.01
FEATURE_SELECTION_METHOD=correlation

# Evaluation
BACKTEST_INITIAL_CAPITAL=10000
BACKTEST_TRANSACTION_COST=0.001
BACKTEST_STRATEGY=long_short
"""
    
    config_file = "config.env"
    if not os.path.exists(config_file):
        with open(config_file, 'w') as f:
            f.write(config_content)
        print(f"   Created: {config_file}")
    else:
        print(f"   Exists: {config_file}")
    
    print("✅ Configuration file ready")

def run_tests():
    """Run installation tests"""
    print("\n🧪 Running installation tests...")
    
    try:
        result = subprocess.run([sys.executable, "test_installation.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ All tests passed")
            return True
        else:
            print("❌ Some tests failed")
            print("Test output:")
            print(result.stdout)
            if result.stderr:
                print("Errors:")
                print(result.stderr)
            return False
            
    except FileNotFoundError:
        print("❌ test_installation.py not found")
        return False

def show_next_steps():
    """Show next steps for the user"""
    print("\n" + "=" * 60)
    print("🎉 Setup Complete!")
    print("=" * 60)
    
    print("\n📋 Next Steps:")
    print("1. Configure API keys (optional):")
    print("   - Edit config.env file")
    print("   - Get free API keys from Alpha Vantage and FRED")
    
    print("\n2. Run the application:")
    print("   python run_app.py")
    print("   or")
    print("   streamlit run main.py")
    
    print("\n3. Try the examples:")
    print("   python examples/basic_usage.py")
    
    print("\n4. Read the documentation:")
    print("   README.md")
    
    print("\n🚀 Happy Trading!")

def main():
    """Main setup function"""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        return
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Setup failed. Please fix the errors above and try again.")
        return
    
    # Create directories
    create_directories()
    
    # Create config file
    create_config_file()
    
    # Run tests
    if not run_tests():
        print("\n⚠️  Some tests failed, but setup may still work.")
        print("   Try running the application anyway.")
    
    # Show next steps
    show_next_steps()

if __name__ == "__main__":
    main() 