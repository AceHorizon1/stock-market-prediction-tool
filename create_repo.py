#!/usr/bin/env python3
"""
GitHub Repository Setup Script
This script helps you set up the Git repository and provides instructions for GitHub.
"""

import os
import subprocess
import sys


def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(
            command, shell=True, check=True, capture_output=True, text=True
        )
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def check_git_installed():
    """Check if Git is installed"""
    try:
        subprocess.run(["git", "--version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def initialize_git_repo():
    """Initialize Git repository and make initial commit"""
    print("🚀 Setting up Git repository...")

    # Check if Git is installed
    if not check_git_installed():
        print("❌ Git is not installed. Please install Git first:")
        print("   - macOS: brew install git")
        print("   - Windows: Download from https://git-scm.com/")
        print("   - Linux: sudo apt-get install git")
        return False

    # Initialize Git repository
    if not run_command("git init", "Initializing Git repository"):
        return False

    # Add all files
    if not run_command("git add .", "Adding files to Git"):
        return False

    # Make initial commit
    if not run_command(
        'git commit -m "Initial commit: Stock Market Prediction Tool"',
        "Making initial commit",
    ):
        return False

    print("✅ Git repository initialized successfully!")
    return True


def show_github_instructions():
    """Show instructions for creating GitHub repository"""
    print("\n" + "=" * 60)
    print("📋 GitHub Repository Setup Instructions")
    print("=" * 60)

    print("\n1. Create a new repository on GitHub:")
    print("   - Go to https://github.com/new")
    print("   - Repository name: stock-market-prediction-tool")
    print(
        "   - Description: A comprehensive AI-powered tool for predicting stock market movements"
    )
    print("   - Make it Public (recommended for open source)")
    print("   - Don't initialize with README (we already have one)")
    print("   - Click 'Create repository'")

    print("\n2. Connect your local repository to GitHub:")
    print("   After creating the repository, GitHub will show you commands like:")
    print(
        "   git remote add origin https://github.com/YOUR_USERNAME/stock-market-prediction-tool.git"
    )
    print("   git branch -M main")
    print("   git push -u origin main")

    print("\n3. Push your code:")
    print("   Replace YOUR_USERNAME with your actual GitHub username and run:")
    print(
        "   git remote add origin https://github.com/YOUR_USERNAME/stock-market-prediction-tool.git"
    )
    print("   git branch -M main")
    print("   git push -u origin main")

    print("\n4. Set up repository features:")
    print("   - Go to Settings > Pages to enable GitHub Pages")
    print("   - Go to Settings > Options to configure repository options")
    print("   - Add topics: stock-prediction, machine-learning, python, finance, ai")

    print("\n5. Create a release:")
    print("   - Go to Releases > Create a new release")
    print("   - Tag: v1.0.0")
    print("   - Title: Stock Market Prediction Tool v1.0.0")
    print("   - Description: Initial release with comprehensive features")

    print("\n6. Optional: Set up GitHub Pages")
    print("   - Go to Settings > Pages")
    print("   - Source: Deploy from a branch")
    print("   - Branch: main, folder: / (root)")
    print("   - Save")


def show_repository_info():
    """Show information about the repository structure"""
    print("\n📁 Repository Structure:")
    print("=" * 40)

    files = [
        ("📄 README.md", "Comprehensive documentation"),
        ("📄 requirements.txt", "Python dependencies"),
        ("📄 LICENSE", "MIT License"),
        ("📄 CONTRIBUTING.md", "Contribution guidelines"),
        ("📄 setup.py", "Installation script"),
        ("📄 run_app.py", "Application launcher"),
        ("📄 demo.py", "Demo script"),
        ("📄 test_installation.py", "Installation test"),
        ("📁 data_collector.py", "Data collection module"),
        ("📁 feature_engineering.py", "Feature engineering module"),
        ("📁 models.py", "Machine learning models"),
        ("📁 evaluation.py", "Model evaluation"),
        ("📁 main.py", "Streamlit web app"),
        ("📁 examples/", "Usage examples"),
        ("📁 .github/", "GitHub templates and workflows"),
    ]

    for file_name, description in files:
        print(f"   {file_name:<25} - {description}")


def show_next_steps():
    """Show next steps after repository setup"""
    print("\n🎯 Next Steps:")
    print("=" * 30)

    steps = [
        "1. Create GitHub repository (follow instructions above)",
        "2. Push code to GitHub",
        "3. Set up repository features (Pages, topics, etc.)",
        "4. Create first release",
        "5. Share with the community!",
        "",
        "📢 Promotion Ideas:",
        "   - Share on Reddit (r/Python, r/MachineLearning, r/Finance)",
        "   - Post on Twitter/LinkedIn",
        "   - Submit to Python Weekly newsletter",
        "   - Add to Awesome Python list",
        "   - Create YouTube tutorial",
    ]

    for step in steps:
        print(f"   {step}")


def main():
    """Main function"""
    print("🚀 Stock Market Prediction Tool - GitHub Setup")
    print("=" * 50)

    # Check if we're in the right directory
    required_files = ["README.md", "main.py", "data_collector.py"]
    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        print("Please run this script from the project root directory")
        return

    # Initialize Git repository
    if not initialize_git_repo():
        print("❌ Failed to initialize Git repository")
        return

    # Show repository information
    show_repository_info()

    # Show GitHub instructions
    show_github_instructions()

    # Show next steps
    show_next_steps()

    print(
        "\n🎉 Setup complete! Follow the instructions above to create your GitHub repository."
    )
    print("Happy coding! 🚀📈")


if __name__ == "__main__":
    main()
