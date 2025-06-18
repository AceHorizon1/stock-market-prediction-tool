# Contributing to Stock Market Prediction Tool

Thank you for your interest in contributing to the Stock Market Prediction Tool! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### 1. Fork the Repository
- Click the "Fork" button on the GitHub repository page
- Clone your forked repository to your local machine

### 2. Set Up Development Environment
```bash
# Clone your fork
git clone https://github.com/your-username/stock-market-prediction-tool.git
cd stock-market-prediction-tool

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy
```

### 3. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### 4. Make Your Changes
- Follow the coding standards (see below)
- Add tests for new functionality
- Update documentation as needed

### 5. Test Your Changes
```bash
# Run the test suite
python test_installation.py

# Run linting
black .
flake8 .

# Run type checking
mypy .
```

### 6. Commit Your Changes
```bash
git add .
git commit -m "feat: add new feature description"
```

### 7. Push and Create a Pull Request
```bash
git push origin feature/your-feature-name
```

## 📋 Coding Standards

### Python Code Style
- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Keep functions and classes focused and well-documented
- Use meaningful variable and function names

### Documentation
- Add docstrings to all functions and classes
- Update README.md for new features
- Include examples in docstrings

### Testing
- Write unit tests for new functionality
- Ensure all tests pass before submitting
- Aim for good test coverage

## 🚀 Development Setup

### Prerequisites
- Python 3.8 or higher
- Git
- pip

### Local Development
1. **Clone and setup**:
   ```bash
   git clone https://github.com/your-username/stock-market-prediction-tool.git
   cd stock-market-prediction-tool
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run tests**:
   ```bash
   python test_installation.py
   ```

3. **Run the application**:
   ```bash
   streamlit run main.py
   ```

4. **Run linting**:
   ```bash
   black .
   flake8 .
   mypy .
   ```

## 🎯 Areas for Contribution

### High Priority
- **Bug fixes**: Fix issues reported in the Issues section
- **Documentation**: Improve README, add examples, fix typos
- **Testing**: Add more comprehensive tests
- **Performance**: Optimize data processing and model training

### Medium Priority
- **New features**: Add new technical indicators, models, or evaluation metrics
- **UI improvements**: Enhance the Streamlit interface
- **Data sources**: Add support for additional data providers
- **Visualization**: Improve charts and graphs

### Low Priority
- **Code refactoring**: Improve code structure and organization
- **Performance monitoring**: Add logging and monitoring
- **Deployment**: Add Docker support or cloud deployment options

## 📝 Pull Request Guidelines

### Before Submitting
- [ ] Code follows the project's style guidelines
- [ ] All tests pass
- [ ] Documentation is updated
- [ ] No sensitive data is included

### Pull Request Template
```markdown
## Description
Brief description of the changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Screenshots (if applicable)
Add screenshots for UI changes

## Checklist
- [ ] My code follows the style guidelines
- [ ] I have performed a self-review
- [ ] I have commented my code where necessary
- [ ] I have made corresponding changes to documentation
- [ ] My changes generate no new warnings
```

## 🐛 Reporting Issues

### Bug Reports
When reporting bugs, please include:
- **Description**: Clear description of the bug
- **Steps to reproduce**: Detailed steps to reproduce the issue
- **Expected behavior**: What you expected to happen
- **Actual behavior**: What actually happened
- **Environment**: OS, Python version, package versions
- **Screenshots**: If applicable

### Feature Requests
When requesting features, please include:
- **Description**: Clear description of the feature
- **Use case**: Why this feature would be useful
- **Implementation ideas**: Any thoughts on how to implement it

## 📞 Getting Help

### Questions and Discussions
- Use GitHub Issues for questions and discussions
- Check existing issues before creating new ones
- Be respectful and constructive in discussions

### Community Guidelines
- Be respectful and inclusive
- Help others learn and grow
- Share knowledge and experiences
- Follow the project's code of conduct

## 🏆 Recognition

Contributors will be recognized in:
- The project's README.md file
- GitHub contributors page
- Release notes

## 📄 License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to the Stock Market Prediction Tool! 🚀📈 