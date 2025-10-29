# Contributing to Home Assistant In Bed Sensor

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to this project.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)
- [Testing](#testing)
- [Documentation](#documentation)

---

## Code of Conduct

### Our Pledge
We are committed to providing a welcoming and inclusive environment for all contributors.

### Expected Behavior
- Be respectful and considerate
- Welcome newcomers and help them learn
- Focus on what's best for the project
- Show empathy towards other contributors

### Unacceptable Behavior
- Harassment or discriminatory language
- Personal attacks or trolling
- Publishing others' private information
- Other unprofessional conduct

---

## Getting Started

### Prerequisites
- Python 3.13 or higher
- Git
- Basic understanding of machine learning concepts
- Familiarity with scikit-learn

### Fork and Clone
1. Fork the repository on GitHub
2. Clone your fork locally:
```bash
git clone https://github.com/brunomigueldasilva/home-assistant-in-bed-sensor.git
cd home-assistant-in-bed-sensor
```

---

## Development Setup

### 1. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install Development Dependencies
```bash
# Optional development tools
pip install pytest pytest-cov pylint black flake8
```

### 4. Verify Setup
```bash
python 08_orchestrator.py --all
```

---

## How to Contribute

### Types of Contributions

#### 🐛 Bug Reports
- Use the bug report template
- Include Python version, OS, and error messages
- Provide steps to reproduce
- Suggest a fix if possible

#### ✨ Feature Requests
- Explain the problem you're trying to solve
- Describe your proposed solution
- Consider alternative approaches
- Explain why this benefits the project

#### 📝 Documentation
- Fix typos or unclear explanations
- Add examples or tutorials
- Improve docstrings
- Translate documentation

#### 💻 Code Contributions
- Bug fixes
- New features
- Performance improvements
- Code refactoring

---

## Coding Standards

### Python Style Guide
Follow **PEP 8** conventions:

```python
# Good
def calculate_accuracy(y_true, y_pred):
    """
    Calculate classification accuracy.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        float: Accuracy score
    """
    return np.mean(y_true == y_pred)


# Bad
def calc_acc(y,yp):
    return np.mean(y==yp)
```

### Code Quality Checklist
- [ ] Follows PEP 8 style guide
- [ ] Includes docstrings for all functions
- [ ] Has meaningful variable names
- [ ] Includes comments for complex logic
- [ ] No hardcoded values (use constants)
- [ ] Handles errors gracefully
- [ ] Is DRY (Don't Repeat Yourself)

### Documentation Strings
Use Google-style docstrings:

```python
def train_model(X_train, y_train, model_type='logistic'):
    """
    Train a classification model.
    
    This function trains a specified model type on the provided
    training data and returns the fitted model object.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels
        model_type (str, optional): Model type to train. 
            Options: 'logistic', 'knn', 'svm'. Defaults to 'logistic'.
    
    Returns:
        object: Fitted scikit-learn model
        
    Raises:
        ValueError: If model_type is not recognized
        
    Example:
        >>> X_train, y_train = load_data()
        >>> model = train_model(X_train, y_train, 'logistic')
        >>> print(model.score(X_test, y_test))
    """
    # Implementation
```

---

## Commit Messages

### Format
```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code formatting (no logic change)
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance tasks

### Examples
```bash
# Good
feat(preprocessing): add SMOTE for handling imbalance
fix(evaluation): correct F1-score calculation for multiclass
docs(readme): update installation instructions

# Bad
fixed stuff
update
changes
```

### Detailed Commit Example
```bash
feat(models): add Random Forest classifier

- Implemented Random Forest with 100 estimators
- Added feature importance extraction
- Updated comparative metrics to include RF
- Added RF to orchestrator pipeline

Closes #23
```

---

## Pull Request Process

### Before Submitting

1. **Create a new branch**:
```bash
git checkout -b feature/your-feature-name
```

2. **Make your changes**

3. **Test thoroughly**:
```bash
# Run the full pipeline
python 08_orchestrator.py --all

# Run specific tests (if available)
pytest tests/
```

4. **Update documentation** if needed

5. **Commit with clear messages**:
```bash
git add .
git commit -m "feat(models): add XGBoost classifier"
```

6. **Push to your fork**:
```bash
git push origin feature/your-feature-name
```

### Submitting the PR

1. Go to the original repository on GitHub
2. Click "New Pull Request"
3. Select your branch
4. Fill out the PR template:
   - **Title**: Clear, concise description
   - **Description**: What changes you made and why
   - **Issue Reference**: Link related issues
   - **Testing**: How you tested your changes
   - **Screenshots**: If applicable

### PR Review Process

1. **Automated checks** will run (if configured)
2. **Maintainer review** (typically within 3-5 days)
3. **Address feedback** if requested
4. **Approval and merge** by maintainer

### PR Checklist
- [ ] Code follows project style guidelines
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Commit messages are clear
- [ ] No merge conflicts
- [ ] PR description is complete

---

## Testing

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=.

# Run specific test file
pytest tests/test_preprocessing.py
```

### Writing Tests
```python
import pytest
import pandas as pd
from preprocessing import clean_sensor_data

def test_remove_unavailable_states():
    """Test that 'unavailable' states are removed."""
    df = pd.DataFrame({
        'state': ['on', 'unavailable', 'off', 'unknown'],
        'sensor': ['light', 'light', 'light', 'light']
    })
    
    result = clean_sensor_data(df)
    
    assert len(result) == 2
    assert 'unavailable' not in result['state'].values
    assert 'unknown' not in result['state'].values
```

---

## Documentation

### Types of Documentation

#### Code Comments
```python
# Use comments to explain WHY, not WHAT
# Good
# Stratified split maintains class proportions in imbalanced data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=42
)

# Bad
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y)
```

#### Docstrings
- Every public function must have a docstring
- Include Args, Returns, Raises, and Examples
- Keep them up-to-date with code changes

#### README Updates
- Update README.md when adding major features
- Keep installation instructions current
- Update examples if API changes

#### CHANGELOG
Document all notable changes:
```markdown
## [1.1.0] - 2025-11-15
### Added
- Random Forest classifier
- Feature importance visualization
- Hyperparameter tuning with GridSearchCV

### Fixed
- Memory leak in data loading
- Incorrect F1-score calculation

### Changed
- Updated scikit-learn to 1.3.0
- Improved error messages
```

---

## Development Workflow

### Typical Workflow
1. **Pick an issue** or create one
2. **Discuss** your approach (for large changes)
3. **Create a branch**
4. **Write code** with tests
5. **Update documentation**
6. **Submit PR**
7. **Address feedback**
8. **Celebrate** when merged! 🎉

### Branch Naming
```bash
feature/add-random-forest      # New features
bugfix/fix-memory-leak         # Bug fixes
docs/update-installation       # Documentation
refactor/optimize-preprocessing # Code improvements
```

---

## Getting Help

### Questions?
- Check existing issues and discussions
- Ask in GitHub Discussions
- Email maintainers (for private matters)

### Stuck?
- Don't be afraid to ask for help
- Provide context and what you've tried
- Be patient and respectful

---

## Recognition

Contributors will be:
- Listed in `AUTHORS.md`
- Mentioned in release notes
- Given credit in the README

Thank you for contributing! 🙏

---

## Additional Resources

- [PEP 8 Style Guide](https://pep8.org/)
- [GitHub Flow](https://guides.github.com/introduction/flow/)
- [Scikit-learn Contributing Guide](https://scikit-learn.org/stable/developers/contributing.html)
- [Writing Good Commit Messages](https://chris.beams.io/posts/git-commit/)

---

**Questions?** Feel free to reach out to the maintainers or open a discussion on GitHub.
