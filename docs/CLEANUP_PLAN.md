# File Cleanup and Organization Plan

## Current Issues
1. Test files are scattered across the root directory and tests/ directory
2. Some duplicate or unnecessary files exist
3. Files are not consistently organized by function

## Organization Structure
```
ml for pm internship/
├── backend/
│   ├── api/
│   ├── models/
│   └── utils/
├── frontend/
│   ├── css/
│   ├── js/
│   ├── assets/
│   └── components/
├── ml_models/
│   ├── models/
│   ├── evaluation/
│   └── utils/
├── interactive/
├── dataset/
├── docs/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── api/
└── scripts/
```

## Cleanup Actions

### 1. Move all test files to tests/ directory
- Move test files from root to tests/ directory
- Organize tests by type (unit, integration, api)

### 2. Remove unnecessary files
- Remove __pycache__ directories
- Remove temp_user_dataset.csv (temporary file)
- Remove duplicate or outdated files

### 3. Organize by function
- Group related files in appropriate directories
- Ensure consistent naming conventions

### 4. Update documentation
- Update README files to reflect new organization
- Create a comprehensive project structure document