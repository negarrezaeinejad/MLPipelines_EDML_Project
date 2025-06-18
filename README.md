# ML Pipeline Evaluation Project

This repository contains an evaluation of an e-commerce ML pipeline for predicting customer purchase behavior.

## Files Overview

- `1st_MLpipeline.py` - Original pipeline (contains critical bugs)
- `fixed_pipeline.py` - Corrected version with proper implementation
- `pipeline_evaluation_report.md` - Comprehensive evaluation report
- `detailed_analysis.md` - In-depth technical analysis and recommendations
- `requirements.txt` - Python dependencies

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the fixed pipeline
python fixed_pipeline.py
```

## Key Findings

### Critical Issues in Original Pipeline
1. **TypeError**: Incorrect use of `LabelEncoder` for categorical features
2. **Data Leakage**: Manual preprocessing outside pipeline
3. **Inconsistent Handling**: Different preprocessing for train/test data

### Performance Results (Fixed Pipeline)
- **Training Accuracy**: 75.1%
- **Test Accuracy**: 49.8%
- **Cross-Validation**: 64.7%
- **Key Issue**: Significant concept drift between training and test periods

### Data Characteristics
- **Training Period**: 56% purchase rate (balanced)
- **Test Period**: 91% purchase rate (heavily skewed)
- **Most Important Features**: time_on_page (45.9%), is_premium_member (35.3%)

## Evaluation Summary

| Aspect | Original | Fixed | Grade |
|--------|----------|-------|-------|
| **Execution** | ❌ Fails | ✅ Works | D+ → B- |
| **Code Quality** | ⚠️ Poor | ✅ Good | |
| **Performance** | N/A | ⚠️ Drift Issues | |
| **Production Ready** | ❌ No | ⚠️ Needs Work | |

## Recommendations

### Immediate Fixes (Implemented)
- ✅ Replace `LabelEncoder` with `OneHotEncoder`
- ✅ Remove manual preprocessing
- ✅ Add comprehensive evaluation metrics
- ✅ Implement proper feature importance analysis

### Next Steps for Production
1. **Address Concept Drift**
   - Add temporal features
   - Implement time-aware cross-validation
   - Consider online learning approaches

2. **Enhance Model Architecture**
   - Hyperparameter tuning
   - Ensemble methods
   - Regularization techniques

3. **Add Monitoring**
   - Drift detection
   - Performance monitoring
   - Automated retraining

4. **Improve Code Quality**
   - Configuration management
   - Unit testing
   - Proper logging

## Technical Details

### Pipeline Architecture
```
Data → Preprocessing → Model → Predictions
       ↓
   ColumnTransformer
   ├── Numerical: StandardScaler
   ├── Categorical: OneHotEncoder
   └── Passthrough: is_premium_member
```

### Model Configuration
- **Algorithm**: Random Forest Classifier
- **Parameters**: 100 estimators, balanced class weights
- **Features**: time_on_page, num_clicks, product_category, is_premium_member
- **Target**: Binary purchase prediction

## Conclusion

The original pipeline had fundamental implementation issues that prevented execution. The fixed version reveals interesting challenges with temporal data drift, providing valuable insights for building robust ML systems in dynamic environments.

**Overall Assessment**: The project demonstrates good ML concepts but requires significant improvements for production deployment. The evaluation process uncovered both technical bugs and important data science challenges related to concept drift.