# Osteoporosis Risk Prediction Using Machine Learning

## Project Overview

This repository contains a comprehensive study on **Osteoporosis Risk Prediction Using Multiple Machine Learning Techniques**. The research aims to identify individuals at high risk of osteoporosis using a range of classification models, supporting early intervention and improved patient care. Osteoporosis—a major public health concern—leads to fragile bones and increased fracture risks, particularly in aging populations. By leveraging data-driven techniques, this study seeks to enhance predictive accuracy and provide healthcare professionals with reliable diagnostic tools.

**Key Findings:**
- **Gradient Boosting** emerged as the best-performing model with an accuracy of **90.81%**.
- A comparative analysis was performed using Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, Support Vector Machine (SVM), and Naive Bayes.
- The research demonstrates that ensemble techniques can significantly improve diagnostic precision for osteoporosis risk.

## Repository Structure

- **/docs**  
  - Contains the research paper (`Osteoporosis-Risk-Prediction-Paper.pdf`) and any supplementary documentation regarding methodology, literature review, and detailed results.
- **/data**  
  - Includes the dataset (`osteoporosis.csv`) sourced from Kaggle. This dataset comprises 1,958 patient records (979 with osteoporosis and 979 without) featuring important attributes such as Age, Gender, Hormonal Changes, Family History, and Calcium Intake.
- **/notebooks**  
  - Contains the Jupyter Notebook (`Osteoporosis_RISK_PREDICTION.ipynb`) used for data preprocessing, model training, evaluation, and visualization.
- **README.md**  
  - This file provides an overview of the project, installation instructions, methodology summary, and citation details.

## Installation and Usage Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/Osteoporosis-Risk-Prediction.git
cd Osteoporosis-Risk-Prediction
```

### 2. Create and Activate a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
Ensure you have Python 3.12+ installed. Then run:
```bash
pip install -r requirements.txt
```
*The `requirements.txt` file includes all necessary libraries such as Scikit-learn, Pandas, Matplotlib, Seaborn, and NumPy.*

### 4. Run the Notebook
Launch Jupyter Notebook to explore the analysis and reproduce the results:
```bash
jupyter notebook notebooks/Osteoporosis_RISK_PREDICTION.ipynb
```

## Methodology and Results Summary

### Methodology

#### Dataset:
- The study uses a Kaggle dataset containing 1,958 patient records with equal representation of osteoporotic and non-osteoporotic cases.
- Features include Age, Gender, Hormonal Changes, Family History, Physical Activity, and Calcium Intake.
- The data underwent thorough preprocessing (handling missing values, encoding categorical data, and feature selection).

#### Machine Learning Models Evaluated:
- **Logistic Regression:** A baseline model for binary classification.
- **Decision Tree Classifier:** Provides interpretability but is prone to overfitting.
- **Random Forest Classifier:** An ensemble method reducing overfitting by aggregating multiple trees.
- **Gradient Boosting:** Achieved the highest accuracy by iteratively correcting errors from previous models.
- **Support Vector Machine (SVM):** Effective in high-dimensional spaces, though dependent on kernel selection.
- **Naive Bayes:** A probabilistic approach that offers fast predictions but assumes feature independence.

#### Evaluation Metrics:
- Models were evaluated using Accuracy, Precision, Recall, F1-score, and AUC-ROC.
- Gradient Boosting achieved a **90.81%** accuracy, proving its effectiveness in capturing complex data patterns.
- Comparative evaluations demonstrated the robustness of ensemble methods.

## Results
- **Gradient Boosting** emerged as the top-performing model with a **90.81%** accuracy.
- Visual comparisons of the confusion matrix and ROC curves are included in the notebook.
- Detailed insights into model performance and feature importance are available in the research paper.

## Citation and Acknowledgements

### Citation
If you use or refer to this work, please cite it as follows:
```arduino
Singh, S. (2024). Osteoporosis Risk Prediction Using Multiple Machine Learning Techniques. GitHub Repository. https://github.com/yourusername/Osteoporosis-Risk-Prediction
```

### Acknowledgements
- **Data Source:**  
  - Dataset obtained from Kaggle.
- **References:**  
  - This research builds upon numerous studies in the field of machine learning and healthcare diagnostics. Please refer to the `Osteoporosis-Risk-Prediction-Paper.pdf` for detailed references.
- **Contributors:**  
  - Thanks to all the mentors, colleagues, and institutions that supported this research.

## License Information

- **Code**: Licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
- **Research Paper**: Licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0). See the [LICENSE-PAPER](LICENSE-PAPER) file for details.


