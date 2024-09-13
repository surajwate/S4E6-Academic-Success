# S4E6 Academic Success - Kaggle Challenge

This project is part of the **Kaggle Playground Series S4E6**, where the goal is to predict students' academic success (Graduate, Dropout, Enrolled) using a multiclass classification model. We used various machine learning models and performed hyperparameter tuning, cross-validation, and model evaluation, ultimately selecting **XGBoost** as the final model.

---

## Project Links
- **Blog Post**: [Classification with an Academic Success Dataset](https://surajwate.com/blog/classification-with-an-academic-success-dataset/)
- **Kaggle Notebook**: [Academic Success XGBoost](https://www.kaggle.com/code/surajwate/academic-success-xgboost)

---

## How to Run the Project

1. **Clone the repository:**
   ```bash
   git clone https://github.com/surajwate/S4E6-Academic-Success.git
   ```

2. **Install Dependencies:**
   Install the necessary dependencies using `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

3. **Create Folds:**
   Before training, split the dataset into K-Folds for cross-validation:
   ```bash
   python src/create_fold.py
   ```

4. **Train the Model:**
   Use the `main.py` script to train a model. You can specify which model to use via the `--model` argument.

   #### Available Models:
   - logistic_regression
   - random_forest
   - decision_tree
   - svm
   - gradient_boosting
   - xgboost

   #### Example Command:
   ```bash
   python main.py --model xgboost
   ```

   The above command will train the XGBoost model using 5-fold cross-validation.

---

## Key Scripts

- **src/create_fold.py**: Splits the dataset into K-Folds for cross-validation.
- **src/train.py**: Trains the model using the specified fold and model type.
- **src/final_model.py**: Finalizes the model for submission.
- **src/model_dispatcher.py**: Dispatches model configurations based on input.
- **main.py**: CLI script to run training for different models.

---

## Results
- **Best Model**: XGBoost
- **Private Score**: 0.83467
- **Public Score**: 0.83454

For more details, check the [Kaggle Notebook](https://www.kaggle.com/code/surajwate/academic-success-xgboost).

---

## License
This project is licensed under the MIT License - see the LICENSE file for details.
