# Game of Thrones Death Prediction

This project uses machine learning methods to predict the probability of death (binary classification) of characters in the TV series “Game of Thrones” based on various characteristics (gender, house status, appearance in books, etc.).

## Project Description

The goal is to build a model that can determine whether a character will survive using data about the characters provided on the website http://awoiaf.westeros.org.

The project includes:
- Data loading and preprocessing
- Exploratory data analysis
- Building a classification model
- Evaluating model quality using accuracy metrics

## Preview

Distribution of the target variable (train data)

![image](https://github.com/user-attachments/assets/465bd4b3-2fe4-4c70-84c9-3ea0668ed936)

Correlation heatmap for selected features and target variable

![image](https://github.com/user-attachments/assets/b9f23ce0-9b88-422e-a810-ff3aa5340da8)


## Technologies Used

- Python 3.12.4
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- sklearn
- Jupyter Notebook

## Algorithms used

Various classification models were tested. AdaBoostClassifier showed the best accuracy.

## Results 
| Model    | Accuracy Value |
| -------- | -----          |
| LogisticRegression | 0.8205 |
| RandomForestClassifier | 0.8269 |
| AdaBoostClassifier  | 0.8494 |
| GaussianProcessClassifier  | 0.8429 |
| KNeighborsClassifier  | 0.8141 |
| SVC  | 0.8365 |
| DecisionTreeClassifier  | 0.8365 |

Most important features: dateOfBirth, age_plus_date, book_4, boolDeadRelations, age.
Accuracy on test data: 0.8175.

## Data
The dataset was provided as part of an educational course and is not included in this repository due to copyright restrictions.
If you're taking DeepLearningSchool course (DLS), you will have access to the data through the course platform.


## Possible Improvements
- Add ROC curve 
- Try cross-validation with GridSearchCV
- Deploy as a simple web app using Streamlit or Flask

## How to run?


### 1. Clone the repository
```bash
git clone https://github.com/your-username/game-of-thrones-death-prediction.git
cd game-of-thrones-death-prediction

### 2. Create virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate   #on Linux/Mac
venv\Scripts\activate      #on Windows

### 3. Install dependencies
```bash
pip install -r requirements.txt

### 4. Run Jupyter Notebook
```bash
jupyter notebook notebooks/game_of_thrones_prediction.ipynb







