import pandas as pd
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from scipy.sparse import hstack
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import time

df = pd.read_csv("movie_reviews_data.csv")

vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
x_tfidf = vectorizer.fit_transform(df["text"].astype(str))

x_numeric = df[[
    "technical_word_count", "rave_word_count", "rant_word_count", 
    "recommendation_word_count", "word_count", "exclamation_count", 
    "question_count", "quote_count", "numerical_rating_mentioned"
]].values

x_numeric_sparse = scipy.sparse.csr_matrix(x_numeric)
X = hstack([x_tfidf, x_numeric_sparse])
y = df["label"].astype(str).factorize()[0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "RandomForest": RandomForestClassifier(n_estimators=70, max_depth=15, n_jobs=-1, class_weight='balanced'),
    "LinearSVC": LinearSVC(C=0.5, dual='auto'),
    "DecisionTree": DecisionTreeClassifier(max_depth=15, class_weight='balanced'),
    "LightGBM": LGBMClassifier(n_estimators=70, max_depth=5, n_jobs=-1, verbose=-1),
    "AdaBoost": AdaBoostClassifier(),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=70, max_depth=3),
    "ExtraTrees": ExtraTreesClassifier(n_estimators=70, max_depth=15, n_jobs=-1, class_weight='balanced')
}

accuracies = []
model_names = []

for name, model in models.items():
    print(f"Running {name}...")
    start = time.time()
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    elapsed = time.time() - start
    
    accuracies.append(acc)
    model_names.append(name)
    
    print(f" {name} Accuracy: {acc:.4f} (Time: {elapsed:.2f}s)")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("-" * 50)

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5]
}

rf_grid = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1), param_grid, cv=3)
rf_grid.fit(X_train, y_train)

print(f"Best Parameters found: {rf_grid.best_params_}")
print(f"Best Grid Accuracy: {rf_grid.best_score_:.4f}")

plt.figure(figsize=(10, 6))
sns.barplot(x=accuracies, y=model_names, palette="viridis")
plt.xlabel("Accuracy")
plt.title("Model Comparison: Sentiment Analysis on Movie Reviews")
plt.xlim(0.6, 1.0)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("Model_Comparison_Chart.png")
print("\nChart saved as 'Model_Comparison_Chart.png'. All done.")