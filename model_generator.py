import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load data
data = pd.read_csv("medical_data.csv")
X = data["text"]
y = data["label"]

# Vectorize text
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))

# Save model and vectorizer
joblib.dump((model, vectorizer), "trained_model.joblib")



# Load the saved model and vectorizer
model, vectorizer = joblib.load("trained_model.joblib")

# Define new input data
# new_data = [
#     "I have been having migraines and headaches. I can't sleep. My whole body is shaking and shivering. I feel dizzy sometimes.",
#     "I have asthma and I get wheezing and breathing problems. I also have fevers, headaches, and I feel tired all the time."
# ]
new_data = ["chest pain and struggle in breathing the air"]

# Transform the new input data using the loaded vectorizer
new_data_vectorized = vectorizer.transform(new_data)


new_predictions = model.predict(new_data_vectorized)


print("Predictions:", new_predictions)
