import joblib
from sklearn.datasets import fetch_20newsgroups

# Load the saved model
model = joblib.load("20newsgroups_model.joblib")

print("Model loaded successfully!")

# Load category names
data = fetch_20newsgroups(
    subset="train",
    remove=("headers", "footers", "quotes")
)

# Ask the user for text
text = [input("Enter a news text: ")]

# Make prediction
prediction = model.predict(text)

# Display predicted category
print("Predicted category:", data.target_names[prediction[0]])