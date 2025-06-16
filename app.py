import pandas as pd
import random
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Step 1: Load dataset
df = pd.read_csv("twitter_dataset.csv")

# Step 2: Show column names
print("Columns in CSV:", df.columns)

# Step 3: Preprocess text
def preprocess_text(text):
    if pd.isnull(text):
        return ""
    text = text.lower()
    return ''.join([char for char in text if char.isalnum() or char.isspace()])

df['Text'] = df['Text'].apply(preprocess_text)

# Step 4: Add dummy sentiment labels if not present
if 'sentiment' not in df.columns:
    print("No 'sentiment' column found. Adding dummy labels.")
    df['sentiment'] = [random.choice(['positive', 'negative']) for _ in range(len(df))]

# Step 5: Split and train
X_train, X_test, y_train, y_test = train_test_split(df['Text'], df['sentiment'], test_size=0.2, random_state=42)
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Step 6: Setup Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['user_input']
    processed_input = preprocess_text(user_input)
    input_vector = vectorizer.transform([processed_input])
    prediction = model.predict(input_vector)[0]
    return render_template('index.html', prediction=prediction, input_text=user_input)

if __name__ == '__main__':
    app.run(debug=True)
