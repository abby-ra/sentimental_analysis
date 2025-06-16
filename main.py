import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
df = pd.read_csv(r"D:\Project\sentimental_analyis\twitter_dataset.csv")  # Update the path if needed

# 2. Assign dummy sentiment values for testing (you can replace this with actual sentiment labels if available)
df['sentiment'] = [random.choice(['positive', 'negative']) for _ in range(len(df))]

# 3. Preprocess the Text data
def preprocess_Text(text):
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])
    return text

df['Text'] = df['Text'].apply(preprocess_Text)

# 4. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['Text'], df['sentiment'], test_size=0.2, random_state=42)

# 5. Vectorize the Text data using Bag of Words
vectorizer = CountVectorizer()
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

# 6. Train a Multinomial Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_bow, y_train)

# 7. Make predictions on the test set
y_pred = model.predict(X_test_bow)

# 8. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 9. Example usage: Predicting sentiment of a new sentence
new_sentence = "This is a fantastic experience!"
new_sentence_bow = vectorizer.transform([preprocess_Text(new_sentence)])
predicted_sentiment = model.predict(new_sentence_bow)[0]
print(f"\nThe sentiment of '{new_sentence}' is: {predicted_sentiment}")
