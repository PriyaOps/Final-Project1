import streamlit as st
import pandas as pd
import string
import nltk
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and w.isalpha()]
    return ' '.join(tokens)

def predict_sentiment(text, model, vectorizer):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    label_map_rev = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return label_map_rev[pred]

st.set_page_config(page_title="Sentiment Analyzer", layout="centered")
st.title(" Customer Review Sentiment Analyzer")

uploaded_file = st.file_uploader("📂 Upload CSV with 'Review_Text' and 'Rating'", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Basic validation
    if 'Review_Text' not in df.columns or 'Rating' not in df.columns:
        st.error("❌ CSV must have 'Review_Text' and 'Rating' columns.")
    else:
        st.success("✅ File uploaded successfully!")

        # Cleaning & labeling
        with st.spinner("Processing..."):
            df['Cleaned_Text'] = df['Review_Text'].apply(clean_text)
            df['Sentiment_Label'] = df['Rating'].apply(
                lambda x: 'Negative' if x <= 2 else 'Neutral' if x == 3 else 'Positive'
            )

            label_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
            y = df['Sentiment_Label'].map(label_map)

            tfidf = TfidfVectorizer(max_features=3000)
            X = tfidf.fit_transform(df['Cleaned_Text'])

            smote = SMOTE(random_state=42)
            X_bal, y_bal = smote.fit_resample(X, y)

            X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.2, random_state=42)
            model = RandomForestClassifier(n_estimators=150, random_state=42)
            model.fit(X_train, y_train)

            #y_pred = model.predict(X_test)
            #accuracy = accuracy_score(y_test, y_pred)
            #report = classification_report(y_test, y_pred, target_names=label_map.keys(), output_dict=True)

            # Save model
            joblib.dump(model, "sentiment_model.pkl")
            joblib.dump(tfidf, "tfidf_vectorizer.pkl")

        #st.subheader("✅ Model Trained Successfully!")
        #st.write(f"**Accuracy:** {accuracy * 100:.2f}%")
        #st.json(report)

        # Input for prediction
        st.markdown("Enter Your Own Review")
        user_input = st.text_area("Write your review here:")
        if st.button("Analyze Sentiment"):
            sentiment = predict_sentiment(user_input, model, tfidf)
            st.success(f"Predicted Sentiment: **{sentiment}**")
else:
    st.info("Please upload a CSV file to get started.")
