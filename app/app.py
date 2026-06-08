import streamlit as st
import pickle
import re
import os

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Restaurant Review Sentiment Analysis",
    page_icon="🍽️",
    layout="wide"
)

# =========================
# Load CSS
# =========================
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "style.css")
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# =========================
# Text Cleaning
# =========================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# =========================
# Load Model and Vectorizer
# =========================
model_path = os.path.join(os.path.dirname(__file__), "..", "models", "sentiment_model.pkl")
vectorizer_path = os.path.join(os.path.dirname(__file__), "..", "models", "vectorizer.pkl")

with open(model_path, "rb") as file:
    model = pickle.load(file)

with open(vectorizer_path, "rb") as file:
    vectorizer = pickle.load(file)

# =========================
# Hero Section
# =========================
st.markdown(
    """
    <div class="hero-section">
        <div class="hero-badge">NLP Powered Review Intelligence</div>
        <h1>🍽️ Restaurant Review Sentiment Analyzer</h1>
        <p>
            Turn customer feedback into instant business insights using Natural Language Processing
            and Machine Learning.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================
# Stats Cards
# =========================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="stat-card">
            <h3>10K+</h3>
            <p>Restaurant Reviews</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class="stat-card">
            <h3>TF-IDF</h3>
            <p>Text Vectorization</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        """
        <div class="stat-card">
            <h3>ML</h3>
            <p>Sentiment Prediction</p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# =========================
# Main Section
# =========================
left_col, right_col = st.columns([1.25, 0.75])

with left_col:
    st.markdown(
        """
        <div class="section-card">
            <h2>📝 Analyze a Customer Review</h2>
            <p class="section-subtitle">
                Enter a restaurant review below and the model will predict whether it is positive or negative.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    sample_reviews = {
        "Positive Example": "The food was delicious and the staff were very friendly.",
        "Negative Example": "The food was cold and the service was very slow.",
        "Mixed Example": "The food was tasty but the waiting time was too long."
    }

    selected_sample = st.selectbox(
        "Choose a sample review or type your own:",
        ["Type my own review"] + list(sample_reviews.keys())
    )

    default_text = ""
    if selected_sample != "Type my own review":
        default_text = sample_reviews[selected_sample]

    review = st.text_area(
        "Customer Review",
        value=default_text,
        height=180,
        placeholder="Example: The food was amazing, but the service was very slow..."
    )

    predict_button = st.button("✨ Analyze Sentiment")

with right_col:
    st.markdown(
        """
        <div class="info-panel">
            <h3>💡 How it works</h3>
            <div class="step-box">
                <span>1</span>
                <p>Review text is cleaned</p>
            </div>
            <div class="step-box">
                <span>2</span>
                <p>Text is converted using TF-IDF</p>
            </div>
            <div class="step-box">
                <span>3</span>
                <p>ML model predicts sentiment</p>
            </div>
            <div class="step-box">
                <span>4</span>
                <p>Business insight is generated</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================
# Prediction Result
# =========================
if predict_button:
    if review.strip() == "":
        st.warning("Please enter a review before analyzing sentiment.")
    else:
        cleaned_review = clean_text(review)
        vectorized_review = vectorizer.transform([cleaned_review])

        prediction = model.predict(vectorized_review)[0]
        probabilities = model.predict_proba(vectorized_review)[0]
        confidence = probabilities.max() * 100

        st.markdown("<br>", unsafe_allow_html=True)

        if prediction == "Positive":
            st.markdown(
                f"""
                <div class="result-card positive-result">
                    <div class="result-icon">😊</div>
                    <h2>Positive Sentiment</h2>
                    <p>This review shows customer satisfaction and a good dining experience.</p>
                    <div class="confidence-box">
                        <h4>Confidence Score</h4>
                        <div class="progress-bg">
                            <div class="progress-fill positive-fill" style="width:{confidence}%;"></div>
                        </div>
                        <h3>{confidence:.2f}%</h3>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                """
                <div class="business-card positive-business">
                    <h3>📈 Business Recommendation</h3>
                    <p>
                        The restaurant should maintain the quality factors mentioned in the review,
                        such as food taste, service, cleanliness, or staff behavior. Positive feedback
                        can also be used for marketing and customer trust building.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:
            st.markdown(
                f"""
                <div class="result-card negative-result">
                    <div class="result-icon">😞</div>
                    <h2>Negative Sentiment</h2>
                    <p>This review indicates dissatisfaction and areas that may need improvement.</p>
                    <div class="confidence-box">
                        <h4>Confidence Score</h4>
                        <div class="progress-bg">
                            <div class="progress-fill negative-fill" style="width:{confidence}%;"></div>
                        </div>
                        <h3>{confidence:.2f}%</h3>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                """
                <div class="business-card negative-business">
                    <h3>⚠️ Business Recommendation</h3>
                    <p>
                        The restaurant should carefully review this feedback and identify possible
                        issues related to service speed, staff behavior, food quality, hygiene,
                        pricing, or customer waiting time.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

# =========================
# Footer
# =========================
st.markdown(
    """
    <div class="footer">
        <p>Built using Python · NLP · TF-IDF · Logistic Regression · Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)