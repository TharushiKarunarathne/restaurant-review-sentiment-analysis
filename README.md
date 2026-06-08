# Restaurant Review Sentiment Analysis using NLP

This project analyzes restaurant customer reviews and predicts whether a review expresses **positive** or **negative** sentiment using Natural Language Processing and Machine Learning.

## Project Overview

Customer reviews contain valuable information about customer satisfaction, food quality, service quality, and overall restaurant experience. This project uses restaurant review text and rating data to build a sentiment classification model.

The trained model is also integrated into a Streamlit web application where users can enter a restaurant review and receive a sentiment prediction with a confidence score and simple business insight.

## Objectives

- Load and explore restaurant review data
- Clean missing and invalid values
- Create sentiment labels from customer ratings
- Preprocess restaurant review text
- Convert text into numerical features using TF-IDF
- Train a sentiment classification model
- Evaluate model performance
- Save the trained model and vectorizer
- Build an interactive Streamlit web application

## Dataset

The dataset contains restaurant customer reviews, ratings, and related metadata.

Main columns used in this project:

| Column | Description |
|---|---|
| Review | Customer review text |
| Rating | Customer rating |

The original dataset does not directly contain sentiment labels. Therefore, sentiment labels were created using the rating column.

| Rating | Sentiment |
|---|---|
| Rating >= 4 | Positive |
| Rating <= 2 | Negative |
| Rating 3 or 3.5 | Removed as neutral |

Neutral ratings were removed to build a clearer binary sentiment classification model.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Natural Language Processing
- TF-IDF Vectorization
- Logistic Regression
- Streamlit

## Project Structure

```text
Restaurant-Review-Sentiment-Analysis/
│
├── data/
│   └── Restaurant reviews.csv
│
├── notebooks/
│   └── restaurant_sentiment_analysis.ipynb
│
├── app/
│   ├── app.py
│   └── style.css
│
├── models/
│   ├── sentiment_model.pkl
│   └── vectorizer.pkl
│
├── images/
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Machine Learning Workflow

1. Loaded the restaurant review dataset
2. Selected the `Review` and `Rating` columns
3. Removed missing review and rating values
4. Converted the rating column into numeric format
5. Created positive and negative sentiment labels from ratings
6. Cleaned review text by removing special characters, numbers, and extra spaces
7. Converted cleaned review text into numerical features using TF-IDF
8. Split the dataset into training and testing sets
9. Trained a Logistic Regression model
10. Evaluated the model using accuracy, classification report, and confusion matrix
11. Saved the trained model and TF-IDF vectorizer
12. Built a Streamlit application for sentiment prediction

## Text Preprocessing

The review text was cleaned using the following steps:

- Converted text to lowercase
- Removed numbers and special characters
- Removed extra spaces
- Removed empty reviews after cleaning

This preprocessing step helps the machine learning model focus on meaningful words in the review text.

## Model Used

The project uses **Logistic Regression** for sentiment classification.

Logistic Regression was selected because it is simple, efficient, and commonly used for text classification tasks when combined with TF-IDF features.

## Streamlit App

The Streamlit application allows users to enter a restaurant review and receive:

- Predicted sentiment
- Confidence score
- Business insight based on the result

Example input:

```text
The food was delicious and the staff were very friendly.
```

Example output:

```text
Positive Sentiment
```

## Business Value

This project can help restaurants understand customer feedback more efficiently. By automatically classifying reviews as positive or negative, restaurants can identify customer satisfaction trends, detect service issues, and make data-driven improvements.

Possible business uses include:

- Monitoring customer satisfaction
- Identifying negative customer experiences
- Improving food and service quality
- Supporting customer experience analysis
- Helping management make better operational decisions

## How to Run the Project

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app/app.py
```

## Sample Reviews for Testing

Positive review:

```text
The food was absolutely delicious and the staff were very friendly.
```

Negative review:

```text
The food was cold and the service was very slow.
```

Mixed review:

```text
The food was delicious but the service was very slow.
```

## Status

Completed.

## Author

**Tharushi Karunarathne**


