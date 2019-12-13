"""
The goal of this challenge is to build a Machine Learning model to predict the
genres of a movie given its synopsis.
"""

from typing import List
import pandas as pd
import re
import sklearn
import nltk
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier

nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")


def remove_from_dataset(df: pd.DataFrame, genres: List[str]) -> pd.DataFrame:
    """
    Remove movies belonging to a list of genres from a DataFrame of movies.

    This function will remove from the given DataFrame df the genres given in
    the list genres. It then returns the pruned DataFrame. Note that it assumes
    that a column 'genres' is in the DataFrame. Genre is case sensitive.
    :param genres: List of genres to be filtered out. e.g.: ['Drama', 'Comedy']
    :param df: DataFrame which needs to be filtered
    :return: pd.DataFrame, df filtered
    """
    for genre in genres:
        df = df[~df["genres"].str.contains(genre)]
    return df


def clean_synopsis(
        synopsis: str,
        stemming: bool = True,
        stop_words: bool = True
        ) -> str:
    """
    Prepare a synopsis for a TF-IDF extractor.

    The function will clean a synopsis by converting it to lowercase, removing
    stop words, keeping only alphanumeric characters, and stemming each
    word. It then returns the "cleaned" synopsis. It assumes that the synopsis
    is in English.
    :param synopsis: string that needs to be cleaned
    :param stemming: Boolean parameter to decide if words should be stemmed
    :param stop_words: Boolean parameter to decide if stop words should be
    removed
    :return: str, synopsis cleaned
    """
    # remove any char that isn't alphanumeric
    synopsis = re.sub(r"[^a-zA-Z0-9]", " ", synopsis)
    # remove whitespaces
    synopsis = " ".join(synopsis.split())
    # convert text to lowercase
    synopsis = synopsis.lower()
    if stemming or stop_words:
        # extract words
        sentence_words = nltk.tokenize.word_tokenize(synopsis)
        if stemming:
            # stem each word
            ps = nltk.stem.PorterStemmer()
            sentence_words = [ps.stem(word) for word in sentence_words]
        if stop_words:
            # removing stop_words
            stop_words_list = set(nltk.corpus.stopwords.words("english"))
            sentence_words = [
                sentence_word
                for sentence_word in sentence_words
                if sentence_word not in stop_words_list
            ]
        # convert back to string
        synopsis = " ".join([str(sentence_word) for sentence_word in sentence_words])
    return synopsis


def train(train_file_name):
    """
    Train the machine learning model on a training dataset.

    This function is called on a dataset saved in the train_file_name csv file.
    The dataset is used for training the model which is then
    saved in global variables.
    :return: empty string
    """
    # Load the dataset
    df = pd.read_csv(train_file_name, sep=",")

    # Remove duplicate rows
    df.drop_duplicates(subset=["synopsis"], keep="first", inplace=True)

    # Remove IMAX as it is not a genre, and Film-noir as we don't have many
    # examples
    df = remove_from_dataset(df, ["IMAX", "Film-Noir"])

    # Get list of genres remaining in the dataset
    genre_vectorizer = sklearn.feature_extraction.text.CountVectorizer(
        tokenizer=lambda x: x.split(" ")
    )
    genre_vectorizer.fit(df["genres"])
    genres: List[str] = genre_vectorizer.get_feature_names()

    # Clean the synopsis of each movie
    df["clean_synopsis"] = df["synopsis"].apply(lambda x: clean_synopsis(x))

    # Extract TFIDF vectors for each movie
    tfidf_vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(
        max_df=0.95, min_df=2, max_features=6000, stop_words="english"
    )
    tfidf = tfidf_vectorizer.fit_transform(df["clean_synopsis"])

    # Extract the label vector for each movie
    y = genre_vectorizer.transform(df["genres"]).toarray()

    # Create the model: LogisticRegression -> NN
    clf_log = OneVsRestClassifier(
        LogisticRegression(C=1.8, solver="lbfgs", penalty="l2", max_iter=250)
    )
    clf_log.fit(tfidf, y)
    y_pred_int = clf_log.predict_proba(tfidf)
    clf_nn = MLPClassifier(
        hidden_layer_sizes=(18,),
        activation="relu",
        solver="adam",
        max_iter=50,
        alpha=0.75,
    )
    clf_nn.fit(y_pred_int, y)

    return tfidf_vectorizer, genres, clf_log, clf_nn


def predict(test_file_name, tfidf_vectorizer, genres, clf_log, clf_nn):
    """
    Predict the 5 most probable genres for a each movie of a given dataset.

    This function can be called on a dataset saved in a test_file_name csv file.
    A model is then used to predict the 5 most probable
    genres of each movie from the dataset.
    :return: empty string
    """
    # Load the dataset
    df = pd.read_csv(test_file_name, sep=",")

    # Clean the synopsis of each movie
    df["clean_synopsis"] = df["synopsis"].apply(lambda x: clean_synopsis(x))

    # Extract tf-idf features for each genre then predict
    tfidf = tfidf_vectorizer.transform(df["clean_synopsis"])
    y_pred_int = clf_log.predict_proba(tfidf)
    y_pred = clf_nn.predict_proba(y_pred_int)

    # Build the dataframe that will contain the predictions
    df_submit = pd.concat(
        [
            pd.DataFrame(
                {
                    "movie_id": [df["movie_id"][i]],
                    "predicted_genres": " ".join(
                        [
                            genre.title()
                            for _, genre in sorted(zip(pred, genres), reverse=True)
                        ][0:6]
                    ),
                }
            )
            for i, pred in enumerate(y_pred)
        ],
        ignore_index=True,
    )

    # Save the predictions in submission.csv
    df_submit.to_csv(path_or_buf="submission.csv", sep=",", index=False)
    return "A submission.csv file has been created"
