"""
The goal of this challenge is to build a Machine Learning model to predict the
genres of a movie given its synopsis.
"""

from typing import List, Tuple, Union
import pandas as pd
import statistics as stat
import re
import sklearn
import nltk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")

dataset = Union[str, pd.DataFrame]


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


def apk(
        actual: List[List[str]],
        predicted: List[List[str]],
        k: int=10
        ) -> float:
    """
    Compute the average precision at k.

    This function will compute the average precision at k between two lists of
    items. Note that the order of the items in the lists does not matter. Also
    the number of items in the two lists can be different.
    :param actual: list of list of items that are to be predicted
    :param predicted: list of list of items that are predicted
    :param k: the maximum number of predicted elements
    :return: float, average precision at k
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(
        actual: List[List[str]],
        predicted: List[List[str]],
        k: int=10
        ) -> float:
    """
    Compute the mean average precision at k.

    This function will compute the mean average precision at k between two
    lists of items. Note that the order of the items in the lists does not
    matter. Also the number of items in the two lists can be different.
    :param actual: list of list of items that are to be predicted
    :param predicted: list of list of items that are predicted
    :param k: the maximum number of predicted elements
    :return: float, mean average precision at k
    """
    return stat.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def train(
        train_dataset: dataset,
        train_validation_split: float=0.0,
        verbose: int=1
        ) -> Tuple[
        List[str],
        TfidfVectorizer,
        OneVsRestClassifier,
        MLPClassifier]:
    """
    Train the machine learning model on a training dataset.

    This function is called on a dataset stored in the train_dataset parameter.
    It can either be a DataFrame or a csv file.
    The dataset is used for training the model which is then returned along
    with the unique genres of the provided dataset.
    :param train_dataset: name of the csv file containing the dataset or a
    Pandas DataFrame
    :param train_validation_split: float defining the percentage of the
    dataset that should be reserved for validation. By default there is not
    split (optional)
    :param verbose: parameter to control how much information you want to have
    during training (optional)
    :return: genres, *model
    """
    # Load the dataset
    df = pd.read_csv(train_dataset, sep=",") if type(train_dataset) == str \
        else train_dataset

    # Remove duplicate rows
    if verbose >= 1:
        print('Removing duplicated data...')
    df.drop_duplicates(subset=["synopsis"], keep="first", inplace=True)

    # Get list of genres remaining in the dataset
    genre_vectorizer = CountVectorizer(
        tokenizer=lambda x: x.split(" ")
    )
    genre_vectorizer.fit(df["genres"])
    genres: List[str] = genre_vectorizer.get_feature_names()
    if verbose >= 1:
        print('Unique genres: {}'.format(genres))

    # Splitting dataset
    if train_validation_split > 0.0:
        if verbose >= 1:
            print('Splitting dataset into a training set and a validation set')
        df, df_val = train_test_split(df, test_size=train_validation_split)
        df_val.reset_index(inplace=True)

    # Clean the synopsis of each movie
    if verbose >= 1:
        print('Cleaning synopses...')
    df["clean_synopsis"] = df["synopsis"].apply(lambda x: clean_synopsis(x))

    # Extract TFIDF vectors for each movie
    if verbose >= 1:
        print('Extracting TFIDF vectors for each movie...')
    tfidf_vectorizer = TfidfVectorizer(
        max_df=0.95, min_df=2, max_features=6000, stop_words="english"
    )
    tfidf = tfidf_vectorizer.fit_transform(df["clean_synopsis"])

    # Extract the label vector for each movie
    y = genre_vectorizer.transform(df["genres"]).toarray()

    # Create the model: LogisticRegression -> NN
    if verbose >= 1:
        print('Creating and training the model...')
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

    # Get the mean average precision score of the model on the validation set
    if train_validation_split > 0.0 and verbose >= 1:
        print('Assessing model\'s performance on validation set...')
        df_pred = predict(df_val, genres, tfidf_vectorizer, clf_log, clf_nn)
        map = mapk(df_val["genres"], df_pred["predicted_genres"], k=5)
        print('Mean Average Precision at 5 on validation set = {}'.format(map))

    return genres, tfidf_vectorizer, clf_log, clf_nn


def predict(
        test_dataset: dataset,
        genres: List[str],
        tfidf_vectorizer: TfidfVectorizer,
        clf_log: OneVsRestClassifier,
        clf_nn: MLPClassifier):
    """
    Predict the 5 most probable genres for a each movie of a given dataset.

    This function can be called on a dataset stored in a csv file or directly
    in a Pandas DataFrame.
    A model is then used to predict the 5 most probable genres of each movie
    from the dataset.
    :param test_dataset: name of the csv file or the Pandas DataFrame
    containing the movies' synopses for which the genres need to be predicted
    :param genres: list of the different unique genres
    :param tfidf_vectorizer: vectorizer to extract features from synopses
    :param clf_log: a OneVsRest classifier for the first part of the model
    :param clf_nn: a neural network for the second and last part of the model
    :return: empty string
    """
    # Load the dataset
    df = pd.read_csv(test_dataset, sep=",") if type(test_dataset) == str \
        else test_dataset

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

    return df_submit
