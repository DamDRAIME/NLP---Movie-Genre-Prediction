# NLP - Movie Genres Prediction
Natural Language Processing aims to interpret natural language data. Here is an implementation of a model designed to predict the most probable genres for a movie given its synopsis.

## Model

## Usage
To train the model you simply need to submit a dataset containing for each movie a synopsis and its associated genres. This dataset will be stored in a csv file. Use the `train` function to train your model.
```python
genres, tfidf_vectorizer, clf_log, clf_nn = train("train.csv")
```
Once the model is trained, the unique genres will be returned as well as the different classifiers and vectorizers constituing the model.
You can now test your model on a new dataset containing only synopses. Use the `predict` function. This function will create a submission.csv file containing the 5 predicted genres for each movie/synopsis.
```python
predict("test.csv", genres, tfidf_vectorizer, clf_log, clf_nn)
```

## Requirements
- Pandas
- NLTK
- Sklearn
