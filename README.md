# NLP - Movie Genres Prediction
Natural Language Processing aims to interpret natural language data. Here is an implementation of a model designed to predict the most probable genres for a movie given its synopsis.

## Model

## Usage
To train the model you simply need to submit a dataset containing for each movie a synopsis and its associated genres. This dataset will be stored in a csv file. Two columns must be part of this dataset: `synopsis`, and `genres`. Other columns could also be included, such as the `id`, and the `title`, but these won't be used to train the model. Below is an example of such a dataset:

| synopsis | genres |
| --- | --- |
| A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival. | Adventure, Drama, Sci-Fi |
| An eight-year-old troublemaker must protect his house from a pair of burglars when he is accidentally left home alone by his family during Christmas vacation. | Comedy, Family |
| After a bitter divorce, an actor disguises himself as a female housekeeper to spend time with his children held in custody by his former wife. | Comedy, Drama, Family |
| In Nazi-occupied France during World War II, a plan to assassinate Nazi leaders by a group of Jewish U.S. soldiers coincides with a theatre owner's vengeful plans for the same. | Adventure, Drama, War |
| In 1954, a U.S. Marshal investigates the disappearance of a murderer who escaped from a hospital for the criminally insane. | Mystery, Thriller |

_Note that the different genres of a movie are separated by commas._

Then pass as argument the name of your csv file to the `train` function to train your model.
```python
genres, tfidf_vectorizer, clf_log, clf_nn = train("train.csv")
```
or
```python
genres, *model = train("train.csv")
```

Once the model is trained, the unique genres will be returned as well as the different classifiers and vectorizers constituing the model.
You can now test your model on a new dataset containing only synopses. Use the `predict` function. This function will create a submission.csv file containing the 5 predicted genres for each movie/synopsis.
```python
predict("test.csv", genres, tfidf_vectorizer, clf_log, clf_nn)
```
or
```python
predict("test.csv", genres, *model)
```

## Requirements
- Pandas
- NLTK
- Sklearn
