# NLP - Movie Genres Prediction
Natural Language Processing aims to interpret natural language data. Here is an implementation of a model designed to predict the most probable genres for a movie given its synopsis.

## Model
Many approaches have been considered when building this model. Here is an explanation of the implementation that gave the best results.
The main steps involved in this model are:
1. Cleaning the dataset
2. Extracting features
3. Classifying

### Cleaning the dataset
#### Remove duplicates
The first step is to make sure there aren't any duplicates in the dataset. This is done my comparing synopses. We do this to avoid submitting the same examples to our model in order to avoid overfitting.
#### Cleaning synopses
The second step cleans synopsis of each movie to increase the robustness of our model. Below is a non-exhaustive list of operations that can be implemented to clean a text.
- Removing stop-words since they do not contain any information
- Stemming or Lemmatizing the text as we don't want our model to learn that "fear" is a word often present in synopses of horror movies and thus is a good indicator for that genre, but fail to correctly label a new movie with the word "fearing" in its synopsis because no movie in the training set contained the word "fearing" in its synopsis.
- Keeping only alphanumeric characters. This depends heavily on the task at hand but in this case, we assumed that characters such #, $, &, ^, etc. didn't bear any relevant information for the classifier. We kept numeric characters as dates can be good indicators to predict war and sci-fi movies.
- Converting the case to lowercase


Hence a synopsis such as:
```
A 17 year-old aristocrat falls in love with a kind but poor artist aboard the luxurious, ill-fated R.M.S. Titanic
```
will be converted to:
```
17 year old aristocrat fall love kind poor artist aboard luxuri ill fate R M S Titan
```

### Extracting features
Many approaches can be considered here to extract features that will be used by the classifier.
Non-Negative Matrix Factorization (NMF) and Latent Dirichlet Allocation (LDA) can be used to extract latent topics from a text. However, one has to decide how many of those topics should be discovered. This is not as straightforward as one thinks. Those two approaches have been implemented but didn't lead to great results. Instead a TFI-IDF approach has been used to extract features. Many parameters also need to be tuned with this approach, though.

### Classifying


## Usage
### Train
To train the model you simply need to submit a dataset containing for each movie a synopsis and its associated genres. This dataset will be stored in a csv file or directly in a Pandas DataFrame. Two columns must be part of this dataset: `synopsis`, and `genres`. Other columns could also be included, such as the `id`, and the `title`, but these won't be used to train the model. Below is an example of such a dataset:

| synopsis | genres |
| --- | --- |
| A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival. | Adventure, Drama, Sci-Fi |
| An eight-year-old troublemaker must protect his house from a pair of burglars when he is accidentally left home alone by his family during Christmas vacation. | Comedy, Family |
| After a bitter divorce, an actor disguises himself as a female housekeeper to spend time with his children held in custody by his former wife. | Comedy, Drama, Family |
| In Nazi-occupied France during World War II, a plan to assassinate Nazi leaders by a group of Jewish U.S. soldiers coincides with a theatre owner's vengeful plans for the same. | Adventure, Drama, War |
| In 1954, a U.S. Marshal investigates the disappearance of a murderer who escaped from a hospital for the criminally insane. | Mystery, Thriller |

_Note that the different genres of a movie are separated by commas._

Then pass as argument the name of your csv file (or your Pandas DataFrame) to the `train` function to train your model.
```python
genres, tfidf_vectorizer, clf_log, clf_nn = train(dataset="train.csv")
```
or
```python
genres, *model = train(dataset=df_train)
```

If you want to split your dataset into a training set and a validation set, you can use the second parameter of the `train` function: `train_validation_split`. It refers to the percentage of the dataset that should be kept for validation. If this parameter's value is greater than 0.0, then the dataset will be split in two. The validation set will be used to assess the performance of your model via the Mean Accuracy Precision at k=5. Here is an example of usage:
```python
genres, *model = train(dataset=df_train, train_validation_split=0.2)
```

### Predict
Once the model is trained, the unique genres will be returned as well as the different classifiers and vectorizers constituing the model.
You can now test your model on a new dataset containing only synopses. Use the `predict` function. This function will create a submission.csv file containing the 5 predicted genres for each movie/synopsis.
```python
predict("test.csv", genres, tfidf_vectorizer, clf_log, clf_nn)
```
or
```python
predict(df_test, genres, *model)
```

## Requirements
- Pandas
- NLTK
- Sklearn
