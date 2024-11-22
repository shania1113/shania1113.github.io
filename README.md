## Steam User Preference and Game Recommendation using Simple Decision Tree Classifier

Lingxi Liu

Department of EPSS

Fall 2024 AOS C204 Final Project

## Introduction 

Steam is the biggest gaming platform, hosting over 50,000 video games and over 132 million users globally [1]. Each user can "recommend" or "not recommend" a game by simply clicking a button while reviewing a game. Using game features such as genres and release date, we can classify a user's reviewed games and find out the user's gaming preference.

The datasets I'm using are [Steam Games](https://www.kaggle.com/datasets/thedevastator/get-your-game-on-metacritic-recommendations-and) dataset provided by The Devastator and [Game Recommendations on Steam](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam) dataset provided by Anton Kozyriev, both on [Kaggle](https://www.kaggle.com/datasets). Steam Games include game features, and Game Recommendations describes user reviews.

I used the scikit-learn decision tree classifier to train and test the model on multiple users. I found that this model only works well for certain user with very clear game preferences (liking the same type of games); if a user plays a variety of different games and like all of them, the model isn't useful in recommending them new games or analyzing their preference. For some users who only give out positive or negative views, the model has a very high accuracy but is meaningless in recommending games.


## Data

1. FEATURES. The [Steam Games](https://www.kaggle.com/datasets/thedevastator/get-your-game-on-metacritic-recommendations-and) dataset contains one CSV file named "games-features-edit.csv" with over 12,000 games, mostly released before January 2017. This data is collected from Steam API and is under the Steam API term of use. The clean-up process of this file can be found in [this script](assets/game_feature_data.ipynb). The pandas data column example is shown below:
![](assets/IMG/featuredatahead.png){: width="500" }

First I cleaned out the games not released, and then I added a release-date feature by extracting the release year using Python's re module since the release-date column is not in uniform date form. Lastly, I filtered out the games that had no recommendation reviews and reduced the data length to 4846, almost 1/3 of the original. The cleaned-up data is exported to [this file](assets/game_feature_data.csv), later used to be combined with the target file.


2. TARGET. The [Game Recommendations on Steam](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam) dataset contains two files, "recommendations.csv" with over 41 million Steam game reviews and "games.csv" with all the game names, release dates, and IDs. The clean-up and compile process can be found in [this script](assets/recommendation.ipynb). The pandas data column example is shown below:
![](assets/IMG/targetdatahead.png){: width="500" }

Each user has a unique user ID, and this ID can be repeated many times (a user even reviewed more than 6000 games, so their reviews would take up 6000 rows in this dataset) for each game that they review. Similarly, the game ID repeats many times as different users review them. First, the two CSV files are merged together by game_id to fetch the game name and release date for later processing. Then, users who viewed more than 1000 games are selected from the original massive dataset, acting as our potential training target so that the model has enough data to be trained. Lastly, games released after 2016 are filtered out to match the features data. When this dataset is ready, the target data is merged with features data by game title, and luckily both dataset has almost identical game name formats. For decision tree processing, all "True/False" values are turned into "1/0". The final product is exported to [this file](assets/data_cleaned.csv), to be imported to the model script for analysis. The list of users is also obtained through the value_count() method for the pandas dataframe, exported to [this file](assets/list_of_users.csv).

## Modelling

First, an individual user is selected. There are 56 users who reviewed more than 1000 games total, and 53 of them reviewed more than 100 games released before 2017. Users are randomly selected for each run, and user id can be modified in a separate [file]((assets/userid.py) to ensure global variable availability, in case we need it in another script.

Here are some more details about the machine learning approach, and why this was deemed appropriate for the dataset. 

The model might involve optimizing some quantity. You can include snippets of code if it is helpful to explain things.

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
# select features and target
feature_columns = [
    'IsFree', 
    'GenreIsNonGame', 
    'GenreIsIndie', 
    'GenreIsAction', 
    'GenreIsAdventure', 
    'GenreIsCasual', 
    'GenreIsStrategy', 
    'GenreIsRPG', 
    'GenreIsSimulation', 
    'GenreIsEarlyAccess', 
    'GenreIsFreeToPlay', 
    'GenreIsSports', 
    'GenreIsRacing', 
    'GenreIsMassivelyMultiplayer', 
    'After2014', 
    'Expensive'
]
X = luckyguy[feature_columns]
y = luckyguy['is_recommended']

# split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

# train 
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# predictions
y_pred = model.predict(X_test)

# evaluate model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
```

This is how the method was developed.

## Results

Figure X shows... [description of Figure X].

## Discussion

From Figure X, one can see that... [interpretation of Figure X].

## Conclusion

Here is a brief summary. From this work, the following conclusions can be made:
* first conclusion
* second conclusion

Here is how this work could be developed further in a future project.

## Special Thanks

The code writing for this project had help from ChatGPT, including asking it coding questions on how to use specific functions, but the author writes major work. 

Some other coding questions are answered by the previous course ICCs and StackOverflow.

## References
[1] [Steam Statistics (2024) —Active Users & Market Share](https://www.demandsage.com/steam-statistics/)

[back](./)
