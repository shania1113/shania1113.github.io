## Steam User Preference and Game Recommendation using Simple Decision Tree Classifier

Lingxi Liu

Department of EPSS

Fall 2024 AOS C204 Final Project

## Introduction 

Steam is the biggest gaming platform, hosting over 50,000 video games and over 132 million users globally [1]. Each user can "recommend" or "not recommend" a game by simply clicking a button while reviewing a game. Using game features such as genres and release dates, we can classify a user's reviewed games and find out the user's gaming preference.

The datasets I'm using are [Steam Games](https://www.kaggle.com/datasets/thedevastator/get-your-game-on-metacritic-recommendations-and) dataset provided by The Devastator and [Game Recommendations on Steam](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam) dataset provided by Anton Kozyriev, both on [Kaggle](https://www.kaggle.com/datasets). Steam Games includes game features, and Game Recommendations describes user reviews.

I used the scikit-learn decision tree classifier to train and test the model on multiple users. I found that this model only works well for certain users with very clear game preferences (liking the same type of games); if a user plays a variety of different games and likes all of them, the model isn't useful in recommending them new games or analyzing their preference. For some users who only give out positive or negative views, the model has a very high accuracy but is meaningless in recommending games.


## Data

#### FEATURES
The [Steam Games](https://www.kaggle.com/datasets/thedevastator/get-your-game-on-metacritic-recommendations-and) dataset contains one CSV file named "games-features-edit.csv" with over 12,000 games, mostly released before January 2017. This data is collected from Steam API and is under the Steam API term of use. The clean-up process of this file can be found in [this script](assets/game_feature_data.ipynb). The example pandas data column is shown below:

<div style="overflow-x: auto;">

| ResponseName              |   ReleaseDate |   Metacritic |   RecommendationCount | IsFree   | GenreIsNonGame   | GenreIsIndie   | GenreIsAction   | GenreIsAdventure   | GenreIsCasual   | GenreIsStrategy   | GenreIsRPG   | GenreIsSimulation   | GenreIsEarlyAccess   | GenreIsFreeToPlay   | GenreIsSports   | GenreIsRacing   | GenreIsMassivelyMultiplayer   |   PriceInitial |   After2014 |   Expensive |
|--------------------------|--------------|-------------|----------------------|---------|-----------------|---------------|----------------|-------------------|----------------|------------------|-------------|--------------------|---------------------|--------------------|----------------|----------------|------------------------------|---------------|------------|------------|
| Counter-Strike            |          2000 |           88 |                 68991 | False    | False            | False          | True            | False              | False           | False             | False        | False               | False                | False               | False           | False           | False                         |           9.99 |           0 |           0 |
| Team Fortress Classic     |          1999 |            0 |                  2439 | False    | False            | False          | True            | False              | False           | False             | False        | False               | False                | False               | False           | False           | False                         |           4.99 |           0 |           0 |
| Day of Defeat             |          2003 |           79 |                  2319 | False    | False            | False          | True            | False              | False           | False             | False        | False               | False                | False               | False           | False           | False                         |           4.99 |           0 |           0 |
| Deathmatch Classic        |          2001 |            0 |                   888 | False    | False            | False          | True            | False              | False           | False             | False        | False               | False                | False               | False           | False           | False                         |           4.99 |           0 |           0 |
| Half-Life: Opposing Force |          1999 |            0 |                  2934 | False    | False            | False          | True            | False              | False           | False             | False        | False               | False                | False               | False           | False           | False                         |           4.99 |           0 |           0 |

</div>

First I cleaned out the games not released, and then I added a release-date feature by extracting the release year using Python's re module since the release-date column is not in uniform date form. Lastly, I filtered out the games that had no recommendation reviews and reduced the data length to 4846, almost 1/3 of the original. The cleaned-up data is exported to [this file](assets/game_feature_data.csv), later used to be combined with the target file.

#### TARGET
The [Game Recommendations on Steam](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam) dataset contains two files, "recommendations.csv" with over 41 million Steam game reviews and "games.csv" with all the game names, release dates, and IDs. The clean-up and compile process can be found in [this script](assets/recommendation.ipynb). The example pandas data column of the merged product is shown below:

<div style="overflow-x: auto;">

|   app_id |   helpful |   funny | date       | is_recommended   |   hours |          user_id |   review_id | date_release        | title                             |   year |
|---------:|----------:|--------:|:-----------|:-----------------|--------:|-----------------:|------------:|:--------------------|:----------------------------------|-------:|
|    13500 |         0 |       0 | 2021-03-29 | True             |     0.1 | 748899           | 1.42596e+07 | 2008-11-21 00:00:00 | Prince of Persia: Warrior Within™ |   2008 |
|    13500 |         7 |       2 | 2018-12-29 | True             |   168.3 |      1.1283e+07  | 1.42807e+07 | 2008-11-21 00:00:00 | Prince of Persia: Warrior Within™ |   2008 |
|    13500 |         8 |       3 | 2021-11-12 | True             |     1.2 |      1.15536e+07 | 1.4281e+07  | 2008-11-21 00:00:00 | Prince of Persia: Warrior Within™ |   2008 |
|    13500 |         3 |       0 | 2020-09-28 | False            |     1.2 |      1.2823e+07  | 1.42925e+07 | 2008-11-21 00:00:00 | Prince of Persia: Warrior Within™ |   2008 |
|    13500 |         0 |       0 | 2013-08-25 | True             |    17.3 |      1.1681e+07  | 3.00331e+07 | 2008-11-21 00:00:00 | Prince of Persia: Warrior Within™ |   2008 |

</div>

Each user has a unique user ID, and this ID can be repeated many times (a user even reviewed more than 6000 games, so their reviews would take up 6000 rows in this dataset) for each game that they review. Similarly, the game ID repeats many times as different users review them. First, the two CSV files are merged together by game_id to fetch the game name and release date for later processing. Then, users who viewed more than 1000 games are selected from the original massive dataset, acting as our potential training target so that the model has enough data to be trained. Lastly, games released after 2016 are filtered out to match the features data. When this dataset is ready, the target data is merged with features data by game title, and luckily both dataset has almost identical game name formats. For decision tree processing, all "True/False" values are turned into "1/0". The final product is exported to [this file](assets/data_cleaned.csv), to be imported to the model script for analysis. The list of users is also obtained through the value_count() method for the pandas dataframe, exported to [this file](assets/list_of_users.csv).

## Modelling

First, an individual user is selected. There are 56 users who reviewed more than 1000 games total, and 53 of them reviewed more than 100 games released before 2017. Users are randomly selected for each run, and the user id can be modified in a separate [file]((assets/userid.py) to ensure global variable availability, in case we need it in another script.

Once the user ID is obtained, a separate dataframe can be created like so:

```python
# import dataset to work with
df = pd.read_csv('data_cleaned.csv')

# extract specific user
luckyguy = df[df['user_id'] == user_id]
```

There are many machine learning tools that are suitable for recommendation systems, such as SVD (single value decomposer)[2] or CF (collaborative filtering)[3]. SVD would be the ideal method for creating an actual recommendation system for every user in the 41 million-long datasets since it compares the recommendations from users who play similar games to provide a reliable prediction of games that anyone may like. However, SVD requires the data in a uniform matrix (for example, if there are 500 games to analyze, then each user will need to have 500 entries; if they never reviewed that game, then the entry value would be zero), and to process our massive amount of data to this shape requires more work that can be done in the limited time, with limited memories on my laptop. Due to the variety of the games needed for this analysis, I want as many games as possible in my training dataset; therefore I need to cut the number of users to make this model work. Cutting the number of users can result in not enough comparable users, thus the best method for this project would be to use a simple decision tree classifier on an individual user. The model script can be found in [this file](assets/decisiontree.ipynb).

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
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

# classification report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# calculate RMSE
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
```

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

The code writing for this project had help from [ChatGPT](https://chatgpt.com/).

Some other coding questions are answered by the previous course ICCs and StackOverflow.

## References
[1] [Steam Statistics (2024) —Active Users & Market Share](https://www.demandsage.com/steam-statistics/)

[2] https://github.com/Michalos88/Game-Recommendation-System

[3] [The Steam Engine: A Recommendation System for Steam Users](https://brandonlin.com/steam.pdf)

[back](./)
