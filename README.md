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
**Hi class, welcome to the AOS C111/204 final project!** <img align="right" width="220" height="220" src="/assets/IMG/template_logo.png">

For this project, you will be applying your skills to train a machine learning model using real-world data, then publishing a report on your own website.

* To get data for your project, you could:
  * use **your own data** from a separate research activity
  * **scour the internet** to find something original, then preprocess it yourself - see the Module Overview on BruinLearn for some resources
  * browse an archive of data designed for machine learning problems, such as the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/datasets)

* Your report should be written in a scientific language and style. [This template page](/project.md) gives an example structure that you could use, but feel free to make it your own. See Bruinlearn for some examples from previous students.

Your website will be a great addition to your CV, and a place to host future projects too since it doubles as a GitHub repository. The first step is to set up a project website like this one by following the instructions below. 

## How does this website work?

First, check out the Github repository for this site: [https://github.com/atmosalex/atmosalex.github.io/](https://github.com/atmosalex/atmosalex.github.io/).

Using GitHub pages, you can write a website using markdown syntax - the same syntax we use to write comments in Google Colab notebooks. GitHub pages then takes the markdown file and renders it as a web page using a Jekyll theme. The markdown source code for this page [is shown here](https://github.com/atmosalex/atmosalex.github.io/blob/main/README.md?plain=1).

## Setting up your Project Website

### How to copy this site as a template
1. Create [a GitHub account](https://github.com/)
2.	Go to [https://github.com/atmosalex/atmosalex.github.io/](https://github.com/atmosalex/atmosalex.github.io/) and click *Use this template*, then **Create a new repository**. [![screenshot][1]][1]
3.	In the box that says *Repository name*, write your **Github username**, followed by **.github.io**, as shown in the screenshot below. Then click **Create repository** at the bottom. [![screenshot][2]][2]
4.	Go to the *Settings* tab, then click *Pages* (under *Code and automation*). In the *Build and deployment* section, under **Branch**, select "main" and click save (if it isn't already selected). It should look like this: [![screenshot][3]][3]
5.	Click the *Actions* tab at the top of the page and check that the build and deployment action has finished. Once it has, navigate to **[your username].github.io** to see your site, which should be a copy of this one! If you cannot see an *Actions* tab, just wait a few minutes then go to your URL to check it is live.

Now you are ready to customize your site! To add your name to the site, go to your repository page on Github, click `_config.yml`, and edit it to replace the temporary title with your name, etc. When we make changes to a project on Github, we have to **commit** the new version of each file. Github keeps track of all the changes we make, making it easy to roll back (i.e. return the project to a previous commit).

[1]: /assets/IMG/instr_new.png
[2]: /assets/IMG/instr_template.png
[3]: /assets/IMG/instr_bd.png

### How to change the theme (optional)
1.	You can choose any theme [listed on this page](https://pages.github.com/themes/), though some do not work as well on mobile devices.
2.	From GitHub, edit `_config.yml` and replace the `theme:` line with `theme: jekyll-theme-name` where `name` is the name of the theme from the above list. **For the `minima` theme, use a shortened preface like so `theme: minima`**, the others seem to need the whole preface `theme: jekyll-theme-`. You can check the *Actions* tab (as in step 5. above) to make sure the site is building successfully.

### How to change your site logo (optional)
1. Some themes, such as `jekyll-theme-minimal`, show a logo. In your repository, upload a logo or profile picture to the `assets/IMG/` directory
2. Open `_config.yml` and modify the line `logo: /assets/IMG/template_logo.png` to point to your new image

***

## Guide to Adding Content
* Your repository's `README.md` file (the file you are reading now) acts like a home page. Replace its contents with whatever you want the world to see by editing the file on GitHub.
* If you want to turn this page into a CV or blog, etc., it may be useful to refer to a [guide for writing Markdown](https://www.markdownguide.org/basic-syntax/).
* You can create other markdown files (.md) in your repository and navigate to them from this page using links, i.e.: [here is a link to another file, `project.md`](project.md)
* When editing a markdown file on GitHub, it is useful to wrap text by selecting the *Soft wrap* option as shown: ![screenshot](/assets/IMG/instr_wrap.png)
* If you want to get even more technical, you can also write HTML in your .md files, and GitHub Pages will render it. For example, the image below is displayed by writing the following (edit this file to see!): `<img align="right" width="200" height="200" src="/assets/IMG/template_frog.png">`
<img align="right" width="337" height="200" src="/assets/IMG/template_frog.png"> 

***

## Delivering your Project

Your final project is delivered in two components: a report and your code.

### Report

Your report should be **delivered via your website**. Submit a link to your website on BruinLearn so that your instructor can browse it to find your report. 

To make this simple, you can write the report using a word processor or Latex, then export it as a .pdf file and upload it to the `assets` directory. You can then link to it [like so](/assets/project_demo.pdf). However, you can also type the report directly onto the website using another markdown page - [here is](/project.md) a template for that.

### Code

A link to your code must be submitted on BruinLearn, and the course instructor must be able to download your code to mark it. The code could be in a Google Colab notebook (make sure to *share* the notebook so access is set to **Anyone with the link**), or you could upload the code into a separate GitHub repository, or you could upload the code into the `assets` directory of your website and link to it. 
