## ARE_Miner: A web based automated tool to extract software requirements from app reviews.

###
The source code for the front end application can be found [here](https://github.com/jaymoz/are-miner-frontend)

### Introduction
This application provides an easy-to-use tool to automatically extract software requirements from app reviews. It uses encoder-decoder based model architecture to extract requirements. 

#### Server
This application provides two APIs as described below.

`/eda` : This API helps to get an exploratory analysis on the input data.

- The input is a CSV file containing `App`, `Review`, and `Date` columns. Sample file is shown below:
    ```
    App,Review,Date
    PhotoEditor,Too many ads and secondly erratic interface.,8/27/2018 20:55
    PhotoEditor,There are so many ads popping up that the app is now unusable because an ad pops up literally every time I close the last.,9/27/2018 20:55
    ```
- The output is a json object in the below format.
    ```
    {
        "avg_word_count": 16.0,
        "sentiment_distribution": {
            "sentiment": number_of_reviews,
            "positive": 2
        },
        "app_distribution": {
            "app_name": number_of_reviews,
            "PhotoEditor": 2
        },
        "time_distribution": {
            "year-month": number_of_reviews,
            "2018-08": 1,
            "2018-09": 1
        },
        "records": {
            "0": {
                "App": "PhotoEditor",
                "Review": "Too many ads and secondly erratic interface.",
                "Date": "8/27/2018 20:55",
                "sentiment": "positive",
                "word_count": 7
            },
            "1": {
                "App": "PhotoEditor",
                "Review": "There are so many ads popping up that the app is now unusable because an ad pops up literally every time I close the last.",
                "Date": "9/27/2018 20:55",
                "sentiment": "positive",
                "word_count": 25
            }
        }
    }
    ```

`/extract_requirements` : This API helps to extract requirements and get exploratory analysis on the output.

- The input is a CSV file containing `App`, `Review`, and `Date` columns. Sample file is shown below:
    ```
    App,Review,Date
    PhotoEditor,Too many ads and secondly erratic interface.,8/27/2018 20:55
    PhotoEditor,There are so many ads popping up that the app is now unusable because an ad pops up literally every time I close the last.,9/27/2018 20:55
    ```
- The output is a json object in the below format.
    ```
    {
        "records": {
            "0": {
                "App": "PhotoEditor",
                "Review": "Too many ads and secondly erratic interface.",
                "Date": "8/27/2018 20:55",
                "requirements": []
            },
            "1": {
                "App": "PhotoEditor",
                "Review": "There are so many ads popping up that the app is now unusable because an ad pops up literally every time I close the last.",
                "Date": "9/27/2018 20:55",
                "requirements": [
                    {
                        "requirement": "so many ads",
                        "sentiment": "positive"
                    },
                    {
                        "requirement": "that",
                        "sentiment": "positive"
                    },
                    {
                        "requirement": "app is",
                        "sentiment": "positive"
                    },
                    {
                        "requirement": "every",
                        "sentiment": "positive"
                    }
                ]
            }
        },
        "distribution_over_apps": {
            "app_name": number_of_requirements,
            "PhotoEditor": 4
        },
        "word_count_distribution": {
            "word_count": number_of_requirements,
            "3": 1,
            "1": 2,
            "2": 1
        },
        "sentiment_distribution": {
            "sentiment": number_of_requirements,
            "positive": 4
        },
        "distribution_over_reviews": {
            "number_of_requirements": number_of_reviews,
            "0": 1,
            "4": 1
        },
        "distribution_over_time": {
            "year-month": number_of_requirements,
            "2018-08": 0,
            "2018-09": 4
        }
    }
    ```
