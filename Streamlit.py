import streamlit as st
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import tweepy
import json
from tweepy import OAuthHandler
import re
import textblob
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import openpyxl
import time
import tqdm

# To Hide Warnings
st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showPyplotGlobalUse', False)
# Viz Pkgs
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')
import seaborn as sns

# sns.set_style('darkgrid')


STYLE = """
<style>
img {
    max-width: 100%;
}
</style> """


def main():
    """ Common ML Dataset Explorer """
    st.title("Hello!, welcome")
    st.subheader("my name's Kuks I am your sentiment analyzer,😀")

    html_temp = """ """
    st.markdown(html_temp, unsafe_allow_html=True)

    consumer_key = "qpon31IjVDvAkGxDCnbstC2c7"
    consumer_secret = "6k14epQwA3S1YxWCeVcJZ0jWmbULoq6x4TorVe6A7KbmVZihrd"
    access_token = "903716376220762113-BKjEbXsGq9rDGoBLnHJJHQLFS9wVVxA"
    access_token_secret = "iEkhnVg1rbE1LT55ck0dIN7KHbvF5ChS1E4S04lDBtDie"
    bearer_token = "AAAAAAAAAAAAAAAAAAAAAKDPbQEAAAAApRgaWrZ%2FnLqFNR9L3J8ViSV8Wuw%3DlDF4neqGi2K2Tjy88KGCjrXndI1f3rh1E7EXKN4WRziSqxd4xi"

    # Use the above credentials to authenticate the API.

    # auth = tweepy.OAuthHandler( consumer_key , consumer_secret )
    # auth.set_access_token( access_token , access_token_secret )
    auth = tweepy.OAuth2BearerHandler(bearer_token)
    api = tweepy.API(auth)

    df = pd.DataFrame(columns=["Date", "User", "IsVerified", "Tweet", "Likes", "RT", 'User_location'])

    # Write a Function to extract tweets:
    def get_tweets(Topic, Count):
        i = 0
        # my_bar = st.progress(100) # To track progress of Extracted tweets
        for tweet in tweepy.Cursor(api.search_tweets, q=Topic, count=100, lang="en", exclude='retweets', tweet_mode='extended').items():
            # time.sleep(0.1)
            # my_bar.progress(i)

            df.loc[i, "Date"] = tweet.created_at
            df.loc[i, "User"] = tweet.user.name
            df.loc[i, "IsVerified"] = tweet.user.verified
            df.loc[i, "Tweet"] = tweet.text
            df.loc[i, "Likes"] = tweet.favorite_count
            df.loc[i, "RT"] = tweet.retweet_count
            df.loc[i, "User_location"] = tweet.user.location
            # df.to_csv("TweetDataset.csv",index=False)
            # df.to_excel('{}.xlsx'.format("TweetDataset"),index=False)   ## Save as Excel
            i = i + 1
            if i > Count:
                break
            else:
                pass
    def get_new_tweets(Topic, Count):
        all_tweets = []
        for tweet in tweepy.Cursor(api.search_tweets, q=Topic, count=100, lang="en", exclude='retweets',tweet_mode='extended').items():
            tweet_details = []
            tweet_details.append(tweet.id_str)
            tweet_details.append(tweet.created_at)
            tweet_details.append(tweet.user.verified)
            tweet_details.append(tweet.full_text)
            all_tweets.append(tweet_details)
        return all_tweets

    # Function to Clean the Tweet.
    def clean_tweet(tweet):
        return ' '.join(re.sub('(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|([RT])', ' ', tweet.lower()).split())

    # Funciton to analyze Sentiment
    def analyze_sentiment(tweet):
        analysis = TextBlob(tweet)
        if analysis.sentiment.polarity > 0:
            return 'Positive'
        elif analysis.sentiment.polarity == 0:
            return 'Neutral'
        else:
            return 'Negative'

    # Function to Pre-process data for Worlcloud
    def prepCloud(Topic_text, Topic):
        Topic = str(Topic).lower()
        Topic = ' '.join(re.sub('([^0-9A-Za-z \t])', ' ', Topic).split())
        Topic = re.split("\s+", str(Topic))
        stopwords = set(STOPWORDS)
        stopwords.update(Topic)  ###Add our topic in Stopwords, so it doesnt appear in wordClous###
        text_new = " ".join([txt for txt in Topic_text.split() if txt not in stopwords])
        return text_new

    #
    from PIL import Image
    #image = Image.open('Logo1.jpg')
    #st.image(image, caption='Twitter for Analytics', use_column_width=True)

    # Collect Input from user :
    Topic = str()
    Topic = str(st.text_input("Enter a topic, product or brand you are interested in (Press Enter once done)"))

    if len(Topic) > 0:

        # Call the function to extract the data. pass the topic and filename you want the data to be stored in.
        with st.spinner("Please wait, Tweets are being extracted"):
            df = pd.DataFrame(get_new_tweets(Topic, Count=20), columns=["Date", "User", "IsVerified", "Tweet"])
        st.success('Tweets have been Extracted !!!!')

        # Call function to get Clean tweets
        df['clean_tweet'] = df['Tweet'].apply(lambda x: clean_tweet(x))

        # Call function to get the Sentiments
        df["Sentiment"] = df["Tweet"].apply(lambda x: analyze_sentiment(x))

        # Write Summary of the Tweets
        st.write("Total Tweets Extracted for Topic '{}' are : {}".format(Topic, len(df.Tweet)))
        st.write("Total Positive Tweets are : {}".format(len(df[df["Sentiment"] == "Positive"])))
        st.write("Total Negative Tweets are : {}".format(len(df[df["Sentiment"] == "Negative"])))
        st.write("Total Neutral Tweets are : {}".format(len(df[df["Sentiment"] == "Neutral"])))

        # See the Extracted Data :
        if st.button("See the Extracted Data"):
            st.markdown(html_temp, unsafe_allow_html=True)
            st.success("Below is the Extracted Data :")
            # st.table(df.head(50))
            sub = df[['Tweet','Sentiment','Date', 'IsVerified']]
            st.table(sub.head(10))
            # print(df)

        # get the countPlot
        if st.button("Get Count Plot for Different Sentiments"):
            st.success("Generating A Count Plot")
            st.subheader(" Count Plot for Different Sentiments")
            st.write(sns.countplot(df["Sentiment"]))
            st.pyplot()

        # Piechart
        if st.button("Get Pie Chart for Different Sentiments"):
            st.success("Generating A Pie Chart")
            a = len(df[df["Sentiment"] == "Positive"])
            b = len(df[df["Sentiment"] == "Negative"])
            c = len(df[df["Sentiment"] == "Neutral"])
            d = np.array([a, b, c])
            explode = (0.1, 0.0, 0.1)
            st.write(
                plt.pie(d, shadow=True, explode=explode, labels=["Positive", "Negative", "Neutral"], autopct='%1.2f%%'))
            st.pyplot()

        # get the countPlot Based on Verified and unverified Users
        if st.button("What influencers say"):
            st.success("Generating A Count Plot (Verified Users)")
            st.subheader(" Count Plot for Different Sentiments for Verified Users")
            st.write(sns.countplot(df["Sentiment"], hue=df.IsVerified))
            st.pyplot()

        ## Points to add 1. Make Backgroud Clear for Wordcloud 2. Remove keywords from Wordcloud

        # Create a Worlcloud
        if st.button("Get WordCloud for all things said about {}".format(Topic)):
            st.success("Generating A WordCloud for all things said about {}".format(Topic))
            text = " ".join(review for review in df.clean_tweet)
            stopwords = set(STOPWORDS)
            text_newALL = prepCloud(text, Topic)
            wordcloud = WordCloud(stopwords=stopwords, max_words=800, max_font_size=70).generate(text_newALL)
            st.write(plt.imshow(wordcloud, interpolation='bilinear'))
            st.pyplot()

        # Wordcloud for Positive tweets only
        if st.button("Get WordCloud for all Positive Tweets about {}".format(Topic)):
            st.success("Generating A WordCloud for all Positive Tweets about {}".format(Topic))
            text_positive = " ".join(review for review in df[df["Sentiment"] == "Positive"].clean_tweet)
            stopwords = set(STOPWORDS)
            text_new_positive = prepCloud(text_positive, Topic)
            # text_positive=" ".join([word for word in text_positive.split() if word not in stopwords])
            wordcloud = WordCloud(stopwords=stopwords, max_words=800, max_font_size=70).generate(text_new_positive)
            st.write(plt.imshow(wordcloud, interpolation='bilinear'))
            st.pyplot()

        # Wordcloud for Negative tweets only
        if st.button("Get WordCloud for all Negative Tweets about {}".format(Topic)):
            st.success("Generating A WordCloud for all Positive Tweets about {}".format(Topic))
            text_negative = " ".join(review for review in df[df["Sentiment"] == "Negative"].clean_tweet)
            stopwords = set(STOPWORDS)
            text_new_negative = prepCloud(text_negative, Topic)
            # text_negative=" ".join([word for word in text_negative.split() if word not in stopwords])
            wordcloud = WordCloud(stopwords=stopwords, max_words=800, max_font_size=70).generate(text_new_negative)
            st.write(plt.imshow(wordcloud, interpolation='bilinear'))
            st.pyplot()

    st.sidebar.header("About App")
    st.sidebar.info("Kuks is sentiment analysis app which analyzes twitter feeds for the topic, products or brand selected by you."
                    "\n""\n"
                    "The extracted data will give you an overview of the brand or products you are looking for."
                   "\n" "\n" 
                    "A different visualizations will help you get a feel of the overall mood of the people on "
                    "twitter regarding to a brand or product they select.")



    #st.sidebar.header("For Any Queries/Suggestions Please reach out at :")
    #st.sidebar.info("made with love")

    # st.sidebar.subheader("Scatter-plot setup")
    # box1 = st.sidebar.selectbox(label= "X axis", options = numeric_columns)
    # box2 = st.sidebar.selectbox(label="Y axis", options=numeric_columns)
    # sns.jointplot(x=box1, y= box2, data=df, kind = "reg", color= "red")
    # st.pyplot()

    if st.button("Exit"):
        st.balloons()


if __name__ == '__main__':
    main()
