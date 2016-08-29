from TwitterSearch import *
import codecs


def crawl(filename, keywords, language):
    f = codecs.open(filename, "w", "utf-8")
    try:
        tso = TwitterSearchOrder()  # create a TwitterSearchOrder object
        tso.set_keywords(keywords)  # let's define all words we would like to have a look for
        tso.set_language(language)  # we want to see German tweets only
        tso.set_include_entities(False)  # and don't give us all those entity information

        # it's about time to create a TwitterSearch object with our secret tokens
        ts = TwitterSearch(
            consumer_key='MozbqzFag8UQMbuw9qkuyG7Fm',
            consumer_secret='c4m8EKOwQb90A3nLLySKSEkV7fVXe8taZq4IjgDrMVKihbNW4s',
            access_token='2684788650-VOzUZGhPItlgye6w5LhX5QMevWLK8WTALcxe8KM',
            access_token_secret='9IeW0F8XFnZ7FV5sCyZIahLEZBQTkzwO4L0q3vqRkl4je'
        )

        # this is where the fun actually starts :)
        for tweet in ts.search_tweets_iterable(tso):
            print('@%s tweeted: %s' % (tweet['user']['screen_name'], tweet['text']))

    except TwitterSearchException as e:  # take care of all those ugly errors if there are some
        print(e)

    f.close()


if __name__ == "__main__":
    crawl("twitter.txt", ["Iphone"], "en")
