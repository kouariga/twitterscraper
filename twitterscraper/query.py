from __future__ import division
import random
import requests
import datetime as dt
import json
from functools import partial
from multiprocessing.pool import Pool

import numpy as np

from twitterscraper.tweet import Tweet
from twitterscraper.ts_logger import logger
from twitterscraper.user import User

HEADERS_LIST = [
    'Mozilla/5.0 (Windows; U; Windows NT 6.1; rv:2.2) Gecko/20110201',
    'Opera/9.80 (X11; Linux i686; Ubuntu/14.10) Presto/2.12.388 Version/12.16',
]

HEADER = {'User-Agent': random.choice(HEADERS_LIST)}

SEARCH_INIT_URL = 'https://twitter.com/search?f=tweets'\
            '&vertical=default&q={query}&l={lang}'

SEARCH_RELOAD_URL = 'https://twitter.com/i/search/timeline?f=tweets'\
             '&vertical=default&include_available_features=1'\
             '&include_entities=1&reset_error_state=false&src=typd'\
             '&max_position={pos}&q={query}&l={lang}'

USER_INIT_URL = 'https://twitter.com/{user}'

USER_RELOAD_URL = 'https://twitter.com/i/profiles/show/{user}/timeline/tweets'\
                  '?include_available_features=1&include_entities=1'\
                  '&max_position={pos}&reset_error_state=false'


def get_url(query, lang, pos=None, from_user=False):
    if from_user:
        return get_user_url(query, pos)
    return get_search_url(query, lang, pos)


def get_search_url(query, lang, pos=None):
    if pos:
        return SEARCH_RELOAD_URL.format(query=query, pos=pos, lang=lang)
    return SEARCH_INIT_URL.format(query=query, lang=lang)


def get_user_url(user, pos=None):
    if pos:
        return USER_RELOAD_URL.format(user=user, pos=pos)
    return USER_INIT_URL.format(user=user)


def query_user(user):
    """Returns the scraped user data from a twitter user page.

    :param user: the twitter user to web scrape its twitter page info
    """
    try:
        logger.info(f"querying {user}'s profile...")
        user_info = _request_user(get_user_url(user))
        logger.info("success" if user_info is not None else "failure")
        return user_info
    except KeyboardInterrupt:
        logger.info('program interrupted by user.')
    except BaseException as e:
        logger.exception(e)


def query_user_tweets(user, limit=None):
    logger.info(f"querying {user}'s tweets...")
    tweets = []
    try:
        pos = None
        while True:
            new_tweets, new_pos = _request_page(user, pos=pos, from_user=True)
            if (pos and int(new_pos) > int(pos)):
                for tweet in new_tweets:
                    if tweets[-1].timestamp > tweet.timestamp:
                        tweets.append(tweet)
                break
            pos = new_pos
            tweets += new_tweets
            if (limit and len(tweets) >= limit):
                break
    except KeyboardInterrupt:
        logger.info('program interrupted by user.')
    except BaseException as e:
        logger.exception(e)
    logger.info(f"got {len(tweets)} tweets from {user}.")
    return tweets


def query_tweets(query, lang='', pos=None, limit=None):
    """
    Scrape the search result at https://twitter.com/search-home

    :param query: https://twitter.com/search-advanced to compile query
    :param pos:   iteration pointer
    :param limit: stop when at least ``limit`` tweets are fetched
    :return:      at least ``limit`` twitterscraper.Tweet objects
    """
    logger.info(f'querying {query}')
    query = query.replace(' ', '%20').replace('#', '%23').replace(':', '%3A')
    num_tweets = 0
    try:
        while True:
            new_tweets, new_pos = _request_page(query, lang, pos)
            if len(new_tweets) == 0:
                logger.info(f'got {num_tweets} tweets for {query}.')
                return
            for t in new_tweets:
                yield (t, pos)
            pos = new_pos
            num_tweets += len(new_tweets)
            if limit and num_tweets >= limit:
                logger.info(f'got {num_tweets} tweets for {query}.')
                return

    except KeyboardInterrupt:
        logger.info('program interrupted by user.')
    except BaseException as e:
        logger.exception(e)
    logger.info(f'got {num_tweets} tweets for {query}.')


def query_tweets_parallel(query, lang='', limit=None, poolsize=20,
                          begindate=dt.date(2006, 3, 21),
                          enddate=dt.date.today()):
    no_days = (enddate - begindate).days
    if poolsize > no_days:
        poolsize = no_days
    dateranges = [begindate + dt.timedelta(days=elem)
                  for elem in np.linspace(0, no_days, poolsize + 1)]
    limit_per_pool = (limit // poolsize) + 1 if limit else None
    queries = [f'{query} since:{since} until:{until}'
               for since, until in zip(dateranges[:-1], dateranges[1:])]

    tweets = []
    try:
        pool = Pool(poolsize)
        logger.info('queries: {}'.format(queries))
        job = partial(_query_tweets_wrapper, limit=limit_per_pool, lang=lang)
        for new_tweets in pool.imap_unordered(job, queries):
            tweets.extend(new_tweets)
            logger.info(f'got {len(tweets)} tweets ({len(new_tweets)} new).')
    except KeyboardInterrupt:
        logger.info('program interrupted by user.')
    finally:
        pool.close()
        pool.join()

    return tweets


def _query_tweets_wrapper(*args, **kwargs):
    res = list(query_tweets(*args, **kwargs))
    if res:
        tweets, _ = zip(*res)
        return tweets
    return []


def _request_user(url, retry=10):
    """Returns the scraped user data from a twitter user page.

    :param url: URL of a Twitter user page.
    :param retry: Number of retries if something goes wrong.
    :return: Returns the scraped user data from a Twitter user page.
    """
    for i in range(retry):
        try:
            response = requests.get(url, headers=HEADER)
            html = response.text or ''
            user = User()
            user_info = user.from_html(html)
            return user_info
        except requests.exceptions.RequestException as e:
            logger.exception(e)
        logger.info(f'retry... ({retry - (i + 1)} left)')

    logger.error(f'no success within {retry} attempts')
    return None


def _request_page(query, lang='', pos=None, retry=10, from_user=False):
    """
    Returns tweets from the given URL.
    :param query: The query parameter of the query url
    :param lang: The language parameter of the query url
    :param pos: The query url parameter that determines where to start looking
    :param retry: Number of retries if something goes wrong.
    :return: The list of tweets, the pos argument for getting the next page.
    """
    for i in range(retry):
        try:
            url = get_url(query, lang, pos, from_user)
            response = requests.get(url, headers=HEADER)
            if pos is None:
                response_json = None
                html = response.text or ''
            else:
                try:
                    response_json = json.loads(response.text)
                    html = response_json['items_html'] or ''
                except ValueError as e:
                    logger.exception(e)
                    html = ''

            tweets = list(Tweet.from_html(html))
            if not tweets:
                pos = response_json['min_position'] if response_json else None
                continue
            return (tweets, tweets[-1].id)

        except requests.exceptions.RequestException as e:
            logger.exception(e)
        except json.decoder.JSONDecodeError as e:
            logger.exception(e)
        logger.info(f'retry... ({retry - (i + 1)} left)')

    logger.error(f'no success within {retry} attempts.')
    return ([], None)
