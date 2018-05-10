from time import time as now

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from langdetect import detect
from sklearn.decomposition import LatentDirichletAllocation

import text_utils
from display_utils import print_time

emotions = ['joy', 'trust', 'sadness', 'disgust', 'anger', 'fear', 'anticipation', 'surprise']


def process_song(row):
    """
    Applied to a DataFrame to clean lyrics and get word count, it also makes any song with lyrics in another
    language be returned with an np.nan row (missing data to easily remove later)

    """
    try:
        lyrics = row['lyrics']
        if detect(lyrics) == 'en':
            cleaned_lyrics = text_utils.clean_text(lyrics)
            if detect(cleaned_lyrics) == 'en':
                row['cleaned_lyrics'] = cleaned_lyrics
                row['old_word_count'] = len(lyrics.strip().split())
                row['new_word_count'] = len(cleaned_lyrics.strip().split())
            else:
                row['lyrics'] = np.nan
        else:
            row['lyrics'] = np.nan
    except:
        row['lyrics'] = np.nan
    return row


def process_data(data: pd.DataFrame, sort: bool = True, dropna: bool = True, drop_duplicates: bool = True):
    """
    Cleans the incoming dataframe, removes duplicate and missing data,
    and sorts the remaining data before returning

    """
    data = data[['artist', 'song', 'genre', 'year', 'lyrics']]

    print('Cleaning lyrics, counting words and detecing language. This may take some time ... ', end='')
    last = now()
    data = data.apply(process_song, axis=1)
    print_time(now() - last)

    if dropna:
        print('Removing data with missing features ... ', end='')
        last = now()
        data = data.dropna(axis=0, how='any')
        print_time(now() - last)

    if drop_duplicates:
        print('Removing duplicate data ... ', end='')
        last = now()
        data = data.drop_duplicates(subset=['cleaned_lyrics'])
        print_time(now() - last)

    if sort:
        print('Sorting data by artist & song title ... ', end='')
        last = now()
        data.sort_values(['artist', 'song'], inplace=True)
        print_time(now() - last)

    return data.reset_index(drop=True)


def get_songs_with_min_length(data: pd.DataFrame, min_word_count: int = 100):
    """
    Remove any songs that do not satisfy the min_word_count requirement from the given dataset

    """
    print('Removing songs with word count less than %d ... ' % min_word_count, end='')
    last = now()
    data = data[data.new_word_count >= min_word_count].reset_index(drop=True)
    print_time(now() - last)
    return data


def trunc_artist(df: pd.DataFrame, artist: str, keep: float = 0.5, random_state: int = None):
    """

    Keeps only the requested portion of songs by the artist
    (this method is not in use anymore)

    """
    data = df.copy()
    df_artist = data[data.artist == artist]
    data = data[data.artist != artist]
    orig_length = len(df_artist)
    try:
        df_artist = df_artist.sample(int(len(df_artist) * keep), random_state=random_state)
    except ValueError:
        pass
    new_length = len(df_artist)
    print("Truncating data for {artist}, original length = {orig}, new length = {new}".format(artist=artist,
                                                                                              orig=orig_length,
                                                                                              new=new_length))
    data = data.append(df_artist)
    return data.reset_index(drop=True)


def get_topics_for_song(topic_model: LatentDirichletAllocation, transformed_lyrics: list, song_index: int,
                        data: pd.DataFrame):
    """

    For a given song's LDA transformed vector, returns the most relevant topic indexes

    """
    topic_relevancy = transformed_lyrics[song_index]
    sorted_relevancy = np.array(sorted(topic_relevancy, reverse=True))
    count = 1
    i = 1
    n_topics = topic_model.n_components
    while i < n_topics:  # check if each successive topic's probability is at > 50% of the highest probability
        if sorted_relevancy[i] / sorted_relevancy[0] > .5:
            count += 1
        i += 1
    if count == 1:
        i = 2
        alt_count = 2
        while i < n_topics:  # check if each successive topic's probability is at > 70% of the second highest probability
            if sorted_relevancy[i] / sorted_relevancy[1] > .7:
                alt_count += 1
            i += 1
        if alt_count != 2:
            count = alt_count
        else:  # if neither, is the second topic at least 30% of the first?
            if sorted_relevancy[1] / sorted_relevancy[0] > .3:
                count = 2
    indexes = np.argpartition(topic_relevancy, -count)[-count:]
    indexes = np.array(sorted(indexes))
    relevancies = [topic_relevancy[i] for i in indexes]
    if data is not None:
        data.at[song_index, 'topics'] = indexes
        data.at[song_index, 'relevancies'] = relevancies
    return indexes, relevancies


def get_songs_for_topic(topic_model: LatentDirichletAllocation, transformed_lyrics: list, topic_index: int,
                        min_relevance: float, data: pd.DataFrame):
    """

    Get songs that represent a given topic index with a specified min_relevance.

    The songs are only returned if they satisfy the min_relevance conditions in get_topics_for_song
    (i.e. songs are not returned if they aren't relevant, even if min_relevance is 0.

    """

    songs = pd.DataFrame(columns=['song_index', 'song', 'artist', 'topics', 'relevancy'])
    for n in range(len(transformed_lyrics)):
        indexes, relevancy = get_topics_for_song(topic_model, transformed_lyrics, n, data)
        if topic_index in indexes:
            i_topic = indexes.tolist().index(topic_index)
            if relevancy[i_topic] > min_relevance:
                songs = songs.append({"song_index": n, 'song': data.song[n], 'artist': data.artist[n],
                                      'topics': indexes, 'relevancy': relevancy}, ignore_index=True)

    return songs


def get_top_words_for_each_topic(model: LatentDirichletAllocation, feature_names, n_top_words: int):
    """
    Return n top words from the given LDA model for each topic
    """
    topic_words = []
    for topic_idx, topic in enumerate(model.components_):
        topic_words.append([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
    return topic_words


def display_topic_details_for_song(model, vec, n, data: pd.DataFrame):
    print(data["song"][n], "by", data["artist"][n])
    indexes, relevancy = get_topics_for_song(model, vec, n, data)
    print(indexes + 1)
    print(relevancy)
    print()


def get_similar_words(w2v_model: Word2Vec, word: str, min_relevance: float = .6):
    """
    Returns words similar to the given word from the given Word2Vec model

    """
    try:
        words = set()
        [words.add(w) for w, relevance in w2v_model.wv.most_similar(word) if relevance > min_relevance]
        return words
    except KeyError:
        return set()
