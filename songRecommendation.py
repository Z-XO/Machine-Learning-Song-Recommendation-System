from sklearn.neighbors import NearestNeighbors
import timeit
from datetime import datetime
import warnings
from scipy.sparse import csr_matrix
from fuzzywuzzy import fuzz
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class Recommender:
    def __init__(self, metric, algorithm, k, data, decode_id_song):
        self.metric = metric
        self.algorithm = algorithm
        self.k = k
        self.data = data
        self.decode_id_song = decode_id_song
        self.data = data
        self.model = self.recommender().fit(data)
    
    def make_recommendation(self, newSong, totalReqs):
        recommended = self.recommend(newSong =newSong, totalReqs = totalReqs)
        print("The almighty recommender recommends these songs, take them at face value")
        return recommended 
    
    
    def mapIndexToSong(self, songID):
        # get reverse mapper
        return {song_id: song_title for song_title, song_id in self.decode_id_song.items()}   
    
    
    # sklearn.neighbors implimentation
    def recommender(self):
        return NearestNeighbors(metric=self.metric, algorithm=self.algorithm, n_neighbors=self.k, n_jobs=-1)
    
    
    def recommend(self, newSong, totalReqs):
        # Get the id of the recommended songs
        songs = []
        songID = self.get_recommendations(newSong = newSong, totalReqs = totalReqs)
        mappedSongs = self.mapIndexToSong(songID)
        # Rank songs based on index
        for i, (j, dist) in enumerate(songID):
            songs.append(mappedSongs[j])
        return songs
                 
    def get_recommendations(self, newSong, totalReqs):
        
        recommendedSongID = self.fuzz_matched(song = newSong)
        
        # Start the recommendation process
        print(f"Searching for similar songs to {newSong} ")
        
        # Return total neighbors for the song id
        distances, indexs = self.model.kneighbors(self.data[recommendedSongID], n_neighbors = totalReqs + 1)
        return sorted(list(zip(indexs.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]
    
    
    def fuzz_matched(self, song):
        matchedSong = []
        # get match
        for title, idx in self.decode_id_song.items():
            ratio = fuzz.ratio(title, song)
            if ratio >= 65:
                matchedSong.append((title, idx, ratio))
        # sort
        matchedSong = sorted(matchedSong, key=lambda x: x[2])[::-1]
        if not matchedSong:
            print(f"The recommendation system could not find a match for {song}")
            return
        return matchedSong[0][1]


#Read userid-songid-listen_count
start=datetime.now()
song_info = pd.read_csv('https://static.turi.com/datasets/millionsong/10000.txt',sep='\t',header=None)
song_info.columns = ['user_id', 'song_id', 'listen_count']

#Read song  metadata
song_actual =  pd.read_csv('https://static.turi.com/datasets/millionsong/song_data.csv')
song_actual.drop_duplicates(['song_id'], inplace=True)

#Merge the two dataframes above to create input dataframe for recommender systems
songs = pd.merge(song_info, song_actual, on="song_id", how="left")


model = Recommender(metric= 'euclidean', algorithm= 'brute', k=10, data = song_info, decode_id_song = song_actual)

# Example song input,
song = 'say my name'

new_recommendations = model.make_recommendation(new_song = song, n_recommendations = 10)

# Prints results
print(f"The recommendations for {song} are:")
print(f"{new_recommendations} \n")


# Check total time taken
print( datetime.now()-start )