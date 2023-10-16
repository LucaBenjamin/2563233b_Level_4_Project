import spotipy
from spotipy.oauth2 import SpotifyOAuth
import requests


with open("Audio_Processing/spotify_scraper/auth.txt") as file:
    auth_file_text = file.read().split(',')


client_id = auth_file_text[0]
client_secret = auth_file_text[1]
redirect_uri = 'http://localhost:8080/callback'


sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=client_id,
                                               client_secret=client_secret,
                                               redirect_uri=redirect_uri,
                                               scope="user-library-read")) 


results = sp.search(q='folk piano', type='track', limit=50)
tracks = results['tracks']['items']

for i, track in enumerate(tracks):
    print(track['name'], "-", track['artists'][0]['name'])
    url = sp.track(track["id"])
    url = url["preview_url"]
    print(url)
    response = requests.get(url)

    with open(f'Audio_Processing/spotify_scraper/saved_mp3s/preview_{i}.mp3', 'wb') as file:
        file.write(response.content)