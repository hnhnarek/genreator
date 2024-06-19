import csv

DATA_PATH = './data/'

with open('metadata_new.csv') as f:
    reader = csv.DictReader(f)
    data = list(reader)

data = [r for r in data if r['ignore'] == 'False']
list_of_genres = [r['style'] for r in data]
list_of_genres = list(set(list_of_genres))
list_of_genres.sort(key=lambda x: x.lower())
