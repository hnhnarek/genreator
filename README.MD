# GENREATOR project

A generative device for musicians, to aid and inspire the writing/production process by generating rhythms based on 3 genres. It should allow interpolation between these genres to create interesting patterns.

Check the demo [here](http://genreator.medlee.studio:8501/).

NOTE: The work was done during the [Generative Music AI Workshop](https://www.upf.edu/web/mtg/generative-music-ai-workshop).

## Dependencies

- Install conda and magenta
```
curl https://raw.githubusercontent.com/tensorflow/magenta/main/magenta/tools/magenta-install.sh > /tmp/magenta-install.sh
bash /tmp/magenta-install.sh
```
- `conda activate magenta`
- `pip install -r requirements.txt`
- Download the model (groovae_4bar) from this [repo](https://github.com/magenta/magenta/tree/main/magenta/models/music_vae#pre-trained-checkpoints)
    - Update model path in `inference.py` or download and untar file in `models/` directory

## Usage [standalone]

You can use it standalone or with MAX for Live. With standalone option you will have a web interface in case of streamlit or API in case of running flask app.

- You can use streamlit demo
```
streamlit run demo.py
```

OR

- Use flask web server (-w for workers count)
```
gunicorn -w 3 -b 0.0.0.0:8989 flask_app:app
```


## Usage with MAX for Live

For MAX for Live besides flask server which is **serving model**, you need a runing local server for communication with MAX patch

So you need 
- flask server for model (instructions are above)
- local server for MAX for Live
- patch in MAX for Live

___

### Below are the instructions for local server

- Adjust configs
```
client_ip = "host for MAX for Live udpreceive | default : 127.0.0.1"
client_port = "port for MAX for Live udpreceive | default : 7374"
SERVER_URL = "ip:port of running flask app for model"
```


- Install dependencies
```
pip install python-osc requests
```


- Run the server
```
python max_server.py
```

- Use flask app host:port to send data from MAX for Live | udpsend

- Use local server client_ip:client_port to receive data in MAX for Live | udpreceive


## TODO

- refactor code : it's shitty all around :))
- experiment with more models
- compile model to use without server
