# GENREATOR project

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

## Usage

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

For MAX for Live you need a patch and runing local server

- Adjust configs
```
client_ip = "host for MAX for Live udpreceive | default : 127.0.0.1"
client_port = "port for MAX for Live udpreceive | default : 7374"
SERVER_URL = "ip:port of running flask app for model"
```

- Run the server
```
python max_server.py
```

- Use flask app host:port to send data from MAX for Live | udpsend

- Use local server client_ip:client_port to receive data in MAX for Live | udpreceive
