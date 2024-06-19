# GENREATOR project

## Dependencies

- Install conda and magenta
```
curl https://raw.githubusercontent.com/tensorflow/magenta/main/magenta/tools/magenta-install.sh > /tmp/magenta-install.sh
bash /tmp/magenta-install.sh
```
- `conda activate magenta`
- `pip install -r requirements.txt`
- Download the model from this [repo](https://github.com/magenta/magenta/tree/main/magenta/models/music_vae#pre-trained-checkpoints)

## Usage

- You can use streamlit demo
```
streamlit run demo.py
```

OR

- Use flask web server
```
python app.py
```

