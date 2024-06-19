import streamlit as st
import numpy as np
import magenta.music as mm
from magenta.models.music_vae import configs
from magenta.models.music_vae.trained_model import TrainedModel
import numpy as np
import os
import tensorflow.compat.v1 as tf
import note_seq
from note_seq.protobuf import music_pb2
from demo_data import list_of_genres, data, DATA_PATH

st.set_page_config(page_title="Genreator | Demo", page_icon="🎵", layout="wide")

models = {
    "hierdec-trio_16bar": {
        "config_key": "hierdec-trio_16bar",
        "model_path": "models/hierdec-trio_16bar.ckpt"
    },
    "groovae_4bar": {
        "config_key": "groovae_4bar",
        "model_path": "models/groovae_4bar/model.ckpt-2721"
    }
}


def load_model(model_name):
    model_info = models[model_name]
    config = configs.CONFIG_MAP[model_info['config_key']]
    path = model_info['model_path']
    model = TrainedModel(config, batch_size=4, checkpoint_dir_or_path=path)
    return model, config

@st.cache_resource
def load_model_st():
    model, config = load_model('groovae_4bar')
    model._config.data_converter._split_bars = 4
    model._config.data_converter._steps_per_quarter = 4
    model._config.data_converter._steps_per_bar = 16
    return model, config

def _slerp(p0, p1, t):
  """Spherical linear interpolation."""
  omega = np.arccos(
      np.dot(np.squeeze(p0/np.linalg.norm(p0)),
             np.squeeze(p1/np.linalg.norm(p1))))
  so = np.sin(omega)
  return np.sin((1.0-t)*omega) / so * p0 + np.sin(t*omega)/so * p1


def mix_two(input_1, input_2, mix_model, mix_config):
    #encoding
    _, mu, _ = mix_model.encode([input_1, input_2])
    #mixing
    z = np.array([
        _slerp(mu[0], mu[1], t) for t in np.linspace(0, 1, 4)])
    #decoding
    results = mix_model.decode(
        length=mix_config.hparams.max_seq_len,
        z=z,
        temperature=0.5)
    return results


def midi_preprocessing(midi_file):
    input_f = note_seq.midi_file_to_note_sequence(midi_file)
    while input_f.total_time < 7:
        input_f = duplicate_notes(input_f)
    if input_f.total_time > 7:
        input_f = cut_to_4_bars(input_f)
    return input_f


def mix_two_midi(input_midi_1, input_midi_2, mix_model, mix_config):
    input_1 = midi_preprocessing(input_midi_1)
    input_2 = midi_preprocessing(input_midi_2)
    return mix_two(input_1, input_2, mix_model, mix_config)


def get_weights_n(weights):
    weights = [int(10*i/sum(weights)) for i in weights]
    return weights



def mix_midis(inputs, mix_model, mix_config):
    inputs = [midi_preprocessing(i) for i in inputs]
    _, mu, _ = mix_model.encode(inputs)
    results = [mu[i] for i in range(len(inputs))]

    while len(results) !=2:
        new_res = []
        for i in range(len(results)-1):
            z = _slerp(results[i], results[i+1], 0.5)
            new_res.append(z)
        results = new_res
    z = np.array([
        _slerp(results[0], results[1], t) for t in np.linspace(0, 1, 4)])
    #decoding
    results_midi = mix_model.decode(
        length=mix_config.hparams.max_seq_len,
        z=z,
        temperature=0.5)
    return results_midi


def duplicate_notes(sequence):
    from note_seq.protobuf import music_pb2

    new_sequence = music_pb2.NoteSequence()

    # Copy original notes
    for note in sequence.notes:
        new_note = new_sequence.notes.add()
        new_note.CopyFrom(note)

    # Duplicate notes with time offset
    for note in sequence.notes:
        new_note = new_sequence.notes.add()
        new_note.pitch = note.pitch
        new_note.start_time = note.start_time + sequence.total_time
        new_note.end_time = note.end_time + sequence.total_time
        new_note.velocity = note.velocity

    new_sequence.total_time = sequence.total_time * 2

    return new_sequence


def cut_to_4_bars(note_seq, time_signature_numerator=4, time_signature_denominator=4):
    """
    Cut a note sequence to the first 4 bars.
    
    Args:
        note_seq: A NoteSequence object.
        time_signature_numerator: The numerator of the time signature (default 4 for 4/4 time).
        time_signature_denominator: The denominator of the time signature (default 4 for 4/4 time).
    
    Returns:
        A NoteSequence object trimmed to the first 4 bars.
    """
    # Calculate the duration of one bar in seconds
    bar_duration = time_signature_numerator * (4.0 / time_signature_denominator)
    
    # Calculate the duration of 4 bars
    four_bars_duration = 4 * bar_duration - 4
    
    # Create a new NoteSequence to store the trimmed notes
    trimmed_sequence = music_pb2.NoteSequence()
    
    # Copy original sequence info to trimmed sequence
    trimmed_sequence.tempos.extend(note_seq.tempos)
    trimmed_sequence.time_signatures.extend(note_seq.time_signatures)
    trimmed_sequence.key_signatures.extend(note_seq.key_signatures)
    
    # Collect notes that are within the first 4 bars
    for note in note_seq.notes:
        if note.start_time < four_bars_duration:
            new_note = trimmed_sequence.notes.add()
            new_note.CopyFrom(note)
            # Ensure the note does not exceed the 4 bars limit
            if new_note.end_time > four_bars_duration:
                new_note.end_time = four_bars_duration
    
    # Set the total time of the trimmed sequence
    trimmed_sequence.total_time = min(note_seq.total_time, four_bars_duration)
    
    return trimmed_sequence


st.sidebar.title("Genreator | Navigation")
st.sidebar.markdown("___")
page = st.sidebar.selectbox("Select a page", ["Select page", "Demo with files", "Demo with genres"])
model,config  = load_model_st()

if page == "Demo with files":
    st.title("Genreator Demo")
    st.header("This is a demo of the genre mixing model")
    st.markdown("___")
    st.warning("Please upload two midi files to generate a new one")

    cols = st.columns(2)
    cols[0].header("First Midi File")
    cols[1].header("Second Midi File")


    file1 = cols[0].file_uploader("Upload the first midi file", type=["mid", "midi"])
    file2 = cols[1].file_uploader("Upload the second midi file", type=["mid", "midi"])

    st.markdown("___")

    if file1 and file2:
        st.header("Weights between")
        st.slider(f" -1 : ({file2.name.split('.')[0]}) -  1 : ({file2.name.split('.')[0]})", -1., 1., 0., 0.1)
        with open("/tmp/input1.mid", "wb") as f:
            f.write(file1.getvalue())
        with open("/tmp/input2.mid", "wb") as f:
            f.write(file2.getvalue())
        file1_path = "/tmp/input1.mid"
        file2_path = "/tmp/input2.mid"
        results = mix_two_midi(file1_path, file2_path, model, config)
        downloadable_files = []
        for _idx,res in enumerate(results):
            f_name = f"/tmp/mix_{file1.name.split('.')[0]}_{file2.name.split('.')[0]}_output{_idx}.mid"
            note_seq.sequence_proto_to_midi_file(res, f_name)
            downloadable_files.append(f_name)
        st.header("Download the generated midi file")
        for file in downloadable_files:
            with open(file, "rb") as f:
                st.download_button(label=f"{file.split('/')[-1]}", data=f, file_name=file)
elif page == "Demo with genres":
    st.title("Genreator Demo")
    st.header("This is a demo of the genre mixing model")
    genres_select = st.multiselect("Select the genres", list_of_genres, max_selections=3)
    if len(genres_select) >1:
        slider_values = []
        cols_demo = st.columns(len(genres_select))
        for idx, genre in enumerate(genres_select):
            cols_demo[idx].header(genre)
            slider_values.append(cols_demo[idx].slider(f"Weight", 0, 100, 100, 5, key=genre))
        w = get_weights_n(slider_values)
        # choose random sample from each selected genre from data
        inputs = []
        for _idx, genre in enumerate(genres_select):
            genre_data = [r for r in data if r['style'] == genre]
            random_sample = np.random.choice(genre_data,size=w[_idx])
            midi_file = [i['midi_filename'] for i in random_sample]
            inputs.extend(midi_file)
        results = mix_midis(inputs, model, config)
        attempts = 10
        while True:
            if [i for i in results if i.notes]:
                break
            attempts -= 1
            print(f"Retrying {attempts}, files : {inputs}")
            inputs = []
            for _idx, genre in enumerate(genres_select):
                genre_data = [r for r in data if r['style'] == genre]
                random_sample = np.random.choice(genre_data,size=w[_idx])
                midi_file = [i['midi_filename'] for i in random_sample]
                inputs.extend(midi_file)
            results = mix_midis(inputs, model, config)
            if attempts == 0:
                break
        downloadable_files = []
        genre_postfix = "_".join(genres_select).replace("/", "-")
        for _idx,res in enumerate(results):
            if not res.notes:
                continue
            f_name = f"/tmp/mix_{genre_postfix}_output{_idx}.mid"
            note_seq.sequence_proto_to_midi_file(res, f_name)
            downloadable_files.append(f_name)
        st.header("Download the generated midi file")
        if not downloadable_files:
            st.error("No midi file generated. taking a look")
        for file in downloadable_files:
            with open(file, "rb") as f:
                st.download_button(label=f"{file.split('/')[-1]}", data=f, file_name=file)

        
