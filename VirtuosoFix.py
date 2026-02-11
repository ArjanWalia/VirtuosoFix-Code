import streamlit as st
from pretty_midi import PrettyMIDI, Instrument, Note
import warnings
import numpy as np
from pydub import AudioSegment
import soundfile as sf
from datetime import datetime
import copy
import csv
import tempfile
import os

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Page configuration
st.set_page_config(page_title="MIDI Alignment Tool", page_icon="🎵", layout="wide")

# Initialize session state
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'wav_path' not in st.session_state:
    st.session_state.wav_path = None
if 'csv_path' not in st.session_state:
    st.session_state.csv_path = None
if 'accuracy' not in st.session_state:
    st.session_state.accuracy = 0
if 'total_notes' not in st.session_state:
    st.session_state.total_notes = 0
if 'correct_notes' not in st.session_state:
    st.session_state.correct_notes = 0
if 'error_type' not in st.session_state:
    st.session_state.error_type = ""
if 'num_errors' not in st.session_state:
    st.session_state.num_errors = 0
if 'timestamp' not in st.session_state:
    st.session_state.timestamp = ""

# Title and description
st.title("🎵 MIDI Alignment Tool")
st.markdown("Upload your files and correct pitch/timing errors in your audio performance")

# Sidebar for file uploads and parameters
st.sidebar.header("📁 File Uploads")

score_midi_file = st.sidebar.file_uploader("Score MIDI", type=['mid', 'midi'], help="Ground truth score MIDI file")
audio_midi_file = st.sidebar.file_uploader("Audio MIDI", type=['mid', 'midi'], help="Performed audio as MIDI file")
audio_mp3_file = st.sidebar.file_uploader("Audio File", type=['mp3', 'wav'], help="Performed audio as MP3 or WAV file")
sf2_file = st.sidebar.file_uploader("SF2 Soundfont", type=['sf2'], help="SoundFont file for synthesis")

st.sidebar.header("⚙️ Parameters")
error_type = st.sidebar.selectbox("Error Type", ["time", "pitch", "both"], help="Type of errors to correct")
num_errors = st.sidebar.number_input("Number of Errors", min_value=0, value=2, step=1,
                                     help="Total number of errors in the audio")


# All the processing functions from the original script
def cut_end(score_list, audio_list):
    for i in range(len(score_list) - 1):
        new_end = score_list[i + 1][1].start
        curr_note = score_list[i][1]
        curr_note.end = new_end

    for i in range(len(audio_list) - 1):
        new_end_audio = audio_list[i + 1][1].start
        curr_note_audio = audio_list[i][1]
        curr_note_audio.end = new_end_audio


def get_volume(mono_score_array, mono_full_audio_array, audio_data):
    audio_note_start_frame = int(audio_data.start * 44100)
    audio_note_end_frame = int(audio_data.end * 44100)
    audio_note_array = mono_full_audio_array[audio_note_start_frame:audio_note_end_frame]

    abs_audio_note_array = set(abs(amp) for amp in audio_note_array)
    abs_score_note_array = set(abs(amp) for amp in mono_score_array)

    avg_audio_value = sum(abs_audio_note_array) / len(abs_audio_note_array)
    avg_score_value = sum(abs_score_note_array) / len(abs_score_note_array)

    volume_factor = avg_audio_value / avg_score_value

    return volume_factor


def get_start(note_num, og_score_list, og_audio_list, error_num_list, difference):
    diff_list = []

    score_audio_list = list(zip(og_score_list, og_audio_list))

    curr_note_index = None
    for i in range(len(score_audio_list)):
        if score_audio_list[i][1][0] == note_num:
            curr_note_index = i
            break

    curr_note_data = score_audio_list[curr_note_index]
    curr_score, curr_audio = curr_note_data
    note_num, curr_score_data, chord_num_score = curr_score
    note_num, curr_audio_data, chord_num_audio = curr_audio

    curr_audio_start = curr_audio_data.start
    curr_score_start = curr_score_data.start
    curr_score_end = curr_score_data.end
    prev_score_start = score_audio_list[curr_note_index - 1][0][1].start

    curr_score_diff = curr_score_start - prev_score_start

    for index in range(1, len(score_audio_list)):
        zip_note_num = index + 1
        zip_score_note_start = score_audio_list[index][0][1].start
        zip_audio_note_start = score_audio_list[index][1][1].start
        prev_zip_score_note_start = score_audio_list[index - 1][0][1].start
        prev_zip_audio_note_start = score_audio_list[index - 1][1][1].start

        diff_diff = (zip_score_note_start - prev_zip_score_note_start) - (curr_score_start - prev_score_start)

        if curr_score_diff == 0:
            continue

        relative_diff = diff_diff / curr_score_diff
        threshold = 0.1

        if (zip_note_num not in error_num_list) and zip_note_num - 1 not in error_num_list and abs(
                relative_diff) <= threshold:
            diff_list.append(zip_audio_note_start - prev_zip_audio_note_start)

    count = 0
    for note_tuple in score_audio_list[curr_note_index::-1]:
        note_number = note_tuple[1][0]
        if note_number in error_num_list:
            count += 1
        else:
            break

    if curr_note_index != len(og_audio_list) - 1:
        next_score_start = score_audio_list[curr_note_index + 1][0][1].start - difference
        new_end = next_score_start
    else:
        new_end = og_score_list[len(og_audio_list) - 1][1].end - difference

    try:
        avg = sum(diff_list) / len(diff_list)
    except ZeroDivisionError:
        new_start = curr_score_start - difference
        new_end = curr_score_end - difference
        return new_start, new_end

    last_correct_audio_start = score_audio_list[curr_note_index - count][1][1].start
    last_correct_score_start = score_audio_list[curr_note_index - count][0][1].start
    score_time_diff = curr_score_start - last_correct_score_start
    new_start = last_correct_audio_start + score_time_diff

    return new_start, new_end


def truncate(start, end, note_array):
    needed_length = end - start
    original_length = len(note_array)
    if needed_length > original_length:
        num_zeroes = needed_length - original_length
        zero_array = np.array([0] * int(num_zeroes))
        note_array = np.concatenate((note_array, zero_array))
    elif needed_length < original_length:
        note_array = note_array[:needed_length]
    return note_array


def cut_dead_time(array, thresh=0.001):
    left = array[:, 0]
    cutoff = 0
    for amp in left:
        if abs(amp) <= thresh:
            cutoff += 1
        else:
            break
    left = left[cutoff:]
    right = left.copy()
    return np.column_stack((left, right))


def add_crossfade(note_array, fade_samples=220):
    if len(note_array) < fade_samples * 2:
        fade_samples = len(note_array) // 4

    if fade_samples > 0:
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)

        if note_array.ndim == 2:
            fade_in = fade_in.reshape(-1, 1)
            fade_out = fade_out.reshape(-1, 1)

        note_array[:fade_samples] *= fade_in
        note_array[-fade_samples:] *= fade_out

    return note_array


def align(score, audio):
    first_score = score[0][1]
    first_audio = audio[0][1]

    factor_score = first_score.start
    factor_audio = first_audio.start

    difference = factor_score - factor_audio

    for note, data, chord_num in score:
        data.start -= difference
        data.end -= difference

    return difference


def n_p_s_e(midi_path):
    midi = PrettyMIDI(midi_path)
    n_p_t_list = []

    for instrument in midi.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                n_p_t_list.append(note)

    n_p_t_list.sort(key=lambda note: (note.start, note.pitch))
    numbered_list = [(i + 1, note) for i, note in enumerate(n_p_t_list)]

    return numbered_list


def file_to_stereo_np(file_path):
    # Determine file type from extension
    file_ext = file_path.lower().split('.')[-1]

    if file_ext == 'wav':
        # Load WAV file directly with soundfile
        audio_data, sample_rate = sf.read(file_path)

        # Normalize if needed (soundfile already returns float values)
        if audio_data.dtype != np.float32 and audio_data.dtype != np.float64:
            audio_data = audio_data.astype(np.float32) / 32768.0

        # Handle channels
        if audio_data.ndim == 1:
            # Mono - convert to stereo
            samples = np.column_stack((audio_data, audio_data))
        elif audio_data.shape[1] == 2:
            # Already stereo
            samples = audio_data
        else:
            # Multi-channel - take first two channels
            samples = audio_data[:, :2]
    else:
        # Load MP3 file with pydub
        audio_data = AudioSegment.from_mp3(file_path)
        samples = np.array(audio_data.get_array_of_samples()) / 32768.0
        channels = audio_data.channels
        if channels == 2:
            samples = samples.reshape((-1, channels))
        else:
            samples = np.column_stack((samples, samples))

    return samples


def identify_chords_audio(audio_list):
    audio_dict = {}
    processed = set()

    for i in range(len(audio_list)):
        if i in processed:
            continue
        elif i == len(audio_list) - 1:
            audio_list[i] += (i + 1,)
            audio_dict[i + 1] = [audio_list[i]]
            break

        audio_list[i] += (i + 1,)
        audio_dict[i + 1] = [audio_list[i]]

        for j in range(i + 1, len(audio_list)):
            diff_audio = audio_list[j][1].start - audio_list[i][1].start
            if diff_audio <= 0.1:
                audio_list[j] += (i + 1,)
                audio_dict[i + 1].append(audio_list[j])
                processed.add(j)
            else:
                break

        processed.add(i)

    return audio_dict


def identify_chords_score(score_list):
    score_dict = {}
    processed = set()

    for i in range(len(score_list)):
        if i in processed:
            continue
        elif i == len(score_list) - 1:
            score_list[i] += (i + 1,)
            score_dict[i] = [score_list[i]]
            break

        score_list[i] += (i + 1,)
        score_dict[i + 1] = [score_list[i]]

        for j in range(i + 1, len(score_list)):
            diff_score = score_list[j][1].start - score_list[i][1].start
            if diff_score <= 0.1:
                score_list[j] += (i + 1,)
                score_dict[i + 1].append(score_list[j])
                processed.add(j)
            else:
                break

        processed.add(i)

    return score_dict


def remove_trail_notes_audio(audio_list, audio_dict):
    for chord_index, chord_notes in audio_dict.items():
        for chord_note in chord_notes[1:]:
            audio_list.remove(chord_note)
    return audio_list


def remove_trail_notes_score(score_list, score_dict):
    for chord_index, chord_notes in score_dict.items():
        for chord_note in chord_notes[1:]:
            score_list.remove(chord_note)
    return score_list


def fix_errors(score, audio, og_score, og_audio, score_dict, audio_dict):
    both_c = []
    pitch_c = []
    time_c = []

    for i in range(min(len(score), len(audio))):
        score_note = score[i][0]
        score_note_d = score[i][1]
        audio_note = audio[i][0]
        audio_note_d = audio[i][1]

        og_score_note_d = og_score[i][1]
        og_audio_note_d = og_audio[i][1]

        chord_num_score = score[i][-1]
        chord_num_audio = audio[i][-1]

        score_notes = score_dict[chord_num_score]
        score_pitches = set(note_data.pitch for note, note_data, chord_num in score_notes)

        audio_notes = audio_dict[chord_num_audio]
        audio_pitches = set(note_data.pitch for note, note_data, chord_num in audio_notes)

        pitch_check = (score_pitches == audio_pitches)
        time_thresh = 0.3

        if i > 0 and score[i - 1][1].start != score_note_d.start:
            relative_time = abs(
                (score_note_d.start - audio_note_d.start) / (score_note_d.start - score[i - 1][1].start))
        else:
            relative_time = 0

        if not pitch_check:
            if relative_time > time_thresh and i != 0:
                both_c.append((audio_note, og_score_note_d, og_audio_note_d, chord_num_score))
            else:
                pitch_c.append((audio_note, og_score_note_d, og_audio_note_d, chord_num_score))
        elif relative_time > time_thresh and i != 0:
            time_c.append((audio_note, og_score_note_d, og_audio_note_d, chord_num_score))

    return both_c, pitch_c, time_c


def convert_p_t_b(p_list, t_list, b_list, left_audio, score_dict, sf2_path):
    left_copy = copy.deepcopy(left_audio)

    p_arrays = []
    t_arrays = []
    b_arrays = []

    for note_b, og_score_data_b, og_audio_data_b, chord_num_b in b_list:
        new_piano = Instrument(program=0)
        new_midi = PrettyMIDI()

        score_notes = copy.deepcopy(score_dict[chord_num_b])

        for note_num, note_data, chord_num in score_notes:
            adjusted_start, adjusted_end = 0, note_data.end - note_data.start
            note_data.start, note_data.end = adjusted_start, adjusted_end
            new_piano.notes.append(note_data)

        new_midi.instruments.append(new_piano)
        new_midi_np_array = new_midi.fluidsynth(sf2_path=sf2_path)

        volume_factor = get_volume(new_midi_np_array, left_copy, og_audio_data_b)

        stereo_note_np_array = np.column_stack((new_midi_np_array, new_midi_np_array))
        stereo_note_np_array = cut_dead_time(stereo_note_np_array)
        stereo_note_np_array *= volume_factor
        stereo_note_np_array = add_crossfade(stereo_note_np_array)

        b_arrays.append((note_b, stereo_note_np_array, og_score_data_b, og_audio_data_b))

    for note_p, og_score_data_p, og_audio_data_p, chord_num_p in p_list:
        new_piano = Instrument(program=0)
        new_midi = PrettyMIDI()

        score_notes = copy.deepcopy(score_dict[chord_num_p])

        for note_num, note_data, chord_num in score_notes:
            adjusted_start, adjusted_end = 0, note_data.end - note_data.start
            note_data.start, note_data.end = adjusted_start, adjusted_end
            new_piano.notes.append(note_data)

        new_midi.instruments.append(new_piano)
        new_midi_np_array = new_midi.fluidsynth(sf2_path=sf2_path)

        volume_factor = get_volume(new_midi_np_array, left_copy, og_audio_data_p)

        stereo_note_np_array = np.column_stack((new_midi_np_array, new_midi_np_array))
        stereo_note_np_array = cut_dead_time(stereo_note_np_array)
        stereo_note_np_array *= volume_factor
        stereo_note_np_array = add_crossfade(stereo_note_np_array)

        p_arrays.append((note_p, stereo_note_np_array, og_score_data_p, og_audio_data_p))

    for note, og_score_data, og_audio_data, chord_num_t in t_list:
        audio_start_frame = int(og_audio_data.start * 44100)
        audio_end_frame = int(og_audio_data.end * 44100)

        audio_array = left_copy[audio_start_frame: audio_end_frame]
        stereo_note_array = np.column_stack((audio_array, audio_array))
        stereo_note_array = cut_dead_time(stereo_note_array)
        stereo_note_array = add_crossfade(stereo_note_array)

        t_arrays.append((note, stereo_note_array, og_score_data, og_audio_data))

    return p_arrays, t_arrays, b_arrays


def insert_p_t_b(p_data, t_data, b_data, difference, og_score_list, og_audio_list, left_audio):
    left_copy = copy.deepcopy(left_audio)

    error_list = set(note for note, _, _, _ in t_data) | set(note for note, _, _, _ in b_data)
    fixed_notes_info = {}

    # BOTH
    for note_b, note_array_b, og_score_data_b, og_audio_data_b in b_data:
        orig_s_f = int(og_audio_data_b.start * 44100)
        orig_e_f = int(og_audio_data_b.end * 44100)
        if orig_s_f < len(left_copy) and orig_e_f <= len(left_copy):
            left_copy[orig_s_f:orig_e_f] = 0

        start, end = get_start(note_b, og_score_list, og_audio_list, error_list, difference)
        s_f, e_f = int(start * 44100), int(end * 44100)
        left_note = truncate(s_f, e_f, note_array_b[:, 0])
        left_copy[s_f:e_f] = left_note

        fixed_notes_info[note_b] = {'start': start, 'end': end, 'pitch': og_score_data_b.pitch}

    # PITCH
    for note_p, note_array_p, og_score_data_p, og_audio_data_p in p_data:
        s_f = int(og_audio_data_p.start * 44100)
        e_f = int(og_audio_data_p.end * 44100)
        if s_f < len(left_copy) and e_f <= len(left_copy):
            left_copy[s_f:e_f] = 0
        left_note = truncate(s_f, e_f, note_array_p[:, 0])
        left_copy[s_f:e_f] = left_note

        fixed_notes_info[note_p] = {'start': og_audio_data_p.start, 'end': og_audio_data_p.end,
                                    'pitch': og_score_data_p.pitch}

    # TIME
    for note_t, audio_array_t, og_score_data_t, og_audio_data_t in t_data:
        orig_s_f = int(og_audio_data_t.start * 44100)
        orig_e_f = int(og_audio_data_t.end * 44100)
        if orig_s_f < len(left_copy) and orig_e_f <= len(left_copy):
            left_copy[orig_s_f:orig_e_f] = 0

        start, end = get_start(note_t, og_score_list, og_audio_list, error_list, difference)
        s_f, e_f = int(start * 44100), int(end * 44100)
        left_note = truncate(s_f, e_f, audio_array_t[:, 0])
        left_copy[s_f:e_f] = left_note

        fixed_notes_info[note_t] = {'start': start, 'end': end, 'pitch': og_audio_data_t.pitch}

    og_audio_array = np.column_stack((left_copy, left_copy))

    return fixed_notes_info, og_audio_array


def generate_csv_comparison(score_list, audio_list, fixed_notes_info, error_type, num_errors, difference):
    csv_rows = []

    comparison_length = min(len(score_list), len(audio_list))

    if len(score_list) != len(audio_list):
        st.warning(
            f'⚠️ Score has {len(score_list)} notes but audio has {len(audio_list)} notes. CSV will only include the first {comparison_length} notes.')

    for i in range(comparison_length):
        note_num = score_list[i][0]
        score_note_data = score_list[i][1]

        score_pitch = score_note_data.pitch
        score_time = score_note_data.start - difference

        if note_num in fixed_notes_info:
            fixed_pitch = fixed_notes_info[note_num]['pitch']
            fixed_time = fixed_notes_info[note_num]['start']
        else:
            audio_note_data = audio_list[i][1]
            fixed_pitch = audio_note_data.pitch
            fixed_time = audio_note_data.start

        pitch_label = 1 if score_pitch == fixed_pitch else 0

        if i == 0:
            time_label = 1
        else:
            score_time_prev = score_list[i - 1][1].start - difference

            if score_time - score_time_prev == 0:
                time_label = 1
            else:
                relative_time_error = abs(fixed_time - score_time) / (score_time - score_time_prev)
                time_label = 1 if relative_time_error <= 0.30 else 0

        combined_label = 1 if (pitch_label == 1 and time_label == 1) else 0

        csv_rows.append({
            'error_type': error_type,
            'num_errors': num_errors,
            'score_pitch': score_pitch,
            'fixed_pitch': fixed_pitch,
            'score_time': f"{score_time:.6f}",
            'fixed_time': f"{fixed_time:.6f}",
            'label': combined_label
        })

    total_notes = len(csv_rows)
    correct_notes = sum(1 for row in csv_rows if row['label'] == 1)
    accuracy = correct_notes / total_notes if total_notes > 0 else 0.0

    for row in csv_rows:
        row['accuracy'] = f"{accuracy:.4f}"

    return csv_rows, accuracy, total_notes, correct_notes


# Main processing function
def process_midi_alignment(score_midi_path, audio_midi_path, audio_mp3_path, sf2_path, error_type, num_errors):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    progress_bar = st.progress(0)
    status_text = st.empty()

    # Step 1: Transcription
    status_text.text("📝 Transcribing MIDI files...")
    progress_bar.progress(10)
    score_list = n_p_s_e(score_midi_path)
    audio_list = n_p_s_e(audio_midi_path)

    # Step 2: Load audio
    status_text.text("🎵 Loading audio...")
    progress_bar.progress(20)
    og_audio_array = file_to_stereo_np(audio_mp3_path)
    left_audio = og_audio_array[:, 0]

    # Step 3: Identify chords
    status_text.text("🎹 Identifying chords...")
    progress_bar.progress(30)
    audio_dict = identify_chords_audio(audio_list)
    score_dict = identify_chords_score(score_list)

    # Step 4: Remove trail notes
    status_text.text("🧹 Cleaning up notes...")
    progress_bar.progress(40)
    audio_list = remove_trail_notes_audio(audio_list, audio_dict)
    score_list = remove_trail_notes_score(score_list, score_dict)

    cut_end(score_list, audio_list)

    og_score_list = copy.deepcopy(score_list)
    og_audio_list = copy.deepcopy(audio_list)

    # Step 5: Alignment
    status_text.text("📐 Aligning score and audio...")
    progress_bar.progress(50)
    difference = align(score_list, audio_list)

    # Padding if needed
    last_score_note_end_frame = (score_list[len(audio_list) - 1][1].end - difference) * 44100
    if last_score_note_end_frame > len(left_audio):
        needed_length = int(last_score_note_end_frame)
        padding_needed = needed_length - len(left_audio)
        left_audio = np.pad(left_audio, (0, padding_needed), mode='constant', constant_values=0)
        og_audio_array = np.column_stack((left_audio, left_audio))

    # Step 6: Fix errors
    status_text.text("🔍 Detecting errors...")
    progress_bar.progress(60)
    both, pitch, time = fix_errors(score_list, audio_list, og_score_list, og_audio_list, score_dict, audio_dict)

    # Step 7: Convert errors
    status_text.text("🔧 Converting errors...")
    progress_bar.progress(70)
    p_data, t_data, b_data = convert_p_t_b(pitch, time, both, left_audio, score_dict, sf2_path)

    # Step 8: Insert corrections
    status_text.text("✨ Applying corrections...")
    progress_bar.progress(80)
    fixed_notes_info, corrected_audio_array = insert_p_t_b(p_data, t_data, b_data, difference, og_score_list,
                                                           og_audio_list, left_audio)

    # Step 9: Generate CSV
    status_text.text("📊 Generating CSV comparison...")
    progress_bar.progress(90)
    csv_rows, accuracy, total_notes, correct_notes = generate_csv_comparison(
        og_score_list, og_audio_list, fixed_notes_info, error_type, str(num_errors), difference
    )

    # Step 10: Save files
    status_text.text("💾 Saving files...")
    progress_bar.progress(95)

    # Save WAV
    wav_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    sf.write(wav_temp.name, corrected_audio_array, samplerate=44100, format='WAV', subtype='PCM_16')

    # Save CSV
    csv_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.csv', mode='w', newline='')
    fieldnames = ['error_type', 'num_errors', 'score_pitch', 'fixed_pitch',
                  'score_time', 'fixed_time', 'label', 'accuracy']
    writer = csv.DictWriter(csv_temp, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(csv_rows)
    csv_temp.close()

    progress_bar.progress(100)
    status_text.text("✅ Processing complete!")

    return wav_temp.name, csv_temp.name, accuracy, total_notes, correct_notes


# Main UI
st.markdown("---")

# If processing is complete, show only the results page
if st.session_state.processing_complete:
    st.header("📊 Results")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Notes", st.session_state.total_notes)
    with col2:
        st.metric("Correct Notes", st.session_state.correct_notes)
    with col3:
        st.metric("Accuracy", f"{st.session_state.accuracy:.2%}")

    st.markdown("---")
    st.header("⬇️ Download Files")

    col1, col2 = st.columns(2)

    with col1:
        with open(st.session_state.wav_path, 'rb') as f:
            wav_data = f.read()
        st.download_button(
            label="🎵 Download Fixed Audio (WAV)",
            data=wav_data,
            file_name=f"{st.session_state.error_type}_{st.session_state.num_errors}_{st.session_state.timestamp}.wav",
            mime="audio/wav",
            use_container_width=True,
            key="wav_download"
        )

    with col2:
        with open(st.session_state.csv_path, 'rb') as f:
            csv_data = f.read()
        st.download_button(
            label="📊 Download CSV Comparison",
            data=csv_data,
            file_name=f"{st.session_state.error_type}_{st.session_state.num_errors}_{st.session_state.timestamp}.csv",
            mime="text/csv",
            use_container_width=True,
            key="csv_download"
        )

    st.markdown("---")
    st.markdown("")  # Add some space
    st.markdown("")

    # Center the button
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("🔄 Start New Analysis", type="primary", use_container_width=True):
            # Clean up temp files
            try:
                os.unlink(st.session_state.wav_path)
                os.unlink(st.session_state.csv_path)
            except:
                pass

            # Reset session state
            st.session_state.processing_complete = False
            st.session_state.wav_path = None
            st.session_state.csv_path = None
            st.session_state.accuracy = 0
            st.session_state.total_notes = 0
            st.session_state.correct_notes = 0
            st.session_state.error_type = ""
            st.session_state.num_errors = 0
            st.session_state.timestamp = ""
            st.rerun()

else:
    # Show the upload and process interface
    # Check if all files are uploaded
    all_files_uploaded = all([score_midi_file, audio_midi_file, audio_mp3_file, sf2_file])

    if all_files_uploaded:
        st.success("✅ All files uploaded successfully!")

        col1, col2 = st.columns([1, 4])

        with col1:
            process_button = st.button("🚀 Process Files", type="primary", use_container_width=True)

        if process_button:
            # Save uploaded files to temporary locations
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mid') as score_temp:
                score_temp.write(score_midi_file.read())
                score_path = score_temp.name

            with tempfile.NamedTemporaryFile(delete=False, suffix='.mid') as audio_midi_temp:
                audio_midi_temp.write(audio_midi_file.read())
                audio_midi_path = audio_midi_temp.name

            # Determine audio file extension
            audio_file_ext = audio_mp3_file.name.lower().split('.')[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{audio_file_ext}') as audio_temp:
                audio_temp.write(audio_mp3_file.read())
                audio_mp3_path = audio_temp.name

            with tempfile.NamedTemporaryFile(delete=False, suffix='.sf2') as sf2_temp:
                sf2_temp.write(sf2_file.read())
                sf2_path_temp = sf2_temp.name

            try:
                # Process the files
                wav_path, csv_path, accuracy, total_notes, correct_notes = process_midi_alignment(
                    score_path, audio_midi_path, audio_mp3_path, sf2_path_temp, error_type, num_errors
                )

                # Store results in session state
                st.session_state.wav_path = wav_path
                st.session_state.csv_path = csv_path
                st.session_state.accuracy = accuracy
                st.session_state.total_notes = total_notes
                st.session_state.correct_notes = correct_notes
                st.session_state.error_type = error_type
                st.session_state.num_errors = num_errors
                st.session_state.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                st.session_state.processing_complete = True

                # Cleanup temp input files (but keep output files)
                os.unlink(score_path)
                os.unlink(audio_midi_path)
                os.unlink(audio_mp3_path)
                os.unlink(sf2_path_temp)

                # Rerun to show results page
                st.rerun()

            except Exception as e:
                st.error(f"❌ Error during processing: {str(e)}")

    else:
        st.info("👆 Please upload all required files in the sidebar to begin")

        st.markdown("### 📋 Required Files:")
        st.markdown("""
        - **Score MIDI**: Ground truth MIDI file
        - **Audio MIDI**: Performed audio transcribed to MIDI
        - **Audio File**: Performed audio recording (MP3 or WAV)
        - **SF2 Soundfont**: SoundFont file for synthesis
        """)

# Footer
st.markdown("---")
st.markdown("*MIDI Alignment Tool - Automatically correct pitch and timing errors in musical performances*")
