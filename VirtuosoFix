from pretty_midi import PrettyMIDI, Instrument, Note
import warnings
import numpy as np
from pydub import AudioSegment
import soundfile as sf
from datetime import datetime
import copy

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
score_ask_m = '/Users/mandeepwalia/Desktop/midi_real/Input midi/mary-had-a-little-lamb_midi.mid'
audio_ask =   '/Users/mandeepwalia/Desktop/midi_real/Input MP3/12:26_2.mp3'
audio_ask_m = '/Users/mandeepwalia/Desktop/midi_real/Input midi/12:26_2_mid.mid'
sf2_path = '/Users/mandeepwalia/Desktop/midi_real/TimGM6mb.sf2'
output_path_final = '/Users/mandeepwalia/Desktop/midi_real/Output final file'


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


def insert(audio_array, note_array, start, end):
    audio_array[start:end] = note_array
    return audio_array


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

        # Prevent division by zero
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

    # FIXED: Use curr_note_index + 1 to get NEXT note, and use og_audio_list
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

    # Count is always >= 1 since current note is guaranteed to be an error
    # Use the last correct note's audio position and add the score-based time difference
    last_correct_audio_start = score_audio_list[curr_note_index - count][1][1].start
    last_correct_score_start = score_audio_list[curr_note_index - count][0][1].start

    # Calculate how much time should pass from last correct note to current note (from score)
    score_time_diff = curr_score_start - last_correct_score_start

    # Apply this timing to the audio position
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

    # ONLY cut from the start
    left = left[cutoff:]
    right = left.copy()

    return np.column_stack((left, right))


def add_crossfade(note_array, fade_samples=220):
    """
    Apply fade in/out to note array to prevent clicks/pops
    fade_samples: number of samples for fade (220 = 5ms at 44100Hz)
    Using 5ms to prevent swallowing notes while still eliminating clicks
    """
    if len(note_array) < fade_samples * 2:
        # If note is too short, use shorter fade
        fade_samples = len(note_array) // 4

    if fade_samples > 0:
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)

        # Handle both mono and stereo arrays
        if note_array.ndim == 2:
            # Stereo: reshape fade to broadcast correctly
            fade_in = fade_in.reshape(-1, 1)
            fade_out = fade_out.reshape(-1, 1)

        note_array[:fade_samples] *= fade_in
        note_array[-fade_samples:] *= fade_out

    return note_array


def silence_note(left, og_audio_data, fade_samples=220):
    """Silence with fade to prevent pops"""
    start_audio = og_audio_data.start
    end_audio = og_audio_data.end
    start_frame = int(start_audio * 44100)
    end_frame = int(end_audio * 44100)

    segment_length = end_frame - start_frame

    # Apply fade out before silencing to prevent clicks
    if segment_length > fade_samples:
        fade_out = np.linspace(1, 0, fade_samples)
        left[start_frame:start_frame + fade_samples] *= fade_out
        left[start_frame + fade_samples:end_frame] = 0
    else:
        # If segment is short, fade the whole thing
        if segment_length > 0:
            fade_out = np.linspace(1, 0, segment_length)
            left[start_frame:end_frame] *= fade_out

    return left


def align(score, audio):
    first_score = score[0][1]
    first_audio = audio[0][1]

    factor_score = first_score.start
    factor_audio = first_audio.start

    difference = factor_score - factor_audio

    for note, data, chord_num in score:
        data.start -= difference
        data.end -= difference

    print('Done: Alignment')
    return difference


def n_p_s_e(midi):
    midi = PrettyMIDI(midi)
    n_p_t_list = []

    # Read ALL instruments (both left and right hand)
    for instrument in midi.instruments:
        if not instrument.is_drum:  # Skip drum tracks if any
            for note in instrument.notes:
                # Don't number yet, just collect all notes
                n_p_t_list.append(note)

    # CRITICAL: Sort all notes by start time, then by pitch (for consistent ordering)
    n_p_t_list.sort(key=lambda note: (note.start, note.pitch))

    # Now number the notes after sorting
    numbered_list = [(i + 1, note) for i, note in enumerate(n_p_t_list)]

    print(f'Done: Transcription - Total notes: {len(numbered_list)}')

    return numbered_list


def file_to_stereo_np(file):
    audio_data = AudioSegment.from_mp3(file)
    samples = np.array(audio_data.get_array_of_samples()) / 32768.0
    channels = audio_data.channels
    if channels == 2:
        return samples.reshape((-1, channels))
    else:
        return np.column_stack((samples, samples))


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


def fix_errors(score, audio, og_score, og_audio):
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

        # FIXED: Prevent division by zero
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

    print('Done: Classification')

    return both_c, pitch_c, time_c


def convert_p_t_b(p_list, t_list, b_list):
    left_copy = copy.deepcopy(left_audio)

    p_arrays = []
    t_arrays = []
    b_arrays = []

    i = 0

    for note_b, og_score_data_b, og_audio_data_b, chord_num_b in b_list:
        i += 1
        new_piano = Instrument(program=0)
        new_midi = PrettyMIDI()

        score_notes = copy.deepcopy(score_dict[chord_num_b])

        # FLUIDSYNTH ONLY WORKS WITH NOTES STARTING AT 0
        for note_num, note_data, chord_num in score_notes:
            adjusted_start, adjusted_end = 0, note_data.end - note_data.start
            note_data.start, note_data.end = adjusted_start, adjusted_end
            new_piano.notes.append(note_data)

        new_midi.instruments.append(new_piano)
        new_midi_np_array = new_midi.fluidsynth(sf2_path=sf2_path)

        # VOLUME CONTROL
        volume_factor = get_volume(new_midi_np_array, left_copy, og_audio_data_b)

        stereo_note_np_array = np.column_stack((new_midi_np_array, new_midi_np_array))
        stereo_note_np_array = cut_dead_time(stereo_note_np_array)
        stereo_note_np_array *= volume_factor

        # ADDED: Apply crossfade to prevent clicks (5ms fade)
        stereo_note_np_array = add_crossfade(stereo_note_np_array)

        b_arrays.append((note_b, stereo_note_np_array, og_score_data_b, og_audio_data_b))

        print(f'Done: Both Conversion {i}')

    i = 0

    for note_p, og_score_data_p, og_audio_data_p, chord_num_p in p_list:
        i += 1
        new_piano = Instrument(program=0)
        new_midi = PrettyMIDI()

        score_notes = copy.deepcopy(score_dict[chord_num_p])

        # FLUIDSYNTH ONLY WORKS WITH NOTES STARTING AT 0
        for note_num, note_data, chord_num in score_notes:
            adjusted_start, adjusted_end = 0, note_data.end - note_data.start
            note_data.start, note_data.end = adjusted_start, adjusted_end
            new_piano.notes.append(note_data)

        new_midi.instruments.append(new_piano)
        new_midi_np_array = new_midi.fluidsynth(sf2_path=sf2_path)

        # VOLUME CONTROL
        volume_factor = get_volume(new_midi_np_array, left_copy, og_audio_data_p)

        stereo_note_np_array = np.column_stack((new_midi_np_array, new_midi_np_array))
        stereo_note_np_array = cut_dead_time(stereo_note_np_array)
        stereo_note_np_array *= volume_factor

        # ADDED: Apply crossfade to prevent clicks (5ms fade)
        stereo_note_np_array = add_crossfade(stereo_note_np_array)

        p_arrays.append((note_p, stereo_note_np_array, og_score_data_p, og_audio_data_p))

        print(f'Done: Pitch Conversion {i}')

    i = 0

    for note, og_score_data, og_audio_data, chord_num_t in t_list:
        i += 1



        audio_start_frame = int(og_audio_data.start * 44100)
        audio_end_frame = int(og_audio_data.end * 44100)

        audio_array = left_copy[audio_start_frame: audio_end_frame]
        stereo_note_array = np.column_stack((audio_array, audio_array))
        stereo_note_array = cut_dead_time(stereo_note_array)

        # ADDED: Apply crossfade to prevent clicks (5ms fade)
        stereo_note_array = add_crossfade(stereo_note_array)

        t_arrays.append((note, stereo_note_array, og_score_data, og_audio_data))



        print(f'Done: Time Conversion {i}')

    return p_arrays, t_arrays, b_arrays


def insert_p_t_b(p_data, t_data, b_data, difference):
    global og_audio_array, left_audio
    left_copy = copy.deepcopy(left_audio)

    # FIXED: Use set for O(1) lookup performance
    error_list = set(note for note, _, _, _ in t_data) | set(note for note, _, _, _ in b_data)

    # BOTH
    for note_b, note_array_b, og_score_data_b, og_audio_data_b in b_data:
        start, end = get_start(note_b, og_score_list, og_audio_list, error_list, difference)
        s_f, e_f = int(start * 44100), int(end * 44100)

        # Silence without fade - direct muting
        if s_f < len(left_copy) and e_f <= len(left_copy):
            left_copy[s_f:e_f] = 0

        # insert new
        left_note = truncate(s_f, e_f, note_array_b[:, 0])
        left_copy[s_f:e_f] = left_note

    # PITCH
    for note_p, note_array_p, og_score_data_p, og_audio_data_p in p_data:
        s_f = int(og_audio_data_p.start * 44100)
        e_f = int(og_audio_data_p.end * 44100)

        # Silence without fade - direct muting
        if s_f < len(left_copy) and e_f <= len(left_copy):
            left_copy[s_f:e_f] = 0

        left_note = truncate(s_f, e_f, note_array_p[:, 0])
        left_copy[s_f:e_f] = left_note

    # TIME
    for note_t, audio_array_t, og_score_data_t, og_audio_data_t in t_data:
        start, end = get_start(note_t, og_score_list, og_audio_list, error_list, difference)
        print(start, end)
        s_f, e_f = int(start * 44100), int(end * 44100)

        # Silence without fade - direct muting
        if s_f < len(left_copy) and e_f <= len(left_copy):
            left_copy[s_f:e_f] = 0

        left_note = truncate(s_f, e_f, audio_array_t[:, 0])
        left_copy[s_f:e_f] = left_note

    og_audio_array = np.column_stack((left_copy, left_copy))




def write():
    #sf.write(output_path_final + f'/#{timestamp}.WAV', og_audio_array, samplerate=44100, format='WAV', subtype='PCM_16')
    print('Done: Export')


# All function calls below
score_list = (n_p_s_e(score_ask_m))
audio_list = (n_p_s_e(audio_ask_m))

og_audio_len = len(audio_list)

og_audio_array = file_to_stereo_np(audio_ask)
left_audio, right_audio = og_audio_array[:, 0], og_audio_array[:, 0]

audio_dict = identify_chords_audio(audio_list)
score_dict = identify_chords_score(score_list)

audio_list = remove_trail_notes_audio(audio_list, audio_dict)
score_list = remove_trail_notes_score(score_list, score_dict)

cut_end(score_list, audio_list)

og_score_list = copy.deepcopy(score_list)
og_audio_list = copy.deepcopy(audio_list)

difference = align(score_list, audio_list)

last_score_note_start_frame, last_score_note_end_frame = (score_list[len(audio_list) - 1][
                                                              1].start - difference) * 44100, (
                                                                 score_list[len(audio_list) - 1][
                                                                     1].end - difference) * 44100

if last_score_note_end_frame > len(left_audio):
    needed_length = int(last_score_note_end_frame)
    padding_needed = needed_length - len(left_audio)

    # Pad with zeros at the end
    left_audio = np.pad(left_audio, (0, padding_needed), mode='constant', constant_values=0)
    right_audio = np.pad(right_audio, (0, padding_needed), mode='constant', constant_values=0)
    og_audio_array = np.column_stack((left_audio, right_audio))

both, pitch, time = fix_errors(score_list, audio_list, og_score_list, og_audio_list)

p_data, t_data, b_data = convert_p_t_b(pitch, time, both)

insert_p_t_b(p_data, t_data, b_data, difference)

write()