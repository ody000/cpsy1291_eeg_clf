import os
import numpy as np

RAW_ROOT = "OSF Data Raw"
PROC_ROOT = "OSF Data Processed unGNN"

GROUPS = ["AD", "Healthy"]
EYE_STATES = ["Eyes_closed", "Eyes_open"]

def load_patient_channels(patient_path):
    """
    Load all .txt channel files from a Paciente folder into a dict.
    Returns:
        ch_names: list of channel names (e.g. ['C3', 'C4', ...])
        data: np.ndarray, shape (n_channels, n_samples)
    """
    ch_names = []
    series_list = []

    for fname in sorted(os.listdir(patient_path)):
        if not fname.lower().endswith(".txt"):
            continue
        ch_name = os.path.splitext(fname)[0]  # 'C3' from 'C3.txt'
        fpath = os.path.join(patient_path, fname)
        # each file is one value per line
        x = np.loadtxt(fpath, dtype=float)
        ch_names.append(ch_name)
        series_list.append(x)

    if not series_list:
        return [], None

    # Ensure all channels have same length
    lengths = {len(s) for s in series_list}
    if len(lengths) != 1:
        raise ValueError(
            f"Channel lengths mismatch in {patient_path}: {lengths}"
        )

    data = np.stack(series_list, axis=0)  # (n_channels, n_samples)
    return ch_names, data


def common_average_reference(data):
    """
    data: (n_channels, n_samples)
    return: re-referenced data with average reference per timepoint
    """
    mean_over_channels = data.mean(axis=0, keepdims=True)
    return data - mean_over_channels


def zscore_per_channel(data, eps=1e-6):
    """
    data: (n_channels, n_samples)
    return: z-scored per channel (mean 0, std 1 per channel)
    """
    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True) + eps
    return (data - mean) / std


def process_patient(raw_patient_path, proc_patient_path):
    # load all channel txt files
    ch_names, data = load_patient_channels(raw_patient_path)
    if data is None:
        print(f"[WARN] No .txt files found in {raw_patient_path}")
        return

    # preprocessing steps
    # take common average reference(x-mean  across channels at each timepoint)
    # then z-score per channel
    data = common_average_reference(data)
    data = zscore_per_channel(data)

    # make output folder
    os.makedirs(proc_patient_path, exist_ok=True)

    # save each channel back to its own txt file
    for i, ch_name in enumerate(ch_names):
        out_path = os.path.join(proc_patient_path, ch_name + ".txt")
        np.savetxt(out_path, data[i], fmt="%.6f")


def main():
    for group in GROUPS:
        for eye_state in EYE_STATES:
            raw_state_path = os.path.join(RAW_ROOT, group, eye_state)
            if not os.path.isdir(raw_state_path):
                # skip combos that don't exist
                continue

            proc_state_path = os.path.join(PROC_ROOT, group, eye_state)
            os.makedirs(proc_state_path, exist_ok=True)

            # each subfolder under this is a Paciente folder
            for patient in sorted(os.listdir(raw_state_path)):
                raw_patient_path = os.path.join(raw_state_path, patient)
                if not os.path.isdir(raw_patient_path):
                    continue

                proc_patient_path = os.path.join(proc_state_path, patient)
                print(f"Processing {raw_patient_path} -> {proc_patient_path}")
                process_patient(raw_patient_path, proc_patient_path)

    print("Done. Processed data saved under:", PROC_ROOT)


if __name__ == "__main__":
    main()
