# %%
import mne
import moabb.datasets as md
import matplotlib.pyplot as plt

import numpy as np

# %%
datasets = [md.BNCI2014_001(),
            md.BNCI2014_004(),
            md.PhysionetMI(),
            md.Shin2017A(accept=True),
            md.Weibo2014(),
            md.Zhou2016()]

fig, axes = plt.subplots(2, len(datasets)//2, figsize=(9, 6), sharey=True)

for i, dataset in enumerate(datasets):
    row = i // 3
    col = i % 3
    data = dataset._get_single_subject_data(subject=1)
    session = list(data)[0]
    run = list(data[session])[0]
    raw = data[session][run]
    raw.pick('eeg')
    montage = mne.channels.make_standard_montage("standard_1005")
    raw.info.set_montage(montage)
    raw.plot_sensors(show_names=False, to_sphere=True,
                     show=False, axes=axes[row, col],
                     sphere=[-0.001, 0.018, 0, 0.095])
    axes[row, col].title.set_text(dataset.__class__.__name__)
plt.tight_layout()

# %% Final positions
fig, axes = plt.subplots(1, 1, figsize=(4,4))

montage = mne.channels.make_standard_montage("standard_1005")
standard_channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'C3',
                     'Cz', 'C4', 'P3', 'Pz', 'P4', 'T3', 'T4', 'T5',
                     'T6']
final_positions = np.array(
    [montage.get_positions()['ch_pos'][ch] for ch in standard_channels])
info_final = mne.create_info(standard_channels, sfreq=200, ch_types='eeg')
info_final.set_montage(montage)
info_final.plot_sensors(show_names=False, to_sphere=True,
                        show=False, axes=axes,
                        sphere=[-0.001, 0.018, 0, 0.095])
axes.title.set_text("Final positions")
plt.show()

# %% All datasets + final positions in one plot

datasets = [md.BNCI2014_001(),
            md.BNCI2014_004(),
            md.PhysionetMI(),
            md.Shin2017A(accept=True),
            md.Weibo2014(),
            md.Zhou2016()]

fig, axes = plt.subplots(2, 4, figsize=(8, 4), sharey=True)

for i, dataset in enumerate(datasets):
    row = i // 3
    col = i % 3
    data = dataset._get_single_subject_data(subject=1)
    session = list(data)[0]
    run = list(data[session])[0]
    raw = data[session][run]
    raw.pick('eeg')
    montage = mne.channels.make_standard_montage("standard_1005")
    raw.info.set_montage(montage)
    raw.plot_sensors(show_names=False, to_sphere=True,
                     show=False, axes=axes[row, col],
                     sphere=[-0.001, 0.018, 0, 0.095])
    axes[row, col].title.set_text(dataset.__class__.__name__)

montage = mne.channels.make_standard_montage("standard_1005")
standard_channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'C3',
                     'Cz', 'C4', 'P3', 'Pz', 'P4', 'T3', 'T4', 'T5',
                     'T6']
final_positions = np.array(
    [montage.get_positions()['ch_pos'][ch] for ch in standard_channels])
info_final = mne.create_info(standard_channels, sfreq=200, ch_types='eeg')
info_final.set_montage(montage)
info_final.plot_sensors(show_names=False, to_sphere=True,
                        show=False, axes=axes[1, 3],
                        sphere=[-0.001, 0.018, 0, 0.095])
axes[1, 3].title.set_text("Final positions")
axes[0, 3].axis('off')
plt.tight_layout()
plt.show()


# %%
datasets = [md.BNCI2015_001(),
            md.BNCI2014_001(),
            md.AlexMI(),
            md.Weibo2014(),
            md.Zhou2016(),
            md.Schirrmeister2017()]

fig, axes = plt.subplots(1, len(datasets), figsize=(15, 3), sharey=True)

for i, dataset in enumerate(datasets):
    data = dataset._get_single_subject_data(subject=1)
    session = list(data)[0]
    run = list(data[session])[0]
    raw = data[session][run]
    raw.pick('eeg')
    montage = mne.channels.make_standard_montage("standard_1005")
    raw.info.set_montage(montage)
    raw.plot_sensors(show_names=False, to_sphere=True,
                     show=False, axes=axes[i],
                     sphere=[-0.001, 0.018, 0, 0.095])
    axes[i].title.set_text(dataset.__class__.__name__)
plt.tight_layout()
# %%
