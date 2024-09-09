import numpy as np
import mne
from pyriemann.estimation import Covariances


def pick_common_channels(source, target):
    common_channels = [
        ch for ch in source['org']['chnames'] if ch in target['org']['chnames']
    ]
    source_index = [
        source['org']['chnames'].index(ch) for ch in common_channels
    ]
    target_index = [
        target['org']['chnames'].index(ch) for ch in common_channels
    ]
    source['org-bas'] = {}
    source['org-bas']['covs'] = source['org']['covs'][:, source_index][
        :, :, source_index
    ]
    source['org-bas']['labels'] = source['org']['labels']
    source['org-bas']['chnames'] = source['org']['chnames']

    target['org-bas'] = {}
    target['org-bas']['covs'] = target['org']['covs'][:, target_index][
        :, :, target_index
    ]
    target['org-bas']['labels'] = target['org']['labels']
    target['org-bas']['chnames'] = target['org']['chnames']

    return source, target


def pick_common_channels_one(data, common_channels):
    data_index = [
        data['org']['chnames'].index(ch) for ch in common_channels
    ]

    data['org-bas'] = {}
    data['org-bas']['covs'] = data['org']['covs'][:, data_index][
        :, :, data_index
    ]
    data['org-bas']['labels'] = data['org']['labels']
    data['org-bas']['chnames'] = data['org']['chnames']
    return data


def splines_interpolation(epochs,
                          current_pos,
                          final_pos,
                          alpha):

    # Interpolation from actual channels to tuab channels
    interpolation = mne.channels.interpolation._make_interpolation_matrix(
        current_pos, final_pos, alpha=alpha
    )
    epochs_interpolated = np.matmul(interpolation, epochs)

    return epochs_interpolated, interpolation


def splines_interpolation_one(epochs,
                              current_pos,
                              final_pos,
                              alpha):

    # Interpolation from actual channels to tuab channels
    interpolation = mne.channels.interpolation._make_interpolation_matrix(
        current_pos, final_pos, alpha=alpha
    )
    epochs_interpolated = np.matmul(interpolation, epochs)

    return epochs_interpolated, interpolation


def mne_interpolation(epochs, info_source, info_target,
                      mode='accurate', origin=(0.0, 0.0, 0.04),
                      reg=1e-1):

    interpolation = mne.channels.interpolation._map_meg_or_eeg_channels(
        info_source, info_target, mode, origin, miss=reg
    )
    epochs_interpolated = np.matmul(interpolation, epochs)

    return epochs_interpolated, interpolation


def mne_interpolation_one(epochs, info_source, info_target,
                          mode='accurate', origin=(0.0, 0.0, 0.04),
                          reg=1e-3):

    interpolation = mne.channels.interpolation._map_meg_or_eeg_channels(
        info_source, info_target, mode, origin, miss=reg
    )
    epochs_interpolated = np.matmul(interpolation, epochs)

    return epochs_interpolated, interpolation


def make_spline_interpolation(source, target, X_source, alpha):

    positions_source = source['org']['positions']
    positions_target = target['org']['positions']

    X_source_ssi, _ = splines_interpolation(
        X_source, positions_source, positions_target, alpha
    )

    source['org-ssi'] = {}
    source['org-ssi']['covs'] = Covariances(estimator='lwf').fit_transform(
        X_source_ssi
    )
    target['org-ssi'] = target['org']

    source['org-ssi']['labels'] = source['org']['labels']
    source['org-ssi']['chnames'] = source['org']['chnames']

    return source, target


def make_spline_interpolation_one(data, epochs, final_positions, alpha):

    positions_data = data['org']['positions']

    epochs_ssi, _ = splines_interpolation(
        epochs, positions_data, final_positions, alpha
    )

    data['org-ssi'] = {}
    data['org-ssi']['covs'] = Covariances(estimator='lwf').fit_transform(
        epochs_ssi
    )
    data['org-ssi']['labels'] = data['org']['labels']
    data['org-ssi']['chnames'] = data['org']['chnames']
    return data


def make_mne_interpolation(source, target, X_source, reg):

    info_source = source['org']['info']
    info_target = target['org']['info']
    # origin = mne.bem._check_origin("auto", info_source)

    X_source_mne, _ = mne_interpolation(
        X_source, info_source, info_target, reg=reg
    )

    source['org-mne'] = {}
    source['org-mne']['covs'] = Covariances(estimator='lwf').fit_transform(
        X_source_mne
    )
    target['org-mne'] = target['org']

    source['org-mne']['labels'] = source['org']['labels']
    source['org-mne']['chnames'] = source['org']['chnames']

    return source, target


def make_mne_interpolation_one(data, epochs, info_final, reg):

    info = data['org']['info']

    epochs_mne, _ = mne_interpolation_one(
        epochs, info, info_final, reg=reg
    )

    data['org-mne'] = {}
    data['org-mne']['covs'] = Covariances(estimator='lwf').fit_transform(
        epochs_mne
    )

    data['org-mne']['labels'] = data['org']['labels']
    data['org-mne']['chnames'] = data['org']['chnames']

    return data
