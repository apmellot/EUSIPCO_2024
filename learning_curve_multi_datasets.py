import argparse
import numpy as np
import pandas as pd
import mne
import moabb.datasets as md
from moabb.paradigms import MotorImagery
from pyriemann.estimation import Covariances
from pyriemann.utils.tangentspace import tangent_space
from pyriemann.utils.mean import mean_covariance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.dummy import DummyClassifier
from utils import dimensionality_transcending as DT
from utils import transfer_learning as TL
from utils import interpolation as INT
from utils import comimp as CI

from joblib import Parallel, delayed

np.random.seed(0)

parser = argparse.ArgumentParser(description="Run unsupervised multi-dataset")
parser.add_argument('-d', '--dataset_id', default=42,
                    help='Index of the target dataset')
args = parser.parse_args()
dataset_id = int(args.dataset_id)
print('DATASET ID: ', dataset_id)


def load_source_data(datasets_source, paradigm):
    source = []
    list_X_source = []
    list_meta_source = []
    for i, dataset_source in enumerate(datasets_source):
        source.append({})
        source[i]['org'] = {}
        data_source = dataset_source._get_single_subject_data(1)
        session_source = list(data_source)[0]
        run_source = list(data_source[session_source])[0]
        raw_source = data_source[session_source][run_source]
        raw_source.pick('eeg')
        X_source, labels_source, meta_source = paradigm.get_data(
            dataset_source)
        source[i]['org']['covs'] = Covariances(estimator='lwf').fit_transform(
            X_source
        )
        source[i]['org']['labels'] = labels_source
        source[i]['org']['chnames'] = [
            chi.upper() for chi in raw_source.ch_names]
        montage = mne.channels.make_standard_montage("standard_1005")
        raw_source.info.set_montage(montage)
        source[i]['org']['positions'] = raw_source._get_channel_positions()
        source[i]['org']['info'] = raw_source.info
        list_X_source.append(X_source)
        list_meta_source.append(meta_source)
    return source, list_X_source, list_meta_source


def load_target_data(dataset_target, paradigm):
    # get data from target
    target = {}
    data_target = dataset_target._get_single_subject_data(1)
    session_target = list(data_target)[0]
    run_target = list(data_target[session_target])[0]
    raw_target = data_target[session_target][run_target]

    raw_target.pick('eeg')
    X_target, labels_target, meta_target = paradigm.get_data(
            dataset_target)
    target['org'] = {}
    target['org']['covs'] = Covariances(estimator='lwf').fit_transform(
        X_target
    )
    target['org']['labels'] = labels_target
    target['org']['chnames'] = [chi.upper() for chi in raw_target.ch_names]
    montage = mne.channels.make_standard_montage("standard_1005")
    raw_target.info.set_montage(montage)
    target['org']['positions'] = raw_target._get_channel_positions()
    target['org']['info'] = raw_target.info

    return target, X_target, meta_target


def run_alignment(source, list_meta_source, target, meta_target, domain_level):
    # apply RPA on baseline and interpolated data
    for s in ['bas', 'comimp', 'aug', 'ssi', 'mne']:
        print(s)
        for i in range(len(source)):
            print("Recentering dataset source ", i)
            source[i][f'rct-{s}'] = TL.RPA_recenter_unsupervised_one(
                    source[i][f'org-{s}'], list_meta_source[i],
                    domain_level=domain_level, target=False
                )
            print("Rescaling dataset source ", i)
            source[i][f'str-{s}'] = TL.RPA_stretch_unsupervised_one(
                    source[i][f'rct-{s}'], list_meta_source[i],
                    domain_level=domain_level
                )
        print("Recentering dataset target")
        target[f'rct-{s}'] = TL.RPA_recenter_unsupervised_one(
            target[f'org-{s}'], meta_target, domain_level=domain_level,
            target=True
            )
        print("Rescaling dataset target")
        target[f'str-{s}'] = TL.RPA_stretch_unsupervised_one(
            target[f'rct-{s}'], meta_target, domain_level=domain_level
            )
    return source, target


def run_predict(clf, dummy_clf, Cr, target, subject_target, meta_target, meth):
    score = []
    print('Subject target n ', subject_target)
    covs_target = target[meth]['covs'][
        meta_target['subject'] == subject_target
    ]
    y_target = target[meth]['labels'][
        meta_target['subject'] == subject_target
    ]
    ts_covs_target = tangent_space(covs_target, Cr)
    y_pred = clf.predict(ts_covs_target)
    y_pred_proba = clf.predict_proba(ts_covs_target)
    accuracy = accuracy_score(y_target, y_pred)
    auc = roc_auc_score(y_target, y_pred_proba[:, 1])
    score.append(dict(method=meth,
                      auc=auc,
                      accuracy=accuracy,
                      subject_target=subject_target))
    # Dummy
    y_dummy = dummy_clf.predict(ts_covs_target)
    y_dummy_proba = dummy_clf.predict_proba(ts_covs_target)
    accuracy_dummy = accuracy_score(y_target, y_dummy)
    auc_dummy = roc_auc_score(y_target, y_dummy_proba[:, 1])
    score.append(dict(method='dummy',
                      auc=auc_dummy,
                      accuracy=accuracy_dummy,
                      subject_target=subject_target))
    return pd.DataFrame(score)


def run_classifier(source, target, meta_target):
    clf = LogisticRegression(max_iter=int(1e4))
    scores = []
    for meth in ['org-bas', 'rct-bas', 'str-bas',
                 'org-comimp', 'rct-comimp', 'str-comimp',
                 'org-aug', 'rct-aug', 'str-aug',
                 'org-ssi', 'rct-ssi', 'str-ssi',
                 'org-mne', 'rct-mne', 'str-mne']:
        print(meth)
        # Train on source
        covs_source = np.concatenate(
            [s[meth]['covs'] for s in source]
            )
        y_source = np.concatenate(
            [s[meth]['labels'] for s in source]
            )
        if 'org' in meth:
            Cr = mean_covariance(covs_source)
        else:
            Cr = np.eye(covs_source.shape[-1])
        ts_covs_source = tangent_space(covs_source, Cr)
        clf.fit(ts_covs_source, y_source)
        # Dummy
        dummy_clf = DummyClassifier()
        dummy_clf.fit(ts_covs_source, y_source)
        # Predict on each subject target
        subjects_target = np.unique(meta_target.subject.values)
        score = Parallel(n_jobs=N_JOBS)(
            delayed(run_predict)(
                clf, dummy_clf, Cr, target, subject_target, meta_target, meth
            ) for subject_target in subjects_target
        )
        scores.append(pd.concat(score))
    scores = pd.concat(scores)
    return scores


def run_learning_curve(source, target, meta_target):
    scores = []
    for i in range(len(source)):
        source_train = source[:i+1]
        score = run_classifier(source_train, target, meta_target)
        score['dataset_target'] = dataset_target.__class__.__name__
        score['n_dataset_train'] = i+1
        scores.append(score)
    scores = pd.concat(scores)
    return scores


def run_one(paradigm, datasets, dataset_target, domain_level):
    # Load source and target data
    datasets_source = [
        ds for ds in datasets if ds != dataset_target
    ]
    source, list_X_source, list_meta_source = load_source_data(
          datasets_source, paradigm)
    target, X_target, meta_target = load_target_data(dataset_target, paradigm)
    # Look for common channels between source and target
    channels = []
    for s in source:
        channels.append(s['org']['chnames'])
    channels.append(target['org']['chnames'])
    common_channels = set.intersection(*[set(ch) for ch in channels])
    # Get the indices of the electrode correspondances between the datasets
    source, target = DT.get_source_target_correspondance_multi(source, target)
    # Set the final positions to interpolation to
    montage = mne.channels.make_standard_montage("standard_1005")
    standard_channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'C3',
                         'Cz', 'C4', 'P3', 'Pz', 'P4', 'T3', 'T4', 'T5',
                         'T6']
    final_positions = np.array(
        [montage.get_positions()['ch_pos'][ch] for ch in standard_channels])
    info_final = mne.create_info(standard_channels, sfreq=200, ch_types='eeg')
    info_final.set_montage(montage)
    # Apply all matching dimension methods to source datasets
    for i in range(len(source)):
        print("Matching dimension of", datasets_source[i].__class__.__name__)
        # Dimensionality transcending
        source[i] = DT.match_dimensions_unsupervised_one(source[i])
        # Pick the common channels to create baseline
        source[i] = INT.pick_common_channels_one(source[i], common_channels)
        # Make SSI and MNE interpolation: positions source to positions target
        source[i] = INT.make_spline_interpolation_one(source[i],
                                                      list_X_source[i],
                                                      final_positions,
                                                      alpha=1e-7)
        source[i] = INT.make_mne_interpolation_one(source[i],
                                                   list_X_source[i],
                                                   info_final,
                                                   reg=1e-3)
    # Apply all matching dimension methods to target
    print("Matching dimension of", dataset_target.__class__.__name__)
    target = DT.match_dimensions_unsupervised_one(target)
    target = INT.pick_common_channels_one(target, common_channels)
    target = INT.make_spline_interpolation_one(target, X_target,
                                               final_positions,
                                               alpha=1e-7)
    target = INT.make_mne_interpolation_one(target, X_target, info_final,
                                            reg=1e-3)
    source, target = CI.comimp(list_X_source, X_target, source, target,
                               meta_target, imputer_type='iterative')
    # Alignment source and target (all subjects)
    source, target = run_alignment(source, list_meta_source, target,
                                   meta_target, domain_level)
    scores = run_learning_curve(source, target, meta_target)
    return scores


datasets = [md.Shin2017A(accept=True),
            md.BNCI2014_004(),
            md.Zhou2016(),
            md.BNCI2014_001(),
            md.Weibo2014(),
            md.PhysionetMI()
            ]
events = ["right_hand", "left_hand"]
paradigm = MotorImagery(events=events, n_classes=len(events), resample=128)
domain_level = 'subject'

N_JOBS = 20

dataset_target = datasets[dataset_id]
ds_name = dataset_target.__class__.__name__
print("DATASET TARGET: ", ds_name)

scores = run_one(paradigm, datasets, dataset_target, domain_level)

scores.to_csv(
    f'./results/learning_curve_{domain_level}_{ds_name}'
)
