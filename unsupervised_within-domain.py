import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from pyriemann.tangentspace import TangentSpace
import moabb.datasets as md
from moabb.paradigms import MotorImagery
from pyriemann.estimation import Covariances

from joblib import Parallel, delayed

np.random.seed(0)
torch.manual_seed(0)


def load_data(dataset, subject, paradigm):
    # get data from target
    data = {}
    X, labels, _ = paradigm.get_data(dataset, subjects=[subject])
    data['covs'] = Covariances(estimator='lwf').fit_transform(
        X
    )
    data['labels'] = labels

    return data


def run_cv(data, train_index, test_index):
    data_train = data['covs'][train_index]
    y_train = data['labels'][train_index]
    data_test = data['covs'][test_index]
    y_test = data['labels'][test_index]
    clf = LogisticRegression(max_iter=int(1e4))
    ts = TangentSpace()
    ts_data_train = ts.fit_transform(data_train)
    ts_data_test = ts.transform(data_test)
    clf.fit(ts_data_train, y_train)
    y_pred = clf.predict(ts_data_test)
    y_pred_proba = clf.predict_proba(ts_data_test)
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    scores = dict(auc=auc, accuracy=accuracy)
    return scores


def run_one_sub(dataset, subject):
    print(f"Loading data of subject {subject}")
    data = load_data(dataset, subject, paradigm)
    cv = KFold(n_splits=2)
    scores = []
    print("Begin cross-validation")
    scores = Parallel(n_jobs=N_JOBS)(
        delayed(run_cv)(
            data, train_index, test_index
        ) for train_index, test_index in cv.split(data['covs'],
                                                  data['labels'])
    )
    print("End cross-validation")
    scores = pd.DataFrame(scores)
    scores['subject'] = subject
    return scores


def run_one_ds(dataset):
    dataset_name = dataset.__class__.__name__
    print(f'Dataset: {dataset_name}')
    subjects = dataset.subject_list
    scores = Parallel(n_jobs=N_JOBS)(
        delayed(run_one_sub)(
            dataset, subject
        ) for subject in subjects
    )
    scores = pd.concat(scores)
    scores['dataset'] = dataset_name
    return scores


datasets = [md.Zhou2016(),
            md.Shin2017A(accept=True),
            md.BNCI2014_001(),
            md.Weibo2014(),
            md.BNCI2014_004(),
            md.PhysionetMI()]
events = ["right_hand", "left_hand"]
paradigm = MotorImagery(events=events, n_classes=len(events), resample=128)

N_JOBS = 10

scores = Parallel(n_jobs=N_JOBS)(
    delayed(run_one_ds)(
        dataset
    ) for dataset in datasets
)
scores = pd.concat(scores)
scores.to_csv(
    './results/unsupervised_within-domains_right_left_hand'
)
