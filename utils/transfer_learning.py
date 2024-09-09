import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

from pyriemann.transfer import (TLCenter, TLStretch,
                                TLRotate, encode_domains)
from pyriemann.utils.tangentspace import tangent_space
from pyriemann.utils.mean import mean_covariance


def get_sourcetarget_split(target, ncovs_train, random_state):

    sss = StratifiedShuffleSplit(n_splits=1,
                                 train_size=2*ncovs_train/len(
                                     target['labels']
                                 ),
                                 random_state=random_state)
    for train_index, test_index in sss.split(target['covs'], target['labels']):
        target_train = {}
        target_train['covs'] = target['covs'][train_index]
        target_train['labels'] = target['labels'][train_index]

        target_test = {}
        target_test['covs'] = target['covs'][test_index]
        target_test['labels'] = target['labels'][test_index]

    return target_train, target_test


def get_domains(meta_source, meta_target=None, ncovs_target_train=0,
                domain_level='subject', supervised=True):

    if supervised:
        # One domain per subject
        if domain_level == 'subject':
            domains_source = list(meta_source['subject'].values)
        # One domain per session per subject
        elif domain_level == 'session':
            domains_source = [str(sub)+'-'+str(ses) for (sub, ses) in zip(
                list(meta_source['subject'].values),
                list(meta_source['session'].values))]
        domains_target = ['target_domain']*ncovs_target_train*2
    else:
        # One domain per subject in both source and target
        if domain_level == 'subject':
            domains_source = list(meta_source['subject'].values)
            domains_target = list(meta_target['subject'].values)
        # One domain per subject per session in both source and target
        elif domain_level == 'session':
            domains_source = [str(sub)+'-'+str(ses) for (sub, ses) in zip(
                list(meta_source['subject'].values),
                list(meta_source['session'].values))]
            domains_target = [str(sub)+'-'+str(ses) for (sub, ses) in zip(
                list(meta_target['subject'].values),
                list(meta_target['session'].values))]
    return domains_source, domains_target


def get_domains_one(meta_data, domain_level='subject'):
    # One domain per subject in
    if domain_level == 'subject':
        domains = list(meta_data['subject'].values)
    # One domain per subject per session
    elif domain_level == 'session':
        domains = [str(sub)+'-'+str(ses) for (sub, ses) in zip(
            list(meta_data['subject'].values),
            list(meta_data['session'].values))]

    return domains


def RPA_recenter(source, target_train, target_test, meta_source,
                 T=1, sample_weight=None, domain_level='subject'):

    domains_source, domains_target = get_domains(
        meta_source, ncovs_target_train=len(target_train['labels']),
        domain_level=domain_level, supervised=False)
    domains = domains_source + domains_target

    covs_all = np.concatenate((source['covs'], target_train['covs']))
    labels_all = np.concatenate((source['labels'], target_train['labels']))
    _, labels_enc = encode_domains(covs_all, labels_all, domains)

    source_rct = {}
    target_train_rct = {}
    target_test_rct = {}
    source_rct['labels'] = source['labels']
    target_train_rct['labels'] = target_train['labels']
    target_test_rct['labels'] = target_test['labels']

    rct = TLCenter(target_domain='target_domain')
    X_rct = rct.fit_transform(covs_all, labels_enc, T, sample_weight)
    source_rct['covs'] = X_rct[:len(source['labels'])]
    target_train_rct['covs'] = X_rct[len(source['labels']):]
    target_test_rct['covs'] = rct.transform(target_test['covs'])

    return source_rct, target_train_rct, target_test_rct


# def RPA_recenter_unsupervised_one(data, meta_data, T=1, sample_weight=None,
#                                   domain_level='subject', target=False):

#     domains = get_domains_one(meta_data, domain_level=domain_level)

#     covs = data['covs']
#     _, labels_enc = encode_domains(covs, data['labels'], domains)
#     data_rct = {}
#     data_rct['labels'] = data['labels']

#     rct = TLCenter(target_domain='target_domain')
#     import ipdb; ipdb.set_trace()
#     data_rct['covs'] = rct.fit_transform(covs, labels_enc,
#                                          T, sample_weight)
#     return data_rct

def RPA_recenter_unsupervised_one(data, meta_data, T=1, sample_weight=None,
                                  domain_level='subject', target=False):

    domains = get_domains_one(meta_data, domain_level=domain_level)

    covs = data['covs']
    labels = data['labels']
    _, labels_enc = encode_domains(covs, labels, domains)
    data_rct = {}
    data_rct['labels'] = labels
    rct = TLCenter(target_domain='target_domain')
    # For target dataset, we don't use all the points for alignment
    if target:
        covs_calibration = []
        labels_calibration = []
        domains_calibration = []
        # Use only first session for recenter for each subject
        for subject in np.unique(meta_data.subject.values):
            # Get the meta data associated with the subject
            meta_data_sub = meta_data[meta_data['subject'] == subject]
            sessions = np.unique(meta_data_sub.session.values)
            runs = np.unique(meta_data_sub.run.values)
            # When only one session and one run, take first half
            if len(sessions) == 1 and len(runs) == 1:
                index_calibration = meta_data_sub.iloc[
                    0:len(meta_data_sub)//2
                ].index.values
            # When one session and several run, take first run
            elif len(sessions) == 1 and len(runs) != 1:
                run_calibration = np.unique(meta_data_sub.run.values)[0]
                index_calibration = meta_data_sub[
                    meta_data_sub['run'] == run_calibration
                ].index.values
            # When several sessions, take first session
            else:
                session_calibration = np.unique(
                    meta_data_sub.session.values
                )[0]
                index_calibration = meta_data_sub[
                    meta_data_sub['session'] == session_calibration
                ].index.values
            domains_calibration = domains_calibration + get_domains_one(
                meta_data_sub.loc[index_calibration],
                domain_level=domain_level)

            # Store covs and labels
            covs_calibration.append(data['covs'][index_calibration])
            labels_calibration = labels_calibration + list(
                labels[index_calibration]
            )
        covs_calibration = np.concatenate(covs_calibration)
        _, labels_enc_calibration = encode_domains(covs_calibration,
                                                   labels_calibration,
                                                   domains_calibration)
        rct.fit(covs_calibration, labels_enc_calibration, T, sample_weight)
        X_rct = np.zeros_like(covs)
        for d in np.unique(domains):
            idx = domains == d
            X_rct[idx] = rct.recenter_[str(d)].transform(covs[idx])
        data_rct['covs'] = X_rct
    # For source dataset with use all the points
    else:
        data_rct['covs'] = rct.fit_transform(covs, labels_enc,
                                             T, sample_weight)
    return data_rct


def RPA_recenter_unsupervised(source, target, meta_source, meta_target,
                              T=1, sample_weight_source=None,
                              sample_weight_target=None,
                              domain_level='subject'):

    domains_source, domains_target = get_domains(
        meta_source, meta_target, domain_level=domain_level, supervised=False)

    covs_source = source['covs']
    covs_target = target['covs']
    _, labels_source_enc = encode_domains(covs_source, source['labels'],
                                          domains_source)
    _, labels_target_enc = encode_domains(covs_target, target['labels'],
                                          domains_target)
    source_rct = {}
    target_rct = {}
    source_rct['labels'] = source['labels']
    target_rct['labels'] = target['labels']

    rct = TLCenter(target_domain='target_domain')
    source_rct['covs'] = rct.fit_transform(covs_source, labels_source_enc,
                                           T, sample_weight_source)
    target_rct['covs'] = rct.fit_transform(covs_target, labels_target_enc,
                                           T, sample_weight_target)

    return source_rct, target_rct


def RPA_stretch(source, target_train, target_test, meta_source,
                T=1, sample_weight=None, domain_level='subject'):

    domains_source, domains_target = get_domains(
        meta_source, ncovs_target_train=len(target_train['labels']),
        domain_level=domain_level, supervised=False)
    domains = domains_source + domains_target

    covs_all = np.concatenate((source['covs'], target_train['covs']))
    labels_all = np.concatenate((source['labels'], target_train['labels']))
    _, labels_enc = encode_domains(covs_all, labels_all, domains)

    source_str = {}
    target_train_str = {}
    target_test_str = {}
    source_str['labels'] = source['labels']
    target_train_str['labels'] = target_train['labels']
    target_test_str['labels'] = target_test['labels']

    str = TLStretch(target_domain='target_domain', centered_data=True)
    X_str = str.fit_transform(covs_all, labels_enc, sample_weight)
    source_str['covs'] = X_str[:len(source['labels'])]
    target_train_str['covs'] = X_str[len(source['labels']):]
    target_test_str['covs'] = str.transform(target_test['covs'])

    return source_str, target_train_str, target_test_str


def RPA_stretch_unsupervised(source, target, meta_source, meta_target,
                             T=1, sample_weight_source=None,
                             sample_weight_target=None,
                             domain_level='subject'):
    domains_source, domains_target = get_domains(
        meta_source, meta_target, domain_level=domain_level, supervised=False)

    covs_source = source['covs']
    covs_target = target['covs']
    _, labels_source_enc = encode_domains(covs_source, source['labels'],
                                          domains_source)
    _, labels_target_enc = encode_domains(covs_target, target['labels'],
                                          domains_target)
    source_str = {}
    target_str = {}
    source_str['labels'] = source['labels']
    target_str['labels'] = target['labels']

    str = TLStretch(target_domain='target_domain', centered_data=True)
    source_str['covs'] = str.fit_transform(covs_source, labels_source_enc,
                                           sample_weight_source)
    target_str['covs'] = str.fit_transform(covs_target, labels_target_enc,
                                           sample_weight_target)

    return source_str, target_str


def RPA_stretch_unsupervised_one(data, meta_data, T=1, sample_weight=None,
                                 domain_level='subject', target=False):

    domains = get_domains_one(meta_data, domain_level=domain_level)

    covs = data['covs']
    labels = data['labels']
    _, labels_enc = encode_domains(covs, labels, domains)
    data_str = {}
    data_str['labels'] = labels
    tl_str = TLStretch(target_domain='target_domain', centered_data=True)
    # For target dataset, we don't use all the points for alignment
    if target:
        covs_calibration = []
        labels_calibration = []
        domains_calibration = []
        # Use only first session for recenter for each subject
        for subject in np.unique(meta_data.subject.values):
            # Get the meta data associated with the subject
            meta_data_sub = meta_data[meta_data['subject'] == subject]
            sessions = np.unique(meta_data_sub.session.values)
            runs = np.unique(meta_data_sub.run.values)
            # When only one session and one run, take first half
            if len(sessions) == 1 and len(runs) == 1:
                index_calibration = meta_data_sub.iloc[
                    0:len(meta_data_sub)//2
                ].index.values
            # When one session and several run, take first run
            elif len(sessions) == 1 and len(runs) != 1:
                run_calibration = np.unique(meta_data_sub.run.values)[0]
                index_calibration = meta_data_sub[
                    meta_data_sub['run'] == run_calibration
                ].index.values
            # When several sessions, take first session
            else:
                session_calibration = np.unique(
                    meta_data_sub.session.values
                )[0]
                index_calibration = meta_data_sub[
                    meta_data_sub['session'] == session_calibration
                ].index.values
            domains_calibration = domains_calibration + get_domains_one(
                meta_data_sub.loc[index_calibration],
                domain_level=domain_level)

            # Store covs and labels
            covs_calibration.append(data['covs'][index_calibration])
            labels_calibration = labels_calibration + list(
                labels[index_calibration]
            )
        covs_calibration = np.concatenate(covs_calibration)
        _, labels_enc_calibration = encode_domains(covs_calibration,
                                                   labels_calibration,
                                                   domains_calibration)
        # import ipdb; ipdb.set_trace()
        tl_str.fit(covs_calibration, labels_enc_calibration, T, sample_weight)
        X_str = np.zeros_like(covs)
        for d in np.unique(domains):
            idx = domains == d
            # stretch
            X_str[idx] = tl_str._strech(
                covs[idx], tl_str.dispersions_[str(d)], 1.0
            )
        data_str['covs'] = X_str
    # For source dataset with use all the points
    else:
        data_str['covs'] = tl_str.fit_transform(covs, labels_enc,
                                                T, sample_weight)
    return data_str


def RPA_rotate(source, target_train, target_test, meta_source,
               T=1, sample_weight=None, domain_level='subject'):
    domains_source, domains_target = get_domains(
        meta_source, ncovs_target_train=len(target_train['labels']),
        domain_level=domain_level, supervised=False)
    domains = domains_source + domains_target

    covs_all = np.concatenate((source['covs'], target_train['covs']))
    labels_all = np.concatenate((source['labels'], target_train['labels']))
    _, labels_enc = encode_domains(covs_all, labels_all, domains)

    source_rot = {}
    target_train_rot = {}
    target_test_rot = {}
    source_rot['labels'] = source['labels']
    target_train_rot['labels'] = target_train['labels']
    target_test_rot['labels'] = target_test['labels']

    rot = TLRotate(target_domain='target_domain')
    X_rot = rot.fit_transform(covs_all, labels_enc, T,
                              sample_weight=sample_weight)
    source_rot['covs'] = X_rot[:len(source['labels'])]
    target_train_rot['covs'] = X_rot[len(source['labels']):]
    target_test_rot['covs'] = rot.transform(target_test['covs'])

    return source_rot, target_train_rot, target_test_rot


def get_score_calibration(clf, target_train, target_test):
    """Get the classification in calibration

    Training dataset: target_train
    Testing dataset: target_test

    Parameters
    ----------
    clf: classifier
    target_train: dict, keys: ['covs','labels']
    target_test: dict, keys: ['covs','labels']

    """

    covs_train = target_train['covs']
    y_train = target_train['labels']
    covs_test = target_test['covs']
    y_test = target_test['labels']
    Cr = mean_covariance(covs_train)
    ts_covs_train = tangent_space(covs_train, Cr)
    ts_covs_test = tangent_space(covs_test, Cr)

    clf.fit(ts_covs_train, y_train)

    y_pred = clf.predict(ts_covs_test)
    y_pred_proba = clf.predict_proba(ts_covs_test)
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    # y_test = np.array([y_test == i for i in np.unique(y_test)]).T
    # y_pred = np.array([y_pred == i for i in np.unique(y_pred)]).T

    return auc, accuracy


def get_score_transferlearning(clf, source, target_train, target_test,
                               recentered=False):
    """Get the transfer learning score

    Training dataset: target_train + source
    Testing dataset: target_test

    Parameters
    ----------
    clf: classifier
    source: dict, keys: ['covs','labels']
    target_train: dict, keys: ['covs','labels']
    target_test: dict, keys: ['covs','labels']

    """

    covs_source, y_source = source['covs'], source['labels']
    covs_target_train, y_target_train = (target_train['covs'],
                                         target_train['labels'])
    covs_target_test, y_target_test = (target_test['covs'],
                                       target_test['labels'])

    covs_train = np.concatenate([covs_source, covs_target_train])
    y_train = np.concatenate([y_source, y_target_train])
    if recentered:
        Cr = np.eye(covs_source.shape[-1])
    else:
        Cr = mean_covariance(covs_train)
    ts_covs_train = tangent_space(covs_train, Cr)
    ts_covs_test = tangent_space(covs_target_test, Cr)

    clf.fit(ts_covs_train, y_train)

    y_pred = clf.predict(ts_covs_test)
    y_pred_proba = clf.predict_proba(ts_covs_test)
    accuracy = accuracy_score(y_target_test, y_pred)
    auc = roc_auc_score(y_target_test, y_pred_proba[:, 1])
    # y_test = np.array([y_test == i for i in np.unique(y_test)]).T
    # y_pred = np.array([y_pred == i for i in np.unique(y_pred)]).T

    return auc, accuracy


def get_score_transferlearning_unsupervised(clf, source, target,
                                            recentered=False):
    """Get the transfer learning score

    Training dataset: source
    Testing dataset: target

    Parameters
    ----------
    clf: classifier
    source: dict, keys: ['covs','labels']
    target: dict, keys: ['covs','labels']

    """

    covs_source, y_source = source['covs'], source['labels']
    covs_target, y_target = (target['covs'], target['labels'])
    if recentered:
        Cr = np.eye(covs_source.shape[-1])
    else:
        Cr = mean_covariance(covs_source)
    ts_covs_source = tangent_space(covs_source, Cr)
    ts_covs_target = tangent_space(covs_target, Cr)

    clf.fit(ts_covs_source, y_source)

    y_pred = clf.predict(ts_covs_target)
    y_pred_proba = clf.predict_proba(ts_covs_target)
    accuracy = accuracy_score(y_target, y_pred)
    auc = roc_auc_score(y_target, y_pred_proba[:, 1])
    # y_test = np.array([y_test == i for i in np.unique(y_test)]).T
    # y_pred = np.array([y_pred == i for i in np.unique(y_pred)]).T

    return auc, accuracy
