import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import Ridge
from pyriemann.estimation import Covariances


def create_exp_signal(X_data, data_chnames, chnames_total):
    idx_order = [
            chnames_total.index(chi) for chi in data_chnames
        ]
    X_exp = np.zeros((X_data.shape[0],
                      len(chnames_total),
                      X_data.shape[2]))
    X_exp[:, idx_order] = X_data
    X_exp_reshape = np.zeros((X_data.shape[0]*X_data.shape[2],
                              len(chnames_total)))
    for i in range(X_exp.shape[0]):
        for j in range(X_exp.shape[-1]):
            X_exp_reshape[i*X_exp.shape[-1] + j] = X_exp[i, :, j]
    return X_exp_reshape


def recover_imp_signal(X_data, X_imp, chnames_total):
    X_data_imp = np.zeros((X_data.shape[0],
                           len(chnames_total),
                           X_data.shape[2]))
    for i in range(X_data_imp.shape[0]):
        for j in range(X_data_imp.shape[-1]):
            X_data_imp[i, :, j] = X_imp[i*X_data_imp.shape[-1] + j]
    return X_data_imp


def comimp(list_X_source, X_target, source, target, meta_target,
           imputer_type='simple', decim=1):
    chnames_total = []
    for s in source:
        chnames_total = chnames_total + s['org']['chnames']
    chnames_total = chnames_total + target['org']['chnames']
    chnames_total = list(np.unique(chnames_total))
    list_X_exp_fit = []
    list_X_exp_transform = []
    # Make expanded signals for source
    for i, X_source in enumerate(list_X_source):
        X_exp_source = create_exp_signal(X_source,
                                         source[i]['org']['chnames'],
                                         chnames_total)
        list_X_exp_fit.append(X_exp_source)
        list_X_exp_transform.append(X_exp_source)
    # Make expanded signals for target only for train
    # Use only first session for recenter for each subject
    for subject in np.unique(meta_target.subject.values):
        # Get the meta data associated with the subject
        meta_data_sub = meta_target[meta_target['subject'] == subject]
        sessions = np.unique(meta_data_sub.session.values)
        runs = np.unique(meta_data_sub.run.values)
        # When only one session and one run, take first half
        if len(sessions) == 1 and len(runs) == 1:
            index_target_train = meta_data_sub.iloc[
                0:len(meta_data_sub)//2
            ].index.values
        # When one session and several run, take first run
        elif len(sessions) == 1 and len(runs) != 1:
            run_target_train = np.unique(meta_data_sub.run.values)[0]
            index_target_train = meta_data_sub[
                meta_data_sub['run'] == run_target_train
            ].index.values
        # When several sessions, take first session
        else:
            session_target_train = np.unique(
                meta_data_sub.session.values
            )[0]
            index_target_train = meta_data_sub[
                meta_data_sub['session'] == session_target_train
            ].index.values
    X_target_train = X_target[index_target_train]
    # X_exp_target = create_exp_signal(X_target, target, chnames_total)
    X_exp_target_train = create_exp_signal(X_target_train,
                                           target['org']['chnames'],
                                           chnames_total)
    X_exp_target = create_exp_signal(X_target,
                                     target['org']['chnames'],
                                     chnames_total)
    list_X_exp_fit.append(X_exp_target_train)
    list_X_exp_transform.append(X_exp_target)
    # Concatenate everything
    X_exp_fit = np.concatenate(list_X_exp_fit)
    X_exp_transform = np.concatenate(list_X_exp_transform)
    # Compute imputation
    if imputer_type == 'simple':
        imputer = SimpleImputer(missing_values=0, strategy='mean')
    elif imputer_type == 'iterative':
        imputer = IterativeImputer(estimator=Ridge(), missing_values=0)
    imputer.fit(X_exp_fit[::decim])
    X_imp = imputer.transform(X_exp_transform)
    X_imp_del = X_imp
    for i, X_source in enumerate(list_X_source):
        n_sub = X_source.shape[0]
        n_T = X_source.shape[2]
        X_imp_s = X_imp_del[:n_sub*n_T]
        X_imp_del = np.delete(X_imp_del,
                              np.linspace(0, n_sub*n_T-1,
                                          n_sub*n_T, dtype=int),
                              0)
        X_imp_source = recover_imp_signal(X_source, X_imp_s,
                                          chnames_total)
        covs_imp_source = Covariances(estimator='lwf').fit_transform(
            X_imp_source)
        source[i]['org-comimp'] = {}
        source[i]['org-comimp']['covs'] = covs_imp_source
        source[i]['org-comimp']['labels'] = source[i]['org']['labels']
    X_imp_target = recover_imp_signal(X_target, X_imp_del, chnames_total)
    covs_imp_target = Covariances(estimator='lwf').fit_transform(
        X_imp_target)
    target['org-comimp'] = {}
    target['org-comimp']['covs'] = covs_imp_target
    target['org-comimp']['labels'] = target['org']['labels']

    return source, target
