import numpy as np


# utility functions to augment matrix dimensions
def augment_matrix_dimension(A, ind):
    '''
    - A : input matrix
    - ind : indices of the new matrix where there should be zeros
    '''

    if len(ind) > 0:

        n = A.shape[0]
        naug = n + len(ind)
        Atilde = np.eye(naug)

        ired = 0
        for iaug in range(naug):
            if iaug not in ind:
                jred = 0
                for jaug in range(naug):
                    if jaug not in ind:
                        Atilde[iaug, jaug] = A[ired, jred]
                        jred = jred + 1
                    else:
                        continue
                ired = ired + 1
            else:
                continue

    else:
        Atilde = A

    return Atilde


def reduce_matrix_dimension(A, ind):

    Atilde = np.delete(A, ind, axis=0)
    Atilde = np.delete(Atilde, ind, axis=1)

    return Atilde


def augment_dataset_dimension(A, ind):
    '''
    - A : input matrix
    - ind : indices of the new matrix where there should be zeros
    '''
    Atilde = []
    for Ai in A:
        Atilde.append(augment_matrix_dimension(Ai, ind))
    Atilde = np.stack(Atilde)
    return Atilde


def get_source_target_correspondance(source, target):

    # get the indices from the expanded matrix
    chnames_total = np.sort(
        list(set(source['chnames']).union(set(target['chnames'])))
    )
    idx = {}

    # get the indices for the electrode names on the source dataset
    source_idx_order = []
    source_idx_fill = []
    for i, chi in enumerate(chnames_total):
        if chi in source['chnames']:
            source_idx_order.append(i)
        else:
            source_idx_fill.append(i)
    idx['source_order'] = source_idx_order
    idx['source_fill'] = source_idx_fill

    # get the indices for the electrode names on the target dataset
    target_idx_order = []
    target_idx_fill = []
    for i, chi in enumerate(chnames_total):
        if chi in target['chnames']:
            target_idx_order.append(i)
        else:
            target_idx_fill.append(i)
    idx['target_order'] = target_idx_order
    idx['target_fill'] = target_idx_fill

    return idx


def match_source_target_dimensions(source_org, target_train_org,
                                   target_test_org, idx):

    # augment the dimensions for source dataset
    source_org_aug = {}
    dsource = source_org['covs'].shape[1]
    daugment = len(idx['source_fill'])
    idx2fill = np.arange(dsource+daugment)[dsource:]
    source_org_aug['covs'] = augment_dataset_dimension(source_org['covs'],
                                                       idx2fill)
    source_org_aug['labels'] = source_org['labels']

    # augment the dimensions for target train dataset
    target_train_org_aug = {}
    dtarget = target_train_org['covs'].shape[1]
    daugment = len(idx['target_fill'])
    idx2fill = np.arange(dtarget+daugment)[dtarget:]
    target_train_org_aug['covs'] = augment_dataset_dimension(
        target_train_org['covs'], idx2fill
    )
    target_train_org_aug['labels'] = target_train_org['labels']

    # augment the dimensions for target testing dataset
    target_test_org_aug = {}
    dtarget = target_test_org['covs'].shape[1]
    daugment = len(idx['target_fill'])
    idx2fill = np.arange(dtarget+daugment)[dtarget:]
    target_test_org_aug['covs'] = augment_dataset_dimension(
        target_test_org['covs'], idx2fill
    )
    target_test_org_aug['labels'] = target_test_org['labels']

    # match the channel orderings for source
    source_org_reo = {}
    idx2order = idx['source_order'] + idx['source_fill']
    # idx2order = np.argsort(idx2order)
    source_org_reo['covs'] = source_org_aug['covs'][
        :, idx2order, :][:, :, idx2order]
    source_org_reo['labels'] = source_org_aug['labels']

    # match the channel orderings for target-train
    target_train_org_reo = {}
    idx2order = idx['target_order'] + idx['target_fill']
    # idx2order = np.argsort(idx2order)
    target_train_org_reo['covs'] = target_train_org_aug['covs'][
        :, idx2order, :][:, :, idx2order]
    target_train_org_reo['labels'] = target_train_org_aug['labels']

    # match the channel orderings for target-test
    target_test_org_reo = {}
    idx2order = idx['target_order'] + idx['target_fill']
    # idx2order = np.argsort(idx2order)
    target_test_org_reo['covs'] = target_test_org_aug['covs'][
        :, idx2order, :][:, :, idx2order]
    target_test_org_reo['labels'] = target_test_org_aug['labels']
    # import ipdb; ipdb.set_trace()
    return source_org_reo, target_train_org_reo, target_test_org_reo


def match_source_target_dimensions_unsupervised(source_org, target_org, idx):

    # augment the dimensions for source dataset
    source_org_aug = {}
    dsource = source_org['covs'].shape[1]
    daugment = len(idx['source_fill'])
    idx2fill = np.arange(dsource+daugment)[dsource:]
    source_org_aug['covs'] = augment_dataset_dimension(source_org['covs'],
                                                       idx2fill)
    source_org_aug['labels'] = source_org['labels']

    # augment the dimensions for target dataset
    target_org_aug = {}
    dtarget = target_org['covs'].shape[1]
    daugment = len(idx['target_fill'])
    idx2fill = np.arange(dtarget+daugment)[dtarget:]
    target_org_aug['covs'] = augment_dataset_dimension(
        target_org['covs'], idx2fill
    )
    target_org_aug['labels'] = target_org['labels']

    # match the channel orderings for source
    source_org_reo = {}
    idx2order = idx['source_order'] + idx['source_fill']
    # idx2order = np.argsort(idx2order)
    source_org_reo['covs'] = source_org_aug['covs'][
        :, idx2order, :][:, :, idx2order]
    source_org_reo['labels'] = source_org_aug['labels']

    # match the channel orderings for target
    target_org_reo = {}
    idx2order = idx['target_order'] + idx['target_fill']
    # idx2order = np.argsort(idx2order)
    target_org_reo['covs'] = target_org_aug['covs'][
        :, idx2order, :][:, :, idx2order]
    target_org_reo['labels'] = target_org_aug['labels']

    return source_org_reo, target_org_reo


def get_source_target_correspondance_multi(source, target):
    # get the indices from the expanded matrix
    chnames_total = []
    for s in source:
        chnames_total = chnames_total + s['org']['chnames']
    chnames_total = chnames_total + target['org']['chnames']
    chnames_total = np.unique(chnames_total)
    # get the indices for the electrode names on the source dataset
    for s in source:
        source_idx_order = []
        source_idx_fill = []
        # for i, chi in enumerate(chnames_total):
        #     if chi in s['org']['chnames']:
        #         source_idx_order.append(i)
        #     else:
        #         source_idx_fill.append(i)
        for chi in s['org']['chnames']:
            source_idx_order.append(
                list(chnames_total).index(chi))
        for chi in chnames_total:
            if chi not in s['org']['chnames']:
                source_idx_fill.append(list(chnames_total).index(chi))

        s['org']['ch_order'] = source_idx_order
        s['org']['ch_fill'] = source_idx_fill

    # get the indices for the electrode names on the target dataset
    target_idx_order = []
    target_idx_fill = []
    # for i, chi in enumerate(chnames_total):
    #     if chi in target['org']['chnames']:
    #         target_idx_order.append(i)
    #     else:
    #         target_idx_fill.append(i)
    for chi in target['org']['chnames']:
        target_idx_order.append(
            list(chnames_total).index(chi))
    for chi in chnames_total:
        if chi not in target['org']['chnames']:
            target_idx_fill.append(list(chnames_total).index(chi))
    target['org']['ch_order'] = target_idx_order
    target['org']['ch_fill'] = target_idx_fill

    return source, target


def match_dimensions_unsupervised_one(data):
    # augment the dimensions
    data_org_aug = {}
    d_data = data['org']['covs'].shape[1]
    d_augment = len(data['org']['ch_fill'])
    idx2fill = np.arange(d_data+d_augment)[d_data:]
    data_org_aug['covs'] = augment_dataset_dimension(
        data['org']['covs'], idx2fill
    )
    data_org_aug['labels'] = data['org']['labels']

    # match the channel orderings
    data['org-aug'] = {}
    idx2order = np.argsort(data['org']['ch_order'] + data['org']['ch_fill'])
    data['org-aug']['covs'] = data_org_aug['covs'][
        :, idx2order, :][:, :, idx2order]
    data['org-aug']['labels'] = data_org_aug['labels']
    return data
