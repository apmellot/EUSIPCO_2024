# %%
from pathlib import Path
import numpy as np
import moabb.datasets as md
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from statannotations.Annotator import Annotator
from matplotlib.patches import Patch

plt.close('all')
# %%
datasets = [md.Shin2017A(accept=True),
            md.BNCI2014_001(),
            md.PhysionetMI()]

setup = 'learning_curve'
scoring = 'accuracy'
domain_level = 'subject'
classif = 'right_left_hand'
mean = 'classic-mean'

RESULTS_FOLDER = Path(
    './results/'
)
all_results = pd.DataFrame()
for i in range(len(datasets)):
    ds_name = datasets[i].__class__.__name__
    all_results_ = pd.read_csv(
        RESULTS_FOLDER / f'{setup}_{domain_level}_{ds_name}',
        index_col=0
    )
    if all_results.empty:
        all_results = all_results_
    else:
        all_results = pd.concat([all_results, all_results_])

all_results.reset_index(inplace=True, drop=True)
FIGURES_FOLDER = Path(f'./figures')
FIGURES_FOLDER.mkdir(parents=True, exist_ok=True)

# %% Remove dummy et re scale

all_results = all_results.drop(
    all_results[all_results['method'] == 'dummy'].index
)

all_results = all_results.drop(
    all_results[all_results['method'] == 'str-aug'].index
)
all_results = all_results.drop(
    all_results[all_results['method'] == 'str-comimp'].index
)
all_results = all_results.drop(
    all_results[all_results['method'] == 'str-bas'].index
)
all_results = all_results.drop(
    all_results[all_results['method'] == 'str-ssi'].index
)
all_results = all_results.drop(
    all_results[all_results['method'] == 'str-mne'].index
)
# %% Remove org if needed
all_results = all_results.drop(
    all_results[all_results['method'] == 'org-aug'].index
)
all_results = all_results.drop(
    all_results[all_results['method'] == 'org-comimp'].index
)
all_results = all_results.drop(
    all_results[all_results['method'] == 'org-bas'].index
)
all_results = all_results.drop(
    all_results[all_results['method'] == 'org-ssi'].index
)
all_results = all_results.drop(
    all_results[all_results['method'] == 'org-mne'].index
)
# %% Add within domain training

within_domain_results = pd.read_csv(
    f'./results/unsupervised_within-domains_{classif}',
    index_col=0
)
within_domain_results.reset_index(inplace=True, drop=True)
within_domain_results = within_domain_results.iloc[0::2]
within_domain_results = within_domain_results.rename(
    columns={"subject": "subject_target", "dataset": "dataset_target"}
)
within_domain_results.reset_index(inplace=True, drop=True)

# %% Rename methods
all_results[['Alignment methods',
             'Dimension matching methods']] = all_results.method.str.split(
                 '-', expand=True)
all_results = all_results.replace(['org',
                                   'rct',
                                   'bas',
                                   'aug',
                                   'comimp',
                                   'ssi',
                                   'mne'
                                   ],
                                  ['No alignment',
                                   'Re-center',
                                   'Common channels',
                                   'Dimensionality Transcending',
                                   'ComImp',
                                   'Spherical spline interpolation',
                                   'Field interpolation'])

# %%
types = ['Supervised', 'Source\ndependent', 'Source free']
colors = [{'Calibration': '#984ea3'},
          {'ComImp': '#377eb8',
           'Dimensionality Transcending': '#ff7f00'},
          {'Common channels': '#4daf4a',
           'Spherical spline interpolation': '#f781bf',
           'Field interpolation': '#a65628'}]
colors_lineplot = {'ComImp': '#377eb8',
                   'Dimensionality Transcending': '#ff7f00',
                   'Common channels': '#4daf4a',
                   'Spherical spline interpolation': '#f781bf',
                   'Field interpolation': '#a65628'}
# %% All points
datasets = [md.Shin2017A(accept=True),
            md.BNCI2014_004(),
            md.Zhou2016(),
            md.BNCI2014_001(),
            md.Weibo2014(),
            md.PhysionetMI()
            ]
X_ticks_labels = [['2\nS', '4\nS+B4',
                   '10\nS+B4+Z', '22\nS+B4+\nZ+W',
                   '22\nS+B4+Z+\nW+P'],
                  ['12\nS', '14\nS+B4',
                   '25\nS+B4+Z', '37\nS+B4+\nZ+B1',
                   '58\nS+B4+Z+\nB1+W'],
                  ['1\nB4', '1\nB4+Z',
                   '2\nB4+Z+B1', '12\nB4+Z+\nB1+W',
                   '12\nB4+Z+B1+\nW+P']]
datasets = np.unique(all_results.dataset_target.values)

sns.set_theme(style="whitegrid", font_scale=2)
sns.set_palette('colorblind')
fig, axes = plt.subplots(6, 3, figsize=(30, 17),
                         gridspec_kw={'height_ratios': [
                             7, 2.7, 1, 2, 3, 1
                         ]})

for i, dataset in enumerate(datasets):
    # Lineplots on first row
    results_0 = all_results[all_results['dataset_target'] == dataset]
    dt_results = results_0[results_0['method'] == 'rct-mne']
    dt_results_ext = []
    for n in range(1, 6):
        dt_results_ext.append(
            pd.concat(
                [dt_results[dt_results['n_dataset_train'] == n]]*5,
                ignore_index=True))
    dt_results_ext = pd.concat(dt_results_ext)
    results_0['acc_diff'] = results_0.accuracy.values - dt_results_ext.accuracy.values
    sns.lineplot(data=results_0,
                 x='n_dataset_train',
                 y='acc_diff',
                 hue='Dimension matching methods',
                 linewidth=6,
                 palette=colors_lineplot,
                 ax=axes[0, i])
    axes[0, i].set_xlabel('Target channels seen in train')
    axes[0, i].set_ylabel(None)
    axes[0, i].set_xticks([1, 2, 3, 4, 5],
                          labels=X_ticks_labels[i])
    axes[0, i].get_legend().remove()
    axes[0, i].set_title(f'Dataset target: {dataset}',
                         y=1.05, fontsize=30)
    # Boxplots on second row
    results_1 = results_0[results_0['n_dataset_train'] == 5]
    within_domain_results_ = within_domain_results[
        within_domain_results['dataset_target'] == dataset
    ]
    within_domain_results_['Alignment methods'] = ['No alignment']*len(
        within_domain_results_
    )
    within_domain_results_['Dimension matching methods'] = ['Calibration']*len(
        within_domain_results_
    )
    within_domain_results_['method'] = ['org']*len(
        within_domain_results_
    )
    results_1 = pd.concat([results_1, within_domain_results_])
    results_1.reset_index(inplace=True, drop=True)
    # Reformate dataframe
    results_1 = results_1.replace(
        ['Common channels',
         'Dimensionality Transcending',
         'ComImp',
         'Spherical spline interpolation',
         'Field interpolation',
         'Calibration'],
        ['Source free_Common channels',
         'Source\ndependent_Dimensionality Transcending',
         'Source\ndependent_ComImp',
         'Source free_Spherical spline interpolation',
         'Source free_Field interpolation',
         'Supervised_Calibration'])

    results_1[['Type',
               'Dimension matching methods']] = results_1[
                   'Dimension matching methods'].str.split(
                   '_', expand=True)
    fig.delaxes(axes[5, i])
    for row in range(1, 5):
        if row == 1:
            fig.delaxes(axes[row, i])
            continue
        type = types[row-2]
        boxcolors = colors[row-2]
        results_ = results_1[results_1['Type'] == type]
        sns.boxplot(data=results_,
                    x=scoring,
                    y='Dimension matching methods',
                    showmeans=True,
                    meanprops={"marker": "o",
                               "markerfacecolor": "white",
                               "markeredgecolor": "black",
                               "markersize": "10"},
                    ax=axes[row, i],
                    palette=boxcolors
                    )
        if row == 4:
            annot = Annotator(
                ax=axes[row, i],
                pairs=[
                    (('Spherical spline interpolation'),
                     ('Field interpolation'))],
                data=results_,
                x=scoring,
                y='Dimension matching methods',
                orient='h'
            )
            annot.configure(test='Wilcoxon', text_format='star', loc='inside')
            annot.apply_and_annotate()
        sns.stripplot(data=results_,
                      x=scoring,
                      y='Dimension matching methods',
                      palette=boxcolors,
                      linewidth=1,
                      ax=axes[row, i])
        axes[row, i].axvline(x=0.5, color='k')
        axes[row, i].set_ylabel(None)
        if row == 4:
            axes[row, i].set_xticks(
                [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                labels=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
            axes[row, i].set_xlabel('Accuracy')
        else:
            axes[row, i].set_xticklabels([])
            axes[row, i].set_xlabel(None)
        axes[row, i].set_xlim(0.3, 1)
        if i != 0:
            axes[row, i].set_yticklabels([])
        else:
            mean_tick = np.mean(axes[row, i].get_yticks())
            axes[row, i].set_yticks([mean_tick], labels=[type])
axes[0, 0].set_ylabel('Accuracy difference')
axes[0, 0].annotate(text='A', xy=(-0.2, 1), xycoords=('axes fraction'),
                    fontsize=40, weight='bold')
axes[2, 0].annotate(text='B', xy=(-0.2, 1.05), xycoords=('axes fraction'),
                    fontsize=40, weight='bold')
plt.subplots_adjust(hspace=0, wspace=0.15)

# Create a global legend
legend_elements = [
    Patch(facecolor='#984ea3', label='Calibration'),
    Patch(facecolor='#377eb8', label='ComImp'),
    Patch(facecolor='#ff7f00', label='Dimensionality Transcending'),
    Patch(facecolor='#4daf4a', label='Common channels'),
    Patch(facecolor='#f781bf', label='Spherical spline interpolation'),
    Patch(facecolor='#a65628', label='Field interpolation')]

fig.legend(handles=legend_elements, title='Dimension matching methods',
           loc='lower center', bbox_to_anchor=(0.52, 0),
           ncol=3)
# plt.tight_layout()
# %%
fig.savefig(
    FIGURES_FOLDER / 'unsupervised_learning-curve_multi-dataset_subject.pdf'
)
# %% Seperate both plots and make all 6 datasets

datasets = [md.Shin2017A(accept=True),
            md.BNCI2014_004(),
            md.Zhou2016(),
            md.BNCI2014_001(),
            md.Weibo2014(),
            md.PhysionetMI()]

setup = 'learning_curve'
scoring = 'accuracy'
domain_level = 'subject'
classif = 'right_left_hand'
mean = 'classic-mean'

RESULTS_FOLDER = Path(
    './results/'
)
all_results = pd.DataFrame()
for i in range(len(datasets)):
    ds_name = datasets[i].__class__.__name__
    all_results_ = pd.read_csv(
        RESULTS_FOLDER / f'{setup}_{domain_level}_{ds_name}',
        index_col=0
    )
    if all_results.empty:
        all_results = all_results_
    else:
        all_results = pd.concat([all_results, all_results_])

all_results.reset_index(inplace=True, drop=True)

# %% Remove dummy et re scale

all_results = all_results.drop(
    all_results[all_results['method'] == 'dummy'].index
)

all_results = all_results.drop(
    all_results[all_results['method'] == 'str-aug'].index
)
all_results = all_results.drop(
    all_results[all_results['method'] == 'str-comimp'].index
)
all_results = all_results.drop(
    all_results[all_results['method'] == 'str-bas'].index
)
all_results = all_results.drop(
    all_results[all_results['method'] == 'str-ssi'].index
)
all_results = all_results.drop(
    all_results[all_results['method'] == 'str-mne'].index
)
#  Remove org if needed
all_results = all_results.drop(
    all_results[all_results['method'] == 'org-aug'].index
)
all_results = all_results.drop(
    all_results[all_results['method'] == 'org-comimp'].index
)
all_results = all_results.drop(
    all_results[all_results['method'] == 'org-bas'].index
)
all_results = all_results.drop(
    all_results[all_results['method'] == 'org-ssi'].index
)
all_results = all_results.drop(
    all_results[all_results['method'] == 'org-mne'].index
)

# Style
all_results[['Alignment methods',
             'Dimension matching methods']] = all_results.method.str.split(
                 '-', expand=True)
all_results = all_results.replace(['org',
                                   'rct',
                                   'bas',
                                   'aug',
                                   'comimp',
                                   'ssi',
                                   'mne'
                                   ],
                                  ['No alignment',
                                   'Re-center',
                                   'Common channels',
                                   'Dimensionality Transcending',
                                   'ComImp',
                                   'Spherical spline interpolation',
                                   'Field interpolation'])

types = ['Supervised', 'Source\ndependent', 'Source free']
colors = [{'Calibration': '#984ea3'},
          {'ComImp': '#377eb8',
           'Dimensionality Transcending': '#ff7f00'},
          {'Common channels': '#4daf4a',
           'Spherical spline interpolation': '#f781bf',
           'Field interpolation': '#a65628'}]
colors_lineplot = {'ComImp': '#377eb8',
                   'Dimensionality Transcending': '#ff7f00',
                   'Common channels': '#4daf4a',
                   'Spherical spline interpolation': '#f781bf',
                   'Field interpolation': '#a65628'}

# %% Only learning curves

X_ticks_labels = [
    ['1\nB4', '1\nB4+Z', '2\nB4+Z+B1', '12\nB4+Z+\nB1+W', '12\nB4+Z+B1\n+W+P'],
    ['1\nS', '3\nS+Z', '3\nS+Z+B1', '3\nS+Z+\nB1+W', '3\nS+Z+B1\n+W+P'],
    ['1\nS', '3\nS+B4', '9\nS+B4+B1', '14\nS+B4+\nB1+W', '14\nS+B4+B1\n+W+P'],
    ['2\nS', '4\nS+B4', '10\nS+B4+Z', '22\nS+B4+\nZ+W', '22\nS+B4+Z\n+W+P'],
    ['12\nS', '14\nS+B4', '25\nS+B4+Z', '37\nS+B4+\nZ+B1', '58\nS+B4+Z\n+B1+P'],
    ['12\nS', '14\nS+B4', '25\nS+B4+Z', '37\nS+B4+\nZ+B1', '58\nS+B4+Z\n+B1+W']
                  
]

datasets = ['Shin2017A', 'BNCI2014_004', 'Zhou2016',
            'BNCI2014_001', 'Weibo2014', 'PhysionetMI']

sns.set_theme(style="whitegrid", font_scale=2.4)
sns.set_palette('colorblind')
fig, axes = plt.subplots(4, 2, figsize=(20, 26),
                         gridspec_kw={'height_ratios': [
                             7, 7, 7, 1
                         ]})
fig.subplots_adjust(hspace=0.6, wspace=0.2)

for i, dataset in enumerate(datasets):
    row = i % 3
    col = i // 3
    results_0 = all_results[all_results['dataset_target'] == dataset]
    dt_results = results_0[results_0['method'] == 'rct-mne']
    dt_results_ext = []
    for n in range(1, 6):
        dt_results_ext.append(
            pd.concat(
                [dt_results[dt_results['n_dataset_train'] == n]]*5,
                ignore_index=True))
    dt_results_ext = pd.concat(dt_results_ext)
    results_0['acc_diff'] = (
        results_0.accuracy.values - dt_results_ext.accuracy.values)
    sns.lineplot(data=results_0,
                x='n_dataset_train',
                y='acc_diff',
                hue='Dimension matching methods',
                linewidth=6,
                palette=colors_lineplot,
                ax=axes[row, col])
    if row == 2:
        axes[row, col].set_xlabel(
            'Target channels seen in train',
            fontsize=28)
    else:
        axes[row, col].set_xlabel(None)
    axes[row, col].set_ylabel(None)
    axes[row, col].set_xticks([1, 2, 3, 4, 5],
                            labels=X_ticks_labels[i],
                            fontsize=24)
    axes[row, col].get_legend().remove()
    axes[row, col].set_title(f'Dataset target: {dataset}',
                            y=1.05, fontsize=34)
axes[0, 0].set_ylabel('Accuracy difference', fontsize=28)
axes[1, 0].set_ylabel('Accuracy difference', fontsize=28)
axes[2, 0].set_ylabel('Accuracy difference', fontsize=28)
fig.delaxes(axes[3, 0])
fig.delaxes(axes[3, 1])

# Create a global legend
legend_elements = [
    Patch(facecolor='#984ea3', label='Calibration'),
    Patch(facecolor='#377eb8', label='ComImp'),
    Patch(facecolor='#ff7f00', label='Dimensionality Transcending'),
    Patch(facecolor='#4daf4a', label='Common channels'),
    Patch(facecolor='#f781bf', label='Spherical spline interpolation'),
    Patch(facecolor='#a65628', label='Field interpolation')]

fig.legend(handles=legend_elements, title='Dimension matching methods',
           loc='lower center', bbox_to_anchor=(0.5, 0.05),
           ncol=3)
# %%
fig.savefig(
    FIGURES_FOLDER / 'learning_curves.pdf'
)
# %%Only box plots

sns.set_theme(style="whitegrid", font_scale=2.4)
sns.set_palette('colorblind')
fig, axes = plt.subplots(12, 2, figsize=(20, 26),
                         gridspec_kw={'height_ratios': [
                             1, 2, 3, 1.3, 1, 2, 3, 1.3, 1, 2, 3, 1
                         ]})

for i, dataset in enumerate(datasets):
    row_ds = i % 3
    col = i // 3

    results_0 = all_results[all_results['dataset_target'] == dataset]
    axes[row_ds*4, col].set_title(f'Dataset target: {dataset}',
                         y=1.05, fontsize=34)
    # Boxplots on second row
    results_1 = results_0[results_0['n_dataset_train'] == 5]
    within_domain_results_ = within_domain_results[
        within_domain_results['dataset_target'] == dataset
    ]
    within_domain_results_['Alignment methods'] = ['No alignment']*len(
        within_domain_results_
    )
    within_domain_results_['Dimension matching methods'] = ['Calibration']*len(
        within_domain_results_
    )
    within_domain_results_['method'] = ['org']*len(
        within_domain_results_
    )
    results_1 = pd.concat([results_1, within_domain_results_])
    results_1.reset_index(inplace=True, drop=True)
    # Reformate dataframe
    results_1 = results_1.replace(
        ['Common channels',
         'Dimensionality Transcending',
         'ComImp',
         'Spherical spline interpolation',
         'Field interpolation',
         'Calibration'],
        ['Source free_Common channels',
         'Source\ndependent_Dimensionality Transcending',
         'Source\ndependent_ComImp',
         'Source free_Spherical spline interpolation',
         'Source free_Field interpolation',
         'Supervised_Calibration'])

    results_1[['Type',
               'Dimension matching methods']] = results_1[
                   'Dimension matching methods'].str.split(
                   '_', expand=True)
    for row_box in range(3):
        row_plot = row_ds*4 + row_box
        type = types[row_box]
        boxcolors = colors[row_box]
        results_ = results_1[results_1['Type'] == type]
        sns.boxplot(data=results_,
                    x=scoring,
                    y='Dimension matching methods',
                    showmeans=True,
                    meanprops={"marker": "o",
                                "markerfacecolor": "white",
                                "markeredgecolor": "black",
                                "markersize": "10"},
                    ax=axes[row_plot, col],
                    palette=boxcolors
                    )
        if row_box == 2:
            annot = Annotator(
                ax=axes[row_plot, col],
                pairs=[
                    (('Spherical spline interpolation'),
                        ('Field interpolation'))],
                data=results_,
                x=scoring,
                y='Dimension matching methods',
                orient='h'
            )
            annot.configure(test='Wilcoxon', text_format='star', loc='inside')
            annot.apply_and_annotate()
        sns.stripplot(data=results_,
                        x=scoring,
                        y='Dimension matching methods',
                        palette=boxcolors,
                        linewidth=1,
                        ax=axes[row_plot, col])
        axes[row_plot, col].axvline(x=0.5, color='k')
        axes[row_plot, col].set_xlabel(None)
        axes[row_plot, col].set_ylabel(None)
        if row_box == 2:
            axes[row_plot, col].set_xticks(
                [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                labels=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        else:
            axes[row_plot, col].set_xticklabels([])
            axes[row_plot, col].set_xlabel(None)
        axes[row_plot, col].set_xlim(0.3, 1)
        if col != 0:
            axes[row_plot, col].set_yticklabels([])
        else:
            mean_tick = np.mean(axes[row_plot, col].get_yticks())
            axes[row_plot, col].set_yticks([mean_tick], labels=[type])

for i in range (2):
    fig.delaxes(axes[3, i])
    fig.delaxes(axes[7, i])
    fig.delaxes(axes[11, i])
    axes[10, i].set_xlabel('Accuracy')
    
plt.subplots_adjust(hspace=0, wspace=0.15)

# Create a global legend
legend_elements = [
    Patch(facecolor='#984ea3', label='Calibration'),
    Patch(facecolor='#377eb8', label='ComImp'),
    Patch(facecolor='#ff7f00', label='Dimensionality Transcending'),
    Patch(facecolor='#4daf4a', label='Common channels'),
    Patch(facecolor='#f781bf', label='Spherical spline interpolation'),
    Patch(facecolor='#a65628', label='Field interpolation')]

fig.legend(handles=legend_elements, title='Dimension matching methods',
           loc='lower center', bbox_to_anchor=(0.5, 0.02),
           ncol=3)
# %%
fig.savefig(
    FIGURES_FOLDER / 'boxplots.pdf'
)
# %%
