import argparse
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import read_latest_run_id, resolve_run_dir


FILEPATH = os.path.dirname(os.path.abspath(__file__))
RESULT_COLUMNS = ['alg', 'gen', 'met', 'tra', 'tes', 'nle', 'tme', 'fid', 'bac']
RESULT_NUMERIC_COLUMNS = ['tra', 'tes', 'ora', 'bac', 'orb', 'nle', 'orn', 'tme', 'fid', 'orf', 'itr', 'npt']
MODEL_GROUPS = {
    'dt': ('dt', 'dtc', 'dtval'),
    'dtp': ('dtp', 'dtcp', 'dtvalp'),
    'dtb': ('dtb', 'dtcb', 'dtvalb'),
    'rules': ('ripper', 'irep'),
    'sd': ('primcv', 'bicv'),
}


@dataclass(frozen = True)
class RunPaths:
    run_dir: str
    raw_dir: str
    derived_dir: str
    figures_dir: str
    res_path: str
    pivot_wil_path: str
    pivot_median_path: str


def parse_args():
    parser = argparse.ArgumentParser(description = 'Post-process one PRELIM experiment run.')
    parser.add_argument('--run-id', default = None, help = 'Run id under experiments/registry/runs/. Defaults to the latest run.')
    parser.add_argument('--run-dir', default = None, help = 'Absolute or relative path to a run directory.')
    parser.add_argument('--skip-figures', action = 'store_true', help = 'Compute derived tables without rendering heatmaps.')
    return parser.parse_args()


def resolve_selected_run(args):
    if args.run_dir is not None:
        return os.path.abspath(args.run_dir)
    run_id = args.run_id or read_latest_run_id()
    if run_id is None:
        raise FileNotFoundError('No run selected and no latest run recorded.')
    return resolve_run_dir(run_id)


def build_run_paths(run_dir):
    derived_dir = os.path.join(run_dir, 'derived')
    figures_dir = os.path.join(run_dir, 'figures')
    os.makedirs(derived_dir, exist_ok = True)
    os.makedirs(figures_dir, exist_ok = True)
    return RunPaths(
        run_dir = run_dir,
        raw_dir = os.path.join(run_dir, 'raw'),
        derived_dir = derived_dir,
        figures_dir = figures_dir,
        res_path = os.path.join(derived_dir, 'res.csv'),
        pivot_wil_path = os.path.join(derived_dir, 'pivot_wil.csv'),
        pivot_median_path = os.path.join(derived_dir, 'pivot_median.csv'),
    )


def qual_change(data, meta):
    data = data.copy()
    baseline_for_p = data[data['alg'].isin(['dt', 'dtc', 'dtval']) & (data['gen'] == 'na')].copy()
    baseline_for_p['alg'] = baseline_for_p['alg'] + 'p'
    data = pd.concat([data, baseline_for_p], ignore_index = True)

    data['fid'] = pd.to_numeric(data['fid'], errors = 'coerce')
    data['bac'] = pd.to_numeric(data['bac'], errors = 'coerce')

    precdefault = meta.loc[meta['alg'] == 'testprec', 'val'].iloc[0]
    score_mask = ~data['alg'].isin(['primcv', 'bicv'])
    data.loc[score_mask, 'tes'] = data.loc[score_mask, 'tes'] - precdefault
    data.loc[score_mask, 'bac'] = data.loc[score_mask, 'bac'] - 0.5

    for alg in data['alg'].unique():
        baseline_rows = data[(data['alg'] == alg) & (data['gen'] == 'na')]
        if baseline_rows.empty:
            continue
        data = data[~((data['alg'] == alg) & (data['gen'] == 'na'))]
        baseline = baseline_rows.iloc[0]
        data.loc[data['alg'] == alg, 'ora'] = baseline['tes']
        data.loc[data['alg'] == alg, 'orn'] = baseline['nle']
        data.loc[data['alg'] == alg, 'orb'] = baseline['bac']

    for alg, met in data[['alg', 'met']].drop_duplicates().itertuples(index = False):
        if alg in ['primcv', 'bicv']:
            data.loc[(data['alg'] == alg) & (data['met'] == met), 'orf'] = np.nan
            continue
        def_fid = meta.loc[meta['alg'] == met + 'fid', 'val'].iloc[0]
        baseline_alg = alg[:-1] if alg in ['dtp', 'dtcp', 'dtvalp'] else alg
        orig_fid = meta.loc[meta['alg'] == met + baseline_alg + 'fid', 'val'].iloc[0] - def_fid
        mask = (data['alg'] == alg) & (data['met'] == met)
        data.loc[mask, 'orf'] = orig_fid
        data.loc[mask, 'fid'] = data.loc[mask, 'fid'] - def_fid

    return data.reset_index(drop = True)


def load_shard_result(paths, raw_filename):
    meta = pd.read_csv(
        os.path.join(paths.raw_dir, raw_filename.split('.')[0] + '_meta.csv'),
        delimiter = ',',
        header = None,
    )
    meta.columns = ['alg', 'val']

    data = pd.read_csv(os.path.join(paths.raw_dir, raw_filename), delimiter = ',', header = None)
    data.columns = RESULT_COLUMNS
    data = qual_change(data, meta)

    dat, itr, npt = raw_filename.split('.')[0].split('_')
    data['dat'] = dat
    data['itr'] = itr
    data['npt'] = npt
    return data, meta.assign(dat = dat, itr = itr, npt = npt)


def load_run_results(paths):
    filenames = sorted(next(os.walk(paths.raw_dir))[2])
    result_frames = []
    meta_frames = []
    raw_filenames = []

    for filename in filenames:
        if 'meta' in filename or 'zeros' in filename:
            continue
        result_frame, meta_frame = load_shard_result(paths, filename)
        result_frames.append(result_frame)
        meta_frames.append(meta_frame)
        raw_filenames.append(filename)

    if not result_frames:
        raise FileNotFoundError('No raw experiment shard CSV files found in %s' % paths.raw_dir)

    results = pd.concat(result_frames, ignore_index = True)
    results.loc[results['gen'] == 'adasyns', 'gen'] = 'adasyn'
    for name in RESULT_NUMERIC_COLUMNS:
        results[name] = pd.to_numeric(results[name])
    return results, meta_frames, raw_filenames


def change_names(data):
    return data.replace({
        'dtval': 'DT*',
        'dtc': 'DT$^{int}$',
        'dt': 'DT',
        'dtvalp': 'DT*p',
        'dtcp': 'DT$^{int}$p',
        'dtp': 'DTp',
        'dtvalb': 'DT*',
        'dtcb': 'DT$^{int}$',
        'dtb': 'DT',
        'adasyn': 'ADASYN',
        'cmmrf': 'CMM',
        'dummy': 'DUMMY',
        'gmm': 'GMM',
        'gmmal': 'GMMAL',
        'kdebw': 'KDE',
        'kdeb': 'KDEB',
        'kdebwm': 'KDEM',
        'munge': 'MUNGE',
        'randn': 'NORM',
        'randu': 'UNIF',
        'rerx': 'RE-RX',
        'smote': 'SMOTE',
        'ssl': 'SSL',
        'vva': 'VVA',
        'rf': 'RF',
        'xgb': 'BT',
        'rfb': 'RF',
        'xgbb': 'BT',
        'bicv': 'BI',
        'primcv': 'PRIM',
        'irep': 'IREP',
        'ripper': 'RIPPER',
    })


def res_aggregate(results, mod, npts, clname, clnameo):
    if mod not in MODEL_GROUPS:
        raise ValueError('{mod} is a wrong mod value'.format(mod = repr(mod)))
    aggregated = results.copy()
    aggregated = aggregated[aggregated['npt'].isin([npts])]
    aggregated = aggregated[aggregated['alg'].isin(MODEL_GROUPS[mod])]
    aggregated = aggregated[['alg', 'gen', 'met', 'npt', clname, clnameo]].groupby(
        ['alg', 'gen', 'met', 'npt'],
        as_index = False,
    ).mean()
    return aggregated


def separate_baseline(aggregated, clname, clnameo):
    baseline = aggregated.drop(columns = [clname]).copy()
    baseline['gen'] = ' NO'
    baseline = baseline.rename(columns = {clnameo: clname})
    baseline = baseline.groupby(['alg', 'gen', 'met', 'npt'], as_index = False).max()
    return baseline


def load_plotter(plotter = None):
    if plotter is not None:
        return plotter
    import seaborn as sns
    return sns


def my_diverging_palette(plotter, r_neg, r_pos, g_neg, g_pos, b_neg, b_pos, sep = 1, n = 6, center = 'light', as_cmap = False):
    palfunc = dict(dark = plotter.dark_palette, light = plotter.light_palette)[center]
    n_half = int(128 - (sep // 2))
    neg = palfunc((r_neg / 255, g_neg / 255, b_neg / 255), n_half, reverse = True, input = 'rgb')
    pos = palfunc((r_pos / 255, g_pos / 255, b_pos / 255), n_half, input = 'rgb')
    midpoint = dict(light = [(.95, .95, .95)], dark = [(.133, .133, .133)])[center]
    mid = midpoint * sep
    return plotter.blend_palette(np.concatenate([neg, mid, pos]), n, as_cmap = as_cmap)


def draw_heatmap(results, figures_dir, npts, clname, clnameo, plotter = None, mlt = 100, pal = 'normal', mod = 'dt', ylbl = True, ytext = '', fsz = 13):
    plotter = load_plotter(plotter)

    def draw_heatmap_c(*args, **kwargs):
        data = kwargs.pop('data')
        center = data[data['gen'] == ' NO'][args[2]].iloc[0]
        pivot = data.pivot(index = args[1], columns = args[0], values = args[2])
        plotter.heatmap(pivot, center = center, **kwargs)

    aggregated = res_aggregate(results, mod, npts, clname, clnameo)
    aggregated = pd.concat([separate_baseline(aggregated, clname, clnameo), aggregated.drop(columns = [clnameo])], ignore_index = True)
    aggregated = change_names(aggregated)
    aggregated[clname] = np.round(aggregated[clname] * mlt, 1)
    if clname == 'nle':
        aggregated.loc[aggregated['alg'] == 'DT', clname] = np.round(aggregated.loc[aggregated['alg'] == 'DT', clname], 0)

    if pal == 'inverse':
        cmap = my_diverging_palette(plotter, 0, 255, 91, 213, 183, 0, sep = 3, as_cmap = True)
    else:
        cmap = my_diverging_palette(plotter, 255, 0, 213, 91, 0, 183, sep = 3, as_cmap = True)

    aspect = 0.42 / 1.2 if ylbl else 0.33 / 1.2
    grid = plotter.FacetGrid(aggregated, row = 'npt', col = 'alg', margin_titles = False, despine = False, height = 4.2, aspect = aspect)
    grid.map_dataframe(draw_heatmap_c, 'met', 'gen', clname, cbar = False, cmap = cmap, annot = True, fmt = 'g')
    if not ylbl:
        grid.set(yticklabels = [])
    grid.set_axis_labels('', ytext, fontsize = fsz)
    grid.set_titles(col_template = '{col_name}', row_template = '{row_name}')
    grid.tight_layout()
    grid.savefig(os.path.join(figures_dir, mod + '_' + clname + str(npts) + '.pdf'))


def available_sizes(results):
    return sorted(int(value) for value in results['npt'].dropna().unique())


def draw_heatmap_suite(results, figures_dir, plotter = None, sizes = None):
    sizes = sizes or available_sizes(results)
    for index, npts in enumerate(sizes):
        ylbl = index == 0
        draw_heatmap(results, figures_dir, npts, 'tes', 'ora', plotter = plotter, fsz = 13, ylbl = ylbl)
        draw_heatmap(results, figures_dir, npts, 'fid', 'orf', plotter = plotter, ylbl = ylbl)
        draw_heatmap(results, figures_dir, npts, 'nle', 'orn', plotter = plotter, mlt = 1, pal = 'inverse', ylbl = ylbl)
        draw_heatmap(results, figures_dir, npts, 'tes', 'ora', plotter = plotter, mod = 'dtp', ylbl = ylbl)
        draw_heatmap(results, figures_dir, npts, 'bac', 'orb', plotter = plotter, mod = 'dtb', ylbl = ylbl)
        draw_heatmap(results, figures_dir, npts, 'tes', 'ora', plotter = plotter, mod = 'rules', ylbl = ylbl)
        draw_heatmap(results, figures_dir, npts, 'fid', 'orf', plotter = plotter, mod = 'rules', ylbl = ylbl)
        draw_heatmap(results, figures_dir, npts, 'nle', 'orn', plotter = plotter, mlt = 1, pal = 'inverse', mod = 'rules', ylbl = ylbl)
        draw_heatmap(results, figures_dir, npts, 'tes', 'ora', plotter = plotter, mod = 'sd', ylbl = ylbl)
        draw_heatmap(results, figures_dir, npts, 'nle', 'orn', plotter = plotter, mlt = 1, pal = 'inverse', mod = 'sd', ylbl = ylbl)


def bb_qual_change(meta):
    precdefault = meta.loc[meta['alg'] == 'testprec', 'val'].iloc[0]
    rows = []
    for _, row in meta.iterrows():
        alg = row['alg']
        if 'acc' in alg and 'acccv' not in alg and 'rfb' not in alg and 'xgbb' not in alg:
            rows.append({
                'alg': alg[:-3],
                'val': row['val'] - precdefault,
                'dat': row['dat'],
                'itr': row['itr'],
                'npt': row['npt'],
            })
    return pd.DataFrame(rows)


def compute_bb_results(meta_frames):
    results = [bb_qual_change(meta) for meta in meta_frames]
    results = [frame for frame in results if not frame.empty]
    if not results:
        return pd.DataFrame(columns = ['BB', 'N', 'BBacc'])
    bb = pd.concat(results, ignore_index = True)
    bb['npt'] = pd.to_numeric(bb['npt'])
    bb = bb[['alg', 'npt', 'val']].groupby(['alg', 'npt'], as_index = False).mean()
    bb = change_names(bb)
    bb.columns = ['BB', 'N', 'BBacc']
    bb['BBacc'] = round(bb['BBacc'], 3) * 100
    return bb


def get_table(data, mod = 'dt'):
    if mod not in MODEL_GROUPS:
        raise ValueError('{mod} is a wrong mod value'.format(mod = repr(mod)))
    data = data[data['alg'].isin(MODEL_GROUPS[mod])].copy().fillna(0)
    for column in [1.0, 0.0, -1.0]:
        if column not in data.columns:
            data[column] = 0
    data['wdl'] = (
        data[1.0].astype(int).astype(str) + '/' +
        data[0.0].astype(int).astype(str) + '/' +
        data[-1.0].astype(int).astype(str)
    )
    data = data[['alg', 'met', 'npt', 'wdl']]
    data = change_names(data)
    data = data.pivot(index = ['met', 'npt'], columns = ['alg'], values = 'wdl').reset_index()
    data = data.rename(columns = {'met': 'BB', 'npt': 'N'})
    data['N'] = pd.to_numeric(data['N'])
    return data


def get_table_wil(data, mod = 'dt'):
    if mod not in MODEL_GROUPS:
        raise ValueError('{mod} is a wrong mod value'.format(mod = repr(mod)))
    data = data[data['alg'].isin(MODEL_GROUPS[mod])].copy()

    from scipy.stats import wilcoxon

    def my_wil(values):
        return wilcoxon(values, alternative = 'greater')[1]

    pvals_dif = data.groupby(['alg', 'met', 'npt']).dif.apply(my_wil).reset_index(name = 'dif')
    pvals_difs = data.groupby(['alg', 'met', 'npt']).difs.apply(my_wil).reset_index(name = 'difs')
    merged = pd.merge(pvals_dif, pvals_difs)
    merged['ad'] = round(merged['dif'], 3).astype(str) + '/' + round(merged['difs'], 3).astype(str)
    merged = merged[['alg', 'met', 'npt', 'ad']]
    merged = change_names(merged)
    merged = merged.pivot(index = ['met', 'npt'], columns = ['alg'], values = 'ad').reset_index()
    merged = merged.rename(columns = {'met': 'BB', 'npt': 'N'})
    merged['N'] = pd.to_numeric(merged['N'])
    return merged


def build_kdebw_comparison(results):
    comparison = results[results['gen'] == 'kdebw'].copy()
    comparison = comparison[['alg', 'met', 'npt', 'dat', 'tes', 'ora']].groupby(
        ['alg', 'met', 'npt', 'dat'],
        as_index = False,
    ).median()
    comparison['dif'] = comparison['tes'] - comparison['ora']
    comparison['difs'] = np.sign(comparison['tes'] - comparison['ora'])
    return comparison


def write_derived_outputs(paths, results, meta_frames):
    results.to_csv(paths.res_path, index = False)
    bb_results = compute_bb_results(meta_frames)

    comparison = build_kdebw_comparison(results)
    pivot_wil = pd.merge(
        pd.merge(bb_results, get_table_wil(comparison, 'dt')),
        pd.merge(get_table_wil(comparison, 'rules'), get_table_wil(comparison, 'sd')),
    )
    pivot_wil.to_csv(paths.pivot_wil_path, index = False)

    wdl = comparison.groupby(['alg', 'met', 'npt']).difs.value_counts().unstack(fill_value = 0).reset_index()
    pivot_median = pd.merge(
        pd.merge(bb_results, get_table(wdl, 'dt')),
        pd.merge(get_table(wdl, 'rules'), get_table(wdl, 'sd')),
    )
    pivot_median.to_csv(paths.pivot_median_path, index = False)
    return {
        'res': results,
        'res_bb': bb_results,
        'pivot_wil': pivot_wil,
        'pivot_median': pivot_median,
    }


def postprocess_run(run_dir, draw_figures = True, plotter = None):
    paths = build_run_paths(run_dir)
    results, meta_frames, _ = load_run_results(paths)
    outputs = write_derived_outputs(paths, results, meta_frames)
    if draw_figures:
        draw_heatmap_suite(results, paths.figures_dir, plotter = plotter)
    return outputs


def main():
    args = parse_args()
    run_dir = resolve_selected_run(args)
    postprocess_run(run_dir, draw_figures = not args.skip_figures)


if __name__ == '__main__':
    main()
