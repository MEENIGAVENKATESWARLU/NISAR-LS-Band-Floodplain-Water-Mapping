#!/usr/bin/env python3
r"""PHASE-WATER 3H: S-band PR/UR radiometric and change-map sensitivity audit.

This is a DIAGNOSTIC, not a radiometric correction or accuracy assessment.
Uses the existing Phase 3G four-scene change maps and original HDF5 files.
No complete source raster is loaded into memory. Original products are read-only.

Windows PowerShell:
  .\.venv\Scripts\python.exe phase3h_sband_ur_pr_radiometric_stress_test.py --mode preflight
  .\.venv\Scripts\python.exe phase3h_sband_ur_pr_radiometric_stress_test.py --mode run

Optional: --strata-rows 8 --strata-cols 8 --tiles-per-stratum 2 --patch 96

Core safeguards:
  - Checks Phase 3G class-count arithmetic.
  - Samples spatially distributed windows from FOUR-SCENE valid coverage.
  - Reproduces the precise Phase 3G S and equal-weight L/S decisions BEFORE
    calculating any sensitivity; stops if values disagree with saved GeoTIFFs.
  - Reports the extreme-low backscatter tail of September S-HV.
  - Varies *only* September S-HV in a +/-3dB sensitivity experiment.
  - Does NOT imply the shift is calibration bias or correct it.
  - Does NOT extrapolate sampled sensitivity percentages to scene-wide area.
  - Does NOT generate reference labels, accuracy scores, or new change maps.
"""
from __future__ import annotations

import argparse
import csv
from contextlib import ExitStack
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import rasterio
from rasterio.windows import Window

EPS = 1e-10
SCORES = ('L_aug', 'S_aug', 'L_sep', 'S_sep')
RASTER_NAMES = {
    'joint': 'phase3g_four_scene_joint_valid.tif',
    'L_aug': 'phase3g_aug_L_only_water.tif',
    'S_aug': 'phase3g_aug_S_only_water.tif',
    'F_aug': 'phase3g_aug_LS_equalweight_water.tif',
    'L_sep': 'phase3g_sep_L_only_water.tif',
    'S_sep': 'phase3g_sep_S_only_water.tif',
    'F_sep': 'phase3g_sep_LS_equalweight_water.tif',
    'S_change': 'phase3g_S_only_change_4class.tif',
    'F_change': 'phase3g_LS_equalweight_change_4class.tif',
}
DELTA_DB = (-3.0, -2.58, -1.5, 0.0, 1.5, 2.58, 3.0)


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def score_from_db(hh_db, hv_db, hh_threshold, hv_threshold, width):
    z_hh = np.clip((hh_threshold - hh_db) / width, -30., 30.)
    z_hv = np.clip((hv_threshold - hv_db) / width, -30., 30.)
    return np.minimum(1. / (1. + np.exp(-z_hh)), 1. / (1. + np.exp(-z_hv)))


def load_report(folder):
    path = folder / 'phase3g_report.json'
    if not path.is_file():
        raise FileNotFoundError(f'Missing {path}: complete Phase 3G --mode run first.')
    data = json.loads(path.read_text(encoding='utf-8'))
    if not data.get('engineering_integrity_pass'):
        raise ValueError('Phase 3G did not report engineering integrity PASS')
    results = data['full_scene_results']
    n = int(results['four_scene_joint_valid_pixels'])
    if n < 1:
        raise ValueError('No four-way valid pixels in Phase 3G')
    class_checks = {}
    for key in ('L', 'S', 'F'):
        model = results['models'][key]
        counts = model['classes_pixels']
        total = sum(int(v) for v in counts.values())
        if total != n:
            raise AssertionError(f'{key} class totals {total} != joint valid {n}')
        px_area = (model['classes_km2']['water_gain'] -
                   model['classes_km2']['water_loss'])
        if abs(px_area - model['net_water_change_km2']) > 0.00011:
            raise AssertionError(f'{key} net change does not equal gain - loss')
        class_checks[key] = {'pixels_sum': total, 'net_change_arithmetic_pass': True}
    contingency = results['L_vs_S_change_class_contingency_pixels']
    if sum(int(x) for x in contingency.values()) != n:
        raise AssertionError('16-cell L/S change contingency totals do not match')
    diagonal = sum(int(contingency[f'L{i}_S{i}']) for i in range(4))
    if n - diagonal != int(results['L_vs_S_change_disagreement_pixels']):
        raise AssertionError('Change disagreement differs from contingency diagonal')
    if (data.get('grid_test') or '').startswith('PASS') is False:
        raise ValueError('Phase 3G grid alignment was not verified')
    thresholds = data['fixed_august_derived_thresholds_db']
    for label in ('L_HH', 'L_HV', 'S_HH', 'S_HV'):
        if not np.isfinite(float(thresholds[label])):
            raise ValueError('Invalid threshold: '+label)
    transition = float(data['score_transition_db'])
    if transition <= 0:
        raise ValueError('Invalid score transition dB')
    return data, class_checks


def paths_for_scenes(report, project_root):
    out = {}
    for key in SCORES:
        saved = Path(report['source_files'][key])
        if saved.is_file():
            out[key] = saved
            continue
        # Recover after moving a project between drives, but require exact basename.
        sub = 'L_band' if key.startswith('L') else 'S_band'
        candidate = project_root / 'data' / 'raw' / sub / saved.name
        if not candidate.is_file():
            raise FileNotFoundError(f'Missing {key}: {candidate}')
        out[key] = candidate
    return out


def h5_channels(h5, key):
    instrument = 'LSAR' if key.startswith('L') else 'SSAR'
    prefix = f'/science/{instrument}/GCOV/grids/frequencyA'
    if prefix not in h5:
        raise RuntimeError(f'{key}: missing {prefix}')
    g = h5[prefix]
    for label in ('HHHH', 'HVHV'):
        if label not in g:
            raise RuntimeError(f'{key} missing {label}')
    return g['HHHH'], g['HVHV']


def raster_preflight(folder, report):
    sgrid = report['source_grids']['S_aug']
    with ExitStack() as stack:
        ras = {key: stack.enter_context(rasterio.open(folder / name))
               for key, name in RASTER_NAMES.items()}
        ref = ras['joint']
        for key, ds in ras.items():
            if (ds.shape != ref.shape or ds.crs != ref.crs or
                not ds.transform.almost_equals(ref.transform) or ds.count != 1):
                raise ValueError(f'Mismatched raster grid in {key}')
        if ref.width != int(sgrid['cols']) or ref.height != int(sgrid['rows']):
            raise ValueError('Raster dimensions inconsistent with Phase 3G report')
        if ref.crs.to_epsg() != int(sgrid['epsg']):
            raise ValueError('Raster CRS inconsistent with Phase 3G report')
        if (int(report['s_to_l_row_col_offset'][0]) < 0 or
            int(report['s_to_l_row_col_offset'][1]) < 0):
            raise ValueError('Unexpected negative L/S offset')
        return {'height': ref.height, 'width': ref.width,
                'epsg': ref.crs.to_epsg(),
                'pixel_m': abs(float(ref.transform.a)),
                'raster_count': len(ras)}


def windows_stratified(height, width, rows, cols, reps, patch, seed):
    rng = np.random.default_rng(seed)
    for sr in range(rows):
        low_r = (sr * height) // rows
        high_r = ((sr + 1) * height) // rows
        for sc in range(cols):
            low_c = (sc * width) // cols
            high_c = ((sc + 1) * width) // cols
            h = min(patch, max(1, high_r - low_r))
            w = min(patch, max(1, high_c - low_c))
            for rep in range(reps):
                rmax = max(low_r, high_r - h)
                cmax = max(low_c, high_c - w)
                r = int(rng.integers(low_r, rmax + 1))
                c = int(rng.integers(low_c, cmax + 1))
                yield (sr, sc, rep, r, c, h, w)


def summarize_percentiles(vals):
    if not vals:
        return {'n': 0}
    values = np.concatenate(vals)
    return {'n': int(values.size),
            'p01_db': round(float(np.quantile(values, .01)), 3),
            'p10_db': round(float(np.quantile(values, .10)), 3),
            'median_db': round(float(np.median(values)), 3),
            'p90_db': round(float(np.quantile(values, .90)), 3),
            'p99_db': round(float(np.quantile(values, .99)), 3),
            'frac_below_minus_30_db': round(float(np.mean(values < -30)), 6),
            'frac_below_minus_40_db': round(float(np.mean(values < -40)), 6),
            'frac_below_minus_50_db': round(float(np.mean(values < -50)), 6)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', choices=['preflight', 'run'], default='preflight')
    p.add_argument('--project', default='.', help='D:\\Hackathon root')
    p.add_argument('--strata-rows', type=int, default=8)
    p.add_argument('--strata-cols', type=int, default=8)
    p.add_argument('--tiles-per-stratum', type=int, default=2)
    p.add_argument('--patch', type=int, default=96)
    p.add_argument('--seed', type=int, default=20261108)
    a = p.parse_args(argv)
    if min(a.strata_rows, a.strata_cols, a.tiles_per_stratum, a.patch) <= 0:
        p.error('Spatial sampling parameters must be positive integers')
    root = Path(a.project).resolve()
    src = root / 'outputs' / 'phase3g'
    report, check = load_report(src)
    grid = raster_preflight(src, report)
    result = report['full_scene_results']
    print('PHASE-WATER 3H | Phase 3G reported 4-scene valid pixels:',
          result['four_scene_joint_valid_pixels'])
    print('Class-count arithmetic and 16-cell cross-tab: PASS')
    print('Source grid:', grid, '| August S: PR | September S: UR')
    print('L vs S temporal change-class disagreement:',
          round(100 * result['L_vs_S_change_disagreement_fraction'], 3), '%')
    print('S HV sample p10 August/September:',
          report['preflight_sample']['db_distribution_on_same_fourway_sample']['S_aug_HV']['p10_db'],
          report['preflight_sample']['db_distribution_on_same_fourway_sample']['S_sep_HV']['p10_db'], 'dB')
    if a.mode == 'preflight':
        print('PREFLIGHT PASS. Run --mode run for distributed sample sensitivity check.')
        return 0
    h5paths = paths_for_scenes(report, root)
    offset_r, offset_c = map(int, report['s_to_l_row_col_offset'])
    thresholds = report['fixed_august_derived_thresholds_db']
    width = float(report['score_transition_db'])
    summary = {str(d): {'N': 0, 'S_sep_water': 0, 'F_sep_water': 0,
                        'S_sep_flips_vs_baseline': 0, 'F_sep_flips_vs_baseline': 0,
                        'S_gain': 0, 'S_loss': 0, 'F_gain': 0, 'F_loss': 0}
               for d in DELTA_DB}
    qa_counts = {'total_window_pixels': 0, 'valid_window_pixels': 0,
                 'windows_attempted': 0, 'windows_with_valid': 0,
                 'reproduction_mismatches': 0, 'reproduction_comparisons': 0}
    db_samples = {key + '_' + pol: [] for key in SCORES for pol in ('HH', 'HV')}
    details = []
    with ExitStack() as stack:
        handles = {key: stack.enter_context(h5py.File(path, 'r'))
                   for key, path in h5paths.items()}
        channels = {key: h5_channels(handles[key], key) for key in SCORES}
        ras = {key: stack.enter_context(rasterio.open(src / name))
               for key, name in RASTER_NAMES.items()}
        for sr, sc, rep, row, col, h, w in windows_stratified(
                grid['height'], grid['width'], a.strata_rows,
                a.strata_cols, a.tiles_per_stratum, a.patch, a.seed):
            qa_counts['windows_attempted'] += 1
            qa_counts['total_window_pixels'] += h * w
            window = Window(col, row, w, h)
            joint = ras['joint'].read(1, window=window) == 1
            n = int(joint.sum())
            if n == 0:
                continue
            qa_counts['windows_with_valid'] += 1
            qa_counts['valid_window_pixels'] += n
            db = {}
            for key in SCORES:
                off_r, off_c = ((offset_r, offset_c) if key.startswith('L')
                                else (0, 0))
                hh_ds, hv_ds = channels[key]
                r, c = row + off_r, col + off_c
                hh = np.asarray(hh_ds[r:r+h, c:c+w], dtype=np.float32)[joint]
                hv = np.asarray(hv_ds[r:r+h, c:c+w], dtype=np.float32)[joint]
                if not (np.all(np.isfinite(hh)) and np.all(hh > 0) and
                        np.all(np.isfinite(hv)) and np.all(hv > 0)):
                    raise ValueError(f'{key}: Phase 3G joint-valid raster contains invalid covariance')
                db[key + '_HH'] = 10 * np.log10(np.maximum(hh, EPS))
                db[key + '_HV'] = 10 * np.log10(np.maximum(hv, EPS))
                db_samples[key + '_HH'].append(db[key + '_HH'])
                db_samples[key + '_HV'].append(db[key + '_HV'])
            score = {}
            for key in SCORES:
                s = key[0]
                score[key] = score_from_db(db[key+'_HH'], db[key+'_HV'],
                                           float(thresholds[s+'_HH']),
                                           float(thresholds[s+'_HV']), width)
            score['F_aug'] = .5*(score['L_aug']+score['S_aug'])
            score['F_sep'] = .5*(score['L_sep']+score['S_sep'])
            baseline = {key: (score[key] >= .5) for key in
                        ('L_aug','S_aug','F_aug','L_sep','S_sep','F_sep')}
            # Require 1:1 reproduction against already-saved Phase 3G masks.
            for key, predicted in baseline.items():
                ref = ras[key].read(1, window=window)[joint]
                diff = int(np.count_nonzero(ref != predicted.astype(np.uint8)))
                qa_counts['reproduction_mismatches'] += diff
                qa_counts['reproduction_comparisons'] += n
                if diff:
                    raise RuntimeError(f'REPRODUCTION FAILED for {key} in tile ({row},{col}):'
                                       f' {diff}/{n} mismatches. STOP sensitivity inference.')
            for key in ('S','F'):
                for name, aug, sep in [('S_change',baseline['S_aug'],baseline['S_sep']),
                                       ('F_change',baseline['F_aug'],baseline['F_sep'])]:
                    if not name.startswith(key):
                        continue
                    actual = ras[name].read(1, window=window)[joint]
                    mapped = np.where(aug, np.where(sep, 1, 3),
                                      np.where(sep, 2, 0)).astype(np.uint8)
                    if np.any(actual != mapped):
                        raise RuntimeError(f'Change-code reconstruction failed in {name}')
            tile_detail = {'stratum_row': sr, 'stratum_col': sc, 'repeat': rep,
                           'row': row, 'col': col, 'valid_pixels': n,
                           'S_sep_HV_below_minus_40_fraction':
                           round(float(np.mean(db['S_sep_HV'] < -40)), 6),
                           'S_aug_HV_below_minus_40_fraction':
                           round(float(np.mean(db['S_aug_HV'] < -40)), 6),
                           'S_aug_water_fraction': round(float(np.mean(baseline['S_aug'])), 6),
                           'S_sep_water_fraction': round(float(np.mean(baseline['S_sep'])), 6),
                           'L_vs_S_change_disagreement_fraction':
                           round(float(np.mean(ras['S_change'].read(1, window=window)[joint] !=
                                               ras['F_change'].read(1, window=window)[joint])), 6)}
            # Last fraction is S vs F change, for clarity in the CSV.
            tile_detail['S_vs_F_change_disagreement_fraction'] = tile_detail.pop(
                'L_vs_S_change_disagreement_fraction')
            details.append(tile_detail)
            for delta in DELTA_DB:
                delta = float(delta)
                s_sep_shifted = score_from_db(
                    db['S_sep_HH'], db['S_sep_HV'] + delta,
                    float(thresholds['S_HH']), float(thresholds['S_HV']), width)
                s_sep_decision = s_sep_shifted >= .5
                f_sep_decision = (.5 * (score['L_sep'] + s_sep_shifted)) >= .5
                bucket = summary[str(delta)]
                bucket['N'] += n
                bucket['S_sep_water'] += int(s_sep_decision.sum())
                bucket['F_sep_water'] += int(f_sep_decision.sum())
                bucket['S_sep_flips_vs_baseline'] += int(
                    np.count_nonzero(s_sep_decision ^ baseline['S_sep']))
                bucket['F_sep_flips_vs_baseline'] += int(
                    np.count_nonzero(f_sep_decision ^ baseline['F_sep']))
                bucket['S_gain'] += int(np.count_nonzero(~baseline['S_aug'] & s_sep_decision))
                bucket['S_loss'] += int(np.count_nonzero(baseline['S_aug'] & ~s_sep_decision))
                bucket['F_gain'] += int(np.count_nonzero(~baseline['F_aug'] & f_sep_decision))
                bucket['F_loss'] += int(np.count_nonzero(baseline['F_aug'] & ~f_sep_decision))
            if qa_counts['windows_attempted'] % max(1, a.strata_cols) == 0:
                print(f'  Audited {qa_counts["windows_attempted"]} spatial windows...')
    if qa_counts['valid_window_pixels'] < 100:
        raise RuntimeError('Too few jointly valid samples: cannot infer even sampled sensitivity')
    output = root / 'outputs' / 'phase3h'
    output.mkdir(parents=True, exist_ok=True)
    stats = {k: summarize_percentiles(v) for k,v in db_samples.items()}
    deltas = []
    for delta in DELTA_DB:
        bucket = summary[str(float(delta))]
        n = bucket['N']
        row = {'S_sep_HV_test_offset_db': float(delta), 'sample_valid_pixels': n,
               'S_sep_water_fraction': bucket['S_sep_water']/n,
               'F_sep_water_fraction': bucket['F_sep_water']/n,
               'S_sep_water_flip_fraction_vs_baseline':
                   bucket['S_sep_flips_vs_baseline']/n,
               'F_sep_water_flip_fraction_vs_baseline':
                   bucket['F_sep_flips_vs_baseline']/n,
               'S_sample_gain_fraction': bucket['S_gain']/n,
               'S_sample_loss_fraction': bucket['S_loss']/n,
               'F_sample_gain_fraction': bucket['F_gain']/n,
               'F_sample_loss_fraction': bucket['F_loss']/n}
        deltas.append(row)
    with (output / 'phase3h_hv_offset_sensitivity.csv').open('w',newline='',encoding='utf-8') as f:
        wr = csv.DictWriter(f,fieldnames=list(deltas[0]))
        wr.writeheader();wr.writerows(deltas)
    with (output / 'phase3h_spatial_window_diagnostics.csv').open('w',newline='',encoding='utf-8') as f:
        wr = csv.DictWriter(f,fieldnames=list(details[0]))
        wr.writeheader();wr.writerows(details)
    report_out = {
        'phase':'3H diagnostic radiometric-tail audit and S-HV sensitivity, NOT calibration',
        'created_utc':now_iso(), 'engineering_report_checks':check,
        'phase3g_four_scene_joint_valid_pixels':result['four_scene_joint_valid_pixels'],
        'phase3g_L_S_change_disagreement_fraction':
            result['L_vs_S_change_disagreement_fraction'],
        'sample_type':'Spatially stratified random windows; not a reference sample, not design-weighted area estimation',
        'sample_design':{'strata_rows':a.strata_rows,'strata_cols':a.strata_cols,
                         'tiles_per_stratum':a.tiles_per_stratum,'patch':a.patch,'seed':a.seed},
        'sample_counts':qa_counts,
        'radiometric_distributions_db_on_same_sample':stats,
        'september_S_HV_offset_stress_test':deltas,
        'essential_limits':[
            'September S-band uses UR while August S-band uses PR; equivalence NOT established.',
            'Adding a dB offset in this experiment is sensitivity testing, NOT a valid radiometric correction.',
            'The sampled fractions must NOT be scaled to claim entire-scene changes or water area.',
            'Scene-level backscatter distribution changes mix hydrology, land cover, incidence/processing and noise floor.',
            'Extremely low S-HV values require scrutiny of processing, quality masks and noise floor.',
            'No independent water labels, accuracy metrics, uncertainty calibration or verified flooding generated.',
        ],
        'result':'DIAGNOSTIC_ONLY',
    }
    (output / 'phase3h_report.json').write_text(
        json.dumps(report_out,indent=2,allow_nan=False),encoding='utf-8')
    print('\nPHASE-WATER 3H COMPLETE:',len(details),'valid sampled windows')
    print('Reconstruction mismatches:',qa_counts['reproduction_mismatches'])
    print('Sampled S-HV low-backscatter tail, Aug:',stats['S_aug_HV'])
    print('Sampled S-HV low-backscatter tail, Sep:',stats['S_sep_HV'])
    print('Sampled S-HV +2.58dB perturbation, change in S Sep water label fraction:',
          round(next(x for x in deltas if x['S_sep_HV_test_offset_db']==2.58)
                ['S_sep_water_flip_fraction_vs_baseline']*100,3),'%')
    print('Outputs:',output)
    print('CAUTION: NOT a sensor calibration and NOT validated water-change accuracy.')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, KeyError, ValueError, RuntimeError, AssertionError) as error:
        print('ERROR:',type(error).__name__,error,file=sys.stderr)
        sys.exit(2)
