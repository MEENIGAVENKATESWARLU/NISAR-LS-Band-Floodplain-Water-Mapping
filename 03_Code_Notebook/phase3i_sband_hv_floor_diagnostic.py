#!/usr/bin/env python3
"""PHASE-WATER 3I: investigate extreme September S-HV low-power tail.

Diagnostic ONLY. Uses the same spatial windows as Phase 3H, on four-scene
joint-valid support. Inspects the original L/S HDF5 data, mask, numberOfLooks,
HHHV, and selected HDF5 attributes. Never edits input products or estimates
reference accuracy, sensor bias, or full-scene artifact area.

Windows PowerShell:
  .\\.venv\\Scripts\\python.exe phase3i_sband_hv_floor_diagnostic.py --mode preflight
  .\\.venv\\Scripts\\python.exe phase3i_sband_hv_floor_diagnostic.py --mode run

Inputs:
  outputs/phase3g/phase3g_report.json
  outputs/phase3g/phase3g_four_scene_joint_valid.tif
  outputs/phase3g/phase3g_sep_S_only_water.tif
  outputs/phase3g/phase3g_S_only_change_4class.tif
  outputs/phase3h/phase3h_spatial_window_diagnostics.csv
  four Phase 3G source HDF5 files

Outputs:
  outputs/phase3i/phase3i_report.json
  outputs/phase3i/phase3i_spatial_tail_windows.csv
  outputs/phase3i/phase3i_spatial_tail_windows.geojson
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import rasterio
from rasterio.windows import Window

SCENE_KEYS = ('L_aug', 'S_aug', 'L_sep', 'S_sep')
ALERT_DB = -40.0
EPS = 1e-30  # Diagnostics must preserve low positive power; no log clipping at 1e-10.


def to_jsonable(value):
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='replace')
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, (int, float, bool, str)) or value is None:
        if isinstance(value, float) and not np.isfinite(value):
            return str(value)
        return value
    if isinstance(value, np.ndarray):
        if value.size > 16:
            return {'dtype': str(value.dtype), 'shape': list(value.shape), 'note': 'large attribute omitted'}
        return [to_jsonable(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value[:16]]
    return str(value)[:512]


def attrs_small(group):
    return {str(k): to_jsonable(v) for k, v in group.attrs.items()}


def src_paths(report, root):
    paths = {}
    for key in SCENE_KEYS:
        saved = Path(report['source_files'][key])
        if saved.is_file():
            paths[key] = saved
        else:
            sub = 'L_band' if key.startswith('L') else 'S_band'
            alt = root / 'data' / 'raw' / sub / saved.name
            if not alt.is_file():
                raise FileNotFoundError(f'Missing source {key}: {alt}')
            paths[key] = alt
    return paths


def source_group(h5, key):
    sensor = 'LSAR' if key.startswith('L') else 'SSAR'
    path = f'/science/{sensor}/GCOV/grids/frequencyA'
    g = h5[path]
    for field in ('HHHH', 'HVHV', 'mask', 'numberOfLooks'):
        if field not in g:
            raise KeyError(f'{key}: missing {path}/{field}')
    return g


def metadata_hints(h5):
    """Names are evidence for further manual study, not interpreted quality flags."""
    hits = []
    terms = ('noise', 'nes0', 'rfi', 'calibrat', 'quality', 'validity', 'processing',
             'productversion', 'product_version', 'radiometric')
    def visitor(name, obj):
        if len(hits) >= 80:
            return
        low = name.lower()
        if isinstance(obj, h5py.Dataset) and any(term in low for term in terms):
            hits.append({'path': '/' + name, 'shape': list(obj.shape), 'dtype': str(obj.dtype)})
    h5.visititems(visitor)
    return hits


def windows_from_3h(path, shape, patch):
    if not path.is_file():
        raise FileNotFoundError(f'Missing Phase 3H spatial sample: {path}. Run Phase 3H first.')
    rows = []
    with path.open('r', newline='', encoding='utf-8-sig') as fh:
        for r in csv.DictReader(fh):
            y, x = int(r['row']), int(r['col'])
            h, w = min(patch, shape[0] - y), min(patch, shape[1] - x)
            if y < 0 or x < 0 or h < 1 or w < 1:
                raise ValueError(f'Invalid window row/col {y}, {x}')
            rows.append((y, x, h, w, r))
    if not rows:
        raise ValueError('Phase 3H window CSV has no sampled valid windows')
    return rows


def db(power):
    """10 log10 of positive GCOV power; avoid artificial negative-100 dB floor."""
    a = np.asarray(power, dtype=np.float64)
    if np.any(~np.isfinite(a)) or np.any(a <= 0):
        raise ValueError('Nonpositive/nonfinite covariance within Phase 3G joint valid raster')
    return 10.0 * np.log10(np.maximum(a, EPS))


def safe_summary(arr):
    v = np.asarray(arr, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {'n': 0}
    return {'n': int(len(v)), 'p01': round(float(np.percentile(v, 1)), 4),
            'p10': round(float(np.percentile(v, 10)), 4),
            'median': round(float(np.median(v)), 4),
            'p90': round(float(np.percentile(v, 90)), 4),
            'p99': round(float(np.percentile(v, 99)), 4)}


def distribution(arr):
    v = np.asarray(arr, dtype=np.float64)
    return {**safe_summary(v),
            'fraction_below_minus_30_db': round(float(np.mean(v < -30)), 6),
            'fraction_below_minus_40_db': round(float(np.mean(v < -40)), 6),
            'fraction_below_minus_50_db': round(float(np.mean(v < -50)), 6)}


def mask_categories(arr):
    n, counts = np.unique(arr, return_counts=True)
    return {str(int(k)): int(v) for k, v in zip(n, counts)}


def top_exact_values(values, limit=8):
    values = np.asarray(values, dtype=np.float32)
    unique, cnt = np.unique(values, return_counts=True)
    chosen = np.argsort(cnt)[-limit:][::-1]
    return [{'raw_linear_power': float(unique[i]), 'count': int(cnt[i])}
            for i in chosen]


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def geo_windows(rows, transform, outfile, epsg):
    features = []
    for item in rows:
        x0, y0 = transform * (item['col'], item['row'])
        x1, y1 = transform * (item['col'] + item['width'], item['row'] + item['height'])
        ring = [[x0,y0],[x1,y0],[x1,y1],[x0,y1],[x0,y0]]
        properties = {k: v for k, v in item.items() if k not in ('row','col','height','width')}
        properties.update({'row': item['row'], 'col': item['col']})
        features.append({'type':'Feature', 'geometry':{'type':'Polygon','coordinates':[ring]},
                         'properties':properties})
    payload = {'type':'FeatureCollection', 'name':'phase3i_s_hv_tail_windows',
               'crs':{'type':'name','properties':{'name':f'EPSG:{epsg}'}},
               'features':features}
    outfile.write_text(json.dumps(payload, allow_nan=False), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                 formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mode', choices=('preflight', 'run'), default='preflight')
    parser.add_argument('--project', default='.', help='Project directory D:\\Hackathon')
    parser.add_argument('--patch', type=int, default=96,
                        help='Must match Phase 3H --patch value (default 96)')
    args = parser.parse_args()
    if args.patch <= 0:
        parser.error('--patch must be positive')
    root = Path(args.project).resolve()
    folder_g = root/'outputs'/'phase3g'
    folder_h = root/'outputs'/'phase3h'
    report_file = folder_g/'phase3g_report.json'
    if not report_file.is_file():
        raise FileNotFoundError(f'Missing Phase 3G report: {report_file}')
    phase_g = json.loads(report_file.read_text(encoding='utf-8'))
    if phase_g.get('engineering_integrity_pass') is not True:
        raise ValueError('Phase 3G engineering integrity check must pass')
    paths = src_paths(phase_g, root)
    with ExitStack() as stack:
        h5 = {k: stack.enter_context(h5py.File(v, 'r')) for k, v in paths.items()}
        groups = {k: source_group(h5[k], k) for k in SCENE_KEYS}
        masks = stack.enter_context(rasterio.open(folder_g/'phase3g_four_scene_joint_valid.tif'))
        s_pred = stack.enter_context(rasterio.open(folder_g/'phase3g_sep_S_only_water.tif'))
        s_change = stack.enter_context(rasterio.open(folder_g/'phase3g_S_only_change_4class.tif'))
        for name, ds in [('S prediction',s_pred),('S change',s_change)]:
            if ds.shape != masks.shape or ds.crs != masks.crs or not ds.transform.almost_equals(masks.transform):
                raise ValueError(f'{name}: misaligned Phase 3G GeoTIFF')
        r_off, c_off = map(int, phase_g['s_to_l_row_col_offset'])
        for key in ('S_aug','S_sep'):
            for ch in ('HHHH','HVHV','mask','numberOfLooks'):
                if groups[key][ch].shape != masks.shape:
                    raise ValueError(f'{key} {ch} raster mismatch')
        for key in ('L_aug','L_sep'):
            shape = groups[key]['HHHH'].shape
            if shape[0] < masks.height + r_off or shape[1] < masks.width + c_off:
                raise ValueError(f'{key} does not cover S scene')
        windows = windows_from_3h(folder_h/'phase3h_spatial_window_diagnostics.csv',
                                  masks.shape, args.patch)
        meta = {}
        for key in ('S_aug','S_sep'):
            group = groups[key]
            meta[key] = {'filename':paths[key].name,
                         'group_attributes':attrs_small(group),
                         'datasets':{k:{'dtype':str(group[k].dtype),
                                          'attributes':attrs_small(group[k])}
                                      for k in ('HHHH','HVHV','HHHV','mask','numberOfLooks',
                                                'rtcGammaToSigmaFactor') if k in group},
                         'metadata_name_hints':metadata_hints(h5[key])}
        print('PHASE-WATER 3I | spatial windows inherited from Phase 3H:',len(windows))
        print('  Phase 3G aligned grid:', masks.width, 'x', masks.height,
              '| EPSG:',masks.crs.to_epsg())
        print('  S products:',paths['S_aug'].name.split('_')[2], 'vs',
              paths['S_sep'].name.split('_')[2])
        if args.mode == 'preflight':
            print('PREFLIGHT PASS. Run --mode run to diagnose raw September S-HV tail.')
            return 0
        results = []
        concat = {x: [] for x in ('S_aug_HV_db','S_sep_HV_db','S_sep_HH_db',
                                   'S_aug_HH_db','L_sep_HV_db','L_aug_HV_db',
                                   'S_sep_looks','S_aug_looks','S_sep_rtc','S_aug_rtc',
                                   'S_sep_HV_raw','S_sep_mask','S_aug_mask',
                                   'S_sep_water','S_change','S_sep_psd_ratio')}
        count = Counter()
        for i, (row,col,height,width,prior) in enumerate(windows,1):
            window = Window(col,row,width,height)
            valid = masks.read(1,window=window) == 1
            n = int(valid.sum())
            if n == 0:
                raise ValueError(f'Window {i} has no four-scene valid pixels; did --patch match Phase 3H?')
            count['pixels_total'] += height * width
            count['valid'] += n
            block = {}
            for sensor in SCENE_KEYS:
                off_r, off_c = (r_off,c_off) if sensor.startswith('L') else (0,0)
                g = groups[sensor]
                y,x=row+off_r,col+off_c
                block[sensor+'_HH'] = np.asarray(g['HHHH'][y:y+height,x:x+width])[valid]
                block[sensor+'_HV'] = np.asarray(g['HVHV'][y:y+height,x:x+width])[valid]
                if sensor.startswith('S'):
                    block[sensor+'_mask'] = np.asarray(g['mask'][y:y+height,x:x+width])[valid]
                    block[sensor+'_looks'] = np.asarray(g['numberOfLooks'][y:y+height,x:x+width])[valid]
                    rtd = g.get('rtcGammaToSigmaFactor')
                    if rtd is not None and rtd.shape == masks.shape:
                        block[sensor+'_rtc'] = np.asarray(rtd[y:y+height,x:x+width])[valid]
            hv_aug_db = db(block['S_aug_HV'])
            hv_sep_db = db(block['S_sep_HV'])
            hh_sep_db = db(block['S_sep_HH'])
            low = hv_sep_db < ALERT_DB
            count['low_sep'] += int(low.sum())
            count['low_aug'] += int(np.count_nonzero(hv_aug_db < ALERT_DB))
            scodes = s_change.read(1,window=window)[valid]
            swater = s_pred.read(1,window=window)[valid]
            if not np.all((scodes>=0)&(scodes<=3)) or not np.all((swater==0)|(swater==1)):
                raise ValueError(f'Unexpected labels inside joint valid raster at window {i}')
            for k,v in [('S_aug_HV_db',hv_aug_db),('S_sep_HV_db',hv_sep_db),
                        ('S_aug_HH_db',db(block['S_aug_HH'])),('S_sep_HH_db',hh_sep_db),
                        ('L_aug_HV_db',db(block['L_aug_HV'])),
                        ('L_sep_HV_db',db(block['L_sep_HV']))]:
                concat[k].append(v)
            for k in ('S_sep_looks','S_aug_looks','S_sep_mask','S_aug_mask',
                      'S_sep_rtc','S_aug_rtc'):
                if k in block:
                    concat[k].append(block[k])
            concat['S_sep_HV_raw'].append(block['S_sep_HV'].astype(np.float32))
            concat['S_sep_water'].append(swater)
            concat['S_change'].append(scodes)
            if 'HHHV' in groups['S_sep']:
                c = np.asarray(groups['S_sep']['HHHV'][row:row+height,col:col+width])[valid]
                den = (np.asarray(block['S_sep_HH'],dtype=np.float64)*
                       np.asarray(block['S_sep_HV'],dtype=np.float64))
                psd = np.abs(c.astype(np.complex128))**2 / np.maximum(den,1e-30)
                concat['S_sep_psd_ratio'].append(psd)
            one = {'window_id':i,'row':row,'col':col,'height':height,'width':width,
                   'n_joint_valid':n, 'sep_s_hv_below_minus40_pct':round(100*np.mean(low),3),
                   'aug_s_hv_below_minus40_pct':round(100*np.mean(hv_aug_db<-40),3),
                   'sep_s_hv_median_db':round(float(np.median(hv_sep_db)),3),
                   'sep_s_hh_median_db':round(float(np.median(hh_sep_db)),3),
                   's_sep_water_pct':round(100*np.mean(swater==1),3),
                   's_sep_water_gain_pct':round(100*np.mean(scodes==2),3),
                   's_sep_water_loss_pct':round(100*np.mean(scodes==3),3),
                   'low_tail_s_water_pct':round(100*np.mean(swater[low]==1),3) if low.any() else None,
                   'low_tail_s_gain_pct':round(100*np.mean(scodes[low]==2),3) if low.any() else None,
                   's_sep_mask_codes_on_low':str(mask_categories(block['S_sep_mask'][low])),
                   'sep_looks_low_median':float(np.median(block['S_sep_looks'][low])) if low.any() else None}
            results.append(one)
            if i%20 == 0 or i == len(windows):
                print(f'  Audited {i}/{len(windows)} Phase 3H windows')
        arrays = {k:(np.concatenate(v) if v else np.array([])) for k,v in concat.items()}
        n = count['valid']
        low = arrays['S_sep_HV_db'] < ALERT_DB
        if n != len(arrays['S_sep_HV_db']):
            raise RuntimeError('Sample count mismatch')
        active = ~low
        report = {
            'phase':'3I: native HDF5 S-band HV low-power tail diagnosis, NO sensor calibration',
            'created_utc':datetime.now(timezone.utc).isoformat(),
            'sampling':'Exactly the Phase 3H selected spatial windows; four-scene jointly valid pixels; not an area-probability sample',
            'sampled_windows':len(windows),'valid_pixels':n,
            'recorded_phase3h_window_count':len(windows),
            'S_aug_HV_db':distribution(arrays['S_aug_HV_db']),
            'S_sep_HV_db':distribution(arrays['S_sep_HV_db']),
            'S_sep_HV_raw_linear_top_repeated_values':top_exact_values(arrays['S_sep_HV_raw']),
            'S_sep_HV_raw_linear_positive_min':float(np.min(arrays['S_sep_HV_raw'])),
            'S_sep_HV_raw_unique_fraction':round(
               float(len(np.unique(arrays['S_sep_HV_raw']))/n),6),
            'S_sep_mask_all_codes':mask_categories(arrays['S_sep_mask']),
            'S_sep_mask_codes_low_tail':mask_categories(arrays['S_sep_mask'][low]),
            'S_sep_mask_codes_not_low_tail':mask_categories(arrays['S_sep_mask'][active]),
            'S_aug_mask_all_codes':mask_categories(arrays['S_aug_mask']),
            'S_sep_looks_low_tail':safe_summary(arrays['S_sep_looks'][low]),
            'S_sep_looks_non_low_tail':safe_summary(arrays['S_sep_looks'][active]),
            'S_aug_looks_same_low_tail_locations':safe_summary(arrays['S_aug_looks'][low]),
            'S_sep_HH_db_low_tail':safe_summary(arrays['S_sep_HH_db'][low]),
            'S_sep_HH_db_non_low_tail':safe_summary(arrays['S_sep_HH_db'][active]),
            'L_sep_HV_db_same_low_tail_locations':safe_summary(arrays['L_sep_HV_db'][low]),
            'L_aug_HV_db_same_low_tail_locations':safe_summary(arrays['L_aug_HV_db'][low]),
            'S_aug_HV_db_same_low_tail_locations':safe_summary(arrays['S_aug_HV_db'][low]),
            'S_sep_rtc_low_tail':safe_summary(arrays['S_sep_rtc'][low]) if arrays['S_sep_rtc'].size else {'n':0},
            'S_sep_rtc_non_low_tail':safe_summary(arrays['S_sep_rtc'][active]) if arrays['S_sep_rtc'].size else {'n':0},
            'S_sep_PSD_ratio_below40':safe_summary(arrays['S_sep_psd_ratio'][low]) if arrays['S_sep_psd_ratio'].size else {'n':0},
            'S_sep_PSD_ratio_violations_gt1p05':int(np.count_nonzero(arrays['S_sep_psd_ratio']>1.05)) if arrays['S_sep_psd_ratio'].size else None,
            's_sep_class_on_low_tail':{
               'water_fraction':float(np.mean(arrays['S_sep_water'][low]==1)) if low.any() else None,
               'gain_fraction':float(np.mean(arrays['S_change'][low]==2)) if low.any() else None,
               'loss_fraction':float(np.mean(arrays['S_change'][low]==3)) if low.any() else None},
            's_sep_class_on_non_low_tail':{
               'water_fraction':float(np.mean(arrays['S_sep_water'][active]==1)) if active.any() else None,
               'gain_fraction':float(np.mean(arrays['S_change'][active]==2)) if active.any() else None,
               'loss_fraction':float(np.mean(arrays['S_change'][active]==3)) if active.any() else None},
            'spatial_window_fraction_above_50pct_low_tail':round(float(np.mean(
                [r['sep_s_hv_below_minus40_pct']>50 for r in results])),6),
            'windows_with_gt5pct_low_tail':int(sum(
                r['sep_s_hv_below_minus40_pct']>5 for r in results)),
            'source_metadata':meta,
            'interpretation':'High low-tail rates are anomalies needing product-processor/QA review; neither sensor calibration errors nor true water can be concluded solely from these data.',
            'limitations':[
                'No absolute instrument noise-equivalent sigma0 / NES0 test performed.',
                'Jointly valid means mask 1..254, positive looks and covariance; it does not imply reliable cross-pol radiometry.',
                'No S-band UR/PR processing equivalence established.',
                'Sampling is spatially distributed but not probability-weighted for estimating full-scene prevalence.',
                'Do not apply constant dB correction or use this sample to estimate water-map accuracy.'
            ],
            'status':'HIGH_PRIORITY_LOW_S_HV_TAIL_INVESTIGATION' if (np.mean(low)>0.05 and
                np.mean(arrays['S_aug_HV_db'] < -40)<0.001) else 'TAIL_DISCONTINUITY_NOT_ESTABLISHED'
        }
        out = root/'outputs'/'phase3i'
        out.mkdir(parents=True,exist_ok=True)
        write_csv(out/'phase3i_spatial_tail_windows.csv',results)
        geo_windows(results,masks.transform,out/'phase3i_spatial_tail_windows.geojson',
                    masks.crs.to_epsg())
        (out/'phase3i_report.json').write_text(
            json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
        print('\nPHASE-WATER 3I DIAGNOSTIC COMPLETE')
        print('  Four-scene-valid pixels sampled:',n)
        print('  S-HV <-40 dB fraction August/September:',
              round(100*np.mean(arrays['S_aug_HV_db']<-40),3),'/',
              round(100*np.mean(low),3),'%')
        print('  September S-HV top repeated linear-power values:',
              report['S_sep_HV_raw_linear_top_repeated_values'][:3])
        print('  September S-HV low-tail window count:',report['windows_with_gt5pct_low_tail'])
        print('  Result:',report['status'])
        print('  Outputs:',out)
        return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, KeyError, ValueError, RuntimeError, AssertionError) as exc:
        print('ERROR:',type(exc).__name__,str(exc),file=sys.stderr)
        sys.exit(2)
