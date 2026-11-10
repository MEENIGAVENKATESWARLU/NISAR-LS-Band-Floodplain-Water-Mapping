#!/usr/bin/env python3
"""PHASE-WATER 3G: four-scene NISAR L/S two-date exploratory change experiment.

Inputs: verified NISAR L2 GCOV L+S pairs from 19 Aug and 24 Sep 2026.
No full HDF5 is loaded into memory. Input files are read-only. The full S-band
rectangular scene grid is processed, not a discovery-region subset.

Modes (run from D:\\Hackathon):
  .\\.venv\\Scripts\\python.exe phase3g_four_scene_dualfrequency_change.py --mode preflight
  .\\.venv\\Scripts\\python.exe phase3g_four_scene_dualfrequency_change.py --mode run

CRITICAL SCIENTIFIC LIMITS:
- The September S2_UR (urgent-response) GCOV may differ from the August S2_PR
  processing chain. Verify radiometric/QA comparability separately.
- Both acquisition dates MUST share identical pixel-center grids per sensor;
  S-band must be integer-offset-aligned with L-band (NO resampling performed).
- All change classes and area statistics are restricted to the same FOUR-WAY
  jointly valid pixel support across August L/S and September L/S.
- Reuse the August-only Phase 3C thresholds for BOTH dates. Do not estimate
  different thresholds for each date, which would confound temporal change.
- The 50:50 L/S score is uncalibrated; neither it nor its changes are validated.
- No accuracy/flooding claims can be made without independent reference labels.
- The S complex cross-covariance HHHV is sampled for a physics-consistency
  diagnostic but is NOT used in this fair HH/HV baseline.

Four-class change encoding: 0=stable_nonwater; 1=stable_water;
2=water_gain; 3=water_loss; 255=invalid_any_of_four_scenes.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys

import h5py
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window

NODATA = np.uint8(255)
EPS = 1e-10
SCENES = {
    'L_aug': ('L_band', 'L', 'NISAR_L2_PR_GCOV_028_098_A_014_4005_DHDH_A_20260819T233811*.h5'),
    'S_aug': ('S_band', 'S', 'NISAR_S2_PR_GCOV_028_098_A_014_3700_DHNA_A_20260819T233810*.h5'),
    'L_sep': ('L_band', 'L', 'NISAR_L2_PR_GCOV_031_098_A_014_4005_DHDH_A_20260924T233810*.h5'),
    'S_sep': ('S_band', 'S', 'NISAR_S2_UR_GCOV_031_098_A_014_3700_DHNA_A_20260924T233810*.h5'),
}
OUTPUTS = {
    'joint': 'phase3g_four_scene_joint_valid.tif',
    'L_aug': 'phase3g_aug_L_only_water.tif',
    'S_aug': 'phase3g_aug_S_only_water.tif',
    'F_aug': 'phase3g_aug_LS_equalweight_water.tif',
    'L_sep': 'phase3g_sep_L_only_water.tif',
    'S_sep': 'phase3g_sep_S_only_water.tif',
    'F_sep': 'phase3g_sep_LS_equalweight_water.tif',
    'L_change': 'phase3g_L_only_change_4class.tif',
    'S_change': 'phase3g_S_only_change_4class.tif',
    'F_change': 'phase3g_LS_equalweight_change_4class.tif',
    'frequency_change_disagreement': 'phase3g_L_vs_S_change_disagreement.tif',
}
CLASSES = {'stable_nonwater': 0, 'stable_water': 1, 'water_gain': 2, 'water_loss': 3}


def locate(raw: Path, subdir: str, glob: str) -> Path:
    matches = sorted((raw / subdir).glob(glob))
    if len(matches) != 1:
        raise RuntimeError(f'Expected exactly one verified file for {glob} under {raw/subdir}; found {len(matches)}')
    return matches[0]


class Scene:
    def __init__(self, path: Path, instrument: str):
        self.path = path
        self.instrument = instrument
        self.h5 = h5py.File(path, 'r')
        prefix = f'/science/{"LSAR" if instrument == "L" else "SSAR"}/GCOV/grids/frequencyA'
        if prefix not in self.h5:
            raise ValueError(f'Missing expected GCOV group {prefix} in {path.name}')
        self.group = self.h5[prefix]
        for name in ('HHHH','HVHV','mask','numberOfLooks','xCoordinates','yCoordinates','projection'):
            if name not in self.group:
                raise ValueError(f'{path.name} missing {prefix}/{name}')
        self.hh = self.group['HHHH']
        self.hv = self.group['HVHV']
        self.mask = self.group['mask']
        self.looks = self.group['numberOfLooks']
        self.cross = self.group.get('HHHV')
        self.rows, self.cols = self.hh.shape
        for key in ('HVHV','mask','numberOfLooks'):
            if self.group[key].shape != (self.rows,self.cols):
                raise ValueError(f'{path.name}: inconsistent dataset dimensions: {key}')
        if self.cross is not None and self.cross.shape != self.hh.shape:
            raise ValueError(f'{path.name}: inconsistent HHHV dimensions')
        xx,yy = self.group['xCoordinates'],self.group['yCoordinates']
        if len(xx)!=self.cols or len(yy)!=self.rows:
            raise ValueError(f'{path.name}: coordinate dimension mismatch')
        if self.cols < 2 or self.rows < 2:
            raise ValueError('Source raster smaller than 2x2')
        self.x0,self.y0 = float(xx[0]),float(yy[0])
        self.dx,self.dy = float(xx[1])-self.x0,float(yy[1])-self.y0
        if self.dx <= 0 or self.dy >= 0 or abs(self.dx+self.dy) > 1e-5:
            raise ValueError(f'{path.name}: unexpected raster orientation or spacing')
        if not (abs(float(xx[-1])-(self.x0+(self.cols-1)*self.dx))<0.001
                and abs(float(yy[-1])-(self.y0+(self.rows-1)*self.dy))<0.001):
            raise ValueError(f'{path.name}: coordinate endpoints inconsistent with constant step')
        projection=self.group['projection']
        try:
            self.epsg=int(projection[()])
        except (TypeError,ValueError):
            self.epsg=int(projection.attrs['epsg_code'])
        self.transform=from_origin(self.x0-self.dx/2,self.y0-self.dy/2,self.dx,-self.dy)

    def close(self):
        self.h5.close()

    def read(self,row: int,col: int,height: int,width: int):
        sl=np.s_[row:row+height,col:col+width]
        hh=np.asarray(self.hh[sl],dtype=np.float32)
        hv=np.asarray(self.hv[sl],dtype=np.float32)
        mask=np.asarray(self.mask[sl])
        looks=np.asarray(self.looks[sl])
        valid=(mask>0)&(mask<255)&np.isfinite(looks)&(looks>0)
        valid&=np.isfinite(hh)&(hh>0)&np.isfinite(hv)&(hv>0)
        return hh,hv,valid

    def info(self):
        return dict(file=self.path.name, instrument=self.instrument,epsg=self.epsg,
                    rows=self.rows,cols=self.cols,dx=self.dx,dy=self.dy,
                    pixel_center_first=[self.x0,self.y0],
                    pixel_center_last=[self.x0+(self.cols-1)*self.dx,self.y0+(self.rows-1)*self.dy],
                    channels=['HHHH']+(['HHHV'] if self.cross is not None else [])+['HVHV'])


def exact_same_grid(a: Scene,b: Scene,title: str):
    if (a.epsg!=b.epsg or a.rows!=b.rows or a.cols!=b.cols
        or not a.transform.almost_equals(b.transform)
        or abs(a.x0-b.x0)>1e-4 or abs(a.y0-b.y0)>1e-4
        or abs(a.dx-b.dx)>1e-6 or abs(a.dy-b.dy)>1e-6):
        raise ValueError(f'{title}: grids differ. STOP: no temporal pixel comparison or nearest-neighbor shortcut.')


def align(l: Scene,s: Scene):
    if l.epsg != s.epsg or abs(l.dx-s.dx)>1e-6 or abs(l.dy-s.dy)>1e-6:
        raise ValueError('L vs S CRS or native pixel spacing differs. STOP; this script never resamples.')
    col_f=(s.x0-l.x0)/l.dx
    row_f=(l.y0-s.y0)/(-l.dy)
    row,col=int(round(row_f)),int(round(col_f))
    if abs(col_f-col)>1e-6 or abs(row_f-row)>1e-6:
        raise ValueError('L/S grid centers are not integer aligned. STOP; check spatial metadata.')
    if row<0 or col<0 or row+s.rows>l.rows or col+s.cols>l.cols:
        raise ValueError('S rectangular grid extends outside L grid. STOP for separate overlap-grid method.')
    return row,col


def read_thresholds(report_path: Path,paths: dict):
    if not report_path.is_file():
        raise FileNotFoundError(f'Missing {report_path}. Run Phase 3C first; do not invent thresholds.')
    report=json.loads(report_path.read_text(encoding='utf-8'))
    original=report.get('sources') or {}
    for key,source in (('L','L_aug'),('S','S_aug')):
        if Path(str(original.get(key,''))).name != paths[source].name:
            raise ValueError(f'Phase 3C {key} source does not match this August HDF5. Refuse threshold reuse.')
    values=report.get('exploratory_threshold_db')
    if not isinstance(values,dict):
        raise ValueError('Phase 3C report has no exploratory_threshold_db')
    result={}
    for key in ('L_HH','L_HV','S_HH','S_HV'):
        try:
            number=float(values[key])
        except (KeyError,ValueError,TypeError):
            raise ValueError('Phase 3C report missing a finite threshold for '+key)
        if not np.isfinite(number) or number < -50 or number > 20:
            raise ValueError('Implausible Phase 3C threshold for '+key)
        result[key]=number
    return result


def sigmoid(v):
    return 1.0/(1.0+np.exp(-np.clip(v,-30.0,30.0)))


def water_score(hh,hv,thh,thv,transition_db):
    hdb=10.0*np.log10(np.maximum(hh,EPS))
    vdb=10.0*np.log10(np.maximum(hv,EPS))
    return np.minimum(sigmoid((thh-hdb)/transition_db),sigmoid((thv-vdb)/transition_db))


def preflight_samples(scenes,offset,patch=64,per_axis=10):
    sr=scenes['S_aug']
    rr,cc=offset
    counts=Counter()
    distribution={k:[] for k in ['L_aug_HH','L_aug_HV','S_aug_HH','S_aug_HV',
                                 'L_sep_HH','L_sep_HV','S_sep_HH','S_sep_HV']}
    for row in np.linspace(0,max(0,sr.rows-patch),per_axis,dtype=int):
        for col in np.linspace(0,max(0,sr.cols-patch),per_axis,dtype=int):
            row,col=int(row),int(col)
            h,w=min(patch,sr.rows-row),min(patch,sr.cols-col)
            tile={}
            for key,scene in scenes.items():
                y,x=(row+rr,col+cc) if key.startswith('L') else (row,col)
                tile[key]=scene.read(y,x,h,w)
            masks={key:values[2] for key,values in tile.items()}
            common=masks['L_aug']&masks['S_aug']&masks['L_sep']&masks['S_sep']
            counts['sample_pixels']+=h*w
            counts['joint_valid_four_scene']+=int(common.sum())
            for key,good in masks.items():
                counts[key+'_valid']+=int(good.sum())
            if not np.any(common):
                continue
            for key,(a,b,good) in tile.items():
                distribution[key+'_HH'].append((10*np.log10(np.maximum(a[common],EPS))).astype(np.float32))
                distribution[key+'_HV'].append((10*np.log10(np.maximum(b[common],EPS))).astype(np.float32))
            for key in ('S_aug','S_sep'):
                if scenes[key].cross is not None:
                    y,x=row,col
                    cross=np.asarray(scenes[key].cross[y:y+h,x:x+w])[common]
                    hh,hv,_=tile[key]
                    ratio=np.abs(cross)**2/np.maximum((hh*hv)[common],EPS)
                    counts[key+'_cross_checked']+=int(len(ratio))
                    counts[key+'_cross_nonfinite']+=int((~np.isfinite(ratio)).sum())
                    counts[key+'_PSD_gt_1p05']+=int((np.isfinite(ratio)&(ratio>1.05)).sum())
    stats={}
    for key,values in distribution.items():
        if values:
            x=np.concatenate(values)
            stats[key]={'median_db':round(float(np.median(x)),3),
                        'p10_db':round(float(np.percentile(x,10)),3),
                        'p90_db':round(float(np.percentile(x,90)),3),
                        'n':int(len(x))}
    if counts['joint_valid_four_scene']<300:
        raise RuntimeError('Insufficient four-scene valid samples; cannot proceed safely')
    shifts={}
    for sensor in ('L','S'):
        for pol in ('HH','HV'):
            shifts[sensor+'_'+pol+'_sep_minus_aug_median_db']=round(
                stats[sensor+'_sep_'+pol]['median_db']-stats[sensor+'_aug_'+pol]['median_db'],3)
    return dict(counts=dict(counts),db_distribution_on_same_fourway_sample=stats,
                median_shift_diagnostics=shifts,
                note='Scene-level distribution changes can reflect hydrology/land cover/processing; not a radiometric calibration test.' )


def make_writer(path: Path,s: Scene):
    kwargs=dict(driver='GTiff',width=s.cols,height=s.rows,count=1,dtype='uint8',
                nodata=255,crs=f'EPSG:{s.epsg}',transform=s.transform,
                tiled=True,blockxsize=512,blockysize=512,compress='deflate',
                predictor=2,zlevel=3,bigtiff='IF_SAFER',num_threads=2)
    return rasterio.open(path,'w',**kwargs)


def calculate_change(a,b,common):
    result=np.full(a.shape,NODATA,dtype=np.uint8)
    result[common & (~a) & (~b)]=0
    result[common & a & b]=1
    result[common & (~a) & b]=2
    result[common & a & (~b)]=3
    return result


def process(scenes,offset,thresholds,out: Path,tile_size=1024,transition_db=2.5):
    s=scenes['S_aug']; ro,co=offset
    counts=Counter()
    class_counts={key:Counter() for key in ('L','S','F')}
    temporally_cross=Counter()
    temp_paths={key:out/(name+'.partial.tif') for key,name in OUTPUTS.items()}
    for p in temp_paths.values():
        if p.exists():
            p.unlink()
    try:
        with ExitStack() as stack:
            writers={key:stack.enter_context(make_writer(p,s)) for key,p in temp_paths.items()}
            for key,writer in writers.items():
                writer.update_tags(1,DESCRIPTION=key,STATUS='EXPLORATORY_UNVALIDATED',
                                   JOINT_SUPPORT='FOUR_SCENE_AUG_SEP_L_S',
                                   CODES=('0=stable_nonwater;1=stable_water;2=gain;3=loss;255=invalid'
                                          if key.endswith('change') else '0=nonwater_or_no;1=water_or_yes;255=invalid'))
            nrow=math.ceil(s.rows/tile_size)
            for i,row in enumerate(range(0,s.rows,tile_size),1):
                h=min(tile_size,s.rows-row)
                for col in range(0,s.cols,tile_size):
                    w=min(tile_size,s.cols-col)
                    window=Window(col,row,w,h)
                    data={}
                    for key,sc in scenes.items():
                        r,c=(row+ro,col+co) if key.startswith('L') else (row,col)
                        data[key]=sc.read(r,c,h,w)
                    good=np.logical_and.reduce([v[2] for v in data.values()])
                    ng=int(good.sum())
                    counts['joint_valid_pixels']+=ng
                    counts['total_S_grid_pixels']+=h*w
                    for key,(hh,hv,valid) in data.items():
                        counts[key+'_valid_pixels']+=int(valid.sum())
                    outputs={key:np.full((h,w),NODATA,dtype=np.uint8) for key in OUTPUTS}
                    outputs['joint'][:]=0
                    outputs['joint'][good]=1
                    if ng:
                        decisions={}
                        score={}
                        for key,(hh,hv,valid) in data.items():
                            prefix='L' if key.startswith('L') else 'S'
                            probability=water_score(hh[good],hv[good],
                                                    thresholds[prefix+'_HH'],thresholds[prefix+'_HV'],
                                                    transition_db)
                            score[key]=probability
                            decisions[key]=probability>=.5
                        score['F_aug']=0.5*(score['L_aug']+score['S_aug'])
                        score['F_sep']=0.5*(score['L_sep']+score['S_sep'])
                        decisions['F_aug']=score['F_aug']>=.5
                        decisions['F_sep']=score['F_sep']>=.5
                        for key in ('L_aug','S_aug','F_aug','L_sep','S_sep','F_sep'):
                            outputs[key][good]=decisions[key].astype(np.uint8)
                            counts[key+'_water_pixels']+=int(decisions[key].sum())
                        for sensor in ('L','S','F'):
                            change=calculate_change(outputs[sensor+'_aug']==1,
                                                    outputs[sensor+'_sep']==1,good)
                            outputs[sensor+'_change']=change
                            vals=np.bincount(change[good],minlength=4)
                            for name,code in CLASSES.items():
                                class_counts[sensor][name]+=int(vals[code])
                        for period in ('aug','sep'):
                            decisions_l=decisions['L_'+period]
                            decisions_s=decisions['S_'+period]
                            counts[period+'_LS_disagreement_pixels']+=int(np.count_nonzero(decisions_l^decisions_s))
                        changed=outputs['L_change'][good]!=outputs['S_change'][good]
                        outarr=outputs['frequency_change_disagreement']
                        outarr[good]=changed.astype(np.uint8)
                        counts['L_vs_S_change_disagreement_pixels']+=int(changed.sum())
                        # Cross-tab is only model agreement, NOT reference accuracy.
                        lc=outputs['L_change'][good].astype(np.uint8)
                        sc=outputs['S_change'][good].astype(np.uint8)
                        encoded=(lc*4+sc)
                        combos=np.bincount(encoded,minlength=16)
                        for code,n in enumerate(combos):
                            temporally_cross[f'L{code//4}_S{code%4}']+=int(n)
                    for key,array in outputs.items():
                        writers[key].write(array,1,window=window)
                print(f'PHASE-WATER 3G full-scene tile row {i}/{nrow}',flush=True)
        if counts['joint_valid_pixels']<=0:
            raise RuntimeError('Zero four-way valid pixels: outputs left as .partial.tif')
        for method in ('L','S','F'):
            if sum(class_counts[method].values())!=counts['joint_valid_pixels']:
                raise RuntimeError(f'Change class totals do not match four-scene joint valid support for {method}')
            if (class_counts[method]['stable_water']+class_counts[method]['water_loss']
                    !=counts[method+'_aug_water_pixels']):
                raise RuntimeError(f'August water totals do not match change classes for {method}')
            if (class_counts[method]['stable_water']+class_counts[method]['water_gain']
                    !=counts[method+'_sep_water_pixels']):
                raise RuntimeError(f'September water totals do not match change classes for {method}')
        if sum(temporally_cross.values())!=counts['joint_valid_pixels']:
            raise RuntimeError('L/S cross-tab does not sum to four-scene joint support')
        for key,partial in temp_paths.items():
            os.replace(partial,out/OUTPUTS[key])
    except Exception:
        print('ERROR during processing. Partial GeoTIFF files may remain in output folder.',file=sys.stderr)
        raise
    pixel_km2=abs(s.dx*s.dy)/1_000_000.0
    models={}
    for model in ('L','S','F'):
        raw=class_counts[model]
        models[model] = {
            'classes_pixels':dict(raw),
            'classes_km2':{k:round(v*pixel_km2,4) for k,v in raw.items()},
            'aug_water_km2':round(counts[model+'_aug_water_pixels']*pixel_km2,4),
            'sep_water_km2':round(counts[model+'_sep_water_pixels']*pixel_km2,4),
            'net_water_change_km2':round((raw['water_gain']-raw['water_loss'])*pixel_km2,4),
        }
    report={
        's_native_grid_pixels':int(s.rows*s.cols),
        'four_scene_joint_valid_pixels':int(counts['joint_valid_pixels']),
        'four_scene_joint_valid_km2':round(counts['joint_valid_pixels']*pixel_km2,4),
        'four_scene_joint_valid_fraction_of_S_rectangle':round(counts['joint_valid_pixels']/(s.rows*s.cols),8),
        'four_source_mask_counts':{k:int(counts[k+'_valid_pixels']) for k in SCENES},
        'per_date_LS_disagreement':{
            p:{'pixels':int(counts[p+'_LS_disagreement_pixels']),
               'fraction_fourway_joint_valid':round(counts[p+'_LS_disagreement_pixels']/counts['joint_valid_pixels'],8)}
            for p in ('aug','sep')},
        'L_vs_S_change_disagreement_pixels':int(counts['L_vs_S_change_disagreement_pixels']),
        'L_vs_S_change_disagreement_fraction':round(counts['L_vs_S_change_disagreement_pixels']/counts['joint_valid_pixels'],8),
        'models':models,
        'L_vs_S_change_class_contingency_pixels':dict(temporally_cross),
        'outputs':{k:str((out/name).resolve()) for k,name in OUTPUTS.items()},
    }
    rows=[]
    for model in ('L','S','F'):
        for class_name in CLASSES:
            rows.append({'method':{'L':'L_only','S':'S_only','F':'LS_equalweight'}[model],
                         'change_class':class_name,
                         'pixel_count':class_counts[model][class_name],
                         'area_km2':round(class_counts[model][class_name]*pixel_km2,4),
                         'reference_status':'UNVALIDATED_EXPLORATORY'})
    with (out/'phase3g_change_class_areas.csv').open('w',newline='',encoding='utf-8-sig') as f:
        wr=csv.DictWriter(f,fieldnames=rows[0].keys());wr.writeheader();wr.writerows(rows)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mode',choices=('preflight','run'),default='preflight')
    parser.add_argument('--raw-dir',type=Path,default=Path('data/raw'))
    parser.add_argument('--phase3c-report',type=Path,default=Path('outputs/phase3c/phase3c_report.json'))
    parser.add_argument('--out',type=Path,default=Path('outputs/phase3g'))
    parser.add_argument('--tile',type=int,default=1024)
    parser.add_argument('--transition-db',type=float,default=2.5,
                        help='Must match the Phase 3C score transition, default 2.5 dB')
    args=parser.parse_args()
    if not 256<=args.tile<=2048:parser.error('--tile must be between 256 and 2048')
    if not 0<args.transition_db<=20:parser.error('--transition-db must be >0 and <=20')
    paths={key:locate(args.raw_dir,entry[0],entry[2]) for key,entry in SCENES.items()}
    thresholds=read_thresholds(args.phase3c_report,paths)
    args.out.mkdir(parents=True,exist_ok=True)
    scenes={}
    try:
        for key,entry in SCENES.items():
            scenes[key]=Scene(paths[key],entry[1])
        exact_same_grid(scenes['L_aug'],scenes['L_sep'],'L Aug/Sep')
        exact_same_grid(scenes['S_aug'],scenes['S_sep'],'S Aug/Sep (PR vs UR)')
        offset=align(scenes['L_aug'],scenes['S_aug'])
        if align(scenes['L_sep'],scenes['S_sep'])!=offset:
            raise ValueError('August versus September L/S offsets differ: STOP')
        print('PHASE-WATER 3G | 4 GCOV scenes opened; aligned 10m grid checked')
        print('Full S-band scene:',scenes['S_aug'].cols,'x',scenes['S_aug'].rows,
              '| EPSG:',scenes['S_aug'].epsg,'| offset S-in-L row/col:',offset)
        print('Phase 3C August-fixed thresholds (dB):',json.dumps(thresholds))
        print('S September processing: UR (urgent response), August: PR; comparability NOT independently verified')
        sample=preflight_samples(scenes,offset)
        print('Four-way QA sample:',json.dumps(sample['counts']))
        print('Matched-sample median dB shifts:',json.dumps(sample['median_shift_diagnostics']))
        report={'phase':'3G: two-date L-only, S-only, 50:50 L/S exploratory change',
                'created_utc':datetime.now(timezone.utc).isoformat(),
                'mode':args.mode,'source_files':{k:str(v.resolve()) for k,v in paths.items()},
                'source_grids':{k:sc.info() for k,sc in scenes.items()},
                's_to_l_row_col_offset':list(offset),
                'grid_test':'PASS: exact within-sensor temporal grid, integer-offset cross-sensor grid',
                'threshold_source':str(args.phase3c_report.resolve()),
                'fixed_august_derived_thresholds_db':thresholds,
                'score_transition_db':args.transition_db,
                'preflight_sample':sample,
                'change_codes':{'0':'stable_nonwater','1':'stable_water','2':'newly_classified_water',
                                '3':'water_loss','255':'invalid_any_of_four_scenes'},
                'classification_method':'Apply August-derived frequency-specific HH/HV Otsu thresholds to both dates using min-logistic score; 50:50 average L/S scores (uncalibrated).',
                'critical_caveats':[
                    'S September UR versus August PR product processing is not demonstrated radiometrically equivalent.',
                    'Median differences over a changing floodplain are NOT sufficient to diagnose sensor calibration.',
                    'Changes are changes in provisional class assignment, not independently verified flooding.',
                    'All maps and areas restricted to identical FOUR-scene valid support on FULL S native rectangular grid.',
                    'Full L native extent outside S rectangle is represented in separate Phase 2A products.',
                    'No interpolating or upsampling; native pixel grids are exactly aligned at 10 m.',
                    'Gamma-nought diagonal covariances treated as positive linear power.',
                    'HHHV is only an S-band PSD diagnostic and is NOT included in the fair HH/HV comparison.',
                    'No independent reference labels or accuracy scores generated.',
                    'Discovery bbox is not used to clip science scenes.'
                ]}
        preflight_path=args.out/'phase3g_preflight_report.json'
        preflight_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print('Preflight report:',preflight_path)
        if args.mode=='preflight':
            print('PREFLIGHT PASS. Review QA shifts, then run --mode run for all four full-scene maps.')
            return
        result=process(scenes,offset,thresholds,args.out,args.tile,args.transition_db)
        report['full_scene_results']=result
        report['engineering_integrity_pass']=True
        dest=args.out/'phase3g_report.json'
        dest.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print('\nPHASE-WATER 3G FOUR-SCENE ANALYSIS COMPLETED')
        print(json.dumps({key:result[key] for key in
                          ('four_scene_joint_valid_pixels','four_scene_joint_valid_km2',
                           'per_date_LS_disagreement','L_vs_S_change_disagreement_fraction','models')},indent=2))
        print('Saved:',dest)
        print('CAUTION: provisional class-change, not validated flood extent/accuracy. S UR/PR consistency needs further review.')
    finally:
        for scene in scenes.values():
            scene.close()


if __name__=='__main__':
    try:
        main()
    except (OSError,ValueError,RuntimeError,KeyError,IndexError,TypeError) as exc:
        print(f'PHASE 3G ERROR: {type(exc).__name__}: {exc}',file=sys.stderr)
        sys.exit(2)
