#!/usr/bin/env python3
r"""PHASE-WATER 4C: audited research-results pack and technical-note draft.

Windows (run inside D:\Hackathon):
  .\.venv\Scripts\python.exe phase4c_build_submission_results.py --mode preflight
  .\.venv\Scripts\python.exe phase4c_build_submission_results.py --mode build

Uses only Python standard library and EXISTING locally generated Phase 3G and
Phase 4B files. No satellite downloads; does not alter source data; no invented
reference truth, accuracy or radiometric calibration.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('.')
OUT = Path('outputs/phase4c')
CLASSES = ('stable_nonwater','stable_water','water_gain','water_loss')
NAMES = {'L':'L-only','S':'S-only','F':'Equal-weight L+S'}

def load_json(p:Path):
    if not p.is_file(): raise FileNotFoundError(f'Missing required file: {p.resolve()}')
    with p.open('r',encoding='utf-8-sig') as f: return json.load(f)

def read_csv(p:Path):
    if not p.is_file(): raise FileNotFoundError(f'Missing required file: {p.resolve()}')
    with p.open('r',encoding='utf-8-sig',newline='') as f:
        rd=csv.DictReader(f)
        if not rd.fieldnames: raise ValueError(f'CSV has no header: {p}')
        return list(rd)

def pct(v):return f'{100*v:.2f}%'
def area(v):return f'{v:,.2f}'
def decimal(v,n=4):return round(float(v),n)
def write_csv(path,fields,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        w.writeheader();w.writerows(rows)

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def analyze(g,b,records,h=None):
    fg=g.get('full_scene_results',g)
    n=int(fg['four_scene_joint_valid_pixels'])
    if n<1:raise ValueError('No 4-way valid pixels')
    if not math.isclose(float(fg['four_scene_joint_valid_km2']),n*0.0001,rel_tol=0,abs_tol=0.0002):
        raise ValueError('Four-way valid area does not agree with 10m pixel count')
    counts=[];mstats={}
    for model in ('L','S','F'):
        r=fg['models'][model]; c=r['classes_pixels']
        if sum(int(c[k]) for k in CLASSES)!=n:raise ValueError(f'{model}: class counts mismatch')
        vals={k:int(c[k])*.0001 for k in CLASSES}
        for k in CLASSES:
            if not math.isclose(vals[k],float(r['classes_km2'][k]),rel_tol=0,abs_tol=0.0002):
                raise ValueError(f'{model}/{k}: km2 and pixel count mismatch')
        aug=vals['stable_water']+vals['water_loss']
        sep=vals['stable_water']+vals['water_gain']
        net=vals['water_gain']-vals['water_loss']
        for name,v in [('aug_water_km2',aug),('sep_water_km2',sep),('net_water_change_km2',net)]:
            if not math.isclose(float(r[name]),v,rel_tol=0,abs_tol=0.0002):
                raise ValueError(f'{model}/{name}: arithmetical inconsistency')
        mstats[model]={'aug_classified_km2':aug,'sep_classified_km2':sep,'net_km2':net,
                       'gain_km2':vals['water_gain'],'loss_km2':vals['water_loss'],
                       'stable_water_km2':vals['stable_water'],'stable_nonwater_km2':vals['stable_nonwater']}
        counts.append({'method':NAMES[model],**{k+'_pixels':int(c[k]) for k in CLASSES},
                       **{k+'_km2':decimal(vals[k]) for k in CLASSES},
                       'aug_classified_water_km2':decimal(aug),
                       'sep_classified_water_km2':decimal(sep),
                       'net_classified_change_km2':decimal(net)})
    tab=fg.get('L_vs_S_change_class_contingency_pixels')
    if not isinstance(tab,dict) or len(tab)!=16:raise ValueError('Missing 16-cell L/S change cross-tab')
    if sum(int(x) for x in tab.values())!=n: raise ValueError('L/S 16-cell contingency count mismatch')
    agreement=sum(int(tab[f'L{i}_S{i}']) for i in range(4))
    disagree=n-agreement
    if int(fg['L_vs_S_change_disagreement_pixels'])!=disagree:raise ValueError('Cross-tab disagrees with disagreement pixel count')
    if not math.isclose(float(fg['L_vs_S_change_disagreement_fraction']),disagree/n,abs_tol=5e-8):
        raise ValueError('Cross-tab disagrees with disagreement fraction')
    if not b.get('no_full_scene_accuracy') or not b.get('reference_label_fields_modified') is False:
        raise ValueError('Phase4B report no longer has expected no-accuracy / untouched-reference safeguards')
    if len(records)!=400:raise ValueError(f'Expected 400 original points, got {len(records)}')
    ids=[r.get('sample_id','') for r in records]
    if len(set(ids))!=400 or any(not x for x in ids):raise ValueError('Duplicate or missing sample IDs')
    official=sum(r.get('reference_aug_water','').strip() in {'0','1'} for r in records)
    if official!=int(b.get('official_reference_labels_preserved',0)):
        raise ValueError('Official reference label count differs between Phase4B report and merged CSV')
    screen=[r for r in records if r.get('ai_screen_category','').strip()]
    categories=Counter(r.get('ai_screen_category','').strip() for r in screen)
    usable=[r for r in screen if r.get('ai_screen_optical_date_water','').strip() in {'0','1'}]
    if len(screen)!=int(b['screened_n']):raise ValueError('Screened count mismatch')
    diag=b['diagnostic_only']
    for key in ('within24h','within72h','all_usable_optical_dates'):
        if key not in diag:raise ValueError(f'Missing optical subset {key}')
    if int(diag['all_usable_optical_dates']['n'])!=len(usable):
        raise ValueError('Phase4B diagnostic usable count differs from reviewed records')
    for key,maxhours in [('within24h',24),('within72h',72)]:
        count=sum(float(r['ai_screen_optical_gap_hours']) <= maxhours for r in usable)
        if count!=int(diag[key]['n']):raise ValueError(f'{key} mismatch: {count} vs {diag[key]["n"]}')
    optical={'original_sample_n':len(records),'screened_n':len(screen),'interpretable_n':len(usable),
             'review_categories':dict(categories),'within_24h_n':int(diag['within24h']['n']),
             'within_72h_n':int(diag['within72h']['n']),'verified_reference_labels_n':official,
             'class_counts':dict(Counter(r['ai_screen_optical_date_water'].strip() for r in usable))}
    hv=None
    if h:
        tail=h.get('radiometric_distributions_db_on_same_sample',{})
        if 'S_aug_HV' in tail and 'S_sep_HV' in tail:
            hv={'aug_below_minus_40_fraction':float(tail['S_aug_HV']['frac_below_minus_40_db']),
                'sep_below_minus_40_fraction':float(tail['S_sep_HV']['frac_below_minus_40_db']),
                'sample_n':int(tail['S_sep_HV']['n']),
                'aug_p10_db':float(tail['S_aug_HV']['p10_db']),
                'sep_p10_db':float(tail['S_sep_HV']['p10_db'])}
    return {'fourway_n':n,'fourway_area_km2':n*0.0001,'method_stats':mstats,'rows':counts,
            'change_disagreement_n':disagree,'change_disagreement_frac':disagree/n,
            'change_agreement_n':agreement,'optical':optical,
            'aug_frequency_disagree_frac':float(fg['per_date_LS_disagreement']['aug']['fraction_fourway_joint_valid']),
            'sep_frequency_disagree_frac':float(fg['per_date_LS_disagreement']['sep']['fraction_fourway_joint_valid']),
            'hv':hv,
            'S_processing_comparability':'unverified (S August PR vs September UR)'}

def make_markdown(summary):
    s=summary; st=s['method_stats'];o=s['optical'];hv=s['hv']
    t=[]
    t += ['# PHASE-WATER | Audited Scientific Results (Phase 4C)','',
          '**Status:** Engineering-verified exploratory classification. This document does not report independently validated flood extent, classification accuracy, precision, recall, IoU, F1, confidence intervals, or improvement from fusion.','',
          '## Data and common analysis support','',
          'Four NISAR L2 GCOV products, including co-acquired L/S pairs from 19 August and 24 September 2026, were compared over a common native 10 m EPSG:32645 grid. The S-band grid is shifted 72 columns east and 288 rows south relative to the L-band grid; exact integer alignment avoids spatial resampling. All three methods share the identical four-scene valid support.','',
          f"**Jointly valid support:** {s['fourway_n']:,} pixels = {area(s['fourway_area_km2'])} km². The full acquisition footprint was processed, not the smaller catalogue-discovery box.",'',
          '## Four-class temporal comparisons (provisional, km²)','',
          '| Method | Stable non-water | Stable water | Classified gain | Classified loss | Net classified change |',
          '|---|---:|---:|---:|---:|---:|']
    for m in ('L','S','F'):
        z=st[m]
        t.append(f"| {NAMES[m]} | {area(z['stable_nonwater_km2'])} | {area(z['stable_water_km2'])} | {area(z['gain_km2'])} | {area(z['loss_km2'])} | {area(z['net_km2'])} |")
    t += ['',f"**L-versus-S four-class disagreement:** {s['change_disagreement_n']:,} pixels ({pct(s['change_disagreement_frac'])}) on shared valid support.",
          f"Per-date L/S provisional binary disagreement was {pct(s['aug_frequency_disagree_frac'])} in August and {pct(s['sep_frequency_disagree_frac'])} in September.",'',
          '## Optical screening—not independent accuracy validation','',
          f"The original sample contains {o['original_sample_n']} sites. {o['screened_n']} had candidate optical observations reviewed by AI; {o['interpretable_n']} yielded interpretable optical-date screening labels (of which {o['within_24h_n']} were within 24 hours and {o['within_72h_n']} within 72 hours). {sum(v for k,v in o['class_counts'].items() if k == '1')} are provisionally water-like and {o['class_counts'].get('0',0)} non-water-like.",
          f"**Confirmed SAR-time reference labels: {o['verified_reference_labels_n']}.** Cloud-screened observations are availability-selected and temporally mismatched, often with 20–30 m optical spatial support. They must not be substituted into the official reference field or reported as scene-wide accuracy.",'',
          '## Radiometric / processing uncertainty','',
          'September S-band was processed as Urgent Response (UR); August S-band is PR. Radiometric equivalence is not independently established.']
    if hv:
        t.append(f"A spatial diagnostic of {hv['sample_n']:,} common valid samples found S-HV values below −40 dB at {pct(hv['aug_below_minus_40_fraction'])} (August) and {pct(hv['sep_below_minus_40_fraction'])} (September); corresponding p10 values were {hv['aug_p10_db']:.2f} and {hv['sep_p10_db']:.2f} dB. This is diagnostic sampling, not an unbiased full-scene tail-prevalence estimate or a sensor calibration.")
    t += ['', '## What the study can and cannot conclude','',
          '- Supported: reproducible same-pass dual-frequency data alignment, full-footprint comparative class maps, explicit frequency disagreements and transparent sensitivity diagnostics.',
          '- Not supported: L/S fusion superiority, true flood/river-water area or change, physical attribution of S-HV anomalies, or independent classification performance metrics.',
          '- Interpret the L/S disagreements as **method sensitivity** rather than proof of accuracy differences. The 50:50 fusion score is not a calibrated posterior probability.',
          '', '## Suggested figure captions','',
          '**Figure 1.** Spatial distributions of provisional surface-water classifications from NISAR L-band, S-band and equal-weight fusion for August and September 2026, on common four-scene valid support (10 m, EPSG:32645).',
          '**Figure 2.** Four-class temporal classification transitions from August to September 2026: stable non-water, stable water, newly classified water and classified water loss. September S-band UR processing limits direct radiometric interpretability.',
          '**Figure 3.** Locations of L/S disagreement in the four-class change maps on the identical jointly valid footprint. Disagreement is not an independent classification error.',
          '', '## Provenance','',
          'Results calculated from outputs/phase3g/phase3g_report.json; August optical diagnostic from outputs/phase4b/phase4b_AI_screened_400.csv and phase4b_AI_screening_diagnostic_report.json. Optional S-HV diagnostic from outputs/phase3h/phase3h_report.json.',
          '']
    return '\n'.join(t)

def make_note(s):
    z=s['method_stats']; o=s['optical']; hv=s['hv']
    note=[
        '# PHASE-WATER | Technical Note Draft',
        '## Physics-aware dual-frequency NISAR GCOV comparison for dynamic surface-water classification over the Bihar Ganga floodplain',
        '', '**Scientific status:** Exploratory, not independently accuracy-validated.','',
        '### 1. Objective and study design','',
        'We investigated how NISAR L-band and S-band SAR observations differ in their provisional mapping of surface-water classes and temporal transitions across a heterogeneous alluvial floodplain. The test was designed around two co-acquired L/S pairs (19 August and 24 September 2026). The goal was to establish a reproducible common-footprint baseline, quantify disagreement and identify vulnerabilities to changes in sensor processing and radiometry. Improved accuracy was a hypothesis for future testing, not an observed result.','',
        '### 2. Data preparation and common-support protocol','',
        'All science inputs were NISAR Level-2 GCOV HDF5 products. L-band diagonal covariance terms HHHH and HVHV were taken from frequencyA at 10 m; the corresponding S-band terms were also read from frequencyA at 10 m. S-band additionally supplied the complex HHHV covariance, used only for physical-consistency diagnostics and not as an unfair extra predictor in the baseline comparison. The L-band frequencyB group is a distinct bandwidth/gridding group of the L-band product, not a separate S-band sensor.','',
        f"Products were verified on the EPSG:32645 grid. Both S-band scenes use the same 10 m pixels as L-band but begin 288 rows and 72 columns into the L raster. Pixel-centre coordinates support exact integer-offset alignment without interpolation. All comparisons were restricted to {s['fourway_n']:,} pixels ({area(s['fourway_area_km2'])} km²) valid in all four source images. The study processed the complete available S-band GCOV rectangle and did not use the smaller search bounding box to clip scientific products.",'',
        '### 3. Exploratory classification and change detection','',
        'A first-pass unsupervised HH/HV classifier used August-derived frequency-specific thresholds fixed across both dates, with logistic transition scores. Common-polarization L-only and S-only baselines were contrasted with an equal-weight fusion of their scores. The fusion values are ordinal classification scores, not calibrated probabilities. Each dated binary map yielded a four-class transition map: stable non-water, stable water, newly classified water and classified water loss. Native-resolution area calculations used 0.0001 km² for each valid 10 m pixel. The same four-scene mask was used for every method.','',
        '### 4. Results','',
        f"The L-only model classified {area(z['L']['gain_km2'])} km² as gain and {area(z['L']['loss_km2'])} km² as loss (net {area(z['L']['net_km2'])} km²). The S-only model classified {area(z['S']['gain_km2'])} km² as gain and {area(z['S']['loss_km2'])} km² as loss (net {area(z['S']['net_km2'])} km²). Equal-weight fusion classified {area(z['F']['gain_km2'])} km² as gain and {area(z['F']['loss_km2'])} km² as loss (net {area(z['F']['net_km2'])} km²). These figures represent changes in model-assigned class, not independently established changes in inundation extent.",'',
        f"The two single-frequency methods disagreed on {s['change_disagreement_n']:,} four-class labels ({pct(s['change_disagreement_frac'])} of four-scene valid pixels). Date-specific binary water masks disagreed on {pct(s['aug_frequency_disagree_frac'])} (August) and {pct(s['sep_frequency_disagree_frac'])} (September) of common support. Engineering checks confirmed the contingency-table totals and water-area accounting.",'',
        '### 5. Independent optical evidence and limitations','',
        f"The original 400-point evaluation design was stratified by provisional change class. Cloud-screened optical image chips were available for 115 of 116 August candidates. AI-assisted, predictor-blinded visual screening yielded {o['interpretable_n']} interpretable optical-date classifications, including {o['within_24h_n']} within 24 hours of NISAR. Unresolved cases were left without labels. Only {o['class_counts'].get('1',0)} interpretations were water-like, so this subset is strongly class-imbalanced. **No independently verified NISAR-time reference labels were established**. Optical screening was used descriptively only, not as an accuracy assessment; cloud selection, time offsets, and 20–30 m optical measurement support may bias comparison with 10 m SAR pixels.",'',
        '### 6. Processing uncertainty and sensitivity','',
        'August S-band was a PR product, whereas September S-band was UR. The S-band HV lower-tail anomaly therefore matters for temporal interpretation. Matched-grid numerical tests revealed no reported complex covariance positive-semidefinite violations in the preflight sample, but that is not a full product calibration test.']
    if hv:
        note.append(f"In spatial diagnostics ({hv['sample_n']:,} jointly valid samples), {pct(hv['sep_below_minus_40_fraction'])} of September S-HV values were below −40 dB, versus {pct(hv['aug_below_minus_40_fraction'])} in August. A fixed dB adjustment is not a defensible correction without independent noise/signal and product-processing evidence. These statistics are diagnostic only, not full-scene prevalence estimates.")
    note += ['', '### 7. Conclusions and next scientific steps','',
            'The study establishes a reproducible multi-date NISAR L/S GCOV processing and comparison pipeline. Its strongest observation is the large discrepancy in provisional temporal water-class assignments across two co-acquired frequencies, compounded by a pronounced September S-HV lower tail. The method is suitable as a transparent baseline for testing physics-guided fusion, but it does not demonstrate improved accuracy or verified hydrological change. Future work requires temporally matched, independently adjudicated reference water labels; validation of mixed land/water boundaries; a source-level UR/PR radiometric comparability assessment; and ablations of cross-polarized/covariance features under controlled equal-support evaluation.','',
            '### Reproducibility and source attribution','',
            'NISAR L2 GCOV sources: NASA Earthdata/ASF for L-band and ISRO NRSC Bhoonidhi for S-band. Computed results are traceable to the saved Phase 3G JSON and GeoTIFF outputs and Phase 3H/3J quality diagnostics. The original raw HDF5 files remain unmodified. Final figure captions must retain the exploratory/uncertain status even if warnings are omitted from the graphic layout.','']
    return '\n'.join(note)

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode',choices=['preflight','build'],default='preflight')
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--output-dir',type=Path,default=OUT)
    a=p.parse_args()
    root=a.root
    gpath=root/'outputs/phase3g/phase3g_report.json'
    bpath=root/'outputs/phase4b/phase4b_AI_screening_diagnostic_report.json'
    cpath=root/'outputs/phase4b/phase4b_AI_screened_400.csv'
    hpath=root/'outputs/phase3h/phase3h_report.json'
    g,b=load_json(gpath),load_json(bpath)
    records=read_csv(cpath)
    h=load_json(hpath) if hpath.is_file() else None
    s=analyze(g,b,records,h)
    print('PHASE-WATER 4C audit passed | full-scene 4-way joint valid pixels:',s['fourway_n'])
    print('Change-class L/S disagreement:',pct(s['change_disagreement_frac']))
    print('AI optical-date interpretations:',s['optical']['interpretable_n'],'| confirmed NISAR-time reference:',s['optical']['verified_reference_labels_n'])
    print('September S-HV PR/UR comparability: NOT VERIFIED')
    if a.mode=='preflight':
        print('PREFLIGHT PASS: no files modified. Now run --mode build.')
        return
    out=a.output_dir if a.output_dir.is_absolute() else root/a.output_dir
    out.mkdir(parents=True,exist_ok=True)
    fields=['method']+[k+'_pixels' for k in CLASSES]+[k+'_km2' for k in CLASSES]+['aug_classified_water_km2','sep_classified_water_km2','net_classified_change_km2']
    write_csv(out/'phase4c_native_change_statistics.csv',fields,s['rows'])
    optical=[]
    for k in ('within24h','within72h','all_usable_optical_dates'):
        d=b['diagnostic_only'][k]
        row={'optical_subset':k,'optical_date_interpretable_n':d['n'],
             'optical_water_like_n':d['optical_class_counts'].get('1',0),
             'optical_nonwater_like_n':d['optical_class_counts'].get('0',0),
             'official_independent_reference_n':s['optical']['verified_reference_labels_n'],
             'interpretation':'DESCRIPTIVE_SCREEN_ONLY_NOT_ACCURACY'}
        for m in ('L','S','LS_equalweight'):
            st=d['model_optical_screen_agreement'][m]
            row[m+'_agreement_count']=st['agreement_count']
            row[m+'_agreement_fraction_descriptive']=st['agreement_fraction_descriptive_only']
        optical.append(row)
    write_csv(out/'phase4c_optical_diagnostics_not_accuracy.csv',list(optical[0]),optical)
    (out/'phase4c_verified_results_and_limitations.md').write_text(make_markdown(s),encoding='utf-8')
    (out/'phase4c_technical_note_draft.md').write_text(make_note(s),encoding='utf-8')
    report={
        'phase':'4C','status':'ENGINEERING_CHECKS_PASS__NO_INDEPENDENT_ACCURACY',
        'generated_utc':datetime.now(timezone.utc).isoformat(),
        'four_scene_joint_valid_pixels':s['fourway_n'],
        'four_scene_joint_valid_km2':decimal(s['fourway_area_km2']),
        'L_vs_S_fourclass_change_disagreement_percent':decimal(s['change_disagreement_frac']*100,4),
        'methods':s['method_stats'],'optical_screening':s['optical'],
        'september_S_HV_tail':s['hv'],
        'limitations':['No verified SAR-time water labels or performance metrics',
                       'AI optical-date labels and agreement are descriptive; not classifier accuracy',
                       'S September UR vs August PR processing comparability unresolved',
                       'Do not interpret class-area differences as measured water gain/loss',
                       'No calibrated water probability or adaptive-fusion superiority demonstrated'],
        'source_checksums_sha256':{str(p.relative_to(root)):sha256(p) for p in (gpath,bpath,cpath) if p.is_file()},
        'figures_in_local_workspace':[str(x) for x in sorted((root/'outputs/phase4a_high_quality').glob('*.png'))]
    }
    (out/'phase4c_verified_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    for f in sorted(out.iterdir()):print('Saved:',f)
    print('PHASE 4C COMPLETE: technical note + auditable results tables. NO ACCURACY CLAIM.')

if __name__=='__main__':
    try:main()
    except (ValueError,KeyError,TypeError,ArithmeticError,OSError) as e:
        print(f'ERROR: {type(e).__name__}: {e}',file=sys.stderr);sys.exit(2)
