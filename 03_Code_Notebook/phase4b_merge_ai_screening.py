#!/usr/bin/env python3
r"""PHASE-WATER: Safely merge AI-assisted optical-date screen into existing Phase4B 400-point review.

Do not confuse AI optical-date interpretations with NISAR-time independent reference truth.
No original reference_aug_water entries are written, populated, or replaced by this script.

Windows / PowerShell (run inside D:\Hackathon):
  .\.venv\Scripts\python.exe phase4b_merge_ai_screening.py --mode preflight
  .\.venv\Scripts\python.exe phase4b_merge_ai_screening.py --mode run

Put `phase4b_ai_august_review_116.csv` and this .py in D:\Hackathon.
This writes the 400-point labelled *screening* comparison file and a self-describing
exploratory SAR/optical agreement summary, NOT final reference validation.
"""
from __future__ import annotations
import argparse,csv,json,math,sys
from pathlib import Path
from collections import Counter,defaultdict

ROOT=Path('.')
SCREEN=Path('phase4b_ai_august_review_116.csv')
TEMPLATE=Path('outputs/phase4b/phase4b_blinded_reference_template.csv')
QUEUE=Path('outputs/phase3e/phase3e_august_400_review_queue.csv')
OUT=Path('outputs/phase4b')
ADDED={
  'ai_screen_optical_date_water': 'provisional_aug_optical_water_label',
  'ai_screen_category': 'ai_visual_interpretation',
  'ai_screen_confidence': 'ai_confidence',
  'ai_screen_optical_gap_hours': 'offset_hours_from_nisar',
  'ai_screen_temporal_flag': 'temporal_eligibility_flag',
  'ai_screen_visual_notes': 'review_explanation',
  'ai_screen_source_zip_status': 'validation_status',
}

def read(p):
    if not p.is_file():raise FileNotFoundError('Required input missing: '+str(p.resolve()))
    with p.open('r',encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if not reader.fieldnames:raise ValueError('Missing CSV header: '+str(p))
        rows=list(reader);return rows,list(reader.fieldnames)

def make_index(rows,path):
    ids=[r.get('sample_id','').strip() for r in rows]
    if any(not x for x in ids) or len(set(ids))!=len(ids):
        raise ValueError('Duplicate or missing sample ID in '+str(path))
    return {r['sample_id']:r for r in rows}

def assert_consistent(base,screen):
    for sid,s in screen.items():
        if sid not in base:raise ValueError(f'Screened sample {sid} not found in template')
        ref=base[sid]
        for field in ('latitude','longitude'):
            if not math.isclose(float(s[field]),float(ref[field]),abs_tol=1e-5):
                raise ValueError(f'{sid}: {field} does not match; refusing merge')
        if (ref.get('optical_candidate_scene','').strip() and
            ref['optical_candidate_scene'].strip()!=s['optical_scene_id'].strip()):
            raise ValueError(f'{sid}: scene mismatch; refusing merge')
        if (ref.get('optical_candidate_datetime_utc','').strip() and
            ref['optical_candidate_datetime_utc'].strip()!=s['optical_datetime_utc'].strip()):
            raise ValueError(f'{sid}: optical timestamp mismatch; refusing merge')

def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode',choices=('preflight','run'),default='preflight')
    ap.add_argument('--screen',type=Path,default=SCREEN)
    ap.add_argument('--template',type=Path,default=TEMPLATE)
    ap.add_argument('--queue',type=Path,default=QUEUE)
    ap.add_argument('--out',type=Path,default=OUT)
    a=ap.parse_args()
    reviewed,fields=read(a.screen)
    original,origfields=read(a.template)
    queue,queuefields=read(a.queue)
    r_index=make_index(reviewed,a.screen);o_index=make_index(original,a.template);q_index=make_index(queue,a.queue)
    if len(original)!=400:raise ValueError('Expected exactly 400 original sample rows; got '+str(len(original)))
    if len(r_index)!=116:raise ValueError('Expected 116 screened image candidates; got '+str(len(r_index)))
    if set(q_index)!=set(o_index):raise ValueError('Phase 3E sample ID set differs from reference template')
    assert_consistent(o_index,r_index)
    if any(k in origfields for k in ADDED):raise ValueError('Template already includes output columns; not replacing it')
    official_existing=sum(str(row.get('reference_aug_water','')).strip() in ('0','1') for row in original)
    print(f'PHASE-WATER AI review merge checks PASSED | 400 original points | 116 screened | {official_existing} pre-existing official reference entries untouched')
    if a.mode=='preflight':return
    a.out.mkdir(parents=True,exist_ok=True)
    merged=[];comparison=[]
    for row in original:
        sid=row['sample_id'];o=dict(row);scr=r_index.get(sid)
        for field,src in ADDED.items():o[field]=scr[src] if scr else ''
        merged.append(o)
        if scr and scr['provisional_aug_optical_water_label'] in {'0','1'}:
            q=q_index[sid]
            gap=float(scr['offset_hours_from_nisar'])
            cr={'sample_id':sid,'reference_status':'AI_SCREEN_AT_OPTICAL_IMAGE_DATE_NOT_NISAR_GROUND_TRUTH',
                'optical_date_label':scr['provisional_aug_optical_water_label'],
                'optical_gap_hours':scr['offset_hours_from_nisar'],
                'optical_confidence':scr['ai_confidence'],
                'LS_class':q.get('LS_class','')}
            for model,source in [('L','L_water_prediction'),('S','S_water_prediction'),('LS_equalweight','LS_fusion_prediction')]:
                p=q.get(source,'').strip();cr[model+'_prediction']=p
                cr[model+'_agrees_with_optical_screen']=(int(p)==int(scr['provisional_aug_optical_water_label'])) if p in {'0','1'} else ''
            comparison.append(cr)
    target=a.out/'phase4b_AI_screened_400.csv'
    outfields=origfields+list(ADDED)
    with target.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=outfields);w.writeheader();w.writerows(merged)
    target2=a.out/'phase4b_AI_optical_vs_SAR_diagnostic.csv'
    if comparison:
        with target2.open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(comparison[0]));w.writeheader();w.writerows(comparison)
    def stats(rows):
        classes=Counter(r['optical_date_label'] for r in rows)
        totals={}
        for model in ('L','S','LS_equalweight'):
            ok=[r[model+'_agrees_with_optical_screen'] for r in rows if isinstance(r[model+'_agrees_with_optical_screen'],bool)]
            totals[model]={'comparable_n':len(ok),'agreement_count':sum(ok),
                           'agreement_fraction_descriptive_only':round(sum(ok)/len(ok),6) if ok else None}
        return {'n':len(rows),'optical_class_counts':dict(classes),'model_optical_screen_agreement':totals}
    buckets={'within24h':[r for r in comparison if float(r['optical_gap_hours'])<=24],
             'within72h':[r for r in comparison if float(r['optical_gap_hours'])<=72],
             'all_usable_optical_dates':comparison}
    report={'phase':'AI-assisted optical image DATE descriptive comparison, not independent reference validation',
            'source_screen':str(a.screen),'source_400_template':str(a.template),
            'reference_label_fields_modified':False,'official_reference_labels_preserved':official_existing,
            'screened_n':len(reviewed),
            'diagnostic_only':{k:stats(v) for k,v in buckets.items()},
            'no_full_scene_accuracy':True,'no_verified_water_area_change':True,
            'warnings':[
               'Model-optical screen agreement is NOT classification accuracy because imagery is not SAR-time independently verified ground truth.',
               'Subset selected by cloud and optical availability, and original sampling is stratified by L-band change; do not scale to full scene.',
               'AI visual-screen labels are not original expert ground-truth; do not insert these into reference_aug_water or call official --mode evaluate with these labels.',
               'At 24h or longer gaps, genuine fast-moving water may change between SAR and optical acquisition; spatial supports also differ.',
               'S Sep UR/PR HV anomaly remains unresolved; August optical diagnostic does not establish September temporal reliability.'
            ]}
    target3=a.out/'phase4b_AI_screening_diagnostic_report.json'
    target3.write_text(json.dumps(report,indent=2),encoding='utf8')
    print('Saved: '+str(target)+' | '+str(target2)+' | '+str(target3))
    print('IMPORTANT: official reference_aug_water untouched; no ground-truth labels or official accuracy generated.')
    print('Diagnostic counts:',json.dumps({k:v['n'] for k,v in report['diagnostic_only'].items()}))

if __name__=='__main__':
    try:main()
    except (ValueError,OSError,KeyError) as exc:
        print('ERROR:',type(exc).__name__,exc,file=sys.stderr);sys.exit(2)
