#!/usr/bin/env python3
"""Batch export of all Phase 3G NISAR L/S water classification and change maps.

Usage (Windows / VS Code):
  python NISAR_Export_All_Maps.py --mode preflight
  python NISAR_Export_All_Maps.py --mode export

Source data MUST be the georeferenced Phase 3G TIFFs. Cropped PNG screenshots
are never used to reconstruct geospatial coordinates. The exported smaller
GeoTIFF overviews are georeferenced but NOT 10 m; full-resolution originals
are copied by default into GeoTIFF/Native_10m/.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.windows import Window

NODATA = 255
SOURCE_NAMES = {
    'L_aug': 'phase3g_aug_L_only_water.tif',
    'L_sep': 'phase3g_sep_L_only_water.tif',
    'S_aug': 'phase3g_aug_S_only_water.tif',
    'S_sep': 'phase3g_sep_S_only_water.tif',
    'F_aug': 'phase3g_aug_LS_equalweight_water.tif',
    'F_sep': 'phase3g_sep_LS_equalweight_water.tif',
    'L_change': 'phase3g_L_only_change_4class.tif',
    'S_change': 'phase3g_S_only_change_4class.tif',
    'F_change': 'phase3g_LS_equalweight_change_4class.tif',
    'disagreement': 'phase3g_L_vs_S_change_disagreement.tif',
}
DESCRIPTIONS = {
    'L_aug': ('Water_Masks', 'L_Band_2026-08-19', 'L-band', '19 August 2026', 'water'),
    'L_sep': ('Water_Masks', 'L_Band_2026-09-24', 'L-band', '24 September 2026', 'water'),
    'S_aug': ('Water_Masks', 'S_Band_2026-08-19', 'S-band', '19 August 2026', 'water'),
    'S_sep': ('Water_Masks', 'S_Band_2026-09-24', 'S-band', '24 September 2026', 'water'),
    'F_aug': ('Water_Masks', 'LS_EqualWeight_2026-08-19', 'L+S equal-weight', '19 August 2026', 'water'),
    'F_sep': ('Water_Masks', 'LS_EqualWeight_2026-09-24', 'L+S equal-weight', '24 September 2026', 'water'),
    'L_change': ('Change_Maps', 'L_Band_Aug_Sep_Change', 'L-band', '19 August - 24 September 2026', 'change'),
    'S_change': ('Change_Maps', 'S_Band_Aug_Sep_Change', 'S-band', '19 August - 24 September 2026', 'change'),
    'F_change': ('Change_Maps', 'LS_EqualWeight_Aug_Sep_Change', 'L+S equal-weight', '19 August - 24 September 2026', 'change'),
    'disagreement': ('Change_Maps', 'L_vs_S_Change_Disagreement', 'L versus S', '19 August - 24 September 2026', 'disagreement'),
}
NEW_WATER = {
    'L_change': ('L_Band_Newly_Classified_Water', 'L-band'),
    'S_change': ('S_Band_Newly_Classified_Water', 'S-band'),
    'F_change': ('LS_EqualWeight_Newly_Classified_Water', 'L+S equal-weight'),
}
PALETTES = {
    'water': ([0,1,255], ['#e7e5dd','#196fae','#ffffff'], ['Non-water','Classified water','Invalid']),
    'change': ([0,1,2,3,255], ['#dedbd3','#196fae','#12a585','#ef9338','#ffffff'],
               ['Stable non-water','Stable water','Newly classified water','Classified water loss','Invalid']),
    'disagreement': ([0,1,255], ['#dedbd3','#a83278','#ffffff'],
                     ['Same change class','Different change class','Invalid']),
    'new': ([0,1,255], ['#e7e5dd','#12a585','#ffffff'],
            ['Other jointly valid pixels','Newly classified water','Invalid']),
}
VALID_VALUES = {'water':{0,1,255}, 'change':{0,1,2,3,255},
                'disagreement':{0,1,255},'new':{0,1,255}}


def get_args():
    p=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', choices=['preflight','export'], default='export')
    p.add_argument('--source-dir', type=Path, default=Path(r'D:\Hackathon\outputs\phase3g'))
    p.add_argument('--output-dir', type=Path, default=Path(r'D:\Hackathon\NISAR_GitHub_Results\02_Maps_Results'))
    p.add_argument('--max-overview-dim', type=int, default=2400,
                   help='Maximum dimension for true georeferenced GIS overviews (NOT 10 m).')
    p.add_argument('--native-mode', choices=['copy','reference'], default='copy',
                   help='copy: copy all 10 native 10 m TIFFs; reference: point to original 10 m TIFFs.')
    p.add_argument('--skip-native-new', action='store_true',
                   help='Skip creating three native 10 m newly classified water masks.')
    p.add_argument('--no-pdf', action='store_true', help='Skip PDF map atlas.')
    p.add_argument('--comparison-figure', type=Path,
                   default=Path(r'D:\Hackathon\phase4a_nisar_3x3_comparison.png'))
    return p.parse_args()


def validate_source(path: Path):
    with rasterio.open(path) as ds:
        if ds.count != 1 or ds.crs is None or ds.crs.to_epsg() != 32645:
            raise ValueError(f'Expected one-band EPSG:32645 GeoTIFF: {path} | CRS {ds.crs}')
        if not (abs(abs(ds.res[0])-10)<0.05 and abs(abs(ds.res[1])-10)<0.05):
            raise ValueError(f'Expected 10 m native resolution for {path}; got {ds.res}')
        if ds.nodata != 255:
            raise ValueError(f'Expected nodata=255 for {path}; got {ds.nodata}')
        if ds.transform.e >= 0 or ds.transform.b != 0 or ds.transform.d != 0:
            raise ValueError(f'Unsupported orientation in {path}: {ds.transform}')
        return {'crs':str(ds.crs),'resolution_m':list(map(abs,ds.res)),
                'width':ds.width,'height':ds.height, 'transform':list(ds.transform)[:6],
                'bounds_m':[ds.bounds.left,ds.bounds.bottom,ds.bounds.right,ds.bounds.top],
                'size_bytes':path.stat().st_size}


def preflight(args):
    if args.max_overview_dim<300:
        raise ValueError('--max-overview-dim must be >= 300')
    missing=[str(args.source_dir/name) for name in SOURCE_NAMES.values()
             if not (args.source_dir/name).is_file()]
    if missing:
        raise FileNotFoundError('Missing required Phase 3G GeoTIFFs:\n  ' + '\n  '.join(missing)
            + '\nRun phase3g_four_scene_dualfrequency_change.py first; do not substitute PNGs.')
    meta={key:validate_source(args.source_dir/name) for key,name in SOURCE_NAMES.items()}
    for k,v in meta.items():
        if v['width']!=meta['L_aug']['width'] or v['height']!=meta['L_aug']['height'] or any(
            abs(x-y)>1e-5 for x,y in zip(v['transform'],meta['L_aug']['transform'])):
            raise ValueError('Raster grid differs from L_aug; cannot do a fair joint-pixel comparison: '+k)
    print('PREFLIGHT PASSED: 10 native georeferenced rasters, all EPSG:32645, 10 m, same grid')
    print('Raster dimensions:',meta['L_aug']['width'],'x',meta['L_aug']['height'])
    print('Source TIFFs total: %.2f GB' % (sum(v['size_bytes'] for v in meta.values())/1e9))
    print('Source folder:',args.source_dir)
    if args.native_mode=='copy':
        size_sources=sum(v['size_bytes'] for v in meta.values())
        # Derived water masks may compress differently from source change rasters.
        reserve=sum(max(25_000_000, meta[k]['size_bytes']) for k in NEW_WATER) if not args.skip_native_new else 0
        disk=shutil.disk_usage(args.output_dir.anchor or args.output_dir.parent)
        print('Approximate additional storage required: %.2f GB (may vary)' % ((size_sources+reserve)/1e9))
        print('Available storage:',round(disk.free/1e9,2),'GB')
        if disk.free < size_sources+reserve+256_000_000:
            raise OSError('Not enough free space for native copies. Use --native-mode reference to keep original native TIFFs in place.')
    return meta


def downsample(ds, max_dim, kind):
    factor=max(1,math.ceil(max(ds.width,ds.height)/max_dim))
    w=math.ceil(ds.width/factor)
    h=math.ceil(ds.height/factor)
    data=ds.read(1,out_shape=(h,w),resampling=Resampling.nearest)
    values=set(int(x) for x in np.unique(data))
    if not values.issubset(VALID_VALUES[kind]):
        raise ValueError(f'Unexpected categorical values {values} in {ds.name}')
    transform=ds.transform*Affine.scale(ds.width/w,ds.height/h)
    return data,transform


def write_geotiff(path, arr, transform, crs, kind, native_name, source_resolution, label):
    path.parent.mkdir(parents=True,exist_ok=True)
    profile={'driver':'GTiff','height':arr.shape[0],'width':arr.shape[1],'count':1,
             'dtype':'uint8','nodata':255,'crs':crs,'transform':transform,
             'compress':'deflate','predictor':2,'tiled':True,'blockxsize':256,
             'blockysize':256,'BIGTIFF':'IF_SAFER'}
    with rasterio.open(path,'w',**profile) as dst:
        dst.write(arr,1)
        dst.update_tags(SOURCE='NISAR L2 GCOV derived from original Phase 3G classification',
                        ACQUISITION='19 August 2026 and/or 24 September 2026',
                        DATA_DESCRIPTION=label,
                        ORIGINAL_NATIVE_SOURCE=native_name,
                        ORIGINAL_NATIVE_RESOLUTION_METERS=str(source_resolution),
                        PRODUCT='Categorical georeferenced browse overview, nearest-neighbor',
                        CLASS_CODING=kind,
                        STATUS='Exploratory classification; not an accuracy-validated flood extent')
    with rasterio.open(path) as chk:
        if chk.crs!=crs or not chk.transform.almost_equals(transform) or chk.nodata!=255:
            raise RuntimeError('Georeferencing verification FAILED for '+str(path))


def robust_save_png(fig,path,fallback_name):
    """Render in memory to avoid PIL filename write Errno 22 on certain Windows systems."""
    buf=io.BytesIO()
    fig.savefig(buf,format='png',dpi=180,facecolor='white')
    content=buf.getvalue()
    if not content.startswith(bytes.fromhex('89504e470d0a1a0a')):
        raise RuntimeError('Matplotlib produced invalid PNG bytes')
    candidates=[path,Path(r'D:\Hackathon\NISAR_Water_Maps')/fallback_name]
    errors=[]
    for target in candidates:
        try:
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('wb') as f: f.write(content)
            return target
        except OSError as e:
            errors.append(f'{target}: {e}')
    raise OSError('Could not save PNG to either path: '+' | '.join(errors))


def make_map(arr,kind,title,transform,crs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap,BoundaryNorm
    from matplotlib.patches import Patch
    vals,colors,labels=PALETTES[kind]
    plot_arr=np.full(arr.shape,len(vals)-1,dtype=np.uint8)
    for idx,code in enumerate(vals):
        plot_arr[arr==code]=idx
    left=transform.c/1000
    top=transform.f/1000
    right=(transform.c+transform.a*arr.shape[1])/1000
    bottom=(transform.f+transform.e*arr.shape[0])/1000
    fig,ax=plt.subplots(figsize=(9.5,8.5))
    ax.imshow(plot_arr,cmap=ListedColormap(colors),
        norm=BoundaryNorm(np.arange(-0.5,len(vals)+.5,1),len(vals)),
        extent=(left,right,bottom,top),origin='upper',interpolation='nearest')
    ax.set_aspect('equal',adjustable='box')
    ax.set_xlabel('Easting (km, UTM 45N)',fontsize=10)
    ax.set_ylabel('Northing (km, UTM 45N)',fontsize=10)
    # Keep heading and source/CRS text in separate top bands. No footer.
    fig.suptitle(title.replace(' | ', '\n', 1),fontsize=13,fontweight='bold',y=.975)
    fig.text(.5,.895,'Source: NISAR L2 GCOV  |  EPSG:32645  |  Native grid: 10 m',
             ha='center',va='center',fontsize=9,color='#4e5c62')
    # Grid north and a 50-km (or proportionally shorter) UTM scale bar.
    ax.annotate('',xy=(.93,.95),xytext=(.93,.81),xycoords='axes fraction',
                arrowprops=dict(arrowstyle='-|>',lw=1.4,color='#193847'))
    ax.text(.93,.965,'Grid N',transform=ax.transAxes,ha='center',va='bottom',fontsize=8,color='#193847')
    width_km=right-left
    bar_km=50.0 if width_km>=110 else 1.0 if width_km>=3 else max(.1,round(width_km/4,1))
    x0=left+width_km*.065
    y0=bottom+(top-bottom)*.075
    ax.plot([x0,x0+bar_km],[y0,y0],color='#193847',lw=2.5,solid_capstyle='butt')
    ax.plot([x0,x0],[y0-(top-bottom)*.01,y0+(top-bottom)*.01],color='#193847',lw=1.5)
    ax.plot([x0+bar_km,x0+bar_km],[y0-(top-bottom)*.01,y0+(top-bottom)*.01],color='#193847',lw=1.5)
    ax.text(x0+bar_km/2,y0+(top-bottom)*.025,f'{bar_km:g} km (UTM)',
            fontsize=8.4,fontweight='bold',ha='center',color='#193847')
    patches=[Patch(facecolor=c,edgecolor='#777777' if v==255 else 'none',label=lab)
             for v,c,lab in zip(vals,colors,labels)]
    fig.legend(handles=patches,loc='lower center',bbox_to_anchor=(.5,.036),
               ncol=3 if len(vals)<5 else 3,frameon=False,fontsize=9)
    fig.subplots_adjust(left=.12,right=.97,top=.84,bottom=.17)
    return fig


def make_native_new(src, destination, name):
    """Stream each native-resolution window: no full-scene memory allocation."""
    destination.parent.mkdir(parents=True,exist_ok=True)
    profile=src.profile.copy()
    profile.update(driver='GTiff',count=1,dtype='uint8',nodata=255,
                   compress='deflate',predictor=2,tiled=True,blockxsize=512,
                   blockysize=512,BIGTIFF='IF_SAFER')
    with rasterio.open(destination,'w',**profile) as dst:
        for _,window in src.block_windows(1):
            old=src.read(1,window=window)
            new=np.where(old==255,255,np.where(old==2,1,0)).astype(np.uint8)
            dst.write(new,1,window=window)
        dst.update_tags(SOURCE='NISAR Phase 3G four-class native change GeoTIFF',
                        CLASS_CODES='0=other jointly valid;1=newly classified water;255=invalid',
                        CLASSIFICATION_STATUS='provisional, not independently validated',
                        DERIVED_FROM=name)
    with rasterio.open(destination) as check:
        if check.crs!=src.crs or not check.transform.almost_equals(src.transform) or check.res!=src.res:
            raise RuntimeError('Native new water GeoTIFF georeferencing verification FAILED')


def compute_area_stats(srcdir, output, source_meta):
    """Prefer original Phase 3G exact statistics; compute if original CSV is absent."""
    orig=srcdir/'phase3g_change_class_areas.csv'
    if orig.exists():
        shutil.copy2(orig,output)
        return 'copied original native 10m Phase 3G class area statistics'
    data=[]
    px_km2=100/1e6
    for key,label in [('L_change','L_only'),('S_change','S_only'),('F_change','LS_equalweight')]:
        counts=np.zeros(4,dtype=np.int64)
        with rasterio.open(srcdir/SOURCE_NAMES[key]) as src:
            px_km2=abs(src.transform.a*src.transform.e)/1e6
            for _,win in src.block_windows(1):
                a=src.read(1,window=win)
                v=a[a!=255]
                counts+=np.bincount(v.astype(np.uint8),minlength=4)[:4]
        for cls,cname in enumerate(['stable_nonwater','stable_water','water_gain','water_loss']):
            data.append({'method':label,'change_class':cname,'pixel_count':int(counts[cls]),
                         'area_km2':round(int(counts[cls])*px_km2,4),
                         'reference_status':'UNVALIDATED_EXPLORATORY'})
    with output.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    return 'computed from original native 10m change rasters'


def run(args,source_meta):
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    root=args.output_dir
    geobrowse=root/'GeoTIFF'/'Browse_Overviews'
    geonative=root/'GeoTIFF'/'Native_10m'
    for d in ['Water_Masks','Change_Maps','Newly_Detected_Water','Comparison_Overview']:
        (root/d).mkdir(parents=True,exist_ok=True)
    geobrowse.mkdir(parents=True,exist_ok=True)
    geonative.mkdir(parents=True,exist_ok=True)
    pdf_file=root/'NISAR_All_Maps_Atlas.pdf'
    atlas=None if args.no_pdf else PdfPages(pdf_file)
    manifest=[]
    try:
        for index,(key,srcname) in enumerate(SOURCE_NAMES.items(),1):
            folder,stem,sensor,date,kind=DESCRIPTIONS[key]
            srcpath=args.source_dir/srcname
            print(f'[{index}/10] {sensor} - {kind}: {srcname}',flush=True)
            with rasterio.open(srcpath) as src:
                data,trans=downsample(src,args.max_overview_dim,kind)
                preview_path=geobrowse/f'{stem}_GEOREFERENCED_OVERVIEW.tif'
                write_geotiff(preview_path,data,trans,src.crs,kind,srcname,10,sensor)
                fullpath=geonative/f'{stem}_NATIVE_10m.tif'
                if args.native_mode=='copy':
                    if not fullpath.exists() or fullpath.stat().st_size!=srcpath.stat().st_size:
                        shutil.copy2(srcpath,fullpath)
                    with rasterio.open(fullpath) as verify:
                        if verify.crs!=src.crs or not verify.transform.almost_equals(src.transform):
                            raise RuntimeError('Native TIFF copy georeferencing mismatch: '+str(fullpath))
                else:
                    fullpath=srcpath
                title=f'NISAR {sensor} | {"Water classification" if kind=="water" else "Water-change classification" if kind=="change" else "Change-class disagreement"} | {date}'
                fig=make_map(data,kind,title,trans,src.crs)
                pngpath=robust_save_png(fig,root/folder/f'{stem}_MAP.png',f'{stem}_MAP.png')
                if atlas: atlas.savefig(fig,dpi=140,facecolor='white')
                plt.close(fig)
                manifest.append({'map_type':kind,'sensor':sensor,'date':date,'source_native':str(srcpath),
                    'native_10m':str(fullpath),'georeferenced_overview':str(preview_path),
                    'overview_resolution_m':[round(abs(trans.a),3),round(abs(trans.e),3)],
                    'cartographic_png':str(pngpath),'crs':'EPSG:32645','native_resolution_m':10,
                    'native_github_upload_note':'Native TIFF often exceeds GitHub browser limit; host externally if necessary.'})
                if key in NEW_WATER:
                    newstem,newsensor=NEW_WATER[key]
                    newnative=geonative/f'{newstem}_NATIVE_10m.tif'
                    if not args.skip_native_new:
                        print('    Deriving newly classified water at true native 10 m resolution...',flush=True)
                        make_native_new(src,newnative,srcname)
                    new=np.where(data==255,255,np.where(data==2,1,0)).astype(np.uint8)
                    newoverview=geobrowse/f'{newstem}_GEOREFERENCED_OVERVIEW.tif'
                    write_geotiff(newoverview,new,trans,src.crs,'new',srcname,10,newsensor)
                    newfig=make_map(new,'new',f'NISAR {newsensor} | Newly classified water | {date}',trans,src.crs)
                    newpng=robust_save_png(newfig,root/'Newly_Detected_Water'/f'{newstem}_MAP.png',f'{newstem}_MAP.png')
                    if atlas: atlas.savefig(newfig,dpi=140,facecolor='white')
                    plt.close(newfig)
                    manifest.append({'map_type':'newly_classified_water','sensor':newsensor,'date':date,
                        'source_native':str(srcpath),'native_10m':str(newnative) if not args.skip_native_new else 'NOT GENERATED',
                        'georeferenced_overview':str(newoverview),'cartographic_png':str(newpng),
                        'crs':'EPSG:32645','native_resolution_m':10,
                        'note':'Newly classified means Aug non-water to Sep water, not independently validated inundation.'})
        stats=root/'Area_Statistics.csv'
        stats_method=compute_area_stats(args.source_dir,stats,source_meta)
        if args.comparison_figure.is_file():
            shutil.copy2(args.comparison_figure,root/'Comparison_Overview'/'Original_3x3_NISAR_Comparison.png')
        with (root/'Geospatial_Output_Manifest.json').open('w',encoding='utf-8') as f:
            json.dump({'source_dir':str(args.source_dir),'native_mode':args.native_mode,
                'source_raster_metadata':source_meta,'area_statistics_provenance':stats_method,
                'water_change_status':'Provisional classifications; radiometric and independent accuracy checks required.',
                'maps':manifest},f,indent=2)
        readme=(root/'README.md')
        readme.write_text('''# NISAR water maps and change detection outputs\n\n'''
            'All maps derive from original Phase 3G 10 m georeferenced GeoTIFFs (EPSG:32645).\n\n'
            '* Water_Masks: 6 PNGs (Aug/Sep L, S, equal-weight L+S).\n'
            '* Change_Maps: 3 four-class PNGs plus L/S disagreement PNG.\n'
            '* Newly_Detected_Water: 3 PNGs and derived GeoTIFF masks coded 0=other, 1=new, 255=invalid.\n'
            '* GeoTIFF/Browse_Overviews: nearest-neighbour sampled, georeferenced GIS TIFFs (coarser than 10 m if downsampled).\n'
            '* GeoTIFF/Native_10m: 10 exact copies of original native TIFFs when --native-mode copy is selected, plus 3 native derived new-water masks.\n'
            '* Geospatial_Output_Manifest.json: source lineage, native grid and browse resolution.\n'
            '* Area_Statistics.csv: original phase3g summary, or windowed calculation if missing.\n\n'
            'Do not interpret differences in water class labels as validated flood extent or classification accuracy.\n'
            'September S-HV urgent-response radiometry requires further evaluation.\n'
            'Native TIFFs may exceed GitHub browser upload limits: use separate hosting and add links to the repository README.\n',encoding='utf-8')
    finally:
        if atlas: atlas.close()
    print('\nBATCH EXPORT COMPLETE')
    print('Maps PNG: 6 water + 3 change + 3 newly classified + 1 disagreement = 13')
    print('GeoTIFF browse GIS layers: 13')
    print('Original 10 m TIFFs:', '10 copies' if args.native_mode=='copy' else 'referenced at original locations')
    print('New 10 m water-gain TIFFs:', '3' if not args.skip_native_new else 'skipped')
    print('Atlas PDF:',str(pdf_file) if atlas else 'not requested')
    print('Output:',root)
    print('CHECK: these are exploratory classifications, not validated hydrological changes.')


def main():
    args=get_args()
    try:
        source_meta=preflight(args)
        if args.mode=='preflight':return
        run(args,source_meta)
    except Exception as exc:
        print('\nERROR:',type(exc).__name__,str(exc),file=sys.stderr)
        sys.exit(1)

if __name__=='__main__': main()
