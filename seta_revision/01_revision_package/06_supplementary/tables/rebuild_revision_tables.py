#!/usr/bin/env python3
"""Rebuild revision supplement tables from the retained consecutive-hour outputs.

No raw observations or external station records are loaded. Run from any cwd.
"""
from pathlib import Path
import hashlib,itertools,json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
EVIDENCE=HERE.parents[2]/'02_evidence_and_analysis'/'analyses'
N_BOOT=50000
PRIMARY='primary_solar_qc'
QC_ORDER=[PRIMARY,'secondary_solar_qc_unverified']
D=pd.read_csv(EVIDENCE/'consecutive_hour_rerun/canonical_station_metrics.csv')
CHECK=pd.read_csv(EVIDENCE/'consecutive_hour_primary_rerun/canonical_station_metrics.csv')
assert D[(D.threshold_wm2h==200)&D.tolerance_h.isin([1,2])].reset_index(drop=True).equals(CHECK)
KEY=['product','month','station_id','threshold_wm2h','tolerance_h']
assert not D.duplicated(KEY).any()

# Cluster resampling preserves all eligible station-month records within station.
# This reproduces the equally weighted station-month estimator even when a high
# threshold leaves different numbers of defined CSI values across stations.
def cluster_interval(delta,seed):
    station=delta.groupby(level='station_id').agg(['sum','count'])
    rng=np.random.default_rng(seed)
    picks=rng.integers(0,len(station),size=(N_BOOT,len(station)))
    boot=station['sum'].to_numpy()[picks].sum(1)/station['count'].to_numpy()[picks].sum(1)
    return np.quantile(boot,[.025,.975])

rows=[];sign=[]
for threshold in sorted(D.threshold_wm2h.unique()):
 for tol in sorted(D.tolerance_h.unique()):
  for qc in QC_ORDER:
   g=D[(D.threshold_wm2h==threshold)&(D.tolerance_h==tol)&(D.solar_qc_group==qc)]
   p=g.pivot(index=['station_id','month'],columns='product',values='csi')
   pair=p[['Tianji_2.5km','Tianji_27.5km']].dropna()
   delta=pair['Tianji_27.5km']-pair['Tianji_2.5km']
   seed=261112+int(tol) if (threshold==200 and qc==PRIMARY) else 261112+int(threshold)*10+int(tol)+(10000 if qc!=PRIMARY else 0)
   lo,hi=cluster_interval(delta,seed)
   rows.append(dict(contrast='Tianji_27.5km_minus_Tianji_2.5km',threshold_wm2h=threshold,tolerance_h=tol,solar_qc_group=qc,stations=delta.index.get_level_values('station_id').nunique(),paired_station_months=len(delta),mean_2_5km_csi=pair['Tianji_2.5km'].mean(),mean_27_5km_csi=pair['Tianji_27.5km'].mean(),mean_delta_csi=delta.mean(),ci95_low=lo,ci95_high=hi,fraction_station_months_with_loss=(delta<0).mean(),bootstrap_replicates=N_BOOT,bootstrap_seed=seed))
   # Exact sign flip of each station-mean contrast, preserving all of its months.
   station=delta.groupby(level='station_id').mean().to_numpy()
   signs=np.array(list(itertools.product([-1,1],repeat=len(station))))
   observed=station.mean()
   null=(signs*station).mean(1)
   pvalue=np.mean(np.abs(null)>=abs(observed)-1e-14)
   sign.append(dict(threshold_wm2h=threshold,tolerance_h=tol,solar_qc_group=qc,stations=len(station),paired_station_months=len(delta),station_mean_delta_csi=observed,sign_combinations=len(signs),p_sign_flip=pvalue))
S2=pd.DataFrame(rows);S2.to_csv(HERE/'canonical_resolution_effects.csv',index=False)
S9=pd.DataFrame(sign);order=np.argsort(S9.p_sign_flip.to_numpy());p=S9.p_sign_flip.to_numpy()[order];q=np.minimum.accumulate((p*len(p)/np.arange(1,len(p)+1))[::-1])[::-1];q=np.minimum(q,1);out=np.empty(len(p));out[order]=q;S9['p_fdr_bh']=out;S9.to_csv(HERE/'resolution_sign_flip_tests.csv',index=False)

# Station means use the same consecutive-hour 200/1h rows as S2.
old=pd.read_csv(HERE/'station_forest_table.csv')
meta=old[['station_id','hubei_site','solar_qc_group','terrain_class']].drop_duplicates('station_id')
g=D[(D.threshold_wm2h==200)&(D.tolerance_h==1)]
s=g.pivot(index=['station_id','month'],columns='product',values='csi')[['Tianji_2.5km','Tianji_27.5km']].groupby(level='station_id').mean().rename(columns={'Tianji_2.5km':'csi_2.5km','Tianji_27.5km':'csi_27.5km'}).reset_index()
s['delta_csi']=s['csi_27.5km']-s['csi_2.5km'];meta.merge(s,on='station_id').to_csv(HERE/'station_forest_table.csv',index=False)

# New declared estimands are copied verbatim to their publication tables.
for source,target in [('footprint_era5_common_mask_summary.csv','footprint_sensitivity.csv'),('fair_mask_common_month_product_summary.csv','common_month_event_metrics.csv'),('timestamp_hourly_alignment_station_bootstrap.csv','timestamp_alignment_station_bootstrap.csv')]:
    pd.read_csv(EVIDENCE/source).to_csv(HERE/target,index=False)

full=D[(D.threshold_wm2h==200)&D.tolerance_h.isin([1,2])&(D.solar_qc_group==PRIMARY)&D['product'].str.startswith('Tianji')]
summary=[]
for (product,tol),g in full.groupby(['product','tolerance_h']):
 r=dict(product=product,tolerance_h=tol,stations=g.station_id.nunique(),station_months=len(g))
 for x in ['obs_events','model_events','hits','misses','false_alarms']:r[x+'_sum']=int(g[x].sum())
 for x in ['csi','pod','far','frequency_bias']:r['macro_'+x]=g[x].mean()
 summary.append(r)
pd.DataFrame(summary).to_csv(HERE/'three_month_primary_event_metrics.csv',index=False)
lomo=[]
for tol in [1,2]:
 for omit in ['2025-04','2025-07','2025-10']:
  g=full[(full.tolerance_h==tol)&(full.month!=omit)]
  p=g.pivot(index=['station_id','month'],columns='product',values='csi')
  lomo.append(dict(omitted_month=omit,remaining_months=';'.join(sorted(g.month.unique())),tolerance_h=tol,stations=7,station_months=14,csi_2_5km=p['Tianji_2.5km'].mean(),csi_27_5km=p['Tianji_27.5km'].mean(),paired_delta_csi=(p['Tianji_27.5km']-p['Tianji_2.5km']).mean()))
pd.DataFrame(lomo).to_csv(HERE/'leave_one_month_out_consecutive.csv',index=False)

# S6 retains the deterministic inventory, without algorithm-dependent clusters.
a=pd.read_csv(HERE/'coarsening_pathway_regridding_ablation.csv')
a=a[['coarsening_pathway','regridding_method','nominal_resolution_km','event_count','up_share']]
a.to_csv(HERE/'coarsening_pathway_regridding_ablation.csv',index=False)

inputs=['consecutive_hour_rerun/canonical_station_metrics.csv','consecutive_hour_primary_rerun/canonical_station_metrics.csv','footprint_era5_common_mask_summary.csv','fair_mask_common_month_product_summary.csv','timestamp_hourly_alignment_station_bootstrap.csv']
meta=dict(source_files={n:hashlib.sha256((EVIDENCE/n).read_bytes()).hexdigest() for n in inputs},n_bootstrap=N_BOOT,bootstrap='station clusters with all eligible months retained; percentile interval of macro paired station-month contrast',sign_flip='exact two-sided sign flip of station-mean contrast',bh_family='7 thresholds x 4 tolerances x 2 QC groups = 56 tests',full_scan_rows=len(D),primary_200_rows_agree=True)
(HERE/'revision_tables_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
print(S2[(S2.threshold_wm2h==200)&S2.tolerance_h.isin([1,2])&(S2.solar_qc_group==PRIMARY)].to_string(index=False))
print('S9 tests',len(S9))
