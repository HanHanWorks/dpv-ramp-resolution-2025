#!/usr/bin/env python3
"""Rebuild SETA revision Figures 2--4 from the copied evidence tables."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WORK = Path(__file__).resolve().parents[2]
ANALYSES = WORK / '02_evidence_and_analysis' / 'analyses'
OUT = WORK / '01_revision_package' / '05_manuscript' / 'latex_source'
PRIMARY = 'primary_solar_qc'
BLUE, ORANGE, GREY = '#0072B2', '#D55E00', '#737373'
PRODUCTS = ['Tianji_2.5km', 'Tianji_5km', 'Tianji_10km', 'Tianji_25km',
            'Tianji_27.5km', 'ERA5_native_27.5km', 'ERA5_regridded_2.5km']
LABELS = ['Tianji 2.5 km', 'Tianji 5 km', 'Tianji 10 km', 'Tianji 25 km',
          'Tianji 27.5 km', 'ERA5 native 27.5 km', 'ERA5 regridded 2.5 km']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9.5,
    'axes.labelsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 8.5, 'axes.linewidth': 0.8, 'pdf.fonttype': 42,
    'ps.fonttype': 42, 'savefig.dpi': 300, 'figure.facecolor': 'white'})

def read(name):
    return pd.read_csv(ANALYSES / name, encoding='utf-8-sig')

def tidy(ax, horizontal=False):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis='x' if horizontal else 'y', color='#dedede', linewidth=0.55)

def panel(ax, text):
    ax.text(0, 1.08, text, transform=ax.transAxes, ha='left', va='bottom',
            fontsize=10, fontweight='bold')

def save(fig, stem):
    fig.savefig(OUT / (stem + '.pdf'), bbox_inches='tight', pad_inches=0.09)
    fig.savefig(OUT / (stem + '.png'), bbox_inches='tight', pad_inches=0.09)
    plt.close(fig)
    print('Wrote', stem + '.pdf/.png')

main = read('consecutive_hour_primary_rerun/canonical_main_results.csv')
main = main[(main.solar_qc_group == PRIMARY) &
            main['product'].isin(['Tianji_2.5km', 'Tianji_5km', 'Tianji_10km',
                                 'Tianji_25km', 'Tianji_27.5km']) &
            (main.threshold_wm2h == 200)]
fair = read('fair_mask_common_month_product_summary.csv')
fair = fair[fair.tolerance_h == 1].set_index('product').loc[PRODUCTS]
contrasts = read('fair_mask_common_month_paired_contrasts.csv')
contrasts = contrasts[contrasts.tolerance_h == 1].set_index('contrast')
foot = read('footprint_era5_common_mask_summary.csv')
foot = foot[foot.solar_qc_group == PRIMARY]
fp = foot[foot.row_type == 'product'].set_index('series')
fc = foot[foot.row_type == 'paired_contrast'].set_index('comparison')
assert len(fair) == 7 and set(fair.stations) == {7}
assert set(fair.station_months) == {14}
assert set(fair.observed_events_sum) == {836}
assert set(fair.paired_hours_sum) == {9950}
transitions = read('fair_mask_common_hour_transition_counts.csv')
assert len(transitions) == 14
assert transitions.station_id.nunique() == 7
assert int(transitions.valid_common_station_hours.sum()) == 9950
assert int(transitions.eligible_consecutive_within_month_transitions.sum()) == 9925
assert np.isclose(fair.loc['Tianji_2.5km', 'macro_csi'],
                  fp.loc['Tianji_2.5km', 'macro_station_month_csi'])

# Figure 2. Panel a is a three-month temporal extension, shown as means only.
# Panel b is a common-month footprint-conditioned contrast; CIs are copied
# from the saved paired station-cluster analysis, not the legacy bootstrap.
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
for tol, marker, line, color in [(1, 'o', '-', BLUE), (2, 's', '--', ORANGE)]:
    d = main[main.tolerance_h == tol].sort_values('support_km')
    axes[0].plot(d.support_km, d.csi_mean, marker=marker, linestyle=line,
                 color=color, linewidth=1.45, markersize=4.5,
                 label=str(tol) + ' h matching')
axes[0].set_xticks([2.5, 5, 10, 25, 27.5])
axes[0].set_xticklabels(['2.5', '5', '10', '25', '27.5'])
axes[0].set_xlim(1, 29)
axes[0].set_ylim(0, 0.27)
axes[0].set_yticks([0, 0.05, 0.10, 0.15, 0.20, 0.25])
axes[0].set_xlabel('Controlled Tianji support (km)')
axes[0].set_ylabel('Macro station-month CSI')
axes[0].legend(loc='lower left', frameon=False)
panel(axes[0], '(a) Three months; 21 station-months')
tidy(axes[0])
delta, low, high = [], [], []
for series in ['Tianji_2.5km', 'Tianji_2.5km_3x3', 'Tianji_2.5km_5x5']:
    if series == 'Tianji_2.5km':
        r = contrasts.loc['Tianji_27.5km minus Tianji_2.5km']
        delta.append(r.mean_paired_station_month_delta_csi)
        low.append(r.station_cluster_95_low)
        high.append(r.station_cluster_95_high)
    else:
        r = fc.loc['Tianji_27.5km minus ' + series]
        delta.append(r.paired_delta_csi)
        low.append(r.paired_delta_ci95_low)
        high.append(r.paired_delta_ci95_high)
delta, low, high = map(np.array, (delta, low, high))
axes[1].errorbar(np.arange(3), delta, yerr=[delta-low, high-delta],
                 color=BLUE, marker='o', linestyle='none', markersize=5,
                 linewidth=1.25, capsize=3)
axes[1].axhline(0, color='#545454', linewidth=0.8)
axes[1].set_xticks([0, 1, 2])
axes[1].set_xticklabels(['1 x 1\n1 cell', '3 x 3\n9 cells', '5 x 5\n25 cells'])
axes[1].set_xlim(-0.45, 2.45)
axes[1].set_ylim(-0.12, 0.012)
axes[1].set_yticks([-0.12, -0.09, -0.06, -0.03, 0])
axes[1].set_xlabel('Tianji fine-end sampling footprint')
axes[1].set_ylabel('CSI: 27.5 km minus fine footprint')
panel(axes[1], '(b) April, July; 14 station-months; 1 h')
tidy(axes[1])
fig.subplots_adjust(left=0.09, right=0.985, bottom=0.26, top=0.82, wspace=0.39)
save(fig, 'fig2_resolution_and_threshold_sensitivity')

# Figure 3. Means across the three sampled months, with descriptive observed
# min--max whiskers. With only three sampled months per station, these are
# shown as ranges rather than labelled as inferential confidence intervals.
station = read('consecutive_hour_primary_rerun/canonical_station_metrics.csv')
station = station[(station.threshold_wm2h == 200) &
                  (station.tolerance_h == 1) &
                  station['product'].isin(['Tianji_2.5km', 'Tianji_27.5km'])]
p = station.pivot_table(index=['station_id', 'solar_qc_group', 'month'],
                        columns='product', values='csi').reset_index()
p['delta'] = p['Tianji_27.5km'] - p['Tianji_2.5km']
sm = p.groupby(['station_id', 'solar_qc_group'])[
    ['Tianji_2.5km', 'Tianji_27.5km', 'delta']].mean().reset_index()
sr = p.groupby(['station_id', 'solar_qc_group'])[
    ['Tianji_2.5km', 'Tianji_27.5km', 'delta']].agg(['min', 'max']).reset_index()
sr.columns = ['station_id', 'solar_qc_group',
              'fine_min', 'fine_max', 'coarse_min', 'coarse_max',
              'delta_min', 'delta_max']
sm = sm.merge(sr, on=['station_id', 'solar_qc_group'], validate='one_to_one')
sm['primary'] = sm.solar_qc_group.eq(PRIMARY)
sm = sm.sort_values(['primary', 'station_id'],
                     ascending=[False, True]).reset_index(drop=True)
assert sm.primary.sum() == 7 and len(sm) == 15 and (sm.delta < 0).all()
fig, axes = plt.subplots(1, 2, figsize=(7.2, 5.2), sharey=True)
for idx, r in sm.iterrows():
    marker = 'o' if r.primary else 's'
    # The whiskers are the observed April/July/October min--max range for
    # each station. They are deliberately not presented as confidence
    # intervals because only three month-level observations are available.
    axes[0].errorbar(r['Tianji_2.5km'], idx,
                     xerr=[[r['Tianji_2.5km'] - r['fine_min']],
                           [r['fine_max'] - r['Tianji_2.5km']]],
                     color=BLUE, linewidth=1.0, capsize=2.2, zorder=1)
    axes[0].errorbar(r['Tianji_27.5km'], idx,
                     xerr=[[r['Tianji_27.5km'] - r['coarse_min']],
                           [r['coarse_max'] - r['Tianji_27.5km']]],
                     color=ORANGE, linewidth=1.0, capsize=2.2, zorder=1)
    axes[0].plot([r['Tianji_27.5km'], r['Tianji_2.5km']], [idx, idx],
                 color='#b8b8b8', linewidth=1, zorder=1)
    axes[0].plot(r['Tianji_2.5km'], idx, color=BLUE, marker=marker,
                 markersize=5, linestyle='none', zorder=2)
    axes[0].plot(r['Tianji_27.5km'], idx, color=ORANGE, marker=marker,
                 markerfacecolor='white', markersize=5, linestyle='none', zorder=3)
    axes[1].errorbar(r.delta, idx,
                     xerr=[[r.delta - r.delta_min],
                           [r.delta_max - r.delta]],
                     color=BLUE if r.primary else GREY,
                     linewidth=1.0, capsize=2.2, zorder=1)
    axes[1].plot(r.delta, idx, color=BLUE if r.primary else GREY,
                 marker=marker, markersize=5, linestyle='none', zorder=2)
for ax in axes:
    ax.axhline(6.5, color='#7c7c7c', linestyle='--', linewidth=0.8)
    # Leave one empty row below the secondary-QC stations so the endpoint
    # legend does not cover the lowest station whiskers.
    ax.set_ylim(15.7, -0.7)
    tidy(ax, horizontal=True)
axes[0].set_yticks(np.arange(15))
axes[0].set_yticklabels([str(i) for i in sm.station_id])
axes[0].set_ylabel('Station ID (primary above dashed line)')
axes[0].set_xlim(0, 0.27)
axes[0].set_xticks([0, 0.1, 0.2])
axes[0].set_xlabel('Three-month mean CSI')
axes[0].plot([], [], color=BLUE, marker='o', linestyle='none', label='2.5 km')
axes[0].plot([], [], color=ORANGE, marker='o', markerfacecolor='white',
             linestyle='none', label='27.5 km')
axes[0].plot([], [], color='#8c8c8c', linewidth=1.0,
             label='3-month observed min--max')
axes[0].legend(loc='lower right', frameon=False, handletextpad=0.4)
axes[1].axvline(0, color='#545454', linewidth=0.8)
axes[1].set_xlim(-0.135, 0.005)
axes[1].set_xticks([-0.12, -0.08, -0.04, 0])
axes[1].set_xlabel('CSI: 27.5 km minus 2.5 km')
axes[1].plot([], [], color=BLUE, marker='o', linestyle='none', label='Primary QC')
axes[1].plot([], [], color=GREY, marker='s', linestyle='none', label='Secondary QC')
axes[1].legend(loc='lower left', frameon=False, handletextpad=0.4)
panel(axes[0], '(a) Station endpoint means')
panel(axes[1], '(b) Paired station-mean contrasts')
fig.subplots_adjust(left=0.15, right=0.985, bottom=0.14, top=0.88, wspace=0.23)
save(fig, 'fig3_station_heterogeneity_and_variance')

# Figure 4. One common month set, one valid-hour mask, one bootstrap unit.
fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.25), sharey=True)
for idx, product in enumerate(PRODUCTS):
    r = fair.loc[product]
    color, marker = (ORANGE, 'D') if product.startswith('ERA5') else (BLUE, 'o')
    x = r.macro_csi
    axes[0].errorbar(x, idx, xerr=[[x-r.csi_station_cluster_95_low],
                                  [r.csi_station_cluster_95_high-x]],
                     color=color, marker=marker, linestyle='none',
                     markersize=4.5, capsize=3, linewidth=1.2)
    if idx == 0:
        axes[1].plot(0, idx, color=color, marker=marker, markerfacecolor='white',
                     markersize=4.5, linestyle='none')
    else:
        c = contrasts.loc[product + ' minus Tianji_2.5km']
        x = c.mean_paired_station_month_delta_csi
        axes[1].errorbar(x, idx, xerr=[[x-c.station_cluster_95_low],
                                      [c.station_cluster_95_high-x]],
                         color=color, marker=marker, linestyle='none',
                         markersize=4.5, capsize=3, linewidth=1.2)
for ax in axes:
    ax.axhline(4.5, color='#7c7c7c', linestyle='--', linewidth=0.8)
    ax.set_ylim(6.7, -0.7)
    tidy(ax, horizontal=True)
axes[0].set_yticks(np.arange(7))
axes[0].set_yticklabels(LABELS)
axes[0].set_xlim(0, 0.29)
axes[0].set_xticks([0, 0.1, 0.2])
axes[0].set_xlabel('Macro station-month CSI')
axes[1].set_xlim(-0.125, 0.012)
axes[1].set_xticks([-0.12, -0.08, -0.04, 0])
axes[1].axvline(0, color='#545454', linewidth=0.8)
axes[1].set_xlabel('CSI minus Tianji 2.5 km')
panel(axes[0], '(a) Common-month product scores')
panel(axes[1], '(b) Paired product contrasts')
axes[0].text(0.04, 0.04, '● Tianji   ◆ ERA5', transform=axes[0].transAxes,
             fontsize=8, va='bottom')
fig.subplots_adjust(left=0.29, right=0.985, bottom=0.18, top=0.82, wspace=0.25)
save(fig, 'fig4_product_benchmark_common_months')
print('Validated: 7 primary stations, 14 common station-months, 9,950 valid hours, 9,925 consecutive transitions, 836 observed events.')
