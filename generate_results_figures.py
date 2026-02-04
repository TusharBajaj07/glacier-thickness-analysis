#!/usr/bin/env python3
"""Generate additional figures for Results & Discussion section."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path

OUTPUT_DIR = Path('/Users/tusharbajaj/GNR_618/results')

# Load the per-glacier results
print("Loading data...")
glaciers_df = pd.read_csv(OUTPUT_DIR / 'per_glacier_results.csv')
glaciers_gdf = gpd.read_file(OUTPUT_DIR / 'glaciers_with_change.shp')

print(f"Loaded {len(glaciers_df)} glaciers with change data")

# ============================================
# Figure 1: Cumulative Distribution Function
# ============================================
print("\nGenerating cumulative distribution plot...")

fig, ax = plt.subplots(figsize=(10, 6))

# Sort values for CDF
sorted_dh = np.sort(glaciers_df['mean_dh'].dropna())
cdf = np.arange(1, len(sorted_dh) + 1) / len(sorted_dh) * 100

ax.plot(sorted_dh, cdf, 'b-', linewidth=2.5)
ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
ax.axhline(y=50, color='gray', linestyle=':', linewidth=1, alpha=0.7)

# Mark key percentiles
median_dh = np.median(sorted_dh)
ax.plot(median_dh, 50, 'ro', markersize=10, zorder=5)
ax.annotate(f'Median: {median_dh:.2f}m', xy=(median_dh, 50), xytext=(median_dh + 8, 55),
            fontsize=11, arrowprops=dict(arrowstyle='->', color='red'))

# Shade thinning region
ax.fill_between(sorted_dh[sorted_dh < 0], 0, cdf[sorted_dh < 0], alpha=0.3, color='red', label='Thinning glaciers')
ax.fill_between(sorted_dh[sorted_dh >= 0], 0, cdf[sorted_dh >= 0], alpha=0.3, color='blue', label='Thickening glaciers')

# Calculate percentage thinning
pct_thinning = np.sum(sorted_dh < 0) / len(sorted_dh) * 100
ax.text(0.02, 0.98, f'{pct_thinning:.1f}% of glaciers\nshow net thinning',
        transform=ax.transAxes, fontsize=12, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax.set_xlabel('Mean Elevation Change (m)', fontsize=12)
ax.set_ylabel('Cumulative Percentage of Glaciers (%)', fontsize=12)
ax.set_title('Cumulative Distribution of Glacier Elevation Changes', fontsize=14)
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xlim(-50, 50)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'cumulative_distribution.png', dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: cumulative_distribution.png")


# ============================================
# Figure 2: Glacier Size vs Thinning Rate
# ============================================
print("\nGenerating size vs thinning plot...")

fig, ax = plt.subplots(figsize=(10, 7))

# Use log scale for area
scatter = ax.scatter(glaciers_df['area_km2'], glaciers_df['mean_dh'],
                     c=glaciers_df['mean_dh'], cmap='RdBu',
                     s=20, alpha=0.6, vmin=-20, vmax=20)

ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
ax.set_xscale('log')

# Add trend line using numpy polyfit
log_area = np.log10(glaciers_df['area_km2'].values)
dh = glaciers_df['mean_dh'].values
mask = ~np.isnan(dh) & ~np.isnan(log_area) & ~np.isinf(log_area)
coeffs = np.polyfit(log_area[mask], dh[mask], 1)
slope, intercept = coeffs

# Calculate R²
y_pred = slope * log_area[mask] + intercept
ss_res = np.sum((dh[mask] - y_pred) ** 2)
ss_tot = np.sum((dh[mask] - np.mean(dh[mask])) ** 2)
r_squared = 1 - (ss_res / ss_tot)

x_trend = np.logspace(-2, 2, 100)
y_trend = slope * np.log10(x_trend) + intercept
ax.plot(x_trend, y_trend, 'k-', linewidth=2, label=f'Trend (R² = {r_squared:.3f})')

ax.set_xlabel('Glacier Area (km²)', fontsize=12)
ax.set_ylabel('Mean Elevation Change (m)', fontsize=12)
ax.set_title('Glacier Size vs. Elevation Change', fontsize=14)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
cbar.set_label('Elevation Change (m)', fontsize=11)

# Add size categories annotation
ax.axvline(x=1, color='gray', linestyle=':', alpha=0.5)
ax.axvline(x=10, color='gray', linestyle=':', alpha=0.5)
ax.text(0.1, -45, 'Small\n(<1 km²)', ha='center', fontsize=9, color='gray')
ax.text(3, -45, 'Medium\n(1-10 km²)', ha='center', fontsize=9, color='gray')
ax.text(50, -45, 'Large\n(>10 km²)', ha='center', fontsize=9, color='gray')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'size_vs_thinning.png', dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: size_vs_thinning.png")


# ============================================
# Figure 3: Top 20 Most Thinning Glaciers
# ============================================
print("\nGenerating top thinning glaciers plot...")

# Get top 20 most thinning (most negative mean_dh)
top_thinning = glaciers_df.nsmallest(20, 'mean_dh')[['rgi_id', 'mean_dh', 'area_km2', 'region']].copy()
top_thinning['short_id'] = top_thinning['rgi_id'].str.split('-').str[-1]

fig, ax = plt.subplots(figsize=(12, 7))

colors_region = {'Sikkim': '#1f77b4', 'Bhutan': '#ff7f0e', 'Arunachal Pradesh': '#2ca02c'}
bar_colors = [colors_region.get(r, 'gray') for r in top_thinning['region']]

bars = ax.barh(range(len(top_thinning)), top_thinning['mean_dh'], color=bar_colors, edgecolor='black', alpha=0.8)

ax.set_yticks(range(len(top_thinning)))
ax.set_yticklabels(top_thinning['short_id'], fontsize=9)
ax.set_xlabel('Mean Elevation Change (m)', fontsize=12)
ax.set_ylabel('Glacier ID (RGI)', fontsize=12)
ax.set_title('Top 20 Glaciers with Highest Surface Lowering', fontsize=14)
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
ax.grid(True, alpha=0.3, axis='x')

# Add area labels
for i, (idx, row) in enumerate(top_thinning.iterrows()):
    ax.text(row['mean_dh'] - 1, i, f"{row['area_km2']:.1f} km²", va='center', ha='right', fontsize=8, color='white')

# Legend for regions
legend_patches = [mpatches.Patch(color=c, label=r) for r, c in colors_region.items()]
ax.legend(handles=legend_patches, loc='lower left', title='Region')

ax.invert_yaxis()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'top_thinning_glaciers.png', dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: top_thinning_glaciers.png")


# ============================================
# Figure 4: Elevation Band Analysis
# ============================================
print("\nGenerating elevation band analysis...")

# Load full shapefile with elevation data
glaciers_full = gpd.read_file(OUTPUT_DIR / 'glaciers_with_change.shp')

if 'zmed_m' in glaciers_full.columns:
    elev_col = 'zmed_m'
elif 'zmed' in glaciers_full.columns:
    elev_col = 'zmed'
else:
    elev_col = None

if elev_col:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Create elevation bands
    glaciers_full['elev_band'] = pd.cut(glaciers_full[elev_col],
                                         bins=[3500, 4500, 5000, 5500, 6000, 6500, 8000],
                                         labels=['3500-4500', '4500-5000', '5000-5500',
                                                '5500-6000', '6000-6500', '>6500'])

    # Left: Box plot by elevation band
    ax1 = axes[0]
    band_data = [glaciers_full[glaciers_full['elev_band'] == band]['mean_dh'].dropna()
                 for band in glaciers_full['elev_band'].cat.categories]

    bp = ax1.boxplot(band_data, labels=glaciers_full['elev_band'].cat.categories, patch_artist=True)

    # Color gradient from low (warm) to high (cool)
    cmap = plt.cm.RdYlBu
    for i, (patch, median) in enumerate(zip(bp['boxes'], bp['medians'])):
        patch.set_facecolor(cmap(i / len(bp['boxes'])))
        patch.set_alpha(0.7)
        median.set_color('black')
        median.set_linewidth(2)

    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.7)
    ax1.set_xlabel('Elevation Band (m a.s.l.)', fontsize=12)
    ax1.set_ylabel('Mean Elevation Change (m)', fontsize=12)
    ax1.set_title('(a) Elevation Change by Altitude Band', fontsize=13)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='x', rotation=45)

    # Right: Mean thinning rate by elevation band
    ax2 = axes[1]
    band_stats = glaciers_full.groupby('elev_band').agg({
        'mean_dh': ['mean', 'std', 'count']
    }).reset_index()
    band_stats.columns = ['elev_band', 'mean', 'std', 'count']
    band_stats = band_stats.dropna()

    x_pos = range(len(band_stats))
    bars = ax2.bar(x_pos, band_stats['mean'],
                   yerr=band_stats['std']/np.sqrt(band_stats['count']),
                   capsize=5, color=[cmap(i/len(band_stats)) for i in range(len(band_stats))],
                   edgecolor='black', alpha=0.8)

    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.7)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(band_stats['elev_band'], rotation=45)
    ax2.set_xlabel('Elevation Band (m a.s.l.)', fontsize=12)
    ax2.set_ylabel('Mean Elevation Change (m)', fontsize=12)
    ax2.set_title('(b) Mean Change by Altitude (±SE)', fontsize=13)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add count labels
    for i, (bar, count) in enumerate(zip(bars, band_stats['count'])):
        ax2.text(bar.get_x() + bar.get_width()/2, 0.5, f'n={int(count)}',
                ha='center', va='bottom', fontsize=9, rotation=90)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'elevation_band_analysis.png', dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Saved: elevation_band_analysis.png")
else:
    print("Elevation column not found, skipping elevation band analysis")


# ============================================
# Figure 5: Volume Change by Region (Pie + Bar)
# ============================================
print("\nGenerating volume change by region...")

# Calculate volume change per glacier (mean_dh * area)
glaciers_df['volume_change_km3'] = glaciers_df['mean_dh'] * glaciers_df['area_km2'] / 1000  # Convert to km³

region_volume = glaciers_df.groupby('region').agg({
    'volume_change_km3': 'sum',
    'area_km2': 'sum',
    'mean_dh': 'mean',
    'rgi_id': 'count'
}).rename(columns={'rgi_id': 'n_glaciers'})

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Stacked bar showing total volume change
ax1 = axes[0]
regions = ['Sikkim', 'Bhutan', 'Arunachal Pradesh']
region_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# Make sure we have the right order
region_volume = region_volume.reindex(regions)

bars = ax1.bar(regions, region_volume['volume_change_km3'], color=region_colors, edgecolor='black', alpha=0.8)
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, height - 0.3 if height < 0 else height + 0.1,
             f'{height:.2f} km³', ha='center', va='top' if height < 0 else 'bottom', fontsize=11, fontweight='bold')

ax1.set_ylabel('Volume Change (km³)', fontsize=12)
ax1.set_title('(a) Total Volume Change by Region', fontsize=13)
ax1.grid(True, alpha=0.3, axis='y')

# Add total annotation
total_vol = region_volume['volume_change_km3'].sum()
ax1.text(0.98, 0.98, f'Total: {total_vol:.2f} km³', transform=ax1.transAxes,
         ha='right', va='top', fontsize=12, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Right: Pie chart of contribution (absolute values)
ax2 = axes[1]
abs_volumes = np.abs(region_volume['volume_change_km3'])
explode = (0.02, 0.02, 0.02)

wedges, texts, autotexts = ax2.pie(abs_volumes, labels=regions, autopct='%1.1f%%',
                                    colors=region_colors, explode=explode,
                                    wedgeprops={'edgecolor': 'black', 'linewidth': 1.5},
                                    textprops={'fontsize': 11})
for autotext in autotexts:
    autotext.set_fontsize(11)
    autotext.set_fontweight('bold')

ax2.set_title('(b) Regional Contribution to Total Volume Loss', fontsize=13)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'volume_change_by_region.png', dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: volume_change_by_region.png")


# ============================================
# Figure 6: Summary Statistics Table
# ============================================
print("\nGenerating summary statistics table figure...")

# Create summary data
summary_data = {
    'Metric': [
        'Total Glaciers Analyzed',
        'Total Glacier Area',
        'Mean Elevation Change (Δh)',
        'Median Elevation Change',
        'Std. Deviation',
        'Mean Thinning Rate',
        'Total Volume Change',
        'Water Equivalent Loss',
        'Glaciers with Thinning',
        'Glaciers with Thickening'
    ],
    'Value': [
        f"{len(glaciers_df):,}",
        f"{glaciers_df['area_km2'].sum():.1f} km²",
        f"{glaciers_df['mean_dh'].mean():.2f} m",
        f"{glaciers_df['mean_dh'].median():.2f} m",
        f"±{glaciers_df['mean_dh'].std():.2f} m",
        f"{glaciers_df['mean_dh'].mean()/8:.3f} m/yr",
        f"{glaciers_df['volume_change_km3'].sum():.2f} km³",
        f"{glaciers_df['volume_change_km3'].sum() * 0.9:.2f} km³",
        f"{(glaciers_df['mean_dh'] < 0).sum():,} ({100*(glaciers_df['mean_dh'] < 0).sum()/len(glaciers_df):.1f}%)",
        f"{(glaciers_df['mean_dh'] >= 0).sum():,} ({100*(glaciers_df['mean_dh'] >= 0).sum()/len(glaciers_df):.1f}%)"
    ]
}

fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')

# Create table
table = ax.table(cellText=[[m, v] for m, v in zip(summary_data['Metric'], summary_data['Value'])],
                 colLabels=['Metric', 'Value'],
                 cellLoc='left',
                 loc='center',
                 colWidths=[0.55, 0.35])

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)

# Style the table
for i in range(len(summary_data['Metric']) + 1):
    for j in range(2):
        cell = table[(i, j)]
        if i == 0:  # Header
            cell.set_facecolor('#2E7D32')
            cell.set_text_props(color='white', fontweight='bold')
        elif i % 2 == 0:
            cell.set_facecolor('#f0f0f0')
        else:
            cell.set_facecolor('white')
        cell.set_edgecolor('#cccccc')

ax.set_title('Summary Statistics: Glacier Elevation Change Analysis\n(Sikkim, Bhutan, Arunachal Pradesh)',
             fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'summary_statistics_table.png', dpi=120, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: summary_statistics_table.png")


# ============================================
# Figure 7: Aspect Analysis (if available)
# ============================================
print("\nGenerating aspect analysis...")

if 'aspect_deg' in glaciers_full.columns or 'aspect_se' in glaciers_full.columns:
    aspect_col = 'aspect_deg' if 'aspect_deg' in glaciers_full.columns else 'aspect_se'

    # Bin aspects into 8 directions
    def get_direction(aspect):
        if pd.isna(aspect):
            return None
        directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        idx = int(((aspect + 22.5) % 360) / 45)
        return directions[idx]

    glaciers_full['direction'] = glaciers_full[aspect_col].apply(get_direction)

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

    directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    angles = np.linspace(0, 2*np.pi, 9)[:-1]  # 8 directions

    # Calculate mean thinning for each direction
    dir_stats = glaciers_full.groupby('direction')['mean_dh'].agg(['mean', 'std', 'count'])

    means = [dir_stats.loc[d, 'mean'] if d in dir_stats.index else 0 for d in directions]
    counts = [dir_stats.loc[d, 'count'] if d in dir_stats.index else 0 for d in directions]

    # Normalize for color
    colors = plt.cm.RdBu([(m + 10) / 20 for m in means])  # Normalize around 0

    bars = ax.bar(angles, np.abs(means), width=0.7, color=colors, edgecolor='black', alpha=0.8)

    ax.set_xticks(angles)
    ax.set_xticklabels(directions, fontsize=12)
    ax.set_title('Mean Elevation Change by Glacier Aspect\n(bar height = |Δh|, color = direction of change)',
                 fontsize=12, pad=20)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='RdBu', norm=plt.Normalize(-10, 10))
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.1)
    cbar.set_label('Elevation Change (m)', fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'aspect_analysis.png', dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Saved: aspect_analysis.png")
else:
    print("Aspect column not found, skipping aspect analysis")


print("\n" + "="*50)
print("All Results & Discussion figures generated!")
print("="*50)
