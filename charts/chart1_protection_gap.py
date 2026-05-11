"""
Chart 1: Global Insurance Protection Gap
For Section 2 - Problem Statement & Opportunity
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Data
categories = ['Global\nProtection Gap', 'Embedded Insurance\nMarket (2034)', 'AI Revenue\nOpportunity']
values = [9.0, 1.464, 0.07]  # in trillions USD
colors = ['#DB0011', '#1A1A1A', '#4A4A4A']  # HSBC red, dark grey, medium grey

fig, ax = plt.subplots(figsize=(10, 6))

# Bar chart
bars = ax.bar(categories, values, color=colors, width=0.5, edgecolor='white', linewidth=0.5)

# Add value labels on bars
for bar, val in zip(bars, values):
    label = f'${val:.1f}T' if val >= 1 else f'${val*1000:.0f}B'
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.15,
            label, ha='center', va='bottom', fontsize=14, fontweight='bold', color='#1A1A1A')

# Styling
ax.set_ylabel('USD (Trillions)', fontsize=12, color='#1A1A1A')
ax.set_title('The Opportunity: Insurance is Massively Underserving Customers', 
             fontsize=14, fontweight='bold', color='#1A1A1A', pad=20)
ax.set_ylim(0, 11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#CCCCCC')
ax.spines['bottom'].set_color('#CCCCCC')
ax.tick_params(colors='#4A4A4A')
ax.yaxis.set_visible(False)

# Add source footnote
fig.text(0.5, 0.02, 'Sources: MAPFRE Economics 2024, Fortune Business Insights 2024, McKinsey 2024', 
         ha='center', fontsize=9, color='#888888')

# Add insight callout
ax.text(0.5, 0.85, '78% of insurance executives see closing the protection gap\nas an ethical obligation (SAS/Swiss Re 2025)',
        transform=ax.transAxes, ha='center', fontsize=10, style='italic', color='#555555',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#F5F5F5', edgecolor='#CCCCCC'))

plt.tight_layout()
plt.savefig('charts/chart1_protection_gap.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("Chart 1 saved: charts/chart1_protection_gap.png")
