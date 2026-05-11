"""
Chart 3: Profitability Improvement - HSBC Revenue Impact
For Section 4 - How HSBC Profits
"""
import matplotlib.pyplot as plt
import numpy as np

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

# LEFT: Loss Ratio Comparison
labels = ['Industry\nAverage', 'Micro Protection\nFluid Target']
loss_ratios = [68.4, 40]
retained = [31.6, 60]
colors_loss = ['#4A4A4A', '#DB0011']
colors_retained = ['#CCCCCC', '#FFB3B3']

x = np.arange(len(labels))
width = 0.35

bars1 = ax1.bar(x - width/2, loss_ratios, width, label='Claims Paid Out (%)', color=['#888888', '#4A4A4A'])
bars2 = ax1.bar(x + width/2, retained, width, label='Revenue Retained (%)', color=['#CCCCCC', '#DB0011'])

# Value labels
for bar in bars1:
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
             f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
for bar in bars2:
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
             f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

ax1.set_ylabel('Percentage of Premium', fontsize=11)
ax1.set_title('Loss Ratio: Industry vs Micro Protection Fluid', fontsize=12, fontweight='bold', color='#1A1A1A')
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=11)
ax1.set_ylim(0, 80)
ax1.legend(loc='upper left', fontsize=10)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_color('#CCCCCC')
ax1.spines['bottom'].set_color('#CCCCCC')

# Arrow showing improvement
ax1.annotate('', xy=(1.175, 60), xytext=(1.175, 31.6),
            arrowprops=dict(arrowstyle='->', color='#DB0011', lw=2))
ax1.text(1.35, 46, '+28.4pp\nmore\nretained', fontsize=9, color='#DB0011', fontweight='bold', ha='center')

# RIGHT: Revenue Impact at Scale
revenue_sources = ['Lower\nLoss Ratio', 'Reduced\nChurn', 'Reinsurance\nSavings', 'Platform\nFees (Gen 2)']
revenue_values = [1.5, 0.5, 0.3, 0.3]  # in billions USD
colors_rev = ['#DB0011', '#FF6B6B', '#4A4A4A', '#888888']

bars = ax2.bar(revenue_sources, revenue_values, color=colors_rev, width=0.5, edgecolor='white', linewidth=0.5)

for bar, val in zip(bars, revenue_values):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
             f'${val:.1f}B', ha='center', va='bottom', fontsize=12, fontweight='bold', color='#1A1A1A')

ax2.set_title('Additional Annual Revenue at Scale (5M customers)', fontsize=12, fontweight='bold', color='#1A1A1A')
ax2.set_ylim(0, 2.0)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_color('#CCCCCC')
ax2.spines['bottom'].set_color('#CCCCCC')
ax2.yaxis.set_visible(False)

# Total callout
total = sum(revenue_values)
ax2.text(0.5, 0.88, f'Total: ${total:.1f}B additional revenue per year',
         transform=ax2.transAxes, ha='center', fontsize=11, fontweight='bold', color='#DB0011',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF0F0', edgecolor='#DB0011'))

plt.suptitle('', fontsize=1)
fig.text(0.5, 0.01, 'Sources: Gitnux 2023 (industry loss ratio 68.4%), HSBC internal estimates at 5M customers, $100/month avg premium',
         ha='center', fontsize=9, color='#888888')

plt.tight_layout()
plt.savefig('charts/chart3_profitability.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("Chart 3 saved: charts/chart3_profitability.png")
