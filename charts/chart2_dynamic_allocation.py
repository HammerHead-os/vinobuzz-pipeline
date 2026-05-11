"""
Chart for Section 3: Dynamic Allocation
Tall single-column layout, big fonts, no overlapping
"""
import matplotlib.pyplot as plt
import numpy as np

scenarios = ['Traditional', 'Work From Home', 'Flying to Tokyo', 'Rock Climbing']

categories = ['Life', 'Health', 'Travel', 'Property', 'Cyber', 'Liability']
colors = ['#DB0011', '#FF9999', '#1A1A1A', '#2E86C1', '#F39C12', '#27AE60']

data = np.array([
    [80, 50, 40, 30, 0, 0],
    [90, 60, 5, 25, 15, 5],
    [90, 40, 35, 10, 5, 20],
    [90, 70, 5, 15, 5, 15],
])

fig, ax = plt.subplots(figsize=(7, 7))

bar_height = 0.6
y_pos = np.arange(len(scenarios))

left = np.zeros(len(scenarios))
for i, (cat, color) in enumerate(zip(categories, colors)):
    bars = ax.barh(y_pos, data[:, i], left=left, height=bar_height,
                   label=cat, color=color, edgecolor='white', linewidth=0.5)
    for j, (bar, val) in enumerate(zip(bars, data[:, i])):
        if val >= 25:
            ax.text(left[j] + val/2, y_pos[j], f'${val}',
                    ha='center', va='center', fontsize=11, color='white', fontweight='bold')
    left += data[:, i]

for j in range(len(scenarios)):
    ax.text(205, y_pos[j], '$200', ha='left', va='center', fontsize=13, fontweight='bold', color='#1A1A1A')

ax.set_xlabel('Allocation (USD)', fontsize=12)
ax.set_title('Same $200. Smarter Allocation.', fontsize=15, fontweight='bold', color='#1A1A1A', pad=15)
ax.set_yticks(y_pos)
ax.set_yticklabels(scenarios, fontsize=12)
ax.set_xlim(0, 235)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#CCCCCC')
ax.spines['bottom'].set_color('#CCCCCC')

# Legend at bottom, below chart
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=3,
          fontsize=10, framealpha=0.9, edgecolor='#CCCCCC')

plt.tight_layout()
plt.savefig('charts/chart2_dynamic_allocation.png', dpi=600, bbox_inches='tight', facecolor='white')
plt.close()
print("Chart 2 saved: charts/chart2_dynamic_allocation.png")
