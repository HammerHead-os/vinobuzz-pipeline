"""
Chart for Section 2: Problem Statement & Opportunity
Just the line chart, big fonts, no overlap
"""
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(7, 7))

days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
activities = ['WFH', 'Deliver', 'Office', 'Book flt', 'Fly', 'Scooter', 'Home']

actual_risk = [25, 75, 45, 55, 85, 92, 40]
static_coverage = [50, 50, 50, 50, 50, 50, 50]

x = np.arange(len(days))

# Shading with interpolation
for i in range(len(days) - 1):
    xs = np.linspace(x[i], x[i+1], 50)
    rs = np.interp(xs, [x[i], x[i+1]], [actual_risk[i], actual_risk[i+1]])
    ss = np.full_like(xs, 50)
    ax.fill_between(xs, ss, rs, where=(rs > ss), alpha=0.15, color='#DB0011', interpolate=True)
    ax.fill_between(xs, rs, ss, where=(ss > rs), alpha=0.35, color='#555555', interpolate=True)

# Lines
ax.plot(x, actual_risk, color='#DB0011', linewidth=3, marker='o',
        markersize=9, label='Actual risk', zorder=5)
ax.plot(x, static_coverage, color='#4A4A4A', linewidth=2.5, linestyle='--',
        label='Static coverage', zorder=4)

# X labels
combined = [f'{d}\n{a}' for d, a in zip(days, activities)]
ax.set_xticks(x)
ax.set_xticklabels(combined, fontsize=13)
ax.tick_params(axis='y', labelsize=12)

ax.set_ylabel('Risk Level', fontsize=14)
ax.set_title('Risk moves. Coverage does not.', fontsize=18, fontweight='bold',
             color='#1A1A1A', pad=15)
ax.set_ylim(0, 110)
ax.set_xlim(-0.3, 6.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#CCCCCC')
ax.spines['bottom'].set_color('#CCCCCC')

ax.legend(loc='lower right', fontsize=13, framealpha=0.95, edgecolor='#CCCCCC')

ax.text(0.97, 0.95, 'Red = unprotected\nGrey = overpaying',
        transform=ax.transAxes, ha='right', va='top', fontsize=12, color='#333333',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF5F5',
                  edgecolor='#DB0011', alpha=0.9))

plt.tight_layout()
plt.savefig('charts/chart_section2_problem.png', dpi=600, bbox_inches='tight', facecolor='white')
plt.close()
print("Chart Section 2 saved")
