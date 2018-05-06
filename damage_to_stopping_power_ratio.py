import read_srim as srim
import matplotlib.pyplot as plt
import os.path


# use energies we have data for
energy_list = [1, 2, 5]  # MeV

# read in stopping and range table
stopping_table = srim.StoppingTable(os.path.join('data', 'Hydrogen in Iron.txt'))

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(16, 9))

for energy in energy_list:
    # read in damage output file
    damage_table = srim.DamageTable(os.path.join('data', str(int(energy))+'MeV-H-in-Fe-KP-40eV', 'VACANCY.txt'))
    figure_label = str(energy)+' MeV H in Fe'

    # get total damage curve
    damage_profile = damage_table.get_total_damage_profile()

    # get nuclear stopping at those depths
    depths = damage_profile.index.values
    energies = stopping_table.energy_from_depth(depths, energy*1e6)
    stopping = stopping_table.nuclear_stopping_from_energy(energies)

    # normalize depths to range
    range_normalized_depths = depths/stopping_table.range_from_energy(energy*1e6)

    # compute ratio of stopping power to damage
    ratios = stopping/damage_profile.values

    # plot ratio but normalize depth to ion range
    ax[0].plot(range_normalized_depths, ratios, label=figure_label)

    # smooth out data and replot
    averages = []
    spreads = []
    normalized_points = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    for point in normalized_points:
        window = ratios[(range_normalized_depths > (point-0.1)) & (range_normalized_depths < (point+0.1))]
        averages.append(window.mean())
        spreads.append(window.std())
    ax[1].errorbar(normalized_points, averages, yerr=spreads, capsize=5, marker='o', label=figure_label+', smoothed')

# make it look nice
plt.xlabel('range-normalized depth')
ax[0].set_ylim([0.0, 500.0])
for axes in ax:
    axes.legend(loc='best', fontsize=8)
    axes.set_xlim([0.0, 1.1])
    axes.set_ylabel('nuclear stopping/damage ratio\n(eV/vacancy)')

plt.tight_layout()
plt.show()
