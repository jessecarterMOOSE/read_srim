import read_srim as srim
import matplotlib.pyplot as plt
import os.path


# use energies we have data for
energy_list = [1, 2, 5]  # MeV

# read in stopping and range table
stopping_table = srim.StoppingTable(os.path.join('data', 'Hydrogen in Iron.txt'))

for energy in energy_list:
    # read in damage output file
    damage_table = srim.DamageTable(os.path.join('data', str(int(energy))+'MeV-H-in-Fe-KP-40eV', 'VACANCY.txt'))

    # get total damage curve
    damage_profile = damage_table.get_total_damage_profile()

    # get nuclear stopping at those depths
    depths = damage_profile.index.values
    energies = stopping_table.energy_from_depth(depths, energy*1e6)
    stopping = stopping_table.nuclear_stopping_from_energy(energies)

    # plot ratio but normalize depth to ion range
    plt.plot(depths/stopping_table.range_from_energy(energy*1e6), stopping/damage_profile.values, label=str(energy)+' MeV')

# make it look nice
plt.legend(loc='best')
plt.ylim(top=500.0)
plt.xlabel('range-normalized depth')
plt.ylabel('nuclear stopping/damage ratio')
plt.tight_layout()
plt.show()
