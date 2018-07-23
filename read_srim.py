from StringIO import StringIO
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import os.path


class SRIMTable(object):
    def __init__(self, filename, column_names, widths=None):
        # store filename as a member variable
        self.filename = filename

        # store target information in a dict member variable, classes can overwrite this
        self.target_info = self.get_target_info()

        # generate file-like object to pass to pandas
        file_io = StringIO()
        for line in self.parse_srim_file():  # classes can overwrite this
            file_io.write(line)
        file_io.seek(0)  # rewind

        if widths:
            self.raw_df = pd.read_fwf(file_io, widths=widths, header=None, names=column_names).fillna(value=0.0)
        else:
            self.raw_df = pd.read_csv(file_io, delim_whitespace=True, header=None, names=column_names)

    def get_srim_table(self):
        return self.raw_df

    def parse_srim_file(self):
        dashed_lines_found = 0
        with open(self.filename, 'r') as f:
            ready_to_yield = False
            for raw_line in f:
                line = raw_line.strip()
                if ready_to_yield:
                    # look for lines starting with numbers
                    if line[:1].isdigit():  # deals with empty strings
                        yield raw_line  # pandas is expecting newlines in "file"
                    else:
                        # we are done
                        break
                elif set(line) == set(['-', ' ']):
                    dashed_lines_found += 1
                    # look for the broken dashed line to signal data is coming
                    if dashed_lines_found == self.which_dashed_line:
                        ready_to_yield = True


class SingleTarget(SRIMTable):
    # class for handling data files with no layer information, e.g. StoppingTable
    def __init__(self, path_name, column_names, filename=None, widths=None):
        if filename:
            path_name = os.path.join(path_name, filename)
        super(SingleTarget, self).__init__(path_name, column_names, widths=widths)


class LayeredTarget(SRIMTable):
    # class for handling data files from SRIM runs where the target can contain many layers
    def __init__(self, path_name, column_names, filename=None, widths=None):
        if filename:
            path_name = os.path.join(path_name, filename)
        self.which_dashed_line = 1  # data comes after second dashed line in output files
        super(LayeredTarget, self).__init__(path_name, column_names, widths=widths)

    def get_target_info(self):
        # read through file and figure out target properties, including multiple layers
        def fields_past(line, search_string, num_fields, num_to_return=1, return_remaining=False):
            # takes a string, splits it, and looks for search_string and returns the num fields past that
            # good for processing data lines where the format is known but not necessary the full line
            fields = line.split()
            for i, field in enumerate(fields):
                if search_string in field:
                    if return_remaining:
                        return ' '.join(map(str, fields[i + num_fields:]))
                    else:
                        return ' '.join(map(str, fields[i + num_fields:i + num_fields + num_to_return]))
            return

        # get ready to capture some info
        out_dict = {}
        kp_mode = False
        layer_name_list = []
        element_set = set()

        # loop over lines of file
        with open(self.filename, 'r') as f:
            for rawline in f.readlines():
                line = rawline.strip()
                fields = line.split()

                # if we found something, try to parse it
                if fields:
                    # look for ion info
                    if fields[0] == 'Ion' and fields[-1] == 'keV':
                        # ion and energy always in the format 'Ion = <element> Energy = <energy> keV'
                        ion = fields[2]
                        energy = float(fields[-2])
                        energy *= 1e3  # energy in file is in keV
                        out_dict['ion'] = ion
                        out_dict['ion_energy'] = energy

                    # look for layer info
                    elif line.startswith('Layer'):
                        # check for new layer and start a new dict for this layer
                        if fields[1].isdigit():
                            layer_num = int(fields[1])
                            # but first make a new dict of the old layer
                            if layer_num > 1:
                                out_dict[layer_name] = {'width': width, 'atom_density': atom_density, 'elements': layer_elements, 'stoich': layer_stoich}
                            # get the name of the layer
                            layer_name = fields_past(line, ':', 1, return_remaining=True)
                            layer_name_list.append(layer_name)
                            # initialize layer composition
                            layer_elements = []
                            layer_stoich = []
                        # check for line width, and density, which sometimes appear on the same line, but sometimes not,
                        # so can't hardcode field numbers
                        if 'Width' in line:
                            # layer width is 2 fields past 'Width'
                            width = float(fields_past(line, 'Width', 2))
                        if 'Density' in line:
                            # layer density is 1 field before units
                            atom_density = float(fields_past(line, 'atoms/cm3', -1))
                        if 'Atomic Percent' in line:
                            # get layer composition
                            element = fields[3]
                            stoich = float(fields[5])/100.0  # convert to fraction
                            layer_elements.append(element)
                            layer_stoich.append(stoich)
                            element_set.add(element)  # add to global list

                    # look for 'Kinchin-Pease' mode
                    if 'Kinchin-Pease' in line:
                        kp_mode = True

                    # get total number of ions ran, and we're done
                    elif 'Total Ions calculated' in line:
                        ions = float(fields[-1][1:])  # get rid of equals sign
                        out_dict['ions'] = ions

                    elif 'Total Target Vacancies' in line:
                        out_dict['total_vacancies'] = int(fields[-2])

                    elif 'Units are' in line:
                        # we are done, so save a few more things before finishing
                        out_dict[layer_name] = {'width': width, 'atom_density': atom_density, 'elements': layer_elements, 'stoich': layer_stoich}
                        out_dict['kp_mode'] = kp_mode
                        out_dict['elements'] = list(element_set)
                        out_dict['layer_names'] = layer_name_list

                        return out_dict


class StoppingTable(SingleTarget):
    def __init__(self, filename):
        column_names = ['energy', 'energy_units', 'electronic_stopping', 'nuclear_stopping', 'range', 'range_units',
                        'longitudinal_straggling', 'longitudinal_straggling_units',
                        'lateral_straggling', 'lateral_straggling_units']
        widths = [7, 4, 13, 11, 8, 3, 9, 3, 9, 3]
        self.which_dashed_line = 2  # data comes after second dashed line
        super(StoppingTable, self).__init__(filename, column_names, widths=widths)

        # convert all values to either Angstroms or eV
        def convert_to_eV(row):
            value = row[0]
            unit = row[1]

            if unit == 'keV':
                return value*1.0e3
            elif unit == 'MeV':
                return value*1.0e6
            else:
                return value

        def convert_to_Ang(row):
            value = row[0]
            unit = row[1]

            if unit == 'um':
                return value*1.0e4
            elif unit == 'mm':
                return value*1.0e7
            else:
                return value

        # loop over columns that have 'units' in name
        for unit_column in [col for col in self.raw_df.columns if 'unit' in col]:
            value_column = self.raw_df.columns[self.raw_df.columns.get_loc(unit_column) - 1]  # get preceding column

            # convert columns
            self.raw_df[value_column] = self.raw_df[[value_column, unit_column]].apply(convert_to_eV, axis=1)
            self.raw_df[value_column] = self.raw_df[[value_column, unit_column]].apply(convert_to_Ang, axis=1)

            # drop unit columns
            self.raw_df.drop(labels=unit_column, axis=1, inplace=True)

        # set range as dataframe index
        self.raw_df.set_index('energy', inplace=True)

        # add a row of zeros at zero energy any if not there already
        if not self.raw_df.index.isin([0.0]).any():
            self.raw_df.loc[0.0] = 0.0
            self.raw_df.sort_index(inplace=True)  # sort it to put zeros on top

        # make some interp1d objects for quick interpolation later
        self.range_interp = interp1d(self.raw_df.index.values, self.raw_df['range'])
        self.straggling_interp = interp1d(self.raw_df.index.values, self.raw_df['longitudinal_straggling'])
        self.nuclear_stopping_interp = interp1d(self.raw_df.index.values, self.raw_df['nuclear_stopping'])
        self.elec_stopping_interp = interp1d(self.raw_df.index.values, self.raw_df['electronic_stopping'])

        # make a new range series with duplicate entires dropped (due to lack of precision in SRIM output)
        ranges = self.raw_df['range']
        ranges = ranges[ranges.diff() > 0.0]
        ranges.loc[0.0] = 0.0  # put this back in
        self.ranges = ranges.sort_index()

    def range_from_energy(self, energy):
        return self.range_interp(energy)

    def straggling_from_energy(self, energy):
        return self.straggling_interp(energy)

    def nuclear_stopping_from_energy(self, energy):
        return self.nuclear_stopping_interp(energy)

    def electronic_stopping_from_energy(self, energy):
        return self.elec_stopping_interp(energy)

    def energy_from_depth(self, depth, initial_energy):
        # find reference depth, which is the range of the initial ion
        reference_depth = self.range_from_energy(initial_energy)

        # subtract ranges from reference depth and keep only positive values
        depths = reference_depth - self.ranges[reference_depth-self.ranges >= 0.0]

        # insert initial_energy if it is not already in there, which will be the case if we are interpolating energy
        if initial_energy not in depths.index:
            depths[initial_energy] = 0.0  # depth with be zero when ion enters slab

        # this gives us depth indexed by energy, so we need to swap it, then sort it
        energy_vs_depth = pd.Series(depths.index.values, index=depths.values).sort_index()

        # now we can interpolate
        return interp1d(energy_vs_depth.index.values, energy_vs_depth.values, bounds_error=False, fill_value=(initial_energy, 0.0))(depth)

    def estimated_damage_from_energy(self, energy, displacement_energy=40.0, factor=0.5):
        return (0.8/2.0/displacement_energy)*self.nuclear_stopping_from_energy(energy) * factor

    def estimated_damage_from_depth(self, depth, initial_energy, displacement_energy=40.0, factor=0.5):
        energy = self.energy_from_depth(depth, initial_energy)
        return self.estimated_damage_from_energy(energy, displacement_energy, factor)

    def estimated_damage_curve(self, initial_energy, displacement_energy=40.0, factor=0.5, trim=False, num=1000):
        energy_profile = self.energy_profile(initial_energy, num=num)
        curve = pd.Series(data=self.estimated_damage_from_energy(energy_profile.values, displacement_energy, factor), index=energy_profile.index)
        if trim:
            # trim past the peak value
            mask = curve.index < curve.idxmax()
            curve = curve[mask]
        return curve

    def energy_profile(self, initial_energy, num=1000):
        ion_range = self.range_from_energy(initial_energy)
        depths = np.linspace(0, ion_range, num=num)
        energies = self.energy_from_depth(depths, initial_energy)
        return pd.Series(data=energies, index=depths)

    def get_target_info(self):
        # get ready to capture some info
        out_dict = {}
        element_list = []
        atom_fraction_list = []
        ready_to_acquire_composition = False

        # loop over lines in file
        with open(self.filename, 'r') as f:
            for rawline in f.readlines():
                line = rawline.strip()
                fields = line.split()

                # we are done if we hit this line and already read in composition
                if ready_to_acquire_composition and set(line) == set(['=']):
                    # store composition
                    out_dict['elements'] = element_list
                    out_dict['atom_fractions'] = atom_fraction_list
                    return out_dict

                # look for ion info
                if 'Ion =' in line and 'Mass =' in line:
                    out_dict['ion'] = fields[2]

                # look for target density
                if 'Target Density' in line:
                    out_dict['atom_density'] = float(fields[-2])

                # get composition if ready
                if ready_to_acquire_composition:
                    element_list.append(fields[0])
                    atom_fraction_list.append(float(fields[2])/100.0)  # convert percent to fraction

                if set(line) == set(['-', ' ']):
                    ready_to_acquire_composition = True

        return out_dict


class RangeTable(LayeredTarget):
    def __init__(self, path_name, filename='RANGE.txt'):
        column_names = ['depth', 'ions', 'recoils']
        super(RangeTable, self).__init__(path_name, column_names, filename=filename)

        # set range as dataframe index
        self.raw_df.set_index('depth', inplace=True)

    def get_ion_range_profile(self):
        # get just the ion range curve
        return self.raw_df['ions']


class DamageTable(LayeredTarget):
    def __init__(self, path_name, filename='VACANCY.txt'):
        column_names = ['depth', 'vacancies_by_ions', 'vacancies_by_recoils']
        super(DamageTable, self).__init__(path_name, column_names, filename=filename)

        # set range as dataframe index
        self.raw_df.set_index('depth', inplace=True)

        # add up columns and make a column of total
        self.raw_df['total_vacancies'] = self.raw_df.sum(axis=1)

        # find area under each curve
        self.sum_dict = {}
        diff_array = np.diff(np.append([0], self.raw_df.index))  # bin widths, need to add a zero point
        for column in self.raw_df.columns:
            self.sum_dict[column] = np.sum(diff_array*self.raw_df[column].values)

    def get_raw_totals(self):
        return self.sum_dict

    def get_total_damage(self):
        return self.sum_dict['total_vacancies']

    def get_total_damage_profile(self):
        return self.raw_df['total_vacancies']


class RecoilTable(LayeredTarget):
    def __init__(self, path_name, filename='E2RECOIL.txt'):
        column_names = ['depth', 'energy_from_ions', 'energy_absorbed_by_recoils']
        super(RecoilTable, self).__init__(path_name, column_names, filename=filename)

        # set range as dataframe index
        self.raw_df.set_index('depth', inplace=True)


class IonizationTable(LayeredTarget):
    def __init__(self, path_name, filename='IONIZ.txt'):
        column_names = ['depth', 'ionization_by_ions', 'ionization_by_recoils']
        super(IonizationTable, self).__init__(path_name, column_names, filename=filename)

        # set range as dataframe index
        self.raw_df.set_index('depth', inplace=True)


class PhononTable(LayeredTarget):
    def __init__(self, path_name, filename='PHONON.txt'):
        column_names = ['depth', 'phonons_by_ions', ' phonons_by_recoils']
        super(PhononTable, self).__init__(path_name, column_names, filename=filename)

        # set range as dataframe index
        self.raw_df.set_index('depth', inplace=True)


class CollisionTable(SingleTarget):
    def __init__(self, path_name, filename='COLLISON.txt'):
        column_names = ['ion_number', 'ion_energy', 'x', 'y', 'z', 'elec_stopping', 'target_atom', 'recoil_energy', 'displacements']
        super(CollisionTable, self).__init__(path_name, column_names, filename=filename)

    def parse_srim_file(self):
        with open(self.filename, 'r') as f:
            for line in f:
                # lines of data have weird superscript 3's as delimiter, and also start with one,
                # so we need to split on that character but also remove empty fields
                if '\xb3' in line.strip():
                    line_split = [item.strip() for item in line.strip().split('\xb3')]
                    fields = [field for field in line_split if field]
                    # some lines might not be data lines
                    if fields[0].isdigit():
                        # convert ion number to int and ion energy from keV to eV
                        fields[0] = str(int(fields[0]))
                        fields[1] = str(float(fields[1])*1e3)
                        yield ' '.join(map(str, fields))+'\n'

    def get_target_info(self):
        out_dict = {}
        element_list = []
        displacement_energy_list = []
        binding_energy_list = []
        kp_mode = False
        with open(self.filename) as f:
            for line in f:
                fields = line.strip().split()
                if 'Ion Name' in line:
                    out_dict['ion'] = fields[-2]
                elif 'Ion Energy' in line:
                    out_dict['ion_energy'] = float(fields[-3][1:])*1e3  # convert to eV
                elif 'Displacement Energy' in line:
                    displacement_energy_list.append(float(fields[-2]))
                    element_list.append(fields[-4])
                elif 'Latt.Binding Energy' in line:
                    binding_energy_list.append(float(fields[-2]))
                elif 'Kinchin-Pease Theory' in line:
                    kp_mode = True
                elif 'COLLISION HISTORY' in line:
                    out_dict['kp_mode'] = kp_mode
                    out_dict['elements'] = element_list
                    out_dict['displacement_energies'] = displacement_energy_list
                    out_dict['binding_energies'] = binding_energy_list

                    return out_dict


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pprint import pprint

    ion_energy = 3e6  # 3 MeV

    # read in srim file
    dirname = os.path.join('data', str(int(ion_energy*1e-6))+'MeV-H-in-Fe-KP-40eV')
    damage_table = DamageTable(dirname)
    range_table = RangeTable(dirname)
    ionization_table = IonizationTable(dirname)
    phonon_table = PhononTable(dirname)
    recoil_table = RecoilTable(dirname)
    stopping_table = StoppingTable(os.path.join('data', 'Hydrogen in Iron.txt'))
    collision_table = CollisionTable(os.path.join(dirname, 'with-collision-data'), filename='COLLISON-truncated.txt')

    # print some basic info
    print 'range for {} MeV ion: {} +/- {} microns'.format(ion_energy*1e-6, stopping_table.range_from_energy(ion_energy)*1e-4, stopping_table.straggling_from_energy(ion_energy)*1e-4)
    print
    print 'some information on target from', damage_table.filename
    pprint(damage_table.target_info)
    print
    print 'some information on target from', stopping_table.filename
    pprint(stopping_table.target_info)
    print
    print 'some information on target from', collision_table.filename
    pprint(collision_table.get_target_info())
    print

    # plot it up
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(12, 8))
    damage_table.get_srim_table().plot(drawstyle='steps-pre', ax=ax[0, 0])
    range_table.get_srim_table().plot(drawstyle='steps-pre', ax=ax[0, 1])
    stopping_table.get_srim_table().plot(loglog=True, ax=ax[1, 0])
    stopping_table.estimated_damage_curve(ion_energy, trim=True).plot(ax=ax[0, 0], label='estimated total damage', c='k', ls='--', lw=1, zorder=0)

    # do a energy vs. depth plot with stopping powers
    # use more points near end of range
    ion_range = float(stopping_table.range_from_energy(ion_energy))
    depth_array = np.linspace(0, 0.9*ion_range, endpoint=False)
    depth_array = np.append(depth_array, np.linspace(0.9*ion_range, ion_range, num=1000))
    energy_array = stopping_table.energy_from_depth(depth_array, ion_energy)  # energies correspond to depths
    ax[1, 1].plot(depth_array, energy_array, label='ion energy')
    ax2 = ax[1, 1].twinx()
    ax2.plot([], [])  # skip a color
    ax2.plot(depth_array, stopping_table.electronic_stopping_from_energy(energy_array), label='electronic stopping')
    ax2.plot(depth_array, stopping_table.nuclear_stopping_from_energy(energy_array), label='nuclear stopping')
    ax2.set_yscale('symlog', linthreshy=1e-3)
    ax2.set_xlim(right=max(ax[0, 1].get_xlim()))  # line up x-limits to match up with plot above
    # combine legends
    lines, labels = ax[1, 1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, fontsize='smaller', loc='lower right')
    ax2.set_ylim(bottom=0)

    # make it look nice
    ax[0, 0].set_ylim([0, damage_table.get_total_damage_profile().max()*1.1])
    ax[1, 0].set_xlabel('energy (eV)')
    for axrow in ax:
        for axes in axrow:
            axes.set_xlim(left=0)
            axes.set_ylim(bottom=0)
            fig.canvas.draw()  # set up tick labels
            if axes is not ax[1, 1]:
                axes.legend(fontsize='smaller', loc='best')
            if axes is not ax[1, 0]:
                axes.set_xlabel('depth (microns)')
                axes.set_xticklabels([int(float(tick.get_text())*1e-4) for tick in axes.get_xticklabels()])
            if axes is ax[1, 1]:
                axes.set_ylabel('energy (MeV)')
                axes.set_yticklabels([float(tick.get_text())*1e-6 for tick in axes.get_yticklabels()])

    # annotate plots
    display_string = str(int(ion_energy*1e-6))+' MeV H in Fe'
    ax[0, 0].text(0.9, 0.9, display_string, transform=ax[0, 0].transAxes, horizontalalignment='right', bbox={'lw': 1, 'facecolor': 'white'})
    ax[1, 1].text(0.9, 0.9, display_string, transform=ax[1, 1].transAxes, horizontalalignment='right', bbox={'lw': 1, 'facecolor': 'white'})
    ax[0, 1].text(0.1, 0.9, display_string, transform=ax[0, 1].transAxes, horizontalalignment='left', bbox={'lw': 1, 'facecolor': 'white'})

    fig.tight_layout()

    # show a plot of energies vs. depth in the collision data
    fig, ax = plt.subplots()
    df = collision_table.get_srim_table()
    ax.scatter(df['x']*1e-4, df['ion_energy']*1e-6, marker='.', s=1, label=None)

    # add lines to show straggling
    straggling = stopping_table.straggling_from_energy(ion_energy)
    energy_profile = stopping_table.energy_profile(ion_energy)
    depths = energy_profile.index.values
    energies = energy_profile.values
    ax.plot(depths*1e-4, energies*1e-6, 'k--', lw=1, label='estimated energy')
    ax.plot(depths*(1.0+2.0*straggling/ion_range)*1e-4, energies*1e-6, c='grey', dashes=[2, 2], lw=1, label=r'range $\pm 2\sigma$')  # energy + 2*straggling
    ax.plot(depths*(1.0-2.0*straggling/ion_range)*1e-4, energies*1e-6, c='grey', dashes=[2, 2], lw=1)  # energy - 2*straggling

    ax.set_xlabel('depth (microns)')
    ax.set_ylabel('ion energy (MeV)')
    ax.legend(loc='lower left')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.text(0.9, 0.9, display_string, transform=ax.transAxes, horizontalalignment='right', bbox={'lw': 1, 'facecolor': 'white'})

    plt.show()
