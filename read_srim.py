from StringIO import StringIO
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d


class SRIMTable(object):
    def __init__(self, filename, header_keywords, column_names, widths=None):
        # generate file-like object to pass to pandas
        file_io = StringIO()
        for line in self.parse_srim_file(filename, header_keywords):
            file_io.write(line)
        file_io.seek(0)  # rewind

        if widths:
            self.raw_df = pd.read_fwf(file_io, widths=widths, header=None, names=column_names).fillna(value=0.0)
        else:
            self.raw_df = pd.read_csv(file_io, delim_whitespace=True, header=None, names=column_names)

        # store filename as a member variable
        self.filename = filename

    def get_srim_table(self):
        return self.raw_df

    @staticmethod
    def parse_srim_file(filename, header_keywords):
        found_headers = False
        yielded = False
        with open(filename, 'r') as f:
            for line in f:
                if yielded and len(line.strip()) > 0 and not line.lstrip()[0].isdigit():
                    break
                elif yielded and len(line.strip()) > 0 and line.lstrip()[0].isdigit():
                    yield line
                elif found_headers and len(line.strip()) > 0 and line.lstrip()[0].isdigit():
                    yielded = True
                    yield line
                num_headers_found = 0
                if not found_headers:
                    for header in header_keywords:
                        if header in line:
                            num_headers_found += 1
                    if num_headers_found == len(header_keywords):
                        found_headers = True


class SingleTarget(SRIMTable):
    # class for handling data files with no layer information, e.g. StoppingTable
    def __init__(self, filename, header_keywords, column_names, widths=None):
        super(SingleTarget, self).__init__(filename, header_keywords, column_names, widths=widths)


class LayeredTarget(SRIMTable):
    # class for handling data files from SRIM runs where the target can contain many layers
    def __init__(self, filename, header_keywords, column_names, widths=None):
        super(LayeredTarget, self).__init__(filename, header_keywords, column_names, widths=widths)


class StoppingTable(SingleTarget):
    def __init__(self, filename):
        header_keywords = ['Energy', 'Elec', 'Nuclear', 'Range', 'Straggling']
        column_names = ['energy', 'energy_units', 'electronic_stopping', 'nuclear_stopping', 'range', 'range_units',
                        'longitudinal_straggling', 'longitudinal_straggling_units',
                        'lateral_straggling', 'lateral_straggling_units']
        widths = [7, 4, 13, 11, 8, 3, 9, 3, 9, 3]
        super(StoppingTable, self).__init__(filename, header_keywords, column_names, widths=widths)

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

    def estimated_damage_from_energy(self, energy, displacement_energy=40.0, ratio=0.5):
        return (0.8/2.0/displacement_energy)*self.nuclear_stopping_from_energy(energy)*ratio

    def estimated_damage_from_depth(self, depth, initial_energy, displacement_energy=40.0, ratio=0.5):
        energy = self.energy_from_depth(depth, initial_energy)
        return self.estimated_damage_from_energy(energy, displacement_energy, ratio)

    def estimated_damage_curve(self, initial_energy, displacement_energy=40.0, ratio=0.5, trim=False):
        ion_range = self.range_from_energy(initial_energy)
        depths = np.linspace(0, ion_range, num=1000)
        curve = pd.Series(data=self.estimated_damage_from_depth(depths, initial_energy, displacement_energy, ratio), index=depths)
        if trim:
            # trim past the peak value
            mask = curve.index < curve.idxmax()
            curve = curve[mask]
        return curve


class RangeTable(LayeredTarget):
    def __init__(self, filename):
        header_keywords = ['DEPTH', 'Recoil']
        column_names = ['depth', 'ions', 'recoils']
        super(RangeTable, self).__init__(filename, header_keywords, column_names)

        # set range as dataframe index
        self.raw_df.set_index('depth', inplace=True)

    def get_ion_range_profile(self):
        # get just the ion range curve
        return self.raw_df['ions']


class DamageTable(LayeredTarget):
    def __init__(self, filename):
        header_keywords = ['TARGET', 'VACANCIES', 'VACANCIES']
        column_names = ['depth', 'vacancies_by_ions', 'vacancies_by_recoils']
        super(DamageTable, self).__init__(filename, header_keywords, column_names)

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
    def __init__(self, filename):
        header_keywords = ['DEPTH', 'from', 'Absorbed']
        column_names = ['depth', 'energy_from_ions', 'energy_absorbed_by_recoils']
        super(RecoilTable, self).__init__(filename, header_keywords, column_names)

        # set range as dataframe index
        self.raw_df.set_index('depth', inplace=True)


class CollisionTable(SingleTarget):
    def __init__(self, filename):
        column_names = ['ion_number', 'ion_energy', 'x', 'y', 'z', 'elec_stopping', 'target_atom', 'recoil_energy', 'displacements']
        super(CollisionTable, self).__init__(filename, None, column_names)

    def parse_srim_file(self, filename, header_keywords):
        with open(filename, 'r') as f:
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


if __name__ == "__main__":
    import os.path
    import matplotlib.pyplot as plt

    ion_energy = 2e6  # 2 MeV

    # read in srim file
    damage_table = DamageTable(os.path.join('data', str(int(ion_energy*1e-6))+'MeV-H-in-Fe-KP-40eV', 'VACANCY.txt'))
    range_table = RangeTable(os.path.join('data', str(int(ion_energy*1e-6))+'MeV-H-in-Fe-KP-40eV', 'RANGE.txt'))
    recoil_table = RecoilTable(os.path.join('data', '78.7keV-Fe-in-Fe-KP-40eV', 'E2RECOIL.txt'))
    stopping_table = StoppingTable(os.path.join('data', 'Hydrogen in Iron.txt'))
    collision_table = CollisionTable(os.path.join('data', str(int(ion_energy * 1e-6)) + 'MeV-H-in-Fe-KP-40eV', 'with-collision-data', 'COLLISON-truncated.txt'))

    # print some basic info
    print 'range for {} MeV ion: {} +/- {} microns'.format(ion_energy*1e-6, stopping_table.range_from_energy(ion_energy)*1e-4, stopping_table.straggling_from_energy(ion_energy)*1e-4)

    # plot it up
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(12, 8))
    damage_table.get_srim_table().plot(drawstyle='steps-post', ax=ax[0, 0])
    range_table.get_srim_table().plot(drawstyle='steps-post', ax=ax[0, 1])
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
    ax.scatter(df['x']*1e-4, df['ion_energy']*1e-6, marker='.', s=1)

    # add lines to show straggling
    straggling = stopping_table.straggling_from_energy(ion_energy)
    depths = np.linspace(0, ion_range)
    energies = stopping_table.energy_from_depth(depths, ion_energy)
    ax.plot(depths*1e-4, energies*1e-6, 'k--', lw=1)
    ax.plot(depths*(1.0+2.0*straggling/ion_range)*1e-4, energies*1e-6, 'k--', lw=1)  # energy + 2*straggling
    ax.plot(depths*(1.0-2.0*straggling/ion_range)*1e-4, energies*1e-6, 'k--', lw=1)  # energy - 2*straggling

    ax.set_xlabel('depth (microns)')
    ax.set_ylabel('ion energy (MeV)')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.text(0.9, 0.9, display_string, transform=ax.transAxes, horizontalalignment='right', bbox={'lw': 1, 'facecolor': 'white'})

    plt.show()
