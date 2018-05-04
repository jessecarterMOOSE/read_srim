from StringIO import StringIO
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d


class SRIMTable (object):
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


class StoppingTable(SRIMTable):
    def __init__(self, filename):
        header_keywords = ['Energy', 'Elec', 'Nuclear', 'Range', 'Straggling']
        column_names = ['energy', 'energy_units', 'electronic_stopping', 'nuclear_stopping', 'range', 'range_units',
                        'longitudinal_straggling', 'longitudinal_straggling_units',
                        'lateral_straggling', 'lateral_straggling_units']
        widths = [7, 4, 13, 11, 8, 3, 9, 3, 9, 3]
        super(StoppingTable, self).__init__(filename, header_keywords, column_names, widths)

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
        self.nuclear_stopping_interp = interp1d(self.raw_df.index.values, self.raw_df['nuclear_stopping'])
        self.elec_stopping_interp = interp1d(self.raw_df.index.values, self.raw_df['electronic_stopping'])

        # make a new range series with duplicate entires dropped (due to lack of precision in SRIM output)
        ranges = self.raw_df['range']
        ranges = ranges[ranges.diff() > 0.0]
        ranges.loc[0.0] = 0.0  # put this back in
        self.ranges = ranges.sort_index()

    def range_from_energy(self, energy):
        return self.range_interp(energy)

    def nuclear_stopping_from_energy(self, energy):
        return self.nuclear_stopping_interp(energy)

    def electronic_stopping_from_energy(self, energy):
        return self.elec_stopping_interp(energy)

    def energy_from_depth(self, depth, initial_energy):
        # find reference depth, which is the range of the initial ion
        reference_depth = self.range_from_energy(initial_energy)

        # subtract ranges from reference depth and keep only positive values
        depths = reference_depth - self.ranges[reference_depth-self.ranges >= 0.0]

        # this gives us depth indexed by energy, so we need to swap it, then sort it
        energy_vs_depth = pd.Series(depths.index.values, index=depths.values).sort_index()

        # now we can interpolate
        return interp1d(energy_vs_depth.index.values, energy_vs_depth.values)(depth)


class RangeTable(SRIMTable):
    def __init__(self, filename):
        header_keywords = ['DEPTH', 'Recoil']
        column_names = ['depth', 'ions', 'recoils']
        super(RangeTable, self).__init__(filename, header_keywords, column_names)

        # set range as dataframe index
        self.raw_df.set_index('depth', inplace=True)

    def get_ion_range_profile(self):
        # get just the ion range curve
        return self.raw_df['ions']


class DamageTable(SRIMTable):
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


if __name__ == "__main__":
    import os.path
    import matplotlib.pyplot as plt

    ion_energy = 2e6  # 2 MeV

    # read in srim file
    damage_table = DamageTable(os.path.join('data', '2MeV H in Fe KP 40eV', 'VACANCY.txt'))
    range_table = RangeTable(os.path.join('data', '2MeV H in Fe KP 40eV', 'RANGE.txt'))
    stopping_table = StoppingTable(os.path.join('data', 'Hydrogen in Iron.txt'))

    # print some basic info
    print 'range for {} MeV ion: {} microns'.format(ion_energy * 1e-6, stopping_table.range_from_energy(ion_energy)*1e-4)

    # plot it up
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(12, 8))
    damage_table.get_srim_table().plot(drawstyle='steps-post', ax=ax[0, 0])
    range_table.get_srim_table().plot(drawstyle='steps-post', ax=ax[0, 1])
    stopping_table.get_srim_table().plot(loglog=True, ax=ax[1, 0])

    # do a energy vs. depth plot with stopping powers
    axes = ax[1, 1]
    # use more points near end of range
    ion_range = float(stopping_table.range_from_energy(ion_energy))
    depth_array = np.linspace(0, 0.9*ion_range, endpoint=False)
    depth_array = np.append(depth_array, np.linspace(0.9*ion_range, ion_range, num=1000))
    energy_array = stopping_table.energy_from_depth(depth_array, ion_energy)  # energies correspond to depths
    axes.plot(depth_array, energy_array, label='ion energy')
    ax2 = axes.twinx()
    ax2.plot([], [])  # skip a color
    ax2.plot(depth_array, stopping_table.electronic_stopping_from_energy(energy_array), label='electronic stopping')
    ax2.plot(depth_array, stopping_table.nuclear_stopping_from_energy(energy_array), label='nuclear stopping')
    ax2.set_yscale('symlog', linthreshy=1e-3)
    # combine legends
    lines, labels = axes.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2)
    ax2.set_ylim(bottom=0)

    # make it look nice
    for axrow in ax:
        for axes in axrow:
            axes.set_xlim(left=0)
            axes.set_ylim(bottom=0)
            if axes is not ax[1, 1]:
                axes.legend(fontsize=8)

    fig.tight_layout()
    plt.show()
