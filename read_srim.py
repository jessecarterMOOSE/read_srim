from StringIO import StringIO
import pandas as pd
import numpy as np


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


class RangeTable(SRIMTable):
    def __init__(self, filename):
        header_keywords = ['DEPTH', 'Fe', 'Recoil']
        column_names = ['depth', 'ions', 'recoils']
        super(RangeTable, self).__init__(filename, header_keywords, column_names)

        # set range as dataframe index
        self.raw_df.set_index('depth', inplace=True)


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


if __name__ == "__main__":
    import os.path
    import matplotlib.pyplot as plt

    # read in srim file
    damage_table = DamageTable(os.path.join('data', '78.7keV Fe in Fe KP 40eV', 'VACANCY.txt'))
    range_table = RangeTable(os.path.join('data', '78.7keV Fe in Fe KP 40eV', 'RANGE.txt'))
    stopping_table = StoppingTable(os.path.join('data', 'Iron in Iron.txt'))

    # plot it up
    fig, ax = plt.subplots(ncols=3, figsize=(12, 4))
    damage_table.get_srim_table().plot(drawstyle='steps-post', ax=ax[0])
    range_table.get_srim_table().plot(drawstyle='steps-post', ax=ax[1])
    stopping_table.get_srim_table().plot(loglog=True, ax=ax[2])

    fig.tight_layout()
    plt.show()
