from StringIO import StringIO
import pandas as pd
import numpy as np


class SRIMTable (object):
    def __init__(self, filename, header_keywords, column_names):
        # generate file-like object to pass to pandas
        file_io = StringIO()
        for line in self.parse_srim_file(filename, header_keywords):
            file_io.write(line)
        file_io.seek(0)  # rewind

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
                    yield line.lstrip()
                elif found_headers and len(line.strip()) > 0 and line.lstrip()[0].isdigit():
                    yielded = True
                    yield line.lstrip()
                num_headers_found = 0
                if not found_headers:
                    for header in header_keywords:
                        if header in line:
                            num_headers_found += 1
                    if num_headers_found == len(header_keywords):
                        found_headers = True


class DamageTable(SRIMTable):
    def __init__(self, filename):
        header_keywords = ['TARGET', 'VACANCIES', 'VACANCIES']
        column_names = ['depth', 'vacancies_by_ions', 'vacancies_by_recoils']
        super(DamageTable, self).__init__(filename, header_keywords, column_names)

        # add up columns and make a column of total
        self.raw_df.set_index('depth', inplace=True)
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
    print 'raw table:', damage_table.get_srim_table()
    print 'total vacancies/ion:', damage_table.get_total_damage()

    # plot it up
    fig, ax = plt.subplots()
    damage_table.get_srim_table().plot(drawstyle='steps-post', ax=ax)

    fig.tight_layout()
    plt.show()
