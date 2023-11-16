import os
import pyarrow.parquet as pq
import pandas as pd

class SanityCheck():
    def __init__(self, file_path, pq_filename_raw):
        self.file_path = file_path
        self.pq_filename = pq_filename_raw
        self.outliers_index = []
        self.outliers_percent = ""
        self.pq_filepath = os.path.join(file_path, pq_filename_raw)
        pass
    
    def find_outliers(self, window, std_factor):

        print("Reading file " + str(self.pq_filepath))
        
        df = pq.read_table(self.pq_filepath).to_pandas()

        # Calculate wall latency (timestamps.client.send.wall - timestamps.server.receive.wall)
        wall_latency = pd.Series(data=(df['timestamps.server.receive.wall'] - df['timestamps.client.send.wall'])/1e6, index=df.index)

        # Rolling mean denoising
        rm = wall_latency.rolling(window).mean()

        # Outlier detection
        roll = rm.rolling(window)

        avg = roll.mean()
        std = roll.std()
        outliers = (rm > avg+std_factor*std) + (rm < avg-std_factor*std)

        self.outliers_index = []
        for oi in range(len(outliers)):
            if outliers[oi]:
                self.outliers_index.append(oi)

        self.outliers_percent = '{:.2%}'.format(len(self.outliers_index)/len(rm))

        return len(self.outliers_index), self.outliers_percent

    def save_outlier_file(self):
        txt_filename = os.path.splitext(self.pq_filename)[0] + "_" + self.outliers_percent.replace(".", "_") + "_outliers.txt"
        filename = os.path.join(self.file_path, txt_filename)

        try:
            os.remove(filename)
            print("Removed already existing file " + filename)
        except OSError:
            pass

        with open(filename, mode='wt', encoding='utf-8') as txtfile:
                txtfile.write('\n'.join(str(x) for x in self.outliers_index))
                txtfile.write('\n')
