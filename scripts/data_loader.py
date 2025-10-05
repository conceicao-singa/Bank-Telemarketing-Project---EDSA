
import pandas as pd
class MLUtils:
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.data = None

    # -------------------
    # Data Loading
    # -------------------
    def load_data(self, path=None):
        """Load CSV data"""
        if path is None:
            path = self.data_path
        self.data = pd.read_csv(path)
        return self.data