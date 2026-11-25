import joblib
import numpy as np
from pathlib import Path

# Folder holding this file and the saved scalers, regardless of where Python was started.
SCALER_DIR = Path(__file__).resolve().parent

pkl_file = {
    'FOODS_1':{
        'input':[SCALER_DIR/'input_pt_FOODS_1.pkl', SCALER_DIR/'input_rs_FOODS_1.pkl'],
        'output':[SCALER_DIR/'output_mm_FOODS_1.pkl']
    },
    'FOODS_2':{
        'input':[SCALER_DIR/'input_pt_FOODS_2.pkl', SCALER_DIR/'input_rs_FOODS_2.pkl'],
        'output':[SCALER_DIR/'output_mm_FOODS_2.pkl']
    },
    'FOODS_3':{
        'input':[SCALER_DIR/'input_pt_FOODS_3.pkl', SCALER_DIR/'input_rs_FOODS_3.pkl'],
        'output':[SCALER_DIR/'output_mm_FOODS_3.pkl']
    },
    'HOBBIES_1':{
        'input':[SCALER_DIR/'input_pt_HOBBIES_1.pkl', SCALER_DIR/'input_rs_HOBBIES_1.pkl'],
        'output':[SCALER_DIR/'output_mm_HOBBIES_1.pkl']
    },
    'HOBBIES_2':{
        'input':[SCALER_DIR/'input_pt_HOBBIES_2.pkl', SCALER_DIR/'input_rs_HOBBIES_2.pkl'],
        'output':[SCALER_DIR/'output_mm_HOBBIES_2.pkl']
    },
    'HOUSEHOLD_1':{
        'input':[SCALER_DIR/'input_pt_HOUSEHOLD_1.pkl', SCALER_DIR/'input_rs_HOUSEHOLD_1.pkl'],
        'output':[SCALER_DIR/'output_mm_HOUSEHOLD_1.pkl']
    },
    'HOUSEHOLD_2':{
        'input':[SCALER_DIR/'input_pt_HOUSEHOLD_2.pkl', SCALER_DIR/'input_rs_HOUSEHOLD_2.pkl'],
        'output':[SCALER_DIR/'output_mm_HOUSEHOLD_2.pkl']
    },
}


class CustomScaler:

    def __init__(self, files=pkl_file, clip=5.0, output_sign=(1.0, -1.0)):
        self.clip = clip
        # Sign of each output column after scaling: [reward (+), wastage (-)].
        self.output_sign = np.asarray(output_sign, dtype=np.float64)
        self.pkl = {
            product: {
                'input':  [joblib.load(filename) for filename in files[product]['input']],
                'output': [joblib.load(filename) for filename in files[product]['output']],
            }
            for product in files
        }

        self.price_range = {
            'FOODS_1':      [1.97,3.33],
            'FOODS_2':      [2.44,3.99],
            'FOODS_3':      [1.44,2.70],
            'HOBBIES_1':    [1.59,5.91],
            'HOBBIES_2':    [0.66,3.97],
            'HOUSEHOLD_1':  [2.96,4.70],
            'HOUSEHOLD_2':  [4.39,5.99],
        }

    @staticmethod
    def _apply(transformers, x):
        x = np.asarray(x, dtype=np.float64)
        single = x.ndim == 1
        if single:
            x = x.reshape(1, -1)           
        for t in transformers:
            x = t.transform(x)
        return x[0] if single else x

    def input_scaler(self, product, x):
        """x = [yesterday_price, yesterday_demand, yesterday_stock], one row (3,) or a batch (n, 3).

        The price column uses price_scaler (DP's price range -> [-1, 1], the same scale as the
        action); demand and stock go through the fitted PowerTransformer -> RobustScaler.
        """
        x = np.asarray(x, dtype=np.float64)
        single = x.ndim == 1
        if single:
            x = x.reshape(1, -1)
        price = self.price_scaler(product, x[:, :1]).astype(np.float64)
        rest = self._apply(self.pkl[product]['input'], x[:, 1:])
        out = np.clip(np.hstack([price, rest]), -self.clip, self.clip).astype(np.float32)
        return out[0] if single else out

    def output_scaler(self, product, y):
        y = self._apply(self.pkl[product]['output'], y)
        return (y * self.output_sign).astype(np.float32)   

    def price_scaler(self, product, price, min_=-1.0, max_=1.0):
        lo, hi = self.price_range[product]
        a = (np.asarray(price, dtype=np.float64) - lo) / (hi - lo) * (max_ - min_) + min_
        return a.astype(np.float32)

    def price_inverse(self, product, action, min_=-1.0, max_=1.0):
        lo, hi = self.price_range[product]
        a = np.asarray(action, dtype=np.float64)
        return np.round(lo + (a - min_) / (max_ - min_) * (hi - lo), 2)