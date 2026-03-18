import numpy as np
from skgstat import DirectionalVariogram
import traceback

def test_variogram():
    # Create dummy data (30 points)
    np.random.seed(42)
    x = np.random.uniform(0, 100, 30)
    y = np.random.uniform(0, 100, 30)
    z = 10 + 2*x + 3*y + np.random.normal(0, 5, 30)
    coords = np.column_stack((x, y))
    
    azimuth = 0 # North
    math_angle = 90 - azimuth
    tolerance = 45
    n_lags = 15
    maxlag = 50
    model = 'spherical'
    
    print(f"Testing Variogram with math_angle={math_angle}, tolerance={tolerance}...")
    
    from skgstat import Variogram
    try:
        # Testing if Variogram accepts direction parameters directly
        dv = Variogram(
            coords, z, 
            azimuth=math_angle, tolerance=tolerance,
            n_lags=n_lags, maxlag=maxlag, model=model
        )
        print("Successfully created Variogram object.")
        print(f"Experimental: {dv.experimental}")
        print(f"Bins: {dv.bins}")
        
    except Exception as e:
        print("Failed to create Variogram!")
        traceback.print_exc()

if __name__ == "__main__":
    test_variogram()
