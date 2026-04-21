# =============================================================================
# extract_model_params.py -- Model Parameter Extractor for SPICE Dataset
#
# Extracts key physical parameters from .pm model files for use in dataset generation and feature engineering.
#
# Main function:
#   - extract_model_params(model_file): Parses a SPICE model file and returns a dictionary of extracted NMOS and PMOS parameters.
# =============================================================================

import re
import os

MODELS_DIR = "models"  # Update if models/ is moved

def extract_model_params(model_file):

    if not os.path.exists(model_file):
        return {}
        
    try:

        with open(model_file, 'r') as f:
            content = f.read()
        
        params = {}
        nmos_match = re.search(r'\.model\s+nmos.*?(?=\.model\s+pmos|$)', content, re.DOTALL | re.IGNORECASE)
        pmos_match = re.search(r'\.model\s+pmos.*?$', content, re.DOTALL | re.IGNORECASE)
        
        if nmos_match:

            nmos_section = nmos_match.group(0)
            params.update(extract_device_params(nmos_section, 'N'))
            
        if pmos_match:

            pmos_section = pmos_match.group(0)
            params.update(extract_device_params(pmos_section, 'P'))
            
        return params
        
    except Exception as e:

        print(f"Warning: Could not extract parameters from {model_file}: {e}")

        return {}

def extract_device_params(section, device_type):

    params = {}
    
    param_patterns = {
        f'VTH0_{device_type}': r'vth0\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
        f'TOX_{device_type}': r'toxe\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
        f'TOXP_{device_type}': r'toxp\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
        f'TOXM_{device_type}': r'toxm\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
        f'U0_{device_type}': r'u0\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
        f'UA_{device_type}': r'ua\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
        f'UB_{device_type}': r'ub\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
        f'UC_{device_type}': r'uc\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
        f'UA1_{device_type}': r'ua1\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
        f'UB1_{device_type}': r'ub1\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
        f'UC1_{device_type}': r'uc1\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
        f'VSAT_{device_type}': r'vsat\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
        f'XJ_{device_type}': r'xj\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
        f'NDEP_{device_type}': r'ndep\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
        f'NSD_{device_type}': r'nsd\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
        f'NF_{device_type}': r'nfactor\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
        f'ETA0_{device_type}': r'eta0\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
        f'ETAB_{device_type}': r'etab\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
        f'VFB_{device_type}': r'vfb\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
        f'K1_{device_type}': r'k1\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
        f'K2_{device_type}': r'k2\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
        f'K3_{device_type}': r'k3\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
        f'K3B_{device_type}': r'k3b\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
        f'DVT0_{device_type}': r'dvt0\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
        f'DVT1_{device_type}': r'dvt1\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
        f'DVT2_{device_type}': r'dvt2\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
        f'DSUB_{device_type}': r'dsub\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
        f'MINV_{device_type}': r'minv\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
        f'VOFF_{device_type}': r'voff\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
        f'PCLM_{device_type}': r'pclm\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
        f'PDIBLC1_{device_type}': r'pdiblc1\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
        f'PDIBLC2_{device_type}': r'pdiblc2\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
        f'DROUT_{device_type}': r'drout\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
        f'RDSW_{device_type}': r'rdsw\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
        f'RSH_{device_type}': r'rsh\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
        f'CGSO_{device_type}': r'cgso\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
        f'CGDO_{device_type}': r'cgdo\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
        f'CGBO_{device_type}': r'cgbo\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
        f'ALPHA0_{device_type}': r'alpha0\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
        f'ALPHA1_{device_type}': r'alpha1\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
        f'BETA0_{device_type}': r'beta0\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
        f'AGIDL_{device_type}': r'agidl\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
        f'BGIDL_{device_type}': r'bgidl\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
        f'CGIDL_{device_type}': r'cgidl\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
        f'EGIDL_{device_type}': r'egidl\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
    }
    
    for param_name, pattern in param_patterns.items():

        match = re.search(pattern, section, re.IGNORECASE)

        if match:

            try:
                params[param_name] = float(match.group(1))

            except ValueError:
                params[param_name] = None

        else:
            params[param_name] = None
    
    return params

def test_extraction():

    models_dir = "models"

    if not os.path.exists(models_dir):

        print("Models directory not found!")

        return
        
    for filename in os.listdir(models_dir):

        if filename.endswith('.pm'):

            model_path = os.path.join(models_dir, filename)
            print(f"\n=== {filename} ===")
            params = extract_model_params(model_path)
            
            for param, value in params.items():

                if value is not None:
                    print(f"{param}: {value:.3e}")

                else:
                    print(f"{param}: Not found")

if __name__ == "__main__":
    test_extraction()
    