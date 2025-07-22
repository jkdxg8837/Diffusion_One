import os
import torch

def svd_weights(weights):
    # Only operate on 2D tensors (matrices)
    if weights.ndim == 2:
        U, S, Vh = torch.linalg.svd(weights, full_matrices=False)
        return {'U': U.cpu(), 'S': S.cpu(), 'V': Vh.cpu()}
    return None

def process_pt_file(filepath):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = torch.load(filepath, map_location=device)
    if isinstance(data, dict) and 'state_dict' in data:
        state_dict = data['state_dict']
    elif isinstance(data, dict):
        state_dict = data
    else:
        return {}

    svd_dict = {}
    for name, weights in state_dict.items():
        svd_result = svd_weights(weights.float())
        if svd_result is not None:
            svd_dict[name] = svd_result
    return svd_dict

def main():
    result = {}
    for fname in os.listdir('/dcs/pg24/u5649209/data/workspace/diffusers/f_named_grads'):
        if fname.endswith('.pt'):
            print(f'Processing {fname}')
            svd_dict = process_pt_file('/dcs/pg24/u5649209/data/workspace/diffusers/f_named_grads/'+ fname)
            
            if svd_dict:  # Only save if there are SVD results
                svd_fname = fname.replace('.pt', '_svd.pt')
                torch.save(svd_dict, svd_fname)

if __name__ == '__main__':
    main()