import pyrfu
pyrfu.plot.set_color_cycle(pal = "pyrfu")

import glob
import matplotlib.pyplot as plt
import pickle

def list_files(extension, verbose = False):

    files = glob.glob(extension)

    if verbose:
        
        for i, file in enumerate(files):
    
            print(i, file)
            
        return
        
    return files

def load_fig(idx_files, extension):

    files = list_files(extension)

    # idxs = [input("Index of figure in the list to plot: ")]
    # idxs = input("List of indices of figures in the list to plot (separate with comma): ").split(",")
    # idxs = [int(idx) for idx in idxs]
    axs = []
    for idx in idx_files:
        
        with open(files[idx],'rb') as fid:
            
            axs.append(pickle.load(fid))

    return axs