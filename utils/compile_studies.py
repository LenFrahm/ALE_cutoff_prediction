import pandas as pd
import nibabel as nb
import numpy as np

def compile_studies(tags, tasks):
    exp_to_use = tasks[tasks.Name == tags[0][1:]].ExpIndex.to_list()
    exp_not_to_use = []
    masks = []
    mask_names = []
    if len(tags) > 1:
        for idx,tag in enumerate(tags[1:]):
            operation = tag[0]
            if operation == '+': # AND
                tag = tag[1:].lower()
                exp_to_use.append(tasks[tasks.Name == tag].ExpIndex.to_list()[0])

            elif operation == '?': # OR
                tag = tag[1:].lower()
                or_union = list(set(tasks[tasks.Name == tag].ExpIndex.to_list()[0] + exp_to_use[idx]))
                exp_to_use.pop()
                exp_to_use.append(or_union)


            elif operation == "-": # NOT
                tag = tag[1:].lower()
                exp_not_to_use.append(tasks[tasks.Name == tag].ExpIndex.to_list()[0])
                
            elif operation == '$':
                mask_file = condition[1:]
                mask = nb.load(mask_file).get_fdata()
                if np.unique(mask).shape[0] == 2:
                    #binary mask
                    masks.append(mask.astype(bool))
                else:
                    masks.append(mask.astype(int))
                mask_names.append(mask_file[:-4])

    if len(exp_to_use) > 1:
        use_sets = map(set, exp_to_use)
        exp_to_use = list(set.intersection(*use_sets))
    else:
        exp_to_use = exp_to_use[0]

    if len(exp_not_to_use) == 1:
        exp_to_use = list(set(exp_to_use) - set(exp_not_to_use[0]))
    elif len(exp_not_to_use) > 1:
        exp_not_to_use = list(set([item for sublist in exp_not_to_use for item in sublist])) # flatten list and remove duplicates
        exp_to_use = list(set(exp_to_use) - set(exp_not_to_use))
        
    return exp_to_use, masks, mask_names