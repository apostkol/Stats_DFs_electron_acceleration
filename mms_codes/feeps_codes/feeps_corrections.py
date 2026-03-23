from pyrfu import mms

def feeps_corrections(dpf_feeps):

    dpf_feeps_alle_clean_sun_removed = []
    
    for ic in range(len(dpf_feeps)):
        
        ### feeps_correct_energies
        ### feeps_flat_field_corrections

        dpf_feeps_correct_energies = mms.feeps_correct_energies(dpf_feeps[ic])
        dpf_feeps_flat_field_corrections = mms.feeps_flat_field_corrections(dpf_feeps_correct_energies)
        dpf_feeps_remove_bad_data = mms.feeps_remove_bad_data(dpf_feeps_flat_field_corrections)

        split_int_ch = mms.feeps_split_integral_ch(dpf_feeps_remove_bad_data)
        dpf_feeps_alle_clean = split_int_ch[0]; dpf_feeps_alle_500kev = split_int_ch[1];
        dpf_feeps_alle_clean_sun_removed.append(dpf_feeps_alle_clean)
        # dpf_feeps_alle_clean_sun_removed.append(mms.feeps_remove_sun(dpf_feeps_alle_clean))

        
    return dpf_feeps_alle_clean_sun_removed

