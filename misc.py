import os

def lcm(x, y):
    """Calculate least common multiple of two integers. 
    :param x: [int] first integer 
    :param y: [int] second integer 
    :returns: [int] least common multiple 
    """
    # Choose greater number. 
    if x > y: greater = x
    else: greater = y

    while(True):
        if((greater % x == 0) and (greater % y == 0)):
            lcm = greater
            break
        greater += 1
    return lcm

def cleanAtom(pdb_file, out_file=None, ext=".clean.pdb"): 
    """Extract ATOM and TER records from PDB file and write them to a new file. 
    :param pdb_file: [str] path of PDB file from which records are to be extracted 
    :param out_file: [str] output filename; defaults to <pdb_file>.clean.pdb 
    :param ext: [str] file extension of output file; defaults to ".clean.pdb". 
    """ 
    # Find all ATOM and TER lines. 
    with open(pdb_file, "r") as fid: 
        good = [l for l in fid if l.startswith(("ATOM", "TER"))] 

    # Default output file to <pdb_file>.clean.pdb. 
    if out_file is None: 
        out_file = os.path.splitext(pdb_file)[0] + ext 
  
    # Write selected records to new file. 
    with open(out_file, "w") as fid: 
        fid.writelines(good) 

def dump_pdb(u, frame, sel="protein"):
    """Dump structures from trajectory for given list of frames.
    :param u:           [mda universe/atomgrp] universe containing the structure
    :param frames:      [int] frame to dump
    :param sel:         [str] selection string to select group to dump, e.g. 'protein'
    """
    #os.makedirs("./sep/", exist_ok=True) 
    # Save structure.
    mda_frame = u.trajectory[frame]
    structure = u.select_atoms(sel)
    #save_as_tmp = f"./sep/{frame}.pdb"
    save_as_tmp=f"./{frame}.pdb"
    structure.write(save_as_tmp)
