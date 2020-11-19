import hyppopy.HyppopyProject
import hyppopy.solvers.DynamicPSOSolver
import numpy
import pyrosetta as ros # Rosetta molecular modeling suite
import subprocess       # subprocess management
import os               # misecellaneous operating system interfaces
import sys              # system-specific parameters and functions
import errno            # standard errno system symbols
import re               # regular expression operations
import glob             # unix style pathname pattern expansion
from mpi4py import MPI
import MDAnalysis as mda
import misc as misc
import watchdog.events
import watchdog.observers
import threading
import contextlib
import io
import numpy
import struct

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = io.BytesIO()
    yield
    sys.stdout = save_stdout

class TrajHandler(watchdog.events.PatternMatchingEventHandler):
    """Event handler class that takes the event dispatched by the observer
    and performs some action. The handler matches given patterns with file paths
    associated with some occurring event. Only *.trr files are considered.
    """
    def __init__(self):
        super().__init__(patterns="*.trr")
        self.cv = threading.Condition()

    # Modification of xtc will be watched.
    def on_modified(self, event):
        with self.cv:
            self.cv.notify_all()


def read_coordinates(path, frame=0):
    """Read coordinates of specific frame from trr file.
    :param path:  [str] path to trr file
    :param frame: [int] frame number
    :returns:     [float array] 3d coordinates of all atoms
    """
    HEADER = 120
    ATOM_OFFSET = 64

    with open(path, 'rb') as handle:
        # Determine number of atoms in TRR file.
        handle.seek(ATOM_OFFSET, os.SEEK_SET)
        atoms = int.from_bytes(handle.read(4), byteorder='big')

        # Find coordinate position in file for frame.
        bytes_to_read = atoms * 3 * 4  # atoms * dimensions (x, y, z) * bytes per coord (float32)
        handle.seek(frame * (bytes_to_read + HEADER) + HEADER, os.SEEK_SET)

        # Read coordinates.
        coordinates_bytes = handle.read(bytes_to_read)
        coordinates = numpy.array(struct.unpack('>{}f'.format(atoms * 3), coordinates_bytes)).reshape(atoms, 3)

        return coordinates

def get_len_trr(path):
    """Get current lenght of trr trajectory.
    """
    HEADER = 120
    ATOM_OFFSET = 64
    try:
        with open(path, 'rb') as handle:
            # Determine number of atoms in TRR file.
            handle.seek(ATOM_OFFSET, os.SEEK_SET)
            atoms = int.from_bytes(handle.read(4), byteorder='big')
            #print(atoms)
            # Determine number of bytes per frame.
            bytes_per_frame = HEADER + atoms * 3 * 4 # atoms * dimensions (x, y, z) * bytes per coord (float32)
            #print(bytes_per_frame)
            # Determine number of bytes in file.
            handle.seek(0, os.SEEK_END)
            total_bytes = handle.tell()
            #print(total_bytes)
            num_frames = int(total_bytes / bytes_per_frame)
            #print(num_frames)
            return num_frames
    except Exception as e:
        print(e)
        return None

def make_pdb(in_path, out_path, coordinates):
    with open(in_path, 'r') as handle:
        start_pdb = handle.readlines()

    # Find first ATOM record.
    for line_number, line in enumerate(start_pdb):
        if line.startswith('ATOM'):
            first_atom_line = line_number
            break
    else:
        raise ValueError('no atoms found')

    # Find last ATOM record.
    for i in range(len(start_pdb) - 1, -1, -1):
        if start_pdb[i].startswith('ATOM') and 'MW' not in start_pdb[i]:
            last_atom_line = i
            break

    # Replace coordinates.
    acc = (last_atom_line - first_atom_line + 1) * [None]
    acc.append('TER')
    for i, line in enumerate(start_pdb[first_atom_line: last_atom_line + 1]):
        head = line[:30] # see PDB file definition
        tail = line[54:] # see PDB file definition
        acc[i] = '{}{:8.3f}{:8.3f}{:8.3f}{}'.format(head, coordinates[i, 0] * 10, coordinates[i, 1] * 10, coordinates[i, 2] * 10, tail)

    with open(out_path, 'w') as handle:
        handle.write(''.join(acc))


def process(path_trr, frame, path_template_pdb):
    """Process frame from trr trajectory, i.e. dump specified frame, 
    relax structure into REF15 energy and calculate final REF15 value.
    :param path_trr:          [str] path to trajectory 
    :param frame:             [int] frame number
    :param path_template_pdb: [str] path to template pdb file
    :returns:     [int] frame number 
                  [float] REF15 value of relaxed structure from dumped frame
    """
    coordinates = read_coordinates(path_trr, frame)
    filename = f"{frame}.pdb"
    os.makedirs("./sep/", exist_ok=True)
    os.chdir("./sep/")
    make_pdb(path_template_pdb, filename, coordinates)
    subprocess.run("echo '1'| gmx trjconv -s ../../conf.gro -f "+filename+" -pbc whole -o "+filename, shell=True)  # Make molecules whole. 
    subprocess.run("echo '1'| gmx trjconv -s ../../conf.gro -f "+filename+" -pbc nojump -o "+filename, shell=True) # Remove jumps. 
    subprocess.run("echo '1' '1' | gmx trjconv -s ../../conf.gro -f "+filename+" -center -o "+filename, shell=True)# Center system in box. 
    
    # Calculate Rosetta REF15 energy score. 
    ros.init(extra_options="-constant_seed")
    scorefxn = ros.get_fa_scorefxn()                            # Load Rosetta REF15 energy function. 
    relax    = ros.rosetta.protocols.relax.FastRelax(scorefxn)  # Load Rosetta FastRelax protocol.
     
    subprocess.run(["gmx", "pdb2gmx", "-f", filename, "-o", filename, "-ff", "amber99sb-ildn", "-water", "none"])   # Protonate molecule.
    misc.cleanAtom(filename, filename)                                                                              # Clean pdb file.
    curr_pose   = ros.pose_from_pdb(filename)                                                                       # Create Rosetta pose object from pdb.
    relax.apply(curr_pose)									                    # Relax structure into REF15 energy.
    ref15       = scorefxn(curr_pose)                                                                               # Append REF15 value to REF15 array.
    remove_list = glob.glob("./*.top*") + glob.glob("./*.itp*") + glob.glob("*.pdb.*")                              # Remove waste files.
    for fpath in remove_list:
        os.remove(fpath)
    os.chdir("../")
    return frame, ref15

def process_mda(u, frame, path='.'):
    """Process frame from xtc trajectory, i.e. dump specified frame, 
    relax structure into REF15 energy and calculate final REF15 value.
    :param u:     [mda universe] MDAnalysis universe 
    :param frame: [int] frame number
    :returns:     [int] frame number 
                  [float] REF15 value of relaxed structure from dumped frame
    """
    os.chdir(path)
    misc.dump_pdb(u, frame, sel="protein")
    filename = f"{frame}.pdb"
    #subprocess.run("echo '1'| gmx trjconv -s ../../conf.gro -f "+filename+" -pbc whole -o "+filename, shell=True)  # Make molecules whole. 
    #subprocess.run("echo '1'| gmx trjconv -s ../../conf.gro -f "+filename+" -pbc nojump -o "+filename, shell=True) # Remove jumps. 
    #subprocess.run("echo '1' '1' | gmx trjconv -s ../../conf.gro -f "+filename+" -center -o "+filename, shell=True)# Center system in box. 
    
    # Calculate Rosetta REF15 energy score. 
    #ros.init(options="-mute protocols.relax.FastRelax core.pack.interaction_graph.interaction_graph_factory core.pack.task core.pack.pack_rotamers core.scoring.etable basic.io.database core.scoring.ScoreFunctionFactory core.chemical.GlobalResidueTypeSet protocols.relax.RelaxScriptManager", extra_options="-constant_seed")
    ros.init(extra_options="-constant_seed")
    scorefxn = ros.get_fa_scorefxn()                            # Load Rosetta REF15 energy function. 
    relax    = ros.rosetta.protocols.relax.FastRelax(scorefxn)  # Load Rosetta FastRelax protocol.
     
    subprocess.run(["gmx", "pdb2gmx", "-f", filename, "-o", filename, "-ff", "amber99sb-ildn", "-water", "none"])   # Protonate molecule.
    misc.cleanAtom(filename, filename)                                                                              # Clean pdb file.
    curr_pose   = ros.pose_from_pdb(filename)                                                                       # Create Rosetta pose object from pdb.
    relax.apply(curr_pose)									                    # Relax structure into REF15 energy.
    ref15       = scorefxn(curr_pose)                                                                               # Append REF15 value to REF15 array.

    return frame, ref15

def updateParamStd(pop_history, num_params_obj):
    """Update objective function parameters according to current state of knowledge.
    Standard deviation and mean of each term are extracted.
    :param pop_history: [list] list of dynamic particle lists from all previous generations
    :param num_params:  [int] number of objective function parameters
    :returns:           [list] list of objective function parameters
    """
    print(pop_history)
    print(pop_history[0][0])
    num_args = len(pop_history[0][0].fargs)
    cut = sys.float_info.max
    fparams = []
    for idx in range(num_args):
        temp = []
        for pop in pop_history:
            for part in pop:
                if part.fargs[idx] != cut: temp.append(part.fargs[idx])
        fparams.append(numpy.mean(numpy.asarray(temp)))
        fparams.append(numpy.std(numpy.asarray(temp)))
    return fparams
    
def updateParamStdFlat(part_history, num_params_obj):
    """Update obj. func. params according to current state of knowledge.
    Standard deviation and mean of each term are extracted.
    :param part_history: [list] list of dynamic particle lists from all previous generations
    :param num_params:   [int] number of obj. func. params
    :returns:            [list] list of obj. func. params
	"""
    #print(part_history)
    #print(part_history[0].fargs)
    print("Execute updateParamStdFlat...")
    num_args = len(part_history[0].fargs)
    cut = sys.float_info.max
    fparams = []
    for idx in range(num_args):
        temp = []
        for part in part_history:
            if part.fargs[idx] != cut: temp.append(part.fargs[idx])
        fparams.append(numpy.mean(numpy.asarray(temp)))
        fparams.append(numpy.std(numpy.asarray(temp)))
    return fparams

def updateParamVar(pop_history, num_params_obj):
    """Update obj. func. params according to current state of knowledge.
    The variance of each term is chosen as its weight.
    :param pop_history: [list] list of dynamic particle lists from all previous generations
    :param num_params:  [int] number of obj. func. params
    :returns:           [list] list of obj. func. params
	"""
    num_args = len(pop_history[0][0].fargs)
    cut = sys.float_info.max
    fparams = []
    for idx in range(num_args):
        temp = []
        for pop in pop_history:
            for part in pop:
                if part.fargs[idx] != cut: temp.append(part.fargs[idx])
        fparams.append(numpy.var(numpy.asarray(temp)))
    return fparams

def updateParamPow10(pop_history, num_params_obj):
    """Update obj. func. params according to current state of knowledge.
    The weights of each term are set according to its order of magnitude expressed
    in powers of ten. If not specified, all params are set to 1 by default.
    :param pop_history: [list] list of dynamic particle lists from all previous generations
    :param num_params:  [int] number of obj. func. params
    :returns:           [list] list of obj. func. params
    """
    def det_power(num):
        """Determine a number's order of magnitude in powers of ten.
        :param num: [float] number whose order of magnitude is to be determined
        :returns:   [int] exponent w.r.t. base 10
        """
        if num == 0: 
            return 0
        expo = numpy.floor(numpy.log10(numpy.abs(num)))
        if num/(10**expo) < 7:
            return expo
        else:
            return expo+1

    num_args = len(pop_history[0][0].fargs)
    minargs = numpy.empty(num_args)
    maxargs = numpy.empty(num_args)
    cut = numpy.abs(sys.float_info.max)
    for pop in pop_history:
        for part in pop:
            for idx in range(num_args):
                if (minargs[idx] is None or part.fargs[idx] < minargs[idx]) and numpy.abs(part.fargs[idx]) != cut:
                    minargs[idx] = part.fargs[idx]
                if (maxargs[idx] is None or part.fargs[idx] > maxargs[idx]) and numpy.abs(part.fargs[idx]) != cut:
                    maxargs[idx] = part.fargs[idx]
    fparams=[ 10**(-0.5*(det_power(mini)+det_power(maxi))) for mini,maxi in zip(minargs, maxargs) ]
    #print("Current obj. func. parameters:", repr(fparams))
    print("Min. args:", repr(minargs))
    print("Max. args:", repr(maxargs))
    #with open("~/dynPSO/params.dat", "a") as fout:
    #    fout.write(repr(fparams) + "\n")
    return fparams

def combineObj(args, params):
    """Calculate scalar fitness according to obj. func., given its args and params.
    If this function is not specified, the scalar product `args[i]*params[i]` is returned.

    :param params:  [vector] params of obj. func.
    :returns:       [float] obj. func. value (scalar fitness)
    """
    loss = sum([a*p for a, p in zip(args, params)])
    #print("Current obj. func. value =", loss)
    #if numpy.abs(args[0]) == numpy.abs(sys.float_info.max):
    #	pass
    #else:
    #	with open("~/dynPSO/loss.dat", "a") as fout:
    #    	fout.write(str(loss) + "\n")
    return loss
    
def combineObjStandardized(args, params):
    """Calculate scalar fitness according to obj. func., given its args and params.
    If this function is not specified, the scalar product `args[i]*params[i]` is returned.

    :param params:  [vector] params of obj. func.
    :returns:       [float] obj. func. value (scalar fitness)
    """
    #print("args:",len(args), args,"; params:",len(params), params)
    if all(a == sys.float_info.max for a in args):
        return sys.float_info.max
    else:
        return sum([(a-params[2*idx])/params[2*idx+1] for idx, a in enumerate(args)])
    print("CombineObjStandardized: Current OF value =", loss)
    return loss


def get_logarithmic_axis_sample(lb, ub, N, dtype):    
    """Return function value f(n) where f is logarithmic function e^x sampling
    the exponent range [log(a), log(b)] linear at N sampling points.
    The function values returned are in the range [a, b].
    
    :param lb: left value range bound
    :param ub: right value range bound
    :param N: discretization of intervall [a,b]
    :param dtype: data type
    
    :return: [list] axis range
    """
    assert lb < ub, "condition lb < ub violated!"
    assert lb > 0, "condition lb > 0 violated!"
    assert isinstance(N, int), "condition N of type int violated!"
    
    # Convert input range into exponent range.
    lexp = numpy.log10(lb)
    rexp = numpy.log10(ub)
    exp_range = numpy.random.uniform(lexp, rexp, N)
    
    data = []
    for n in range(exp_range.shape[0]):
        x = numpy.power(10, exp_range[n])
        if dtype is int:
            data.append(int(x))
        elif dtype is float:
            data.append(x)
        else:
            raise AssertionError("dtype {} not supported for uniform sampling!".format(dtype))
    return data
