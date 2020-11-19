import mpi4py
mpi4py.rc.initialize=False
from mpi4py import MPI
MPI.Init()
import pickle
import hyppopy.HyppopyProject
import hyppopy.solvers.DynamicPSOSolver
import xsmdMPI
import misc
import subprocess
import numpy
import watchdog
import re
import os
import MDAnalysis as mda
import time
import sys
import glob

# When a program is run with MPI, all processes are grouped in so-called communicators.
# A communicator is a box grouping processes together, allowing them to communicate.
# Every communication is linked to a communicator, allowing the communication to reach
# different processes. Communications can be either of two types:
#  - point-to-point: two processes in the same communicator are going to communicate
#  - collective: all processes in a communicator are going to communicate together
# The default communicator is called MPI_COMM_WORLD. It basically groups all processes
# when a program is started. It is possible to create custom (sub-) communicators.
# The number in a communicator does not change once created (size of communicator).
# Each process inside a communicator has a unique number to identify it, i.e. the
# rank of the process

# MPI-related stuff
size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()

# Parameters to set by user depending on optimization problem and (HPC) system:
block_size           = 100                # block size (block = 1 dyn. PSO sim. node + workers)
num_particles        = 1                  # number of (local) particles per block (never change this)
num_generations      = 15                 # number of generations (iterations per particle)
num_particles_global = size // block_size # number of particles in swarm
WS_path              = sys.argv[1]        # path to permanent workspace /pfs/work6/workspace/scratch/ku4408-LAO_ha-0/setup/ 
ws_path              = sys.argv[2]        # path to temporary job BeeOND file system
seed                 = sys.argv[3]        # random seed
beeond               = sys.argv[4] == 'b' # BeeOND used? b / nb = yes / no

# parameter box constraints to specify search space
tempa  = [10,90]          # temperature constraints
chichi = [10**-11,10**-8] # bias weight constraints

# intra-block communicator (divide different blocks from one another)
intra_comm_color = rank // block_size
intra_comm_key   = rank % block_size
comm_intra       = MPI.COMM_WORLD.Split(color=intra_comm_color, key=intra_comm_key)

# inter-block communicator (divide simulators from bitches)
inter_comm_color = int(rank % block_size == 0)
inter_comm_key   = rank // block_size
comm_inter       = MPI.COMM_WORLD.Split(color=inter_comm_color, key=inter_comm_key)

# closure to pass intra-block communicator to black-box w/o modifying its signature
def f(k_chi, T):
    return obj(k_chi, T, comm_intra)

# actual black-box
def obj(k_chi, T, comm_intra):
    """Black-box function returning unweighted terms contributing to objective function. 
    :param k_chi:[float] bias weight (hyperparameter to be optimized)
    :param T:	 [float] temperature (hyperparameter to be optimized)
    		 k_chi and T are stored in position attribute of dynamic particle.
    :returns:	 [list] unweighted terms contributing to objective function  
    """ 
    # Generate .mdp file with k_chi and T set accordingly.
    print("Generate mdp file...")
    with open("../template.mdp", "r") as fmdp:  # Read template mdp file line by line. 
        mdp_lines = fmdp.readlines() 
     
    mdp_new = mdp_lines 
  
    for idx, line in enumerate(mdp_lines): 
        if line.startswith("dt"):                   # Extract time step. 
            dt = float(re.search(r"\d+.\d+", line).group(0)) 
        if line.startswith("nsteps"):               # Extract number of simulation steps. 
            nsteps = int(re.search(r"\d+", line).group(0)) 
        if line.startswith("nstxout"):   # Extract coordinate output frequency. 
            xout = int(re.search(r"\d+", line).group(0)) 
        if line.startswith("nstenergy"):            # Extract energy output frequency. 
            eout = int(re.search(r"\d+", line).group(0)) 
        if line.startswith("waxs-fc"):              # Insert current bias weight (mdp option "waxs-fc") 
            mdp_new[idx] = "waxs-fc = " + str(k_chi) + "\n" 
        if line.startswith("ref-t"):                # Insert current temperature (mdp options "ref-t" 
            mdp_new[idx] = "ref-t = " + str(T) + "\n"              # and "gen-temp") 
        if line.startswith("gen-temp"): 
            mdp_new[idx] = "gen-temp = " + str(T) + "\n" 
      
    nout = misc.lcm(xout, eout) 
    n_frames = int(nsteps/xout + 1)
    n_workers = int(comm_intra.Get_size()-1)
    n_workrounds = int(n_frames / n_workers)
    print("Number of workrounds per worker:", n_workrounds)

    with open("run.mdp", "w+") as fmdp:             # Write modified mdp options to file. 
        fmdp.writelines(mdp_new) 
    
    print("Sucessfully generated mdp file...")

    grompp = ["gmx", "grompp", "-f", "run.mdp", "-c", "../saxs_conf.gro", "-p", "../topol.top", "-o", "run.tpr"]
    mdrun  = ["gmx", "mdrun", "-deffnm", "run", "-waxs_diff", "../ha_error_reduced.dat",\
              "-waxs_ref", "../1LST_error_reduced.dat",\
              "-sfac", "/home/fh2-project-mddyn/ku4408/gromacs_XSMD/install/share/gromacs/top/sfactor_amino_acid_ds_Fourier.xml",\
              "-waxs_out", "waxs_out", "-waxs_alpha", "waxs_alpha", "-ntomp", "1", "-reprod"]
    print("Start sim...") 
    subprocess.run(grompp)  # Preprocess system.
    subprocess.run(mdrun)   # Run sim.
    print("Sim complete...")
 
    # To be sent: dummy for simulation rank; REF15 values only calculated by workers
    send_frame = numpy.ones(n_workrounds, dtype=numpy.float64)
    send_ref15 = numpy.ones(n_workrounds, dtype=numpy.float64)

    recv_frame = numpy.empty([comm_intra.Get_size(),n_workrounds], dtype=numpy.float64)
    recv_ref15 = numpy.empty([comm_intra.Get_size(),n_workrounds], dtype=numpy.float64)

    print("NEXT GATHERING...")

    comm_intra.Gather(send_frame, recv_frame, root=0)
    comm_intra.Gather(send_ref15, recv_ref15, root=0)
    
    #print("Gathered frames:", recv_frame)
    #print("Gathered REF15:", recv_ref15)    

    frames = recv_frame[1::]  # Discard data from simulation rank 0.
    ref15  = recv_ref15[1::]  # Discard data from simulation rank 0.

    #print("Frames after discarding rank 0:", frames)
    #print("REF15 after discarding rank 0:", ref15)

    frames = frames.flatten() # Flatten array.
    ref15  = ref15.flatten()  # Flatten array.

    #print("Flattened frames:", frames)
    #print("Flattened REF15:", ref15)

    times = nout*dt*numpy.sort(frames)  # sorted time array
    ref15 = ref15[numpy.argsort(frames)]# REF15 array (sorted by time) 
   
    with open("./ref15.p","wb") as ref15p:
        print("Dump REF15...")
        pickle.dump(ref15, ref15p)
 
    #print("Sorted times of length",len(times))#,":",times)
    #print("Sorted REF15 of length",len(ref15))#,":",ref15)
  
    # Extract bias energy from sim.
    subprocess.run("echo '11' '0' | gmx energy -f run.edr -o waxs_debye", shell=True)
    vxs  = numpy.loadtxt("./waxs_debye.xvg", comments = ["#", "@"], usecols=(1,)) 
    print("Bias energy extracted...")
    
    # Calculate unweighted terms from quantities of interest.
    try:
        term1 = numpy.median(vxs) / k_chi
        term2 = k_chi / numpy.mean(vxs)
        term3 = numpy.mean(ref15)
        print("Unweighted obj. func. terms:", repr([term1, term2, term3]))
        return [term1, term2, term3]

    except Exception as e:
        print(e)

with open(ws_path+"template.mdp", "r") as fmdp:  # Read template mdp file line by line. 
    mdp_lines = fmdp.readlines()

for idx, line in enumerate(mdp_lines):
    if line.startswith("nsteps"):               # Extract number of simulation steps. 
        nsteps = int(re.search(r"\d+", line).group(0))
    if line.startswith("nstxout"):   # Extract coordinate output frequency. 
        xout = int(re.search(r"\d+", line).group(0))

total = int(nsteps/xout + 1)
n_workers = comm_intra.size - 1                     # number of workers in block

if rank == 0:
    print("---------------------------------------")
    print("- Dynamic Particle Swarm Optimization -")
    print("---------------------------------------")
    print("MPI.COMM_WORLD size:", size)
    print("Number of generations:", num_generations)
    print("Block size:", block_size)
    print("Number of particles per swarm: ", num_particles_global)
    print("Number of frames per sim: ", total)
    print("Number of workers per block: ", n_workers)

if rank % block_size == 0:
    project = hyppopy.HyppopyProject.HyppopyProject()
    project.add_hyperparameter(name="k_chi", domain="loguniform", data=chichi, type=float)
    project.add_hyperparameter(name="T", domain="uniform", data=tempa, type=float)
    project.add_setting(name="num_particles", value=num_particles)
    project.add_setting(name="num_generations", value=num_generations)
    project.add_setting(name="num_particles_global", value=num_particles_global)
    project.add_setting(name="num_params_obj", value=6) 
    project.add_setting(name="num_args_obj", value=3) 
    project.add_setting(name="combine_obj", value=xsmdMPI.combineObjStandardized)  
    project.add_setting(name="update_param", value=xsmdMPI.updateParamStdFlat)
    project.add_setting(name="phi1", value=2.0)#1.5
    project.add_setting(name="phi2", value=1.5)#2.0
    project.add_setting(name="comm_inter", value=comm_inter)
    project.add_setting(name="comm_intra", value=comm_intra)
    project.add_setting(name="workspace", value=ws_path)
    project.add_setting(name="seed", value=int(seed)+int(rank))
    solver=hyppopy.solvers.DynamicPSOSolver.DynamicPSOSolver(project)
    solver.blackbox=f
    solver.run()

else: # WORK, BITCH!
    for g in range(num_generations):
        path = None
        path = comm_intra.bcast(path, root=0)               # Get path to simulation folder.
        os.chdir(path)
        print("Worker changed to sim. directory:", os.getcwd())
        rr = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", path)
        T_curr = float(rr[-2])
        k_chi_curr = float(rr[-1])

        # Current particle violates box constraints (parameters out of search space).
        if T_curr < tempa[0] or T_curr > tempa[1] or k_chi_curr < chichi[0] or k_chi_curr > chichi[1]:
            print("Particle violates box constraints (parameters out of search space).")
        # Current particle lives within search space.
        else:
            print("Particle lives within search space.")
            trj = path+"/run.trr"
            top = ws_path+"saxs_conf.gro"
            path_r = path+"/rank_"+str(comm_intra.Get_rank())+"/"
            os.makedirs(path_r, exist_ok=True)
            frames = []                                     # local frame values from one worker
            ref15  = []                                     # local REF15 values from one worker
            handler = xsmdMPI.TrajHandler()                 # event handler taking event dispatched by observer + performing some action
            handler.processing = comm_intra.Get_rank() - 1  # initialize variable storing frame currently being processed 
            observer = watchdog.observers.Observer()        # observer thread scheduling watched dirs + dispatching calls to event handler
            observer.schedule(handler, path, recursive=False)
            observer.start()
            #print("Start observer...")
       
            while not os.path.exists(trj):
                time.sleep(1)
            print("Worker awake!")
            
            while handler.processing < total: # While not all frames have been processed...
                with handler.cv:
                    while True:
                        try:
                            while handler.processing >= len(mda.Universe(top, trj, refresh_offsets=True).trajectory):
                                handler.cv.wait()
                        except Exception as e:
                            print(e)
                            print("First try again...")
                            continue
                        else:
                            break

                    print("Start processing frame", handler.processing,"for particle",int(rank/block_size+1),"...")
                    while True:
                        try:
                            os.chdir(path)
                            frame, r15 = xsmdMPI.process_mda(mda.Universe(top, trj, refresh_offsets=True), handler.processing, path_r)
                            os.chdir(path)
                        except Exception as e:
                            print(e)
                            print("Second try again...")
                            continue
                        else:
                            break
                        
                    frames.append(frame)
                    ref15.append(r15)
                    print("Frame",frame,"for particle",int(rank/block_size+1),"processed: REF15 =",r15)
                    handler.processing += n_workers    # static scheduling: frames pre-distributed to different ranks
        
            print("PARTICLE",int(rank/block_size+1),": PROCESSING COMPLETE.")
            observer.stop()
            observer.join()
            print("PARTICLE",int(rank/block_size+1),": OBSERVER STOPPED.")
            frames = numpy.array(frames, dtype=numpy.float64)
            ref15 = numpy.array(ref15, dtype=numpy.float64)
            print("Within block: Frames of shape",frames.shape,":", frames, "; REF15 of shape",ref15.shape,":", ref15)
            print("PARTICLE",int(rank/block_size+1),": NEXT GATHERING...")

            # To be sent:
            send_frame = frames
            send_ref15 = ref15
            # To be received: nothing on workers; only for intra-block rank 0 (see obj. func.)
            recv_frame = None
            recv_ref15 = None
            
            comm_intra.Gather(send_frame, recv_frame, root=0)
            comm_intra.Gather(send_ref15, recv_ref15, root=0)

        print("Worker waiting for MPI barrier...") 
        comm_intra.Barrier()

        # Remove waste files.
        if os.path.exists(path_r): # Only exists for particles living in search space!
            remove_list = glob.glob(path_r+"*.top*") + glob.glob(path_r+"*.itp*") + glob.glob(path_r+"*.pdb.*")
            for fpath in remove_list:
                if os.path.exists(fpath): os.remove(fpath)

        # Copy files back to permanent workspace if ODFS is used.
        if beeond:
            if comm_intra.Get_rank() == 1: # Copy directory of this particle to permanent workspace.
                print(os.getcwd())
                cmds = ['rsync', '-av', os.getcwd(), WS_path]
                print('Copy sim. folder to permanent workspace...')
                start = time.time()
                subprocess.call(cmds)
    
                print('Copy log files and history to permanent workspace...')
                for f in glob.glob(ws_path+'*.p'):
                    cmds = ['rsync', '-av', f, WS_path]
                    subprocess.call(cmds)
                for f in glob.glob(ws_path+'*.log'):
                    cmds = ['rsync', '-av', f, WS_path]
                    subprocess.call(cmds)
    
                stop = time.time()
                duration = stop-start
                print("rsync duration:",duration,"s")
        MPI.COMM_WORLD.Barrier()
        print("Passed...")
