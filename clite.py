import os
import time
import shlex
import numpy as np
import random as rd
import subprocess as sp
from scipy import stats
from scipy.stats import norm
from scipy.optimize import minimize
import sklearn.gaussian_process as gp

# Number of LC apps available
TOT_LC_APPS   = 5

# Number of BG apps available
TOT_BG_APPS   = 6

# Number of QPS categories
N_QPS_CAT     = 10

# LC apps
APP_NAMES     = [
                'img-dnn'  ,
                'masstree' ,
                'memcached',
                'specjbb'  ,
                'xapian'
                ]

# QoS requirements of LC apps (time in seconds)
APP_QOSES     = {
                'img-dnn'  : 3.0  ,
                'masstree' : 2.0  ,
                'memcached': 225.0,
                'specjbb'  : 0.5  ,
                'xapian'   : 12.0 
                }
# QPS levels
APP_QPSES     = {
                'img-dnn'  : list(range(300, 3300, 300))      ,
                'masstree' : list(range(100, 1100, 100))      ,
                'memcached': list(range(20000, 220000, 20000)),
                'specjbb'  : list(range(800, 9600, 800))      ,
                'xapian'   : list(range(800, 9600, 800))
                }

# BG apps
BCKGRND_APPS  = [
                'blackscholes' ,
                'canneal'      ,
                'fluidanimate' ,
                'freqmine'     ,
                'streamcluster',
                'swaptions'
                ]

# Number of times acquisition function optimization is restarted
NUM_RESTARTS  = 1

# Number of maximum iterations (max configurations sampled)
MAX_ITERS     = 100

# Shared Resources hardware configuration:
# Number of Cores (10 units)
# Number of Ways (11 units)
# Percent Memory Bandwidth (10 units)

# Number of resources controlled
NUM_RESOURCES = 3

# Max values of each resources
NUM_CORES     = 10
NUM_WAYS      = 11
MEMORY_BW     = 100

# Max units of (cores, LLC ways, memory bandwidth)
NUM_UNITS     = [10, 11, 10]

# Configuration formats
CONFIGS_CORES = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
CONFIGS_CWAYS = ["0x1", "0x3", "0x7", "0xf", "0x1f", "0x3f", "0x7f", "0xff", "0x1ff", "0x3ff", "0x7ff"]
CONFIGS_MEMBW = ["10", "20", "30", "40", "50", "60", "70", "80", "90", "100"]

# Commands to set hardware allocations
TASKSET       = "sudo taskset -acp "
COS_CAT_SET1  = "sudo pqos -e \"llc:%s=%s\""
COS_CAT_SET2  = "sudo pqos -a \"llc:%s=%s\""
COS_MBG_SET1  = "sudo pqos -e \"mba:%s=%s\""
COS_MBG_SET2  = "sudo pqos -a \"core:%s=%s\""
COS_RESET     = "sudo pqos -R"

# Commands to get MSRs
WR_MSR_COMM       = "wrmsr -a "
RD_MSR_COMM       = "rdmsr -a -u "

# MSR register requirements
IA32_PERF_GBL_CTR = "0x38F"  # Need bits 34-32 to be 1
IA32_PERF_FX_CTRL = "0x38D"  # Need bits to be 0xFFF
MSR_PERF_FIX_CTR0 = "0x309"

# Amount of time to sleep after each sample
SLEEP_TIME    = 2

# Suppress application outputs
# FNULL         = open(os.devnull, 'w')

# Path to the base directory (if required)
BASE_DIR      = "/path/to/base/directory/"
# All the LC apps being run
LC_APPS       = ['img-dnn', 'xapian']

# Path to the latency files of applications
# LATS_FILES    = [BASE_DIR + 'tailbench-v0.9/img-dnn/lats.bin', BASE_DIR + 'tailbench-v0.9/xapian/lats.bin']

# ALl the BG jobs being runs
BG_APPS       = ['blackscholes']

APPS = LC_APPS + BG_APPS

# PIDs of all the applications in order of APPS
APP_PIDS      = ["11475", "11783"]

# QoSes of LC apps
APP_QOSES     = [APP_QOSES[a] for a in LC_APPS]

# Number of apps currently running
NUM_LC_APPS   = len(LC_APPS)

NUM_BG_APPS   = len(BG_APPS)

NUM_APPS      = NUM_LC_APPS + NUM_BG_APPS

# Total number of parameters
NUM_PARAMS    = NUM_RESOURCES*(NUM_APPS-1)

# Set expected value threshold for termination
EI_THRESHOLD  = 0.01**NUM_APPS

# Global variable to hold baseline performances
BASE_PERFS    = [0.0]*NUM_APPS

# Required global variables
BOUNDS        = None

CONSTS        = None

MODEL         = None

OPTIMAL_PERF  = None

class Lat(object):

    def __init__(self, fileName):
        f = open(fileName, 'rb')
        a = np.fromfile(f, dtype=np.uint64)
        self.reqTimes = a.reshape((int(a.shape[0]/3.0), 3))
        f.close()

    def parseQueueTimes(self):
        return self.reqTimes[:, 0]
    
    def parseSvcTimes(self):
        return self.reqTimes[:, 1]
    
    def parseSojournTimes(self):
        return self.reqTimes[:, 2]

def getLatPct(latsFile):

    assert os.path.exists(latsFile)
    
    latsObj = Lat(latsFile)
    
    sjrnTimes = [l/1e6 for l in latsObj.parseSojournTimes()]

    mnLt = np.mean(sjrnTimes)
    
    p95  = stats.scoreatpercentile(sjrnTimes, 95.0)
    
    return p95

def gen_bounds_and_constraints():
    print("����")

    global BOUNDS, CONSTS

    # Generate the bounds and constraints required for the optimizer
    BOUNDS = np.array([[[1, NUM_UNITS[r]-(NUM_APPS-1)] for a in range(NUM_APPS-1)] \
             for r in range(NUM_RESOURCES)]).reshape(NUM_PARAMS, 2).tolist()
    print("BOUNDS", BOUNDS)
    CONSTS = []
    for r in range(NUM_RESOURCES):
        CONSTS.append({'type':'eq', 'fun':lambda x: sum(x[r*(NUM_APPS-1):(r+1)*(NUM_APPS-1)]) - (NUM_APPS-1)})
        CONSTS.append({'type':'eq', 'fun':lambda x: -sum(x[r*(NUM_APPS-1):(r+1)*(NUM_APPS-1)]) + (NUM_UNITS[r]-1)})
    print("CONSTS", CONSTS)

def gen_initial_configs():

    # Generate the maximum allocation configurations for all applications
    configs = [[1]*NUM_PARAMS for j in range(NUM_APPS)]
    for j in range(NUM_APPS-1):
        for r in range(NUM_RESOURCES):
            configs[j][j+((NUM_APPS-1)*r)] = NUM_UNITS[r] - (NUM_APPS-1)

    # Generate the equal partition configuration
    equal_partition = []
    for r in range(NUM_RESOURCES):
        for j in range(NUM_APPS-1):
            equal_partition.append(int(NUM_UNITS[r]/NUM_APPS))
    configs.append(equal_partition)

    return configs

def get_baseline_perfs(configs):

    global BASE_PERFS

    for i in range(NUM_APPS):
        p = configs[i]
    
        # Core allocations of each job
        app_cores = [""]*NUM_APPS
        s = 0
        for j in range(NUM_APPS-1):
            app_cores[j] = ",".join([str(c) for c in list(range(s,s+p[j]))+list(range(s+NUM_UNITS[0],s+p[j]+NUM_UNITS[0]))])
            s += p[j]
        app_cores[NUM_APPS-1] = ",".join([str(c) for c in list(range(s,NUM_UNITS[0]))+list(range(s+NUM_UNITS[0],NUM_UNITS[0]+NUM_UNITS[0]))])
        
        # L3 cache ways allocation of each job
        app_cways = [""]*NUM_APPS
        s = 0
        for j in range(NUM_APPS-1):
            app_cways[j] = str(hex(int("".join([str(1) for w in list(range(p[j+NUM_APPS-1]))]+[str(0) for w in list(range(s))]),2)))
            s += p[j+NUM_APPS-1]
        app_cways[NUM_APPS-1] = str(hex(int("".join([str(1) for w in list(range(NUM_UNITS[1]-s))]+[str(0) for w in list(range(s))]),2)))
        
        # Memory bandwidth allocation of each job
        app_membw = [""]*NUM_APPS
        s = 0
        for j in range(NUM_APPS-1):
            app_membw[j] = str(p[j+2*(NUM_APPS-1)]*10)
            s += p[j+2*(NUM_APPS-1)]*10
        app_membw[NUM_APPS-1] = str(NUM_UNITS[2]*10-s)
        
        # Set the allocations
        for j in range(NUM_APPS):
            taskset_cmnd = TASKSET + app_cores[j] + " " + APP_PIDS[j]
            cos_cat_set1 = COS_CAT_SET1 % (str(j+1), app_cways[j])
            cos_cat_set2 = COS_CAT_SET2 % (str(j+1), app_cores[j])
            cos_mBG_set1 = COS_MBG_SET1 % (str(j+1), app_membw[j])
            cos_mBG_set2 = COS_MBG_SET2 % (str(j+1), app_cores[j])
            sp.check_output(shlex.split(taskset_cmnd), stderr=FNULL)
            sp.check_output(shlex.split(cos_cat_set1), stderr=FNULL)
            sp.check_output(shlex.split(cos_cat_set2), stderr=FNULL)
            sp.check_output(shlex.split(cos_mBG_set1), stderr=FNULL)
            sp.check_output(shlex.split(cos_mBG_set2), stderr=FNULL)

        if i >= NUM_LC_APPS:
            # Reset the IPS counters
            os.system(WR_MSR_COMM + MSR_PERF_FIX_CTR0 + " 0x0")
    
        time.sleep(SLEEP_TIME)
    
        if i < NUM_LC_APPS:
            BASE_PERFS[i] = getLatPct(LATS_FILES[i])
        else:
            # Get the IPS counters  
            ipsP = os.popen(RD_MSR_COMM + MSR_PERF_FIX_CTR0)
        
            # Calculate the IPS
            IPS = 0.0
            cor = [int(c) for c in app_cores[i].split(',')]
            ind = 0
            for line in ipsP.readlines():
                if ind in cor:
                    IPS += float(line)
                ind += 1
            
            BASE_PERFS[i] = IPS

def gen_random_config():

    # Generate a random configuration
    config = []
    for r in range(NUM_RESOURCES):
        total = 0
        remain_apps = NUM_APPS
        for j in range(NUM_APPS-1):
            alloc = rd.randint(1, NUM_UNITS[r] - (total+remain_apps-1))
            config.append(alloc)
            total += alloc
            remain_apps -= 1

    return config

def sample_perf(p):


    # Core allocations of each job
    # app_cores = [""]*NUM_APPS
    # s = 0
    # for j in range(NUM_APPS-1):
    #     app_cores[j] = ",".join([str(c) for c in list(range(s,s+p[j]))+list(range(s+NUM_UNITS[0],s+p[j]+NUM_UNITS[0]))])
    #     s += p[j]
    # app_cores[NUM_APPS-1] = ",".join([str(c) for c in list(range(s,NUM_UNITS[0]))+list(range(s+NUM_UNITS[0],NUM_UNITS[0]+NUM_UNITS[0]))])
    
    # # L3 cache ways allocation of each job
    # app_cways = [""]*NUM_APPS
    # s = 0
    # for j in range(NUM_APPS-1):
    #     app_cways[j] = str(hex(int("".join([str(1) for w in list(range(p[j+NUM_APPS-1]))]+[str(0) for w in list(range(s))]),2)))
    #     s += p[j+NUM_APPS-1]
    # app_cways[NUM_APPS-1] = str(hex(int("".join([str(1) for w in list(range(NUM_UNITS[1]-s))]+[str(0) for w in list(range(s))]),2)))
    
    # # Memory bandwidth allocation of each job
    # app_membw = [""]*NUM_APPS
    # s = 0
    # for j in range(NUM_APPS-1):
    #     app_membw[j] = str(p[j+2*(NUM_APPS-1)]*10)
    #     s += p[j+2*(NUM_APPS-1)]*10
    # app_membw[NUM_APPS-1] = str(NUM_UNITS[2]*10-s)
    
    # # Set the allocations
    # for j in range(NUM_APPS):
    #     taskset_cmnd = TASKSET + app_cores[j] + " " + APP_PIDS[j]
    #     cos_cat_set1 = COS_CAT_SET1 % (str(j+1), app_cways[j])
    #     cos_cat_set2 = COS_CAT_SET2 % (str(j+1), app_cores[j])
    #     cos_mBG_set1 = COS_MBG_SET1 % (str(j+1), app_membw[j])
    #     cos_mBG_set2 = COS_MBG_SET2 % (str(j+1), app_cores[j])
    #     # sp.check_output(shlex.split(taskset_cmnd), stderr=FNULL)
    #     # sp.check_output(shlex.split(cos_cat_set1), stderr=FNULL)
    #     # sp.check_output(shlex.split(cos_cat_set2), stderr=FNULL)
    #     # sp.check_output(shlex.split(cos_mBG_set1), stderr=FNULL)
    #     # sp.check_output(shlex.split(cos_mBG_set2), stderr=FNULL)

    # if NUM_BG_APPS != 0:
    #     # Reset the IPS counters
    #     os.system(WR_MSR_COMM + MSR_PERF_FIX_CTR0 + " 0x0")

    # # Wait for some cycles
    # time.sleep(SLEEP_TIME)

    qv = [rd.random()]*NUM_LC_APPS
    sd = [rd.random()]*NUM_LC_APPS
    # for j in range(NUM_LC_APPS):
    #     p95 = getLatPct(LATS_FILES[j])
    #     if p95 > APP_QOSES[j]:
    #         qv[j] = APP_QOSES[j] / p95
    #         sd[j] = BASE_PERFS[j] / p95

    # # Return the final objective function score if QoS not met
    # if stats.mstats.gmean(qv) != 1.0:
    #     return qv, 0.5*stats.mstats.gmean(qv)

    # # Return the final objective function score if QoS met
    # if NUM_BG_APPS == 0:
    #     return qv, 0.5*(min(1.0,stats.mstats.gmean(sd))+1.0)

    # # Get the IPS counters  
    # ipsP = os.popen(RD_MSR_COMM + MSR_PERF_FIX_CTR0)

    # sd = [0.0]*NUM_BG_APPS
    # for j in range(NUM_BG_APPS):
    #     # Calculate the IPS
    #     IPS = 0.0
    #     cor = [int(c) for c in app_cores[j+NUM_LC_APPS].split(',')]
    #     ind = 0
    #     for line in ipsP.readlines():
    #         if ind in cor:
    #             IPS += float(line)
    #         ind += 1
            
    #     sd[j] = min(1.0, IPS / BASE_PERFS[j+NUM_LC_APPS])

    # # Return the final objective function score if BG jobs are present
    return qv, 0.5*(min(1.0,stats.mstats.gmean(sd))+1.0)


def expected_improvement(c, exp=0.01):
    print("c: ", c)

    # Calculate the expected improvement for a given configuration 'c'
    mu, sigma = MODEL.predict(np.array(c).reshape(-1, NUM_PARAMS), return_std=True)
    val = 0.0
    with np.errstate(divide='ignore'):
        Z = (mu - OPTIMAL_PERF - exp) / sigma
        val = (mu - OPTIMAL_PERF - exp) * norm.cdf(Z) + sigma * norm.pdf(Z)
        val[sigma == 0.0] = 0.0

    return -1 * val

def find_next_sample(x, q, y):

    # Generate the configuration which has the highest expected improvement potential
    max_config = None
    max_result = 1

    # Multiple restarts to find the global optimum of the acquisition function
    for n in range(NUM_RESTARTS):

        val = None

        # Perform dropout 1/4 of the times
        if rd.choice([True, True, True, False]):

            x0 = gen_random_config()
            print("x0", x0)

            val = minimize(fun=expected_improvement,
                           x0=x0,
                           bounds=BOUNDS,
                           constraints=CONSTS,
                           method='SLSQP')
            print("val: ", val)
        else:
            ind = rd.choice(list(range(len(y))))
            app = q[ind].index(max(q[ind]))

            if app == (NUM_APPS-1):

                consts = []
                for r in range(NUM_RESOURCES):
                    units = sum(x[ind][r*(NUM_APPS-1):(r+1)*(NUM_APPS-1)])
                    consts.append({'type':'eq', 'fun':lambda x: sum(x[r*(NUM_APPS-1):(r+1)*(NUM_APPS-1)]) - units})
                    consts.append({'type':'eq', 'fun':lambda x: -sum(x[r*(NUM_APPS-1):(r+1)*(NUM_APPS-1)]) + units})
                print("x[ind]2: ", x[ind])
                val = minimize(fun=expected_improvement,
                               x0=x[ind],
                               bounds=BOUNDS,
                               constraints=consts,
                               method='SLSQP')

            else:

                bounds = [[b[0], b[1]] for b in BOUNDS]

                for r in range(NUM_RESOURCES):
                    bounds[app+r*(NUM_APPS-1)][0] = x[ind][app+r*(NUM_APPS-1)]
                    bounds[app+r*(NUM_APPS-1)][1] = x[ind][app+r*(NUM_APPS-1)]
                print("x[ind]2: ", x[ind])
                val = minimize(fun=expected_improvement,
                               x0=x[ind],
                               bounds=bounds,
                               constraints=CONSTS,
                               method='SLSQP')

        if val.fun < max_result:
            max_config = val.x
            max_result = val.fun
    print("[int(c) for c in max_config]: ", [int(c) for c in max_config])
    return -max_result, [int(c) for c in max_config]

def bayesian_optimization_engine(x0, alpha=1e-5):

    global MODEL, OPTIMAL_PERF

    x_list = []
    q_list = []
    y_list = []

    # Sample initial configurations
    for params in x0:
        x_list.append(params)
        q, y = sample_perf(params)
        q_list.append(q)
        y_list.append(y)
    
    xp = np.array(x_list)
    yp = np.array(y_list)
    print('xp', xp, 'yp', yp)
    print("q_list", q_list)
    # Create the Gaussian process model as the surrogate model
    kernel = gp.kernels.Matern(length_scale=1.0, nu=1.5)
    print("kernel: ", kernel)
    MODEL  = gp.GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=10, normalize_y=True)
    
    # Iterate for specified number of iterations as maximum
    for n in range(MAX_ITERS):

        # Update the surrogate model
        MODEL.fit(xp, yp)
        OPTIMAL_PERF = np.max(yp)

        # Find the next configuration to sample
        ei, next_sample = find_next_sample(x_list, q_list, y_list)
        print("ei: ", ei, "next_sample: ", next_sample)

        # If the configuration is already sampled, carefully replace the sample
        mind = 0
        while next_sample in x_list:
            if mind == len(y_list):
                next_sample = gen_random_config()
                continue
            print("sorted(enumerate(y_list), key = lambda x:x[1]): ", sorted(enumerate(y_list), key = lambda x:x[1]))
            ind = sorted(enumerate(y_list), key = lambda x:x[1])[mind][0]
            print("ind: ", ind)
            print("q_list: ", q_list)
            print("q_list[ind]: ", q_list[ind])
            if stats.mstats.gmean(q_list[ind]) == 1.0:
                mind += 1
                continue
            boxes = sum([q==1.0 for q in q_list[ind]])
            if boxes == 0:
                mind += 1
                continue
            next_sample = [x for x in x_list[ind]]
            for r in range(NUM_RESOURCES):
                avail = NUM_UNITS[r]
                for a in range(NUM_APPS-1):
                    if q_list[ind][a] == 1.0:
                        flip = rd.choice([True, False])
                        if flip and next_sample[r*(NUM_APPS-1)+a] != 1.0:
                            next_sample[r*(NUM_APPS-1)+a] -= 1
                        avail -= next_sample[r*(NUM_APPS-1)+a]
                if q_list[ind][NUM_APPS-1] == 1.0:
                    flip = rd.choice([True, False])
                    unit = NUM_UNITS[r]-sum(next_sample[r*(NUM_APPS-1):(r+1)*(NUM_APPS-1)])
                    if flip and unit != 1.0:
                        avail -= (unit - 1)
                    else:
                        avail -= unit
                cnf = [int(float(avail)/float(NUM_APPS-boxes)) for b in range(NUM_APPS-boxes)]
                cnf[-1] += avail - sum(cnf)
                i = 0
                for a in range(NUM_APPS-1):
                    if q_list[ind][a] != 1.0:
                        next_sample[r*(NUM_APPS-1)+a] = cnf[i]
                        i += 1
            mind += 1
        print("next_sample: ", next_sample)
        # Sample the new configuration
        x_list.append(next_sample)
        q, y = sample_perf(next_sample)
        q_list.append(q)
        y_list.append(y)

        xp = np.array(x_list)
        yp = np.array(y_list)

        # Terminate if the termination requirements are met
        if ei < EI_THRESHOLD or np.max(yp) >= 0.99:
            break

    return n+1, np.max(yp)

def c_lite():
    
    # Generate the bounds and constraints required for optimization
    gen_bounds_and_constraints()

    # Generate the initial set of configurations
    init_configs = gen_initial_configs()

    print("init_configs: ", init_configs)

    # Get the baseline performances with maximum allocations for each application
    # get_baseline_perfs(init_configs)

    # Perform Bayesian optimization
    num_iters, obj_value = bayesian_optimization_engine(x0=init_configs)

    return num_iters, obj_value

def main():

    # Switch on the performance counters
    # os.system(WR_MSR_COMM + IA32_PERF_GBL_CTR + " 0x70000000F")
    # os.system(WR_MSR_COMM + IA32_PERF_FX_CTRL + " 0xFFF")

    # # Print the header
    # st = ''
    # for a in range(NUM_APPS):
    #     st += 'App' + str(a) + ','
    # st += 'ObjectiveValue' + ','
    # st += '#Iterations'

    # print(st)

    # Execute C-LITE
    num_iters, obj_value = c_lite()

    # Print the final results
    # st = ''
    # for a in LC_APPS:
    #     st += a + ','
    # for a in BG_APPS:
    #     st += a + ','
    # st += '%.2f'%obj_value + ','
    # st += '%.2f'%num_iters

    # print(st)

if __name__ == '__main__':

    # Invoke the main function
    #print("gen_random_config: ", gen_random_config())
    #main()
    print(gen_initial_configs())
