#
#  NAME
#    problem_set2_solutions.py
#
#  DESCRIPTION
#    Open, view, and analyze action potentials recorded during a behavioral
#    task.  In Problem Set 2, you will write create and test your own code to
#    create tuning curves.
#

#Helper code to import some functions we will use
from __future__ import division
import numpy as np
import matplotlib.pylab as plt
import matplotlib.mlab as mlab
from scipy import optimize
from scipy import stats

def load_experiment(filename):
    """
    load_experiment takes the file name and reads in the data.  It returns a
    two-dimensional array, with the first column containing the direction of
    motion for the trial, and the second column giving you the time the
    animal began movement during thaht trial.
    """
    data = np.load(filename)[()];
    return np.array(data)

def load_neuraldata(filename):
    """
    load_neuraldata takes the file name and reads in the data for that neuron.
    It returns an arary of spike times.
    """
    data = np.load(filename)[()];
    return np.array(data)
    
def bin_spikes(trials, spk_times, time_bin):
    """
    bin_spikes takes the trials array (with directions and times) and the spk_times
    array with spike times and returns the average firing rate for each of the
    eight directions of motion, as calculated within a time_bin before and after
    the trial time (time_bin should be given in seconds).  For example,
    time_bin = .1 will count the spikes from 100ms before to 100ms after the 
    trial began.
    
    dir_rates should be an 8x2 array with the first column containing the directions
    (in degrees from 0-360) and the second column containing the average firing rate
    for each direction
    """
    motion = np.unique(trials[:,0])
    dir_rates = np.zeros((motion.size,2))
    dir_rates[:,0] = motion
    
    for i in range(0,motion.size):
        tmp = trials[:,1][trials[:,0] == dir_rates[i,0]]
        ind_size = 0
        for n in range(0,tmp.size):
            ind = spk_times[(spk_times>(tmp[n] - time_bin)) & (spk_times<(tmp[n] + time_bin))]
            ind_size = ind_size + ind.size
        dir_rates[i,1] = (ind_size / tmp.size / (2*time_bin))
    
    return dir_rates
    
def plot_tuning_curves(direction_rates, title):
    """
    This function takes the x-values and the y-values  in units of spikes/s 
    (found in the two columns of direction_rates) and plots a histogram and 
    polar representation of the tuning curve. It adds the given title.
    """
    plt.subplot(2,2,1)
    plt.bar(direction_rates[:,0],direction_rates[:,1],width=45)
    plt.xlabel('Direction of Motions (Degrees)')
    plt.ylabel('Firing Rates (Spikes/sec)')
    plt.title(title)
    plt.axis([0, 360, 0, 40])
    plt.xticks(direction_rates[:,0])
    plt.subplot(2,2,2,polar=True)
    spkiecount = np.append(direction_rates[:,1],direction_rates[0,1])
    plt.polar(np.arange(0,361,45)*np.pi/180,spkiecount,label='Firing Rate (spike/s)')
    plt.legend(loc=8)
    plt.title(title)

    
def roll_axes(direction_rates):
    """
    roll_axes takes the x-values (directions) and y-values (direction_rates)
    and return new x and y values that have been "rolled" to put the maximum
    direction_rate in the center of the curve. The first and last y-value in the
    returned list should be set to be the same. (See problem set directions)
    Hint: Use np.roll()
    """
    max_roll_index = np.argmax(direction_rates[:,1])
    new_xs = np.roll(direction_rates[:,0],max_roll_index)
    new_ys = np.roll(direction_rates[:,1],max_roll_index)
    new_xs = np.append(new_xs,new_xs[0])    
    new_ys = np.append(new_ys,new_ys[0]) 
    zero_index = np.where(new_xs == 0)
    new_xs[0:zero_index[0][0]] = new_xs[0:zero_index[0][0]] - 360    
    
    roll_degrees = np.argmax(direction_rates[:,1])
    
    return new_xs, new_ys, roll_degrees    
    

def normal_fit(x,mu, sigma, A):
    """
    This creates a normal curve over the values in x with mean mu and
    variance sigma.  It is scaled up to height A.
    """
    n = A*mlab.normpdf(x,mu,sigma)
    return n

def fit_tuning_curve(centered_x,centered_y):
    """
    This takes our rolled curve, generates the guesses for the fit function,
    and runs the fit.  It returns the parameters to generate the curve.
    """
    max_x = centered_x[np.argmax(centered_y)]
    max_y = np.amax(centered_y)
    sigma = 90
    
    p, conv = optimize.curve_fit(normal_fit,centered_x,centered_y,p0=[max_x, sigma, max_y])
    
#    centered_x = np.deg2rad(centered_x)
#    p, conv = optimize.curve_fit(von_mises_fitfunc,centered_x,centered_y,p0=[max_y,4,max_x, 1])

    return p
    


def plot_fits(direction_rates,fit_curve,title):
    """
    This function takes the x-values and the y-values  in units of spikes/s 
    (found in the two columns of direction_rates and fit_curve) and plots the 
    actual values with circles, and the curves as lines in both linear and 
    polar plots.
    """
    curve_xs = np.arange(direction_rates[0,0], direction_rates[-1,0])
    fit_ys2 = normal_fit(curve_xs,fit_curve[0],fit_curve[1],fit_curve[2])
    
    
    plt.subplot(2,2,3)
    plt.plot(direction_rates[:,0],direction_rates[:,1],'o',hold=True)
    plt.plot(curve_xs,fit_ys2,'-')
    plt.xlabel('Direction of Motions (Degrees)')
    plt.ylabel('Firing Rates (Spikes/sec)')
    plt.title(title)
    plt.axis([0, 360, 0, 40])
    plt.xticks(direction_rates[:,0])
    
    fit_ys = normal_fit(direction_rates[:,0],fit_curve[0],fit_curve[1],fit_curve[2])
    plt.subplot(2,2,4,polar=True)
    spkiecount = np.append(direction_rates[:,1],direction_rates[0,1])
    plt.polar(np.arange(0,361,45)*np.pi/180,spkiecount,'o',label='Firing Rate (spike/s)')
    plt.hold(True)
    spkiecount_y = np.append(fit_ys,fit_ys[0])
    plt.plot(np.arange(0,361,45)*np.pi/180,spkiecount_y,'-')    
    plt.legend(loc=8)
    plt.title(title)
    
    fit_ys2 = np.transpose(np.vstack((curve_xs,fit_ys2)))    
    
    return(fit_ys2)

def von_mises_fitfunc(x, A, kappa, l, s):
    """
    This creates a scaled Von Mises distrubition.
    """
    return A*stats.vonmises.pdf(x, kappa, loc=l, scale=s)


    
def preferred_direction(fit_curve):
    """
    The function takes a 2-dimensional array with the x-values of the fit curve
    in the first column and the y-values of the fit curve in the second.  
    It returns the preferred direction of the neuron (in degrees).
    """
    pd = np.argmax(fit_curve[:,1])

    return pd
    
        
##########################
#You can put the code that calls the above functions down here    
if __name__ == "__main__":
    trials = load_experiment('trials.npy')   
    spk_times = load_neuraldata('neuron2.npy') 
    time_bin = 0.3
    direction_rates = bin_spikes(trials, spk_times, time_bin)
    plot_tuning_curves(direction_rates,"Tuning Curve")
    
    centered_x,centered_y,roll_degrees = roll_axes(direction_rates)
    fit_curve = fit_tuning_curve(centered_x,centered_y)
    fit_ys = plot_fits(direction_rates,fit_curve,"Tuning Curve")
    pref = preferred_direction(fit_ys)
   
   
   
   
