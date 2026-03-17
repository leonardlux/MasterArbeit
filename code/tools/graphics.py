import numpy as np
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit

def save_circuit_diagram(circuit,savepath):
    diagram = circuit.diagram("timeline-svg")
    with open(savepath, 'w') as f:
        f.write(str(diagram))

def determine_slope(
        noise,
        log_prob,
        yerr=[],
        plot=False,
        plotpath="",):

    def linear(x,a,b):
        return a*x + b

    def clean_array(x,*args):
        """
        cleans out NaNs and infs from the x array,
        and applies the same mask to all args
        """
        mask = np.logical_and(np.logical_not(np.isnan(x)),np.isfinite(x))
        return x[mask], *[arg[mask] for arg in args] 


    # clean out NaNs and infite values, after log
    y, x= clean_array(np.log(log_prob), np.log(noise))

    # fit to curve on log scale
    popt, pcov = curve_fit(linear,x,y)
    exponent = popt[0]
    const = popt[1]
    
    if plot:
        plt.figure()
        if len(yerr)!=0:
            plt.errorbar(noise,log_prob,yerr=yerr,label="data points")
        else:
            plt.plot(noise,log_prob,label="data points")
        plt.plot(noise,np.exp(linear(np.log(noise),exponent,const)),label=f"fit: exp = {exponent:.4}")
        plt.xlabel('Physical error rate')
        plt.ylabel('Logical error rate')
        plt.legend(loc = 'upper left')
        plt.yscale('log')
        plt.xscale('log')
        plt.show()
        if plotpath!= "":
            plt.savefig(plotpath)
    return exponent, const

def plot_diff_noise_level(
        log_error_rates,
        y_errs,
        distances,
        noise_set,
        filename = "",
        plot_path = "/home/leo/Documents/MasterArbeit/code/images",
        fit_slopes = False,
        reference_lines = False,
        title="",
        seperate_figure=True,
        prelabel="",
        p_th = None,
    ):
    cm = 1/2.54 # to convert inches to cm
    if seperate_figure:
        plt.figure()
        plt.subplots(figsize=(20*cm,10*cm))
        plt.loglog()
        plt.xlabel("physical error rate")
        plt.ylabel("logical error rate")

    for i, log_error_prob in enumerate(log_error_rates):

        log_error_prob = np.array(log_error_prob)
        y_err = y_errs[i] 

        plt.errorbar(
            noise_set,
            log_error_prob,
            yerr=y_err,
            label=f"{prelabel}d={distances[i]}",
            )
        
        # TODO: add fitted reference lines!
        if fit_slopes:
            cutoff = int(len(noise_set)/2)
            exponent, const = determine_slope(noise_set[:cutoff],log_error_prob[:cutoff],plot=False)
            x = noise_set
            y = np.exp((np.log(noise_set)*exponent + const))
            plt.plot(
                x,
                y,
                label=f"fit d={distances[i]}: exp = {exponent:.4}",
                linestyle="dotted",
                alpha=0.5,
                )

    if reference_lines:
        plt.plot(noise_set,noise_set**2,label="$p^2$")
        plt.plot(noise_set,noise_set,label="$p$",c="green")

    if p_th != None:
        plt.axvline(p_th,label="$p_{th}$")

    if filename != "":
        plt.savefig(plot_path +"/"+ filename +".pdf") # # TODO clean up using OS or similar
    if seperate_figure:
        plt.title(title)
        plt.grid()
        plt.legend()
        plt.show()
    pass

def overlay_different_slopes(
        list_log_error_rates,
        list_yerrs,
        distances,
        noise_set,
        titles=[""]*100, # hacky
        title="",
        reference_lines=True,
        fit_slopes=False,
        ):
    cm = 1/2.54 # to convert inches to cm
    plt.figure()
    plt.subplots(figsize=(20*cm,10*cm))
    plt.loglog()
    plt.xlabel("physical error rate")
    plt.ylabel("logical error rate")
    for i in range(len(list_log_error_rates)):
        plot_diff_noise_level(
            log_error_rates=list_log_error_rates[i],
            y_errs=list_yerrs[i],
            distances=distances,
            noise_set=noise_set,
            prelabel=titles[i] +" ",
            fit_slopes=fit_slopes,
            seperate_figure=False,
        )
    
    
    if reference_lines:
        plt.plot(noise_set,noise_set**2,label="$p^2$")
        plt.plot(noise_set,noise_set,label="$p$",c="green")
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.show()
    pass

def plot_fssa_results(xs,ys,yerrs,pc,nu,distances):
    def fit_func(x,d):
        return d**(1/nu)*(x-pc)
    plt.figure()
    for x, y, yerr, d in zip(xs,ys,yerrs,distances):
        plt.errorbar(fit_func(x,d),y,yerr=yerr,label=f"d={d}") 
    plt.legend()
    plt.show()
