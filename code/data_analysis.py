# === DAEMONIC ERGOTROPY AND CAPACITY (AND RESPECTIVE MOMENTA) PLOT GENERATION FOR CONTINUOUSLY MONITORED OPEN QUANTUM SYSTEMS ===

# importing required libraries
import os
import numpy as np
import matplotlib.pyplot as plt

# variables initialization
alpha_over_kappa = None
eta = None
instate = None
t_f = None
dt = None
NUMBER_OF_TRAJECTORIES = None

with open("input.dat", "r") as f:
    # cycle on every line of the file
    for line in f:
        # words separated by a space within a line become the elements of a list
        parts = line.split()
        # it skips empty lines, controls that there are exactly two elements per line (otherwise it skips the line) and skip the comments
        if not line or len(parts) != 2 or line.startswith("#"):
            continue
        key, value = parts
        if key == "INSTATE":
            instate = value
        elif key == "ALPHA":
            alpha_over_kappa = float(value)
        elif key == "ETA":
            eta = float(value)
        elif key == "FINALT":
            t_f = float(value)
        elif key == "dt":
            dt = float(value)
        elif key == "NTRAJ":
            NUMBER_OF_TRAJECTORIES = int(value)

NUMBER_OF_TIMEINTERVALS = int(t_f / dt) # number of time intervals
resultsdir = "results/"                 # results directory
plotsdir = "plots/"                     # output directory for saved figures
os.makedirs(plotsdir, exist_ok = True)

# labels
unr = ["pd", "hod0", "hod90", "hed"]                                                                    # unravellings
unr_colors = ["red", "blue", "orange", "green"]                                                         # colors for the unravellings
obs = ["erg", "cap"]                                                                                    # observables
obs_labels = [r"\bar{\epsilon}", r"\bar{\mathcal{C}}"]                                                  # observable labels
unr_labels = ["PD", r"HoD \left< \hat{\sigma_x} \right>", r"HoD \left< \hat{\sigma_y} \right>", "HeD"]  # unravelling labels

# === EXPECTATION VALUEs ===

# prefixes and labels of unconditional dynamics
unc_prefixes = {"erg": "erg_unc_", "cap": "cap_unc_"}
unc_labels = {"erg": r"$\epsilon_{unc}(t)$", "cap": r"$\mathcal{C}_{unc}(t)$"}

# loop over physical quantities (ergotropy and capacity)
for i in range(len(obs)):
    # plot initialization
    fig, ax = plt.subplots(figsize=(6, 5))
    # loop over the unravellings
    for j in range(len(unr)):
        data = []
        data = np.loadtxt(resultsdir + obs[i] + "_" + unr[j] + "_" + instate + "_eta" + str(eta) + "_alpha" + str(alpha_over_kappa) + ".dat", delimiter="\t")
        ax.plot(data[:,0], data[:,1], color = unr_colors[j], label = r"$" + obs_labels[i] + r"_{" + unr_labels[j] + ", " + str(eta) + r"}$")
    # reading energy evolution from file
    if obs[i] == "erg":
        data = []
        data = np.loadtxt(resultsdir + "en_unc_" + instate + "_alpha" + str(alpha_over_kappa) + ".dat")
        ax.plot(data[:,0], data[:,1], color = "gray", label = r"$E_{unc}(t)$")
    # reading unconditional quantities evolution
    data = []
    data = np.loadtxt(resultsdir + unc_prefixes[obs[i]] + instate + "_alpha" + str(alpha_over_kappa) + ".dat")
    ax.plot(data[:,0], data[:,1], color = "black", label = unc_labels[obs[i]])

    # costumize axes and grids
    ax.set_xlabel(r"$\kappa t$")
    ax.set_ylabel(r"$" + obs_labels[i] + r"_{unr, \eta} (t) / \omega_0$")
    ax.grid(True, linestyle = ':', alpha = 0.6)

    # costumize legend
    ax.legend(loc = "upper right", ncol = 2)

    fig.suptitle(
            f"Mean values ({obs[i]}): initial pure state, "  r"$\alpha / \kappa$" f" = {alpha_over_kappa},\n" f"{NUMBER_OF_TIMEINTERVALS : .1e} time intervals and{NUMBER_OF_TRAJECTORIES : .1e} trajectories."
        )

    # layout optimization and saving
    plt.tight_layout()
    plt.savefig(plotsdir + "mean_" + obs[i] + "_" + instate + "_eta" + str(eta) + "_alpha" + str(alpha_over_kappa) + ".png", dpi = 300)
    plt.close(fig)

# === VARIANCES ===

for i in range(len(obs)):
    fig, ax = plt.subplots(figsize=(6, 5))
    # loop over the unravellings
    for j in range(len(unr)):
        data = []
        data = np.loadtxt(resultsdir + "var_" + obs[i] + "_" + unr[j] + "_" + instate + "_eta" + str(eta) + "_alpha" + str(alpha_over_kappa) + ".dat", delimiter="\t")
        ax.plot(data[:,0], data[:,1], color = unr_colors[j], label = r"$" + obs_labels[i] + r"_{" + unr_labels[j] + ", " + str(eta) + r"}$")

    # costumize axes and grids
    ax.set_xlabel(r"$\kappa t$")
    ax.set_ylabel(r"$\sigma^2_" + obs_labels[i] + " (t)$")
    ax.grid(True, linestyle = ':', alpha = 0.6)

    ax.legend(loc = "upper right", ncol = 2)

    fig.suptitle(
            f"Variance ({obs[i]}): initial pure state, "  r"$\alpha / \kappa$" f" = {alpha_over_kappa},\n" f"{NUMBER_OF_TIMEINTERVALS : .1e} time intervals and{NUMBER_OF_TRAJECTORIES : .1e} trajectories."
        )

    # layout optimization and saving
    plt.tight_layout()
    plt.savefig(plotsdir + "var_" + obs[i] + "_" + instate + "_eta" + str(eta) + "_alpha" + str(alpha_over_kappa) + ".png", dpi = 300)
    plt.close(fig)

# === NORMALIZED VARIANCES ===

for i in range(len(obs)):
    fig, ax = plt.subplots(figsize=(6, 5))
    # loop over the unravellings
    for j in range(len(unr)):
        mean = []
        var = []
        mean = np.loadtxt(resultsdir + obs[i] + "_" + unr[j] + "_" + instate + "_eta" + str(eta) + "_alpha" + str(alpha_over_kappa) + ".dat", delimiter="\t")
        var = np.loadtxt(resultsdir + "var_" + obs[i] + "_" + unr[j] + "_" + instate + "_eta" + str(eta) + "_alpha" + str(alpha_over_kappa) + ".dat", delimiter="\t")
        norm_var = []
        tlist = []
        for x, y in zip(mean, var):
            # we append new elements only if the mean is non zero (we cannot divide by zero)
            if x[1] != 0:
                tlist.append(x[0])
                norm_var.append(y[1] / (x[1]**2))
        ax.plot(tlist, norm_var, color = unr_colors[j], label = r"$" + obs_labels[i] + r"_{" + unr_labels[j] + ", " + str(eta) + r"}$")

    # costumize axes and grids
    ax.set_xlabel(r"$\kappa t$")
    ax.set_ylabel(r"$\frac{\sigma^2_" + obs_labels[i] + " (t)}{" +  obs_labels[i] + "^2}$")
    ax.grid(True, linestyle = ':', alpha = 0.6)

    ax.legend(loc = "upper right", ncol = 2)

    fig.suptitle(
            f"Normalized variance ({obs[i]}): initial " + instate + " state, "  r"$\alpha / \kappa$" f" = {alpha_over_kappa},\n" f"{NUMBER_OF_TIMEINTERVALS : .1e} time intervals and{NUMBER_OF_TRAJECTORIES : .1e} trajectories."
        )

    # layout optimization and saving
    plt.tight_layout()
    plt.savefig(plotsdir + "norm_var_" + obs[i] + "_" + instate + "_eta" + str(eta) + "_alpha" + str(alpha_over_kappa) + ".png", dpi = 300)
    plt.close(fig)

# === SKEWNESSES ===

for i in range(len(obs)):
    fig, ax = plt.subplots(figsize=(6, 5))
    # loop over the unravellings
    for j in range(len(unr)):
        data = []
        data = np.loadtxt(resultsdir + "skw_" + obs[i] + "_" + unr[j] + "_" + instate + "_eta" + str(eta) + "_alpha" + str(alpha_over_kappa) + ".dat", delimiter="\t")
        ax.plot(data[:,0], data[:,1], color = unr_colors[j], label = r"$" + obs_labels[i] + r"_{" + unr_labels[j] + ", " + str(eta) + r"}$")

    # costumize axes and grids
    ax.set_xlabel(r"$\kappa t$")
    ax.set_ylabel(r"$\mu^3_" + obs_labels[i] + " (t)$")
    ax.grid(True, linestyle = ':', alpha = 0.6)

    ax.legend(loc = "upper right", ncol = 2)

    fig.suptitle(
            f"Skewness ({obs[i]}): initial pure state, "  r"$\alpha / \kappa$" f" = {alpha_over_kappa}\n," f"{NUMBER_OF_TIMEINTERVALS : .1e} time intervals and{NUMBER_OF_TRAJECTORIES : .1e} trajectories."
        )

    # layout optimization and saving
    plt.tight_layout()
    plt.savefig(plotsdir + "skw_" + obs[i] + "_" + instate + "_eta" + str(eta) + "_alpha" + str(alpha_over_kappa) + ".png", dpi = 300)
    plt.close(fig)

# === NORMALIZED SKEWNESSES ===

for i in range(len(obs)):
    fig, ax = plt.subplots(figsize=(6, 5))

    for j in range(len(unr)):
        var = []
        skw = []
        var = np.loadtxt(resultsdir + "var_" + obs[i] + "_" + unr[j] + "_" + instate + "_eta" + str(eta) + "_alpha" + str(alpha_over_kappa) + ".dat", delimiter="\t")
        skw = np.loadtxt(resultsdir + "skw_" + obs[i] + "_" + unr[j] + "_" + instate + "_eta" + str(eta) + "_alpha" + str(alpha_over_kappa) + ".dat", delimiter="\t")
        norm_skw = []
        tlist = []
        for x, y in zip(var, skw):
            # we append new elements only if the variance is non zero (we cannot divide by zero)
            if x[1] != 0:
                tlist.append(x[0])
                norm_skw.append(y[1] / (x[1]**1.5))
        ax.plot(tlist, norm_skw, color = unr_colors[j], label = r"$" + obs_labels[i] + r"_{" + unr_labels[j] + ", " + str(eta) + r"}$")

    # costumize axes and grids
    ax.set_xlabel(r"$\kappa t$")
    ax.set_ylabel(r"$\frac{\mu^3_" + obs_labels[i] + r" (t)}{(\sigma^2_" + obs_labels[i] + ")^{3/2}}$")
    ax.grid(True, linestyle = ':', alpha = 0.6)

    ax.legend(loc = "upper right", ncol = 2)

    fig.suptitle(
            f"Normalized skewness ({obs[i]}): initial " + instate + " state, "  r"$\alpha / \kappa$" f" = {alpha_over_kappa},\n" f"{NUMBER_OF_TIMEINTERVALS : .1e} time intervals and{NUMBER_OF_TRAJECTORIES : .1e} trajectories."
        )

    # layout optimization and saving
    plt.tight_layout()
    plt.savefig(plotsdir + "norm_skw_" + obs[i] + "_" + instate + "_eta" + str(eta) + "_alpha" + str(alpha_over_kappa) + ".png", dpi = 300)
    plt.close(fig)