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
target_times = None

with open("input.dat", "r") as f:
    # cycle on every line of the file
    for line in f:
        # words separated by a space within a line become the elements of a list
        parts = line.split()
        nparts = len(parts)
        # it skips empty lines, controls that there are exactly two elements per line (otherwise it skips the line) and skip the comments
        if not line or line.startswith("#"):
            continue
        key = parts[0]
        if key != "HISTOTIME":
            value = parts[1]
        else:
            value = [float(parts[i]) for i in range(1,nparts)]
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
        elif key == "HISTOTIME":
            target_times = value

NUMBER_OF_TIMEINTERVALS = int(t_f / dt) # number of time intervals
resultsdir = "results/"                 # results directory
plotsdir = "plots/"                     # output directory for saved figures
os.makedirs(plotsdir, exist_ok = True)

# labels
unr = ["pd", "hod0", "hod90", "hed"]                                                                    # unravellings
unr_colors = ["red", "blue", "orange", "green"]                                                         # colors for the unravellings
unr_labels = ["PD", r"HoD \left< \hat{\sigma_x} \right>", r"HoD \left< \hat{\sigma_y} \right>", "HeD"]  # unravelling labels

process = instate + "_eta" + str(eta) + "_alpha" + str(alpha_over_kappa)

# ============================== MEAN VALUES ==============================

def plot_mean_erg():
    """Plots the mean (expectation value) of the ergotropy for every unravelling, plus E_unc(t) and erg_unc(t)."""
    fig, ax = plt.subplots(figsize = (6, 5))
    for j in range(len(unr)):
        data = np.loadtxt(resultsdir + "erg_" + unr[j] + "_" + process + ".dat", delimiter = "\t")
        ax.plot(data[:,0], data[:,1], color = unr_colors[j], label = r"$\bar{\epsilon}_{" + unr_labels[j] + ", " + str(eta) + r"}$")

    # unconditional energy
    data = np.loadtxt(resultsdir + "en_unc_" + instate + "_alpha" + str(alpha_over_kappa) + ".dat")
    ax.plot(data[:,0], data[:,1], color = "gray", label = r"$E_{unc}(t)$")

    # unconditional ergotropy
    data = np.loadtxt(resultsdir + "erg_unc_" + instate + "_alpha" + str(alpha_over_kappa) + ".dat")
    ax.plot(data[:,0], data[:,1], color = "black", label = r"$\epsilon_{unc}(t)$")

    ax.set_xlabel(r"$\kappa t$")
    ax.set_ylabel(r"$\bar{\epsilon}_{unr, \eta} (t) / \omega_0$")
    ax.grid(True, linestyle = ':', alpha = 0.6)
    ax.set_ylim(0, 0.3) if (instate == "m" and alpha_over_kappa == 0.4 and eta == 1.0) else None
    ax.legend(loc = "upper right" if (instate == "p" and alpha_over_kappa == 1.0 and eta == 0.4) else "lower right", ncol = 2)

    fig.suptitle(
            f"Mean values (erg): initial " + instate + " state, "  r"$\alpha / \kappa$" f" = {alpha_over_kappa},\n" f"{NUMBER_OF_TIMEINTERVALS : .1e} time intervals and{NUMBER_OF_TRAJECTORIES : .1e} trajectories."
        )

    plt.tight_layout()
    plt.savefig(plotsdir + "mean_erg_" + process + ".png", dpi = 300)
    plt.close(fig)


def plot_mean_cap():
    """Plots the mean (expectation value) of the capacity for every unravelling, plus cap_unc(t)."""
    fig, ax = plt.subplots(figsize = (6, 5))

    for j in range(len(unr)):
        data = np.loadtxt(resultsdir + "cap_" + unr[j] + "_" + process + ".dat", delimiter = "\t")
        ax.plot(data[:,0], data[:,1], color = unr_colors[j], label = r"$\bar{\mathcal{C}}_{" + unr_labels[j] + ", " + str(eta) + r"}$")

    data = np.loadtxt(resultsdir + "cap_unc_" + instate + "_alpha" + str(alpha_over_kappa) + ".dat")
    ax.plot(data[:,0], data[:,1], color = "black", label = r"$\mathcal{C}_{unc}(t)$")

    ax.set_xlabel(r"$\kappa t$")
    ax.set_ylabel(r"$\bar{\mathcal{C}}_{unr, \eta} (t) / \omega_0$")
    ax.grid(True, linestyle = ':', alpha = 0.6)
    ax.legend(loc = "upper right" if (instate == "p" and alpha_over_kappa == 1.0 and eta == 0.4) else "center right", ncol = 2)

    fig.suptitle(
            f"Mean values (cap): initial " + instate + " state, "  r"$\alpha / \kappa$" f" = {alpha_over_kappa},\n" f"{NUMBER_OF_TIMEINTERVALS : .1e} time intervals and{NUMBER_OF_TRAJECTORIES : .1e} trajectories."
        )

    plt.tight_layout()
    plt.savefig(plotsdir + "mean_cap_" + process + ".png", dpi = 300)
    plt.close(fig)


# ============================== VARIANCES ==============================

def plot_var_erg():
    """Plots the variance of the ergotropy for every unravelling."""
    fig, ax = plt.subplots(figsize = (6, 5))

    for j in range(len(unr)):
        data = np.loadtxt(resultsdir + "var_erg_" + unr[j] + "_" + process + ".dat", delimiter = "\t")
        ax.plot(data[:,0], data[:,1], color = unr_colors[j], label = r"$\bar{\epsilon}_{" + unr_labels[j] + ", " + str(eta) + r"}$")

    ax.set_xlabel(r"$\kappa t$")
    ax.set_ylabel(r"$\sigma^2(t)$")
    ax.grid(True, linestyle = ':', alpha = 0.6)

    ax.legend(loc = "upper right" if (instate == "p" and eta == 0.4 and alpha_over_kappa == 1.0) else "upper left", ncol = 2)

    fig.suptitle(
            f"Variance (erg): initial " + instate + " state, "  r"$\alpha / \kappa$" f" = {alpha_over_kappa},\n" f"{NUMBER_OF_TIMEINTERVALS : .1e} time intervals and{NUMBER_OF_TRAJECTORIES : .1e} trajectories."
        )

    plt.tight_layout()
    plt.savefig(plotsdir + "var_erg_" + process + ".png", dpi = 300)
    plt.close(fig)


def plot_var_cap():
    """Plots the variance of the capacity for every unravelling."""
    fig, ax = plt.subplots(figsize = (6, 5))

    for j in range(len(unr)):
        data = np.loadtxt(resultsdir + "var_cap_" + unr[j] + "_" + process + ".dat", delimiter = "\t")
        ax.plot(data[:,0], data[:,1], color = unr_colors[j], label = r"$\bar{\mathcal{C}}_{" + unr_labels[j] + ", " + str(eta) + r"}$")

    ax.set_xlabel(r"$\kappa t$")
    ax.set_ylabel(r"$\sigma^2(t)$")
    ax.grid(True, linestyle = ':', alpha = 0.6)

    ax.legend(loc = "upper right", ncol = 2)

    fig.suptitle(
            f"Variance (cap): initial " + instate + " state, "  r"$\alpha / \kappa$" f" = {alpha_over_kappa},\n" f"{NUMBER_OF_TIMEINTERVALS : .1e} time intervals and{NUMBER_OF_TRAJECTORIES : .1e} trajectories."
        )

    plt.tight_layout()
    plt.savefig(plotsdir + "var_cap_" + process + ".png", dpi = 300)
    plt.close(fig)


# ============================== NORMALIZED VARIANCES ==============================

def plot_norm_var_erg():
    """Plots the variance of the ergotropy, normalized by the squared mean, for every unravelling."""
    fig, ax = plt.subplots(figsize = (6, 5))

    for j in range(len(unr)):
        mean = np.loadtxt(resultsdir + "erg_" + unr[j] + "_" + process + ".dat", delimiter = "\t")
        var = np.loadtxt(resultsdir + "var_erg_" + unr[j] + "_" + process + ".dat", delimiter = "\t")
        norm_var = []
        tlist = []
        for x, y in zip(mean, var):
            # we append new elements only if the mean is non zero (we cannot divide by zero)
            if x[1] != 0:
                tlist.append(x[0])
                norm_var.append(y[1] / (x[1]**2))
        ax.plot(tlist, norm_var, color = unr_colors[j], label = r"$\bar{\epsilon}_{" + unr_labels[j] + ", " + str(eta) + r"}$")

    ax.set_xlabel(r"$\kappa t$")
    ax.set_ylabel(r"$\sigma^2 / \bar{\epsilon}^2 (t)$")
    ax.grid(True, linestyle = ':', alpha = 0.6)
    ax.set_yscale("log") if (instate == "m" and eta == 1.0 and alpha_over_kappa == 0.4) else None
    ax.legend(loc = "upper left" if (instate == "p" and eta == 1.0 and alpha_over_kappa == 0.4) else "upper right", ncol = 2)

    fig.suptitle(
            f"Normalized variance (erg): initial " + instate + " state, "  r"$\alpha / \kappa$" f" = {alpha_over_kappa},\n" f"{NUMBER_OF_TIMEINTERVALS : .1e} time intervals and{NUMBER_OF_TRAJECTORIES : .1e} trajectories."
        )

    plt.tight_layout()
    plt.savefig(plotsdir + "norm_var_erg_" + process + ".png", dpi = 300)
    plt.close(fig)


def plot_norm_var_cap():
    """Plots the variance of the capacity, normalized by the squared mean, for every unravelling."""
    fig, ax = plt.subplots(figsize = (6, 5))

    for j in range(len(unr)):
        mean = np.loadtxt(resultsdir + "cap_" + unr[j] + "_" + process + ".dat", delimiter = "\t")
        var = np.loadtxt(resultsdir + "var_cap_" + unr[j] + "_" + process + ".dat", delimiter = "\t")
        norm_var = []
        tlist = []
        for x, y in zip(mean, var):
            # we append new elements only if the mean is non zero (we cannot divide by zero)
            if x[1] != 0:
                tlist.append(x[0])
                norm_var.append(y[1] / (x[1]**2))
        ax.plot(tlist, norm_var, color = unr_colors[j], label = r"$\bar{\mathcal{C}}_{" + unr_labels[j] + ", " + str(eta) + r"}$")

    ax.set_xlabel(r"$\kappa t$")
    ax.set_ylabel(r"$\sigma^2 / \bar{\mathcal{C}}^2(t)$")
    ax.grid(True, linestyle = ':', alpha = 0.6)
    ax.set_yscale("log") if (instate == "m" and eta == 1.0 and alpha_over_kappa == 0.4) else None
    ax.legend(loc = "upper right", ncol = 2)

    fig.suptitle(
            f"Normalized variance (cap): initial " + instate + " state, "  r"$\alpha / \kappa$" f" = {alpha_over_kappa},\n" f"{NUMBER_OF_TIMEINTERVALS : .1e} time intervals and{NUMBER_OF_TRAJECTORIES : .1e} trajectories."
        )

    plt.tight_layout()
    plt.savefig(plotsdir + "norm_var_cap_" + process + ".png", dpi = 300)
    plt.close(fig)


# ============================== SKEWNESSES ==============================

def plot_skw_erg():
    """Plots the (third central moment) skewness of the ergotropy for every unravelling."""
    fig, ax = plt.subplots(figsize = (6, 5))

    for j in range(len(unr)):
        data = np.loadtxt(resultsdir + "skw_erg_" + unr[j] + "_" + process + ".dat", delimiter = "\t")
        ax.plot(data[:,0], data[:,1], color = unr_colors[j], label = r"$\bar{\epsilon}_{" + unr_labels[j] + ", " + str(eta) + r"}$")

    ax.set_xlabel(r"$\kappa t$")
    ax.set_ylabel(r"$\gamma (t)$")
    ax.grid(True, linestyle = ':', alpha = 0.6)
    ax.legend(loc = "upper right" if (instate == "p" and eta == 0.4 and alpha_over_kappa == 1.0) else "upper left", ncol = 2)

    fig.suptitle(
            f"Skewness (erg): initial " + instate + " state, "  r"$\alpha / \kappa$" f" = {alpha_over_kappa},\n" f"{NUMBER_OF_TIMEINTERVALS : .1e} time intervals and{NUMBER_OF_TRAJECTORIES : .1e} trajectories."
        )

    plt.tight_layout()
    plt.savefig(plotsdir + "skw_erg_" + process + ".png", dpi = 300)
    plt.close(fig)


def plot_skw_cap():
    """Plots the (third central moment) skewness of the capacity for every unravelling."""
    fig, ax = plt.subplots(figsize = (6, 5))

    for j in range(len(unr)):
        data = np.loadtxt(resultsdir + "skw_cap_" + unr[j] + "_" + process + ".dat", delimiter = "\t")
        ax.plot(data[:,0], data[:,1], color = unr_colors[j], label = r"$\bar{\mathcal{C}}_{" + unr_labels[j] + ", " + str(eta) + r"}$")

    ax.set_xlabel(r"$\kappa t$")
    ax.set_ylabel(r"$\gamma (t)$")
    ax.grid(True, linestyle = ':', alpha = 0.6)

    ax.legend(loc = "upper right", ncol = 2)

    fig.suptitle(
            f"Skewness (cap): initial " + instate + " state, "  r"$\alpha / \kappa$" f" = {alpha_over_kappa},\n" f"{NUMBER_OF_TIMEINTERVALS : .1e} time intervals and{NUMBER_OF_TRAJECTORIES : .1e} trajectories."
        )

    plt.tight_layout()
    plt.savefig(plotsdir + "skw_cap_" + process + ".png", dpi = 300)
    plt.close(fig)


# ============================== NORMALIZED SKEWNESSES ==============================

def plot_norm_skw_erg():
    """Plots the skewness of the ergotropy, normalized by variance^(3/2) (i.e. std^3), for every unravelling."""
    fig, ax = plt.subplots(figsize = (6, 5))

    for j in range(len(unr)):
        var = np.loadtxt(resultsdir + "var_erg_" + unr[j] + "_" + process + ".dat", delimiter = "\t")
        skw = np.loadtxt(resultsdir + "skw_erg_" + unr[j] + "_" + process + ".dat", delimiter = "\t")
        norm_skw = []
        tlist = []
        for x, y in zip(var, skw):
            # we append new elements only if the variance is non zero (we cannot divide by zero)
            if x[1] != 0:
                tlist.append(x[0])
                norm_skw.append(y[1] / (x[1]**1.5))
        ax.plot(tlist, norm_skw, color = unr_colors[j], label = r"$\bar{\epsilon}_{" + unr_labels[j] + ", " + str(eta) + r"}$")

    ax.set_xlabel(r"$\kappa t$")
    ax.set_ylabel(r"$\gamma / \sigma^2 (t)$")
    ax.grid(True, linestyle = ':', alpha = 0.6)
    ax.legend(loc = "center right", ncol = 2)

    fig.suptitle(
            f"Normalized skewness (erg): initial " + instate + " state, "  r"$\alpha / \kappa$" f" = {alpha_over_kappa},\n" f"{NUMBER_OF_TIMEINTERVALS : .1e} time intervals and{NUMBER_OF_TRAJECTORIES : .1e} trajectories."
        )

    plt.tight_layout()
    plt.savefig(plotsdir + "norm_skw_erg_" + process + ".png", dpi = 300)
    plt.close(fig)


def plot_norm_skw_cap():
    """Plots the skewness of the capacity, normalized by variance^(3/2) (i.e. std^3), for every unravelling."""
    fig, ax = plt.subplots(figsize = (6, 5))

    for j in range(len(unr)):
        var = np.loadtxt(resultsdir + "var_cap_" + unr[j] + "_" + process + ".dat", delimiter = "\t")
        skw = np.loadtxt(resultsdir + "skw_cap_" + unr[j] + "_" + process + ".dat", delimiter = "\t")
        norm_skw = []
        tlist = []
        for x, y in zip(var, skw):
            # we append new elements only if the variance is non zero (we cannot divide by zero)
            if x[1] != 0:
                tlist.append(x[0])
                norm_skw.append(y[1] / (x[1]**1.5))
        ax.plot(tlist, norm_skw, color = unr_colors[j], label = r"$\bar{\mathcal{C}}_{" + unr_labels[j] + ", " + str(eta) + r"}$")

    ax.set_xlabel(r"$\kappa t$")
    ax.set_ylabel(r"$\gamma / \sigma^2 (t)$")
    ax.grid(True, linestyle = ':', alpha = 0.6)

    ax.legend(loc = "upper right", ncol = 2)

    fig.suptitle(
            f"Normalized skewness (cap): initial " + instate + " state, "  r"$\alpha / \kappa$" f" = {alpha_over_kappa},\n" f"{NUMBER_OF_TIMEINTERVALS : .1e} time intervals and{NUMBER_OF_TRAJECTORIES : .1e} trajectories."
        )

    plt.tight_layout()
    plt.savefig(plotsdir + "norm_skw_cap_" + process + ".png", dpi = 300)
    plt.close(fig)

def plot_histograms():
    """Histogram generation for ergotropy and capacity at target times.
    For every target time, all unravellings are overlaid on the same axes
    and drawn as step outlines (histtype='step') instead of filled bars,
    so that the distributions can be compared directly.
    """
 
    for t in target_times:
        # ---------- ergotropy ----------
        fig, ax = plt.subplots(figsize=(6, 5))
        for j in range(len(unr)):
            data_erg = np.loadtxt(resultsdir + "histo_erg_" + unr[j] + "_" + process + "_t" + str(t) + ".dat")
            ax.hist(
                data_erg,
                bins = 50,
                histtype = "step",
                linewidth = 1.5,
                color = unr_colors[j],
                label = r"$\bar{\epsilon}_{" + unr_labels[j] + ", " + str(eta) + r"}$",
                # density = True,
            )
        ax.set_title(f"Trajectories' ergotropy distribution at t = {t} s:\n initial " + instate + " state, "  r"$\alpha / \kappa$" f" = {alpha_over_kappa},\n" f"{NUMBER_OF_TIMEINTERVALS : .1e} time intervals and{NUMBER_OF_TRAJECTORIES : .1e} trajectories.")
        ax.set_xlabel(r"$\bar{\epsilon}_{unr, \eta}$")
        # ax.set_ylabel("Density")
        ax.grid(True, linestyle = ':', alpha = 0.6)
        ax.legend(loc = "upper right")
        plt.tight_layout()
        plt.savefig(plotsdir + "histo_erg_" + process + "_t" + str(t) + ".png", dpi = 300)
        plt.close(fig)
 
        # ---------- capacity ----------
        fig, ax = plt.subplots(figsize=(6, 5))
        for j in range(len(unr)):
            data_cap = np.loadtxt(resultsdir + "histo_cap_" + unr[j] + "_" + process + "_t" + str(t) + ".dat")
            ax.hist(
                data_cap,
                bins = 50,
                histtype = "step",
                linewidth = 1.5,
                color = unr_colors[j],
                label = r"$\bar{\mathcal{C}}_{" + unr_labels[j] + ", " + str(eta) + r"}$",
                # density = True,
            )
        ax.set_title(f"Trajectories' capacity distribution at t = {t} s:\n initial " + instate + " state, "  r"$\alpha / \kappa$" f" = {alpha_over_kappa},\n" f"{NUMBER_OF_TIMEINTERVALS : .1e} time intervals and{NUMBER_OF_TRAJECTORIES : .1e} trajectories.")
        ax.set_xlabel(r"$\bar{\mathcal{C}}_{unr, \eta}$")
        # ax.set_ylabel("Density")
        ax.grid(True, linestyle = ':', alpha = 0.6)
        ax.legend(loc = "upper left" if (instate == "m" and eta == 1.0 and alpha_over_kappa == 0.4) else "upper right")
        plt.tight_layout()
        plt.savefig(plotsdir + "histo_cap_" + process + "_t" + str(t) + ".png", dpi = 300)
        plt.close(fig)

def plot_ss_erg():
    """Plots the steady states (expectation value) of the ergotropy against the rate between the driving field intensity and the emission rate for every unravelling, plus E^ss_unc(t) and erg^ss_unc(t)"""
    fig, ax = plt.subplots(figsize=(7,5))
    etavalues = [0.1, 0.7]
    etalinestyle = [":", "-."]
    for j in range(len(unr)):
        for i in range(len(etavalues)):
            data = np.loadtxt(resultsdir + "ss_erg_" + unr[j] + "_eta" + str(etavalues[i]) + ".dat")
            ax.plot(data[:,0], data[:,1], color = unr_colors[j], label = r"$\epsilon^{ss}_{" + unr_labels[j] + ", " + str((etavalues[i])) + r"}$", linestyle = etalinestyle[i])

    data = np.loadtxt(resultsdir + "ss_erg_unc.dat")
    ax.plot(data[:,0], data[:,1], color = "black", label = r"$\epsilon^{ss}_{unc}$")

    data = np.loadtxt(resultsdir + "ss_en_unc.dat")
    ax.plot(data[:,0], data[:,1], color = "gray", label = r"$E^{ss}_{unc}$")

    ax.set_xlabel(r"$\kappa / \alpha$")
    ax.set_ylabel(r"$\bar{\epsilon}_{ss}$")
    ax.grid(True, linestyle = ":", alpha = 0.6)
    ax.legend(loc = "lower right", ncol = 4)

    plt.suptitle(
        f"Steady state daemonic ergotropy for different unravellings as function of " r"$\alpha / \kappa$" f"\n with{NUMBER_OF_TIMEINTERVALS: .1e} time intervals and{NUMBER_OF_TRAJECTORIES : .1e} trajectories."
    )


    plt.tight_layout()
    plt.savefig(plotsdir + "ss_erg.png", dpi = 300)
    plt.close(fig)

def plot_ss_cap():
    """Plots the steady states (expectation value) of the capacity against the rate between the driving field intensity and the emission rate for every unravelling, plus E^ss_unc(t) and erg^ss_unc(t)"""
    fig, ax = plt.subplots(figsize=(7,5))
    etavalues = [0.1, 0.7]
    etalinestyle = [":", "-."]
    for j in range(len(unr)):
        for i in range(len(etavalues)):
            data = np.loadtxt(resultsdir + "ss_cap_" + unr[j] + "_eta" + str(etavalues[i]) + ".dat")
            ax.plot(data[:,0], data[:,1], color = unr_colors[j], label = r"$\epsilon^{ss}_{" + unr_labels[j] + ", " + str((etavalues[i])) + r"}$", linestyle = etalinestyle[i])

    data = np.loadtxt(resultsdir + "ss_cap_unc.dat")
    ax.plot(data[:,0], data[:,1], color = "black", label = r"$\epsilon^{ss}_{unc}$")
    ax.set_xlabel(r"$\kappa / \alpha$")
    ax.set_ylabel(r"$\bar{\epsilon}_{ss}$")
    ax.grid(True, linestyle = ":", alpha = 0.6)
    ax.legend(loc = "lower left", ncol = 2)

    plt.suptitle(
        f"Steady state daemonic capacity for different unravellings as function of " r"$\alpha / \kappa$" f"\n with{NUMBER_OF_TIMEINTERVALS: .1e} time intervals and{NUMBER_OF_TRAJECTORIES : .1e} trajectories."
    )

    plt.tight_layout()
    plt.savefig(plotsdir + "ss_cap.png", dpi = 300)
    plt.close(fig)

def plot_power_ev():
    """Plot the average power evolution for different unravelling against time and the energy thresholds"""
    fig, ax = plt.subplots(figsize = (6, 5))
    for j in range(len(unr)):
        data = np.loadtxt(resultsdir + "avepower_" + unr[j] + "_" + process + "_against_time.dat", delimiter = "\t")
        ax.plot(data[:,0], data[:,1], color = unr_colors[j], label = r"$\bar{\mathcal{P}}_{" + unr_labels[j] + ", " + str(eta) + r"}$")

    # unconditional power
    data = np.loadtxt(resultsdir + "pw_unc_" + instate + "_alpha" + str(alpha_over_kappa) + ".dat", delimiter = "\t")
    ax.plot(data[:,0], data[:,1], color = "gray", label = r"$P_{unc}(t)$")

    # unconditional ergotropic energy
    data = np.loadtxt(resultsdir + "erg_pw_unc_" + instate + "_alpha" + str(alpha_over_kappa) + ".dat", delimiter = "\t")
    ax.plot(data[:,0], data[:,1], color = "black", label = r"$\mathcal{P}_{unc}(t)$")

    ax.set_xlabel(r"$\kappa t$")
    ax.set_ylabel(r"$\bar{\mathcal{P}}_{unr, \eta} (t) / \omega_0$")
    ax.grid(True, linestyle = ":", alpha = 0.6)
    ax.legend(loc = "upper right")
    ax.set_yscale("log") if (instate == "m" and alpha_over_kappa == 0.4 and eta == 1.0) else None

    fig.suptitle(
        f"Average powers as function of time: initial " + instate + f" state, " r"$\alpha / \kappa$" f" = {alpha_over_kappa},\n" f"{NUMBER_OF_TIMEINTERVALS : .1e} time intervals and {NUMBER_OF_TRAJECTORIES : .1e} trajectories"
        )
    
    plt.tight_layout()
    plt.savefig(plotsdir + "avepower_" + process + "_against_time.png", dpi = 300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize = (6, 5))
    for j in range(len(unr)):
        data = np.loadtxt(resultsdir + "avepower_" + unr[j] + "_" + process + "_against_energy_threshold.dat", delimiter = "\t")
        ax.plot(data[:,0], data[:,1], color = unr_colors[j], label = r"$\bar{\mathcal{P}}_{" + unr_labels[j] + ", " + str(eta) + r"}$")

    ax.set_xlabel(r"$\epsilon_{threshold}$")
    ax.set_ylabel(r"$\bar{\mathcal{P}}_{unr, \eta} (t) / \omega_0$")
    ax.grid(True, linestyle = ":", alpha = 0.6)
    ax.legend(loc = "lower right")
    ax.set_yscale("log") if (instate == "m" and alpha_over_kappa == 0.4 and eta == 1.0) else None

    fig.suptitle(
        f"Average powers as function of energy thresholds: initial " + instate + f" state,\n" r"$\alpha / \kappa$" f" = {alpha_over_kappa}," f"{NUMBER_OF_TIMEINTERVALS : .1e} time intervals and {NUMBER_OF_TRAJECTORIES : .1e} trajectories"
        )
    
    plt.tight_layout()
    plt.savefig(plotsdir + "avepower_" + process + "_against_energy_threshold.png", dpi = 300)
    plt.close(fig)

# ============================== MAIN ==============================

if __name__ == "__main__":
    """
    plot_mean_erg()
    plot_mean_cap()
    plot_var_erg()
    plot_var_cap()
    plot_norm_var_erg()
    plot_norm_var_cap()
    plot_skw_erg()
    plot_skw_cap()
    plot_norm_skw_erg()
    plot_norm_skw_cap()
    plot_histograms()
    plot_ss_erg()
    plot_ss_cap()
    """
    plot_power_ev()