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

    data = np.loadtxt(resultsdir + "en_unc_" + instate + "_alpha" + str(alpha_over_kappa) + ".dat")
    ax.plot(data[:,0], data[:,1], color = "gray", label = r"$E_{unc}(t)$")

    data = np.loadtxt(resultsdir + "erg_unc_" + instate + "_alpha" + str(alpha_over_kappa) + ".dat")
    ax.plot(data[:,0], data[:,1], color = "black", label = r"$\epsilon_{unc}(t)$")

    ax.set_xlabel(r"$\kappa t$")
    ax.set_ylabel(r"$\bar{\epsilon}_{unr, \eta} (t) / \omega_0$")
    ax.grid(True, linestyle = ':', alpha = 0.6)

    ax.legend(loc = "upper right", ncol = 2)

    fig.suptitle(
            f"Mean values (erg): initial pure state, "  r"$\alpha / \kappa$" f" = {alpha_over_kappa},\n" f"{NUMBER_OF_TIMEINTERVALS : .1e} time intervals and{NUMBER_OF_TRAJECTORIES : .1e} trajectories."
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

    ax.legend(loc = "upper right", ncol = 2)

    fig.suptitle(
            f"Mean values (cap): initial pure state, "  r"$\alpha / \kappa$" f" = {alpha_over_kappa},\n" f"{NUMBER_OF_TIMEINTERVALS : .1e} time intervals and{NUMBER_OF_TRAJECTORIES : .1e} trajectories."
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
    ax.set_ylabel(r"$\sigma^2_{\bar{\epsilon}} (t)$")
    ax.grid(True, linestyle = ':', alpha = 0.6)

    ax.legend(loc = "upper right", ncol = 2)

    fig.suptitle(
            f"Variance (erg): initial pure state, "  r"$\alpha / \kappa$" f" = {alpha_over_kappa},\n" f"{NUMBER_OF_TIMEINTERVALS : .1e} time intervals and{NUMBER_OF_TRAJECTORIES : .1e} trajectories."
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
    ax.set_ylabel(r"$\sigma^2_{\bar{\mathcal{C}}} (t)$")
    ax.grid(True, linestyle = ':', alpha = 0.6)

    ax.legend(loc = "upper right", ncol = 2)

    fig.suptitle(
            f"Variance (cap): initial pure state, "  r"$\alpha / \kappa$" f" = {alpha_over_kappa},\n" f"{NUMBER_OF_TIMEINTERVALS : .1e} time intervals and{NUMBER_OF_TRAJECTORIES : .1e} trajectories."
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
    ax.set_ylabel(r"$\frac{\sigma^2_{\bar{\epsilon}} (t)}{\bar{\epsilon}^2}$")
    ax.grid(True, linestyle = ':', alpha = 0.6)

    ax.legend(loc = "upper right", ncol = 2)

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
    ax.set_ylabel(r"$\frac{\sigma^2_{\bar{\mathcal{C}}} (t)}{\bar{\mathcal{C}}^2}$")
    ax.grid(True, linestyle = ':', alpha = 0.6)

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
    ax.set_ylabel(r"$\mu^3_{\bar{\epsilon}} (t)$")
    ax.grid(True, linestyle = ':', alpha = 0.6)

    ax.legend(loc = "upper right", ncol = 2)

    fig.suptitle(
            f"Skewness (erg): initial pure state, "  r"$\alpha / \kappa$" f" = {alpha_over_kappa},\n" f"{NUMBER_OF_TIMEINTERVALS : .1e} time intervals and{NUMBER_OF_TRAJECTORIES : .1e} trajectories."
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
    ax.set_ylabel(r"$\mu^3_{\bar{\mathcal{C}}} (t)$")
    ax.grid(True, linestyle = ':', alpha = 0.6)

    ax.legend(loc = "upper right", ncol = 2)

    fig.suptitle(
            f"Skewness (cap): initial pure state, "  r"$\alpha / \kappa$" f" = {alpha_over_kappa},\n" f"{NUMBER_OF_TIMEINTERVALS : .1e} time intervals and{NUMBER_OF_TRAJECTORIES : .1e} trajectories."
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
    ax.set_ylabel(r"$\frac{\mu^3_{\bar{\epsilon}} (t)}{(\sigma^2_{\bar{\epsilon}})^{3/2}}$")
    ax.grid(True, linestyle = ':', alpha = 0.6)
    # ax.set_ylim(-7,5)
    # ax.set_xlim(0,4)

    ax.legend(loc = "lower right", ncol = 2)

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
    ax.set_ylabel(r"$\frac{\mu^3_{\bar{\mathcal{C}}} (t)}{(\sigma^2_{\bar{\mathcal{C}}})^{3/2}}$")
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
                label = unr_labels[j],
                density = True,
            )
        ax.set_title(f"Ergotropy distribution at t = {t} s: initial " + instate + " state, "  r"$\alpha / \kappa$" f" = {alpha_over_kappa},\n" f"{NUMBER_OF_TIMEINTERVALS : .1e} time intervals and{NUMBER_OF_TRAJECTORIES : .1e} trajectories.")
        ax.set_xlabel(r"$\epsilon$")
        ax.set_ylabel("Density")
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
                label = unr_labels[j],
                density = True,
            )
        ax.set_title(f"Capacity distribution at t = {t} s: initial " + instate + " state, "  r"$\alpha / \kappa$" f" = {alpha_over_kappa},\n" f"{NUMBER_OF_TIMEINTERVALS : .1e} time intervals and{NUMBER_OF_TRAJECTORIES : .1e} trajectories.")
        ax.set_xlabel(r"$\mathcal{C}$")
        ax.set_ylabel("Density")
        ax.grid(True, linestyle = ':', alpha = 0.6)
        ax.legend(loc = "upper right")
        plt.tight_layout()
        plt.savefig(plotsdir + "histo_cap_" + process + "_t" + str(t) + ".png", dpi = 300)
        plt.close(fig)

# ============================== MAIN ==============================

if __name__ == "__main__":
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