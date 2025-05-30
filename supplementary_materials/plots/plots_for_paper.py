# Author: Andrey Latyshev
# 
# This script together with `npy` and `pkl` data are a part of supplementary
# material to the paper "Expressing general constitutive models in FEniCSx using
# external operators and algorithmic automatic differentiation" (preprint:
# hal.science/hal-04735022v1) and generate plots for the paper.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm # colorbar
import matplotlib.colors as mcolors # colorbar
from mpl_toolkits.axes_grid1 import make_axes_locatable # colorbar
from mpltools import annotation  # slope markers

rc_fonts = {
    "figure.figsize": (4, 3),
    "font.size": 9.625,
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}\RequirePackage[ttscale=.875,oldstyle]{libertine}\RequirePackage[libertine]{newtxmath}",
    "font.family": "libertine",
}
plt.rcParams.update(rc_fonts)

def plot_von_mises_displacements_curve():
    """Plots pressure-displacement curve of the inner boundary for the von Mises
    problem.
    Figure 4 in the paper.
    """
    # two-rows arrays with the displacements in the first row and the pressures in the second row
    results = np.load("data/results_von_mises.npy")
    results_pure_ufl = np.load("data/results_von_mises_pure_ufl.npy")

    fig, ax = plt.subplots(figsize=(4, 3))

    ax.plot(results_pure_ufl[:, 0], results_pure_ufl[:, 1], "o-", label="pure UFL")
    ax.plot(results[:, 0], results[:, 1], "*-", label="dolfinx-external-operator (Numba)")
    ax.set_xlabel(r"Displacement of inner boundary $u_x$ at $(R_i, 0)$ [mm]")
    ax.set_ylabel(r"Applied pressure $q/q_{\text{lim}}$ [-]")
    ax.grid()
    ax.legend()
    fig.tight_layout()
    fig.savefig("output/von_mises_results.pdf")

def plot_mohr_coulomb_yield_surface():
    """Plots tracing of the Mohr-Coulomb yield surface with apex smoothing.
    Figure 6 in the paper.
    The results are projected for the deviatoric plane p = 0.1.
    """
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(4, 4))

    # Mohr-Coulomb yield surface with apex smoothing

    # Lode angles restored after the return mapping
    theta_returned = np.load("data/theta_returned.npy") 
    # radial coordinate restored after the return mapping
    rho_returned = np.load("data/rho_returned.npy")

    N_loads = len(theta_returned)
    colormap = cm.plasma
    colors = colormap(np.linspace(0.0, 1.0, N_loads))
    for i, color in enumerate(colors):
        rho_total = np.array([])
        theta_total = np.array([])
        for j in range(12):
            angles = j * np.pi / 3 - j % 2 * theta_returned[i] + (1 - j % 2) * theta_returned[i]
            theta_total = np.concatenate([theta_total, angles])
            rho_total = np.concatenate([rho_total, rho_returned[i]])

        ax.plot(theta_total, rho_total, ".", color=color)

    norm = mcolors.Normalize(vmin=0.7, vmax=0.7*9)
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical')
    cbar.set_label(r'Magnitude of the stress path deviator, $\rho$ [MPa]')

    # Standard Mohr-Coulomb yield surface (without smoothing)

    # Lode angles uniformly generated from -pi/6 to pi/6
    theta_values = np.load("data/theta_values.npy")
    # radial coordinate for the standard Mohr-Coulomb yield surface computed
    # analytically 
    rho_standard_MC = np.load("data/rho_standard_MC.npy")

    theta_standard_MC_total = np.array([])
    rho_standard_MC_total = np.array([])
    for j in range(12):
        angles = j * np.pi / 3 - j % 2 * theta_values + (1 - j % 2) * theta_values
        theta_standard_MC_total = np.concatenate([theta_standard_MC_total, angles])
        rho_standard_MC_total = np.concatenate([rho_standard_MC_total, rho_standard_MC])
    ax.plot(theta_standard_MC_total, rho_standard_MC_total, "-", color="black")
    ax.set_yticklabels([])

    fig.savefig("output/MC_yield_surface.pdf", bbox_inches='tight')

def plot_taylor_test():
    """Plots curves establishing the convergence of the Taylor remainders.
    Figure 7 in the paper.
    """
    k_list = np.load("data/k_list.npy")
    reminders_data = np.load("data/taylor_reminders_data.npy")

    # Each array contains the values of dual norms of the Taylor remainders.
    zero_order_remainder_elastic = reminders_data[0]
    first_order_remainder_elastic = reminders_data[1]
    zero_order_remainder_plastic = reminders_data[2]
    first_order_remainder_plastic = reminders_data[3]

    fig, axs = plt.subplots(1, 2, figsize=(5.8, 3))

    axs[0].loglog(k_list, zero_order_remainder_elastic, "o-", label=r"$\|r_k^0\|_{V^\prime}$")
    axs[0].loglog(k_list, first_order_remainder_elastic, "o-", label=r"$\|r_k^1\|_{V^\prime}$")
    annotation.slope_marker((2e-4, 5e-5), (1, 1), ax=axs[0], poly_kwargs={"facecolor": "tab:blue"})
    axs[0].text(0.5, -0.2, "(a) Elastic phase", transform=axs[0].transAxes, ha="center", va="top")

    axs[1].loglog(k_list, zero_order_remainder_plastic, "o-", label=r"$\|r_k^0\|_{V^\prime}$")
    annotation.slope_marker((2e-4, 5e-5), (1, 1), ax=axs[1], poly_kwargs={"facecolor": "tab:blue"})
    axs[1].loglog(k_list, first_order_remainder_plastic, "o-", label=r"$\|r_k^1\|_{V^\prime}$")
    annotation.slope_marker((2e-4, 5e-13), (2, 1), ax=axs[1], poly_kwargs={"facecolor": "tab:orange"})
    axs[1].text(0.5, -0.2, "(b) Plastic phase", transform=axs[1].transAxes, ha="center", va="top")

    for i in range(2):
        axs[i].set_xlabel(r"$k$")
        axs[i].set_ylabel("Taylor remainder norm")
        axs[i].legend()
        axs[i].grid()
    fig.tight_layout()
    
    fig.savefig("output/taylor_test.pdf")

def plot_mohr_coulomb_displacements_curve():
    """Plots the soil self-weight as a function of the displacement of the
    slope.
    Figure 8 in the paper.
    """
    # two-rows array with the displacements in the first row and the soil
    # self-weight in the second row
    results = np.load("data/results_mohr_coulomb.npy")
    results_supp = np.load("data/results_mohr_coulomb_non_associative.npy")

    fig, ax = plt.subplots(figsize=(4, 3))

    # analytical solution in the case of the rectangular slope with friction
    # angle to 30 deg (W. F. Chen et al. 1990, p. 368).
    ax.plot(results[:, 0], results[:, 1], ".-", color="sandybrown", label=r"associative flow  $(\phi = \psi = 30^\circ)$")
    import matplotlib.path as mpath

    star = mpath.Path.unit_regular_star(6)
    ax.plot(results_supp[:, 0], results_supp[:, 1], "-", marker=star, color="seagreen", label=r"non-associative flow  $(\phi = 30^\circ, \psi = 10^\circ)$")
    l_lim = 6.69
    H = 1.0
    c = 3.45
    gamma_lim = l_lim / H * c
    ax.axhline(y=gamma_lim, color='r', linestyle='--', label=r"$\gamma_\text{lim}$")
    ax.set_xlabel(r"Displacement of the slope $u_x$ at $(0, H)$ [mm]")
    ax.set_ylabel(r"Soil self-weight $\gamma$ [MPa/mm$^3$]")
    ax.grid()
    ax.legend()

    fig.tight_layout()
    fig.savefig("output/mohr_coulomb_results.pdf")

def plot_slope():
    """Plots the deformed slope with the magnitude of displacements.
    Figure 9 in the paper.
    """
    # slope image generated via pyvista
    img = np.load("data/slope.npy")
    # values of displacements in mesh nodes
    u = np.load("data/slope_displacement.npy")

    fig, ax = plt.subplots(figsize=(4, 4))

    ax.imshow(img)
    ax.axis('off')

    u_mag = np.sqrt(np.sum(u**2, axis=1))
    norm = mcolors.Normalize(vmin=u_mag.min(), vmax=u_mag.max())
    sm = plt.cm.ScalarMappable(norm=norm)
    sm.set_array([])

    divider = make_axes_locatable(ax)  
    colorbar_axes = divider.append_axes("right", 
                                        size="5%", 
                                        pad=0.1) 

    cbar = fig.colorbar(sm, cax=colorbar_axes, orientation='vertical')
    cbar.set_label(r'Magnitude of displacements, $\|u\|_2$ [mm]')
    ticks = cbar.get_ticks().tolist()[:-1] 
    ticks.append(u_mag.max())
    cbar.set_ticks(ticks)
    fig.savefig("output/slope.pdf", dpi=600, bbox_inches="tight")

def plot_strong_scaling():
    """Plots the strong scaling results for the Mohr-Coulomb problem.
    Makes two plots: the line plot for the whole modelling and the bar plot
    for the last loading step.
    Figures 10 and 11 in the appendix of the paper.
    """
    import pandas as pd
    import pickle

    cols = ["matrix_assembling", "linear_solver", "constitutive_model_update"]
    def get_sum(performance_monitor):
        summary_monitor = pd.DataFrame({
            "loading_step": np.array([], dtype=np.int64),
            "matrix_assembling": np.array([], dtype=np.float64),
            "vector_assembling": np.array([], dtype=np.float64),
            "linear_solver": np.array([], dtype=np.float64),
            "constitutive_model_update": np.array([], dtype=np.float64),
        })
        num_increments = performance_monitor["loading_step"].max() + 1
        tmp_monitor = {}
        for i in range(num_increments):
            tmp_monitor["loading_step"] = i
            for col in cols:
                tmp_monitor[col] = performance_monitor[performance_monitor["loading_step"]==i][col].sum()
            summary_monitor.loc[len(summary_monitor.index)] = tmp_monitor
        return summary_monitor


    n_list = [1, 2, 4, 8, 16, 32, 64]
    all_data = np.empty(len(n_list), dtype=object)
    summed_data = np.empty(len(n_list), dtype=object)
    # raw data of the first ten loading steps of the Mohr-Coulomb problem
    # is stored in a dictionary and dumped via the pickle module.
    for j, n in enumerate(n_list):
        data = f"data/performance_data_200x200_n_{n}.pkl"
        with open(data, "rb") as f:
            performance_data = pickle.load(f)
        all_data[j] = performance_data
        summed_data[j] = get_sum(performance_data["performance_monitor"])

    total_data = pd.DataFrame()
    total_data["n"] = n_list
    total_data["total_time"] = [all_data[j]["total_time"] for j in range(len(n_list))]
    for col in cols[:3]:
        times = np.array([summed_data[i][col].sum() for i in range(len(n_list))])
        total_data[col] = times
    total_data.set_index("n", inplace=True)

    # make the line plot for the whole modelling
    colors = plt.get_cmap('Set2').colors
    fig, ax = plt.subplots(figsize=(6, 3))
    labels = ["Jacobian assemblies", "Newton inner linear solves", "External operators and operand \n evaluations", "Total time"]
    total_data.plot(use_index=True, y=cols+["total_time"], ax=ax, marker=".",label=labels, color=colors)
    ax.set_yscale('log')
    ax.set_xscale('log', base=2)
    ax.set_xlabel(r"Number of processes, $n$")
    ax.set_ylabel(r"Max wall time [s]")
    annotation.slope_marker((5, 70), (-1, 2), ax=ax, poly_kwargs={"facecolor": colors[3]})
    annotation.slope_marker((5, 3), (-1, 1), ax=ax, poly_kwargs={"facecolor": colors[0]})
    ax.grid()
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    fig.savefig("output/strong_scaling_line.pdf")

    # make the bar plot for the last loading step
    extracted_rows = []
    for j, n in enumerate(n_list):
        last_row = summed_data[j].iloc[-1]
        extracted_rows.append((n, last_row))
    last_loading_step = pd.DataFrame([row[1] for row in extracted_rows], index=[row[0] for row in extracted_rows])
    labels = ["Jacobian assemblies", "Newton inner linear solves", "External operators and operand \n evaluations"]

    fig, ax = plt.subplots(figsize=(4, 3))
    last_loading_step.plot(use_index=True, y=cols, kind="bar", stacked=True, ax=ax, color=colors,
    label=labels)

    ax.set_xlabel(r"Number of processes, $n$")
    ax.set_ylabel(r"Max wall time [s]")
    ax.tick_params(axis='x', rotation=0)

    fig.tight_layout()
    fig.savefig("output/strong_scaling_bar.pdf")

plot_von_mises_displacements_curve()
plot_mohr_coulomb_yield_surface()
plot_taylor_test()
plot_mohr_coulomb_displacements_curve()
plot_slope()
plot_strong_scaling()
