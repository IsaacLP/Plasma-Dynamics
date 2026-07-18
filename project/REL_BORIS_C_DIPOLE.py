"""
Relativistic equation of motion integrated with the Boris-C algorithm
Zenitani & Umeda, Physics of Plasmas 25, 112110 (2018)

Particles : proton, electron, alpha particle
Field     : Earth's tilted dipole magnetic field (11.7 deg tilt)
"""

from matplotlib.patches import Circle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse


# ── Physical constants ──────────────────────────────────────────────────────
c       = 299792458.0       # speed of light                      [m/s]
RE      = 6378137.0         # Earth radius                        [m]
m_e     = 9.10938356e-31    # electron mass                       [kg]
m_p     = 1.6726219e-27     # proton mass                         [kg]
m_alpha = 6.6446573e-27     # alpha particle (4He nucleus) mass   [kg]
q_e     = 1.6021766210e-19  # elementary charge / eV->J factor    [C / J eV⁻¹]

sinphi = np.sin(11.7 * np.pi / 180.0)   # dipole tilt (11.7 deg)
cosphi = np.cos(11.7 * np.pi / 180.0)

# Timestep as a fraction of the local relativistic cyclotron period
F_DT = 0.03


# ── Earth's tilted dipole magnetic field ────────────────────────────────────
def dipole_B(r):
    """Return the dipole B-field vector [T] at position r [m]."""
    x, y, z = r
    r5 = (x**2 + y**2 + z**2)**2.5
    Bx = -7.965626e15 * (3*x*z*cosphi + 3*x*y*sinphi) / r5
    By = -7.965626e15 * (3*y*z*cosphi + 2*sinphi*y**2
                         - sinphi*(x**2 + z**2)) / r5
    Bz = -7.965626e15 * (2*cosphi*z**2 - cosphi*(x**2 + y**2)
                         + 3*z*y*sinphi) / r5
    return np.array([Bx, By, Bz])


# ── Boris-C step (Zenitani & Umeda 2018) ───────────────────────────────────
def boris_C_step(r, u, q, m, dt):
    """
    One Boris-C integration step in a pure magnetic dipole field (E = 0).

    u = gamma * v  (relativistic four-velocity, spatial part)  [m/s]

    Key Boris-C modification with respect to the standard Boris solver:
    the Lorentz factor used for the B-rotation is

        gamma_C = sqrt(gamma^{-2} + |tau|^2)

    where  tau = (q * dt / 2m) * B,  instead of  gamma^-  alone.
    This preserves |u| more accurately in the relativistic regime.

    Returns: r_new [m], u_new [m/s], gamma_new [-]
    """
    B = dipole_B(r)

    # Half-rotation vector (unnormalised):  tau = (q dt / 2m) B
    tau = (q * dt / (2.0 * m)) * B

    # Lorentz factor from u^n  (= u^- because E = 0)
    gamma_minus = np.sqrt(1.0 + np.dot(u, u) / c**2)

    # Boris-C corrected Lorentz factor
    gamma_C = np.sqrt(gamma_minus**2 + np.dot(tau, tau))

    # Normalised rotation vectors
    t = tau / gamma_C
    s = 2.0 * t / (1.0 + np.dot(t, t))

    # Boris rotation  u^+ = R(2*theta) u^-
    u_star = u + np.cross(u, t)
    u_new  = u + np.cross(u_star, s)

    # Position update  r^{n+1} = r^n + v^{n+1} dt  (v = u / gamma)
    gamma_new = np.sqrt(1.0 + np.dot(u_new, u_new) / c**2)
    r_new = r + (dt / gamma_new) * u_new

    return r_new, u_new, gamma_new


# ── RK4 step (comparison integrator, not symplectic) ────────────────────────
def rk4_step(r, v, q, m, dt):
    """One RK4 step for dv/dt = (q/m)·sqrt(1-v^2/c^2)·(v x B), dr/dt = v  (E=0)."""
    def accel(r_, v_):
        inv_gamma = np.sqrt(max(0.0, 1.0 - np.dot(v_, v_) / c**2))
        return (q / m) * inv_gamma * np.cross(v_, dipole_B(r_))

    k1r = dt * v
    k1v = dt * accel(r, v)
    k2r = dt * (v + 0.5*k1v)
    k2v = dt * accel(r + 0.5*k1r, v + 0.5*k1v)
    k3r = dt * (v + 0.5*k2v)
    k3v = dt * accel(r + 0.5*k2r, v + 0.5*k2v)
    k4r = dt * (v + k3v)
    k4v = dt * accel(r + k3r, v + k3v)

    return (r + (k1r + 2*k2r + 2*k3r + k4r) / 6.0,
            v + (k1v + 2*k2v + 2*k3v + k4v) / 6.0)


# ── Integrators ──────────────────────────────────────────────────────────────
def simulate(name, q, m, r0, v0, dt, T_sim, store_every=1):
    """Boris-C integrator. Returns trajectory [m] and gamma history."""
    n_steps = int(T_sim / dt)
    n_store = n_steps // store_every + 1

    beta2  = np.dot(v0, v0) / c**2
    gamma0 = 1.0 / np.sqrt(1.0 - beta2)
    u = gamma0 * v0

    r = r0.copy()
    traj   = np.empty((n_store, 3))
    gammas = np.empty(n_store)
    traj[0]   = r
    gammas[0] = gamma0

    j = 1
    for i in tqdm(range(1, n_steps + 1), desc=f"Boris-C  {name:<14}"):
        r, u, gamma = boris_C_step(r, u, q, m, dt)
        if i % store_every == 0 and j < n_store:
            traj[j]   = r
            gammas[j] = gamma
            j += 1

    return traj[:j], gammas[:j]


def simulate_rk4(name, q, m, r0, v0, dt, T_sim):
    """RK4 integrator. Returns only (gamma0, gamma_final) for error comparison."""
    n_steps = int(T_sim / dt)

    beta2  = np.dot(v0, v0) / c**2
    gamma0 = 1.0 / np.sqrt(1.0 - beta2)

    r = r0.copy()
    v = v0.copy()
    for _ in tqdm(range(n_steps), desc=f"RK4      {name:<14}"):
        r, v = rk4_step(r, v, q, m, dt)

    v2          = np.dot(v, v)
    gamma_final = 1.0 / np.sqrt(max(1e-30, 1.0 - v2 / c**2))
    return gamma0, gamma_final


# –– Simulation and plotting functions ––––––––––––––––––––––––––––––––––––––––
def run_simulation(particles_list):
    """Run Boris-C and RK4 for every particle; print a unified energy-error table."""
    results = {}
    summary = []   # (name, gamma0, gamma_final_bc, gamma_final_rk4)

    for p in particles_list:
        traj, gammas        = simulate(
            p['name'], p['q'], p['m'], p['r0'], p['v0'],
            p['dt'], p['T_sim'], p['store_every'])
        gamma0, gamma_final = simulate_rk4(
            p['name'], p['q'], p['m'], p['r0'], p['v0'],
            p['dt'], p['T_sim'])

        summary.append((p['name'], gammas[0], gammas[-1], gamma_final))
        results[p['name']] = dict(traj=traj, gammas=gammas, color=p['color'])

    # ── Unified energy-error table ────────────────────────────────────────────
    hdr = f"  {'Particle':<16} {'Method':<9} {'gamma_0':>8} {'gamma_final':>10} {'gamma_error':>10}"
    print(f"\n{hdr}")
    print("  " + "─" * (len(hdr) - 2))
    for name, g0, gf_bc, gf_rk4 in summary:
        dg_bc  = abs(gf_bc  - g0) / g0
        dg_rk4 = abs(gf_rk4 - g0) / g0
        print(f"  {name:<16} {'Boris-C':<9} {g0:>8.4f} {gf_bc:>10.4f} {dg_bc:>10.2e}")
        print(f"  {'':16} {'RK4':<9} {g0:>8.4f} {gf_rk4:>10.4f} {dg_rk4:>10.2e}")
    print()

    return results


def plot_results(results,save=True,show=True,lim=5,out_dir='figures',suffix=''):
    """Plot the 3D trajectories, 2D projections, and energy conservation."""

    # ── 3-D trajectory plot ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 10))
    ax  = fig.add_subplot(111, projection='3d')

    u_ang = np.linspace(0, 2*np.pi, 60)
    v_ang = np.linspace(0, np.pi,   30)
    xs = np.outer(np.cos(u_ang), np.sin(v_ang))
    ys = np.outer(np.sin(u_ang), np.sin(v_ang))
    zs = np.outer(np.ones_like(u_ang), np.cos(v_ang))
    ax.plot_surface(xs, ys, zs, color='steelblue', alpha=0.6, linewidth=0)

    for name, d in results.items():
        tr = d['traj']
        ax.plot(tr[:, 0]/RE, tr[:, 1]/RE, tr[:, 2]/RE,
                color=d['color'], linewidth=0.7, label=name)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_xlabel(r'$x\;[R_E]$')
    ax.set_ylabel(r'$y\;[R_E]$')
    ax.set_zlabel(r'$z\;[R_E]$')
    ax.legend(fontsize=11)
    ax.set_box_aspect((1, 1, 1))
    plt.tight_layout()
    if save:
        plt.savefig(f'{out_dir}/boris_C_trajectories_3D{suffix}.png', dpi=150, bbox_inches='tight')
        print(f"\nSaved: {out_dir}/boris_C_trajectories_3D{suffix}.png")
    if show:
        plt.show()

    # ── xy and xz projections ───────────────────────────────────────────────
    fig2, (ax_xy, ax_xz) = plt.subplots(1, 2, figsize=(14, 6))

    for name, d in results.items():
        tr = d['traj']
        x  = tr[:, 0] / RE
        y  = tr[:, 1] / RE
        z  = tr[:, 2] / RE
        ax_xy.plot(x, y, color=d['color'], linewidth=0.7, label=name)
        ax_xz.plot(x, z, color=d['color'], linewidth=0.7, label=name)
        ax_xy.add_patch(Circle((0, 0), 1.0, color='steelblue', alpha=0.8, zorder=3))
        ax_xz.add_patch(Circle((0, 0), 1.0, color='steelblue', alpha=0.8, zorder=3))

    for ax, xlabel, ylabel in [(ax_xy, r'$x\;[R_E]$', r'$y\;[R_E]$'),
                                (ax_xz, r'$x\;[R_E]$', r'$z\;[R_E]$')]:
        ax.set_aspect('equal')
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=10)
        ax.grid(True, linewidth=0.4)

    plt.tight_layout()
    if save:
        plt.savefig(f'{out_dir}/boris_C_trajectories_2D{suffix}.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {out_dir}/boris_C_trajectories_2D{suffix}.png")
    if show:
        plt.show()

    # ── Energy conservation (relative Lorentz factor error) ─────────────────
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    for name, d in results.items():
        g0  = d['gammas'][0]
        err = (d['gammas'] - g0) / g0
        ax3.plot(err, color=d['color'], linewidth=0.8, label=name)

    ax3.axhline(0.0, color='k', linewidth=0.5, linestyle='--')
    ax3.set_xlabel('Stored step index')
    ax3.set_ylabel(r'$\Delta\gamma\;/\;\gamma_0$')
    ax3.legend(fontsize=11)
    plt.tight_layout()

    if save:
        plt.savefig(f'{out_dir}/boris_C_energy{suffix}.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {out_dir}/boris_C_energy{suffix}.png")

    if show:
        plt.show()


# ── Particle configurations and initial conditions ──────────────────────────
def init_cond(alpha_eq_deg=60.0):
    """
    Build the particle list with initial conditions for the simulation.
    The equatorial pitch angle is the same for all particles.
    """
    alpha_eq = np.radians(alpha_eq_deg)

    particles = [
        dict(name='Proton',
             q= q_e, m= m_p,
             T_sim=6.0,dt=0.0001,store_every=10,
             r0=np.array([2.5*RE, 0.0, 0.0]),
             v0=np.array([0.0, 0.616 * c * np.sin(alpha_eq), 0.616 * c * np.cos(alpha_eq)]),
             color='royalblue'),
        dict(name='Electron',
             q=-q_e, m=m_e,
             T_sim=0.6,dt=0.00001,store_every=10,
             r0=np.array([4.5*RE, 0.0, 0.0]),
             v0=np.array([0.0, 0.94 * c * np.sin(alpha_eq), 0.94 * c * np.cos(alpha_eq)]),
             color='crimson'),
        dict(name='Alpha particle',
             q=2*q_e, m=m_alpha,
             T_sim=6.0,dt=0.0001,store_every=10,
             r0=np.array([2.5*RE, 0.0, 0.0]),
             v0=np.array([0.0, 0.5 * c * np.sin(alpha_eq), 0.5 * c * np.cos(alpha_eq)]),
             color='seagreen'),
    ]

    return particles


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Relativistic Boris-C particle simulation in Earth's dipole field.")
    parser.add_argument(
        '--pitch_angle', type=float, default=60.0,
        help='Equatorial pitch angle in degrees for all particles (default: 60)')
    parser.add_argument(
        '--plot_limit', type=int, default=5,
        help='Limit for 3D and 2D plots in Earth radii (default: 5)')
    parser.add_argument(
        '--suffix', type=str, default='',
        help='Optional suffix for output figure filenames (default: empty)')
    args = parser.parse_args()

    particles   = init_cond(args.pitch_angle)
    results     = run_simulation(particles)
    plot_results(results, lim=args.plot_limit, suffix=args.suffix)
