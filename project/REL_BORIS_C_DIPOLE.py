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
c       = 299792458.0       # speed of light                     [m/s]
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


# ── Simulation runner ───────────────────────────────────────────────────────
def simulate(name, q, m, r0, v0, dt, T_sim, store_every=1):
    """
    Integrate the relativistic equation of motion with Boris-C.

    v0          : initial velocity  [m/s]  (NOT u = gamma*v)
    store_every : store one point every this many steps (memory saving)

    Returns trajectory array  [n, 3]  in metres, and gamma history.
    """
    n_steps = int(T_sim / dt)
    n_store = n_steps // store_every + 1

    # Convert velocity to four-velocity u = gamma * v
    beta2  = np.dot(v0, v0) / c**2
    gamma0 = 1.0 / np.sqrt(1.0 - beta2)
    u = gamma0 * v0

    r = r0.copy()
    traj   = np.empty((n_store, 3))
    gammas = np.empty(n_store)
    traj[0]   = r
    gammas[0] = gamma0

    j = 1
    for i in tqdm(range(1, n_steps + 1), desc=f"Simulating {name}"):
        r, u, gamma = boris_C_step(r, u, q, m, dt)
        if i % store_every == 0 and j < n_store:
            traj[j]   = r
            gammas[j] = gamma
            j += 1

    traj   = traj[:j]
    gammas = gammas[:j]

    dg = abs(gammas[-1] - gamma0) / gamma0
    print(f"  {name:<14s}  gamma_0 = {gamma0:.4f}  "
          f"gamma_final = {gammas[-1]:.4f}  "
          f"rel. energy error = {dg:.2e}")
    return traj, gammas


def run_simulation(particles_list):
    """Run the Boris-C simulation for all particles and return results.

    Returns a dictionary with particle names as keys and trajectory/gamma data.
    """
    results = {}
    for p in particles_list:
        traj, gammas = simulate(
            p['name'], p['q'], p['m'], p['r0'], p['v0'],
            p['dt'], p['T_sim'], p['store_every'])
        results[p['name']] = dict(traj=traj, gammas=gammas, color=p['color'])
    return results


def plot_results(results, show_plots=True,lim='auto'):
    """Plot the 3D trajectories, 2D projections, and energy conservation."""

    if lim == 'auto':
        # Auto-scale to the furthest point in any trajectory
        max_r = max(np.linalg.norm(d['traj'], axis=1).max() for d in results.values())
        lim   = max(2, int(np.ceil(max_r / RE)) + 1)
    else:
        lim = float(lim)

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
    plt.savefig('boris_C_trajectories_3D.png', dpi=150, bbox_inches='tight')
    print("\nSaved: boris_C_trajectories_3D.png")

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
    plt.savefig('boris_C_trajectories_2D.png', dpi=150, bbox_inches='tight')
    print("Saved: boris_C_trajectories_2D.png")

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
    plt.savefig('boris_C_energy.png', dpi=150, bbox_inches='tight')
    print("Saved: boris_C_energy.png")

    if show_plots:
        plt.show()


# ── Helpers ──────────────────────────────────────────────────────────────────
def _ke_to_kinematics(KE_J, m):
    """Kinetic energy [J] → (gamma, beta, v [m/s])."""
    gamma = 1.0 + KE_J / (m * c**2)
    beta  = np.sqrt(1.0 - 1.0 / gamma**2)
    return gamma, beta, beta * c


# ── Particle configurations and initial conditions ──────────────────────────
def init_cond(alpha_eq_deg=60.0, beta_override=None):
    """
    Build the particle list with all derived simulation parameters.

    alpha_eq_deg  : equatorial pitch angle [degrees], applied to all particles.
    beta_override : if given, replaces per-particle kinetic energy with a fixed
                    beta = v/c for all particles (CLI --beta flag).

    Velocity convention (B ~ ẑ at the equatorial crossing on the x-axis):
        vy = v * sin(alpha_eq)   perpendicular to B
        vz = v * cos(alpha_eq)   field-aligned (parallel to B)

    Derived parameters per particle:
        dt          = F_DT * T_c,   T_c = 2*pi*gamma*m / (|q|*|B(r0)|)
        T_sim       = N_bounce * tau_b,
                      tau_b ~ (L*RE / v) * (3.7 - 1.6*sin alpha_eq)
        store_every = max(1, int(T_c / (20*dt)))   → ~20 stored pts per T_c
    """
    alpha_eq = np.radians(alpha_eq_deg)

    # Per-particle physical setup.
    # KE_eV: kinetic energy in electron-volts (1 eV = q_e J, species-independent).
    raw = [
        dict(name='Proton',
             q= q_e, m= m_p, KE_eV=250e6, # 250 MeV
             r0=np.array([2.5*RE, 0.0, 0.0]),
             N_bounce=100, color='royalblue'),
        dict(name='Electron',
             q= -q_e, m=    m_e, KE_eV=1e6, # 1 MeV 
             r0=np.array([4.5*RE, 0.0, 0.0]),
             N_bounce= 20, color='crimson'),
        dict(name='Alpha particle',
             q=2*q_e, m=m_alpha, KE_eV=1e9, # 1 GeV
             r0=np.array([2.5*RE, 0.0, 0.0]),
             N_bounce=100, color='seagreen'),
    ]

    # ── Summary table header ─────────────────────────────────────────────────
    hdr = (f"  {'Particle':<14}  {'KE':>9}  {'gamma':>6}  {'beta':>6}  "
           f"{'|B|(uT)':>7}  {'Tc(ms)':>7}  {'dt(us)':>7}  "
           f"{'tau_b(s)':>7}  {'T_sim(s)':>8}  {'n_steps':>10}")
    print(f"\n{hdr}")
    print("  " + "─" * (len(hdr) - 2))

    particles = []
    for p in raw:
        m, q   = p['m'], p['q']
        r0_p   = p['r0']

        # Kinematics from kinetic energy (or beta override)
        if beta_override is not None:
            beta  = float(beta_override)
            gamma = 1.0 / np.sqrt(1.0 - beta**2)
            v     = beta * c
            KE_J  = (gamma - 1.0) * m * c**2
        else:
            KE_J           = p['KE_eV'] * q_e
            gamma, beta, v = _ke_to_kinematics(KE_J, m)

        # Initial velocity: v_perp = v sinα (vy), v_par = v cosα (vz ~ B direction)
        v0 = np.array([0.0, v * np.sin(alpha_eq), v * np.cos(alpha_eq)])

        # Field and timescales at r0
        B_mag = np.linalg.norm(dipole_B(r0_p))
        T_c   = 2.0 * np.pi * gamma * m / (abs(q) * B_mag)
        dt    = F_DT * T_c
        L     = np.linalg.norm(r0_p) / RE
        tau_b = (L * RE / v) * (3.7 - 1.6 * np.sin(alpha_eq))
        T_sim = p['N_bounce'] * tau_b
        n_steps     = int(T_sim / dt)
        store_every = max(1, int(T_c / (20.0 * dt)))


        # Format KE for display
        KE_MeV = KE_J / q_e / 1e6
        ke_str = f"{KE_MeV*1e3:.4g} keV" if KE_MeV < 1.0 else f"{KE_MeV:.4g} MeV"

        print(f"  {p['name']:<14}  {ke_str:>9}  {gamma:>6.4f}  {beta:>6.4f}  "
              f"{B_mag*1e6:>7.3f}  {T_c*1e3:>7.3f}  {dt*1e6:>7.2f}  "
              f"{tau_b:>7.3f}  {T_sim:>8.3f}  {n_steps:>10,}")

        particles.append(dict(
            name=p['name'], q=q, m=m, color=p['color'],
            r0=r0_p, v0=v0, dt=dt, T_sim=T_sim, store_every=store_every,
        ))

    print()
    return particles


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Relativistic Boris-C particle simulation in Earth's dipole field.")
    parser.add_argument(
        '--pitch_angle', type=float, default=60.0,
        help='Equatorial pitch angle in degrees for all particles (default: 60)')
    parser.add_argument(
        '--beta', type=float, default=None,
        help='Override initial speed beta=v/c for all particles, '
             'ignoring per-particle kinetic energies')
    parser.add_argument(
        '--plot_limit', type=str, default='auto',
        help='Limit for 3D and 2D plots in Earth radii (default: \'auto\', scales to furthest trajectory point)')
    args = parser.parse_args()

    particles   = init_cond(args.pitch_angle, args.beta)
    results     = run_simulation(particles)
    plot_results(results, lim=args.plot_limit)
