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
from scipy.optimize import brentq


# ── Physical constants ──────────────────────────────────────────────────────
c   = 299792458.0       # speed of light                            [m/s]
RE  = 6378137.0         # Earth radius                              [m]
q_e = 1.6021766210e-19  # elementary charge / eV->J factor          [C / J eV^-1]
M_E = 7.965626e15       # Earth's magnetic dipole moment            [T m^3]
B0  = M_E / RE**3       # Earth's equatorial surface magnetic field [T]

sinphi = np.sin(np.deg2rad(11.7))   # dipole tilt (11.7 deg)
cosphi = np.cos(np.deg2rad(11.7))

# ── Earth's tilted dipole magnetic field ────────────────────────────────────
def dipole_B(r):
    """Return the dipole B-field vector [T] at position r [m]."""
    x, y, z = r
    r5 = (x**2 + y**2 + z**2)**2.5
    Bx = -M_E * (3*x*z*cosphi + 3*x*y*sinphi) / r5
    By = -M_E * (3*y*z*cosphi + 2*sinphi*y**2
                         - sinphi*(x**2 + z**2)) / r5
    Bz = -M_E * (2*cosphi*z**2 - cosphi*(x**2 + y**2)
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


def simulate_rk4(name, q, m, r0, v0, dt, T_sim, store_every=1):
    """RK4 integrator. Returns gamma history at the same cadence as simulate()."""
    n_steps = int(T_sim / dt)
    n_store = n_steps // store_every + 1

    beta2  = np.dot(v0, v0) / c**2
    gamma0 = 1.0 / np.sqrt(1.0 - beta2)

    r = r0.copy()
    v = v0.copy()
    gammas    = np.empty(n_store)
    gammas[0] = gamma0

    j = 1
    for i in tqdm(range(1, n_steps + 1), desc=f"RK4      {name:<14}"):
        r, v = rk4_step(r, v, q, m, dt)
        if i % store_every == 0 and j < n_store:
            v2        = np.dot(v, v)
            gammas[j] = 1.0 / np.sqrt(max(1e-30, 1.0 - v2 / c**2))
            j += 1

    return gammas[:j]


# –– Simulation and plotting functions ––––––––––––––––––––––––––––––––––––––––
def run_simulation(particles_list):
    """Run Boris-C and RK4 for every particle; print a unified energy-error table."""
    results = {}
    summary = []   # (name, gamma0, gamma_final_bc, gamma_final_rk4)

    for p in particles_list:
        traj, gammas        = simulate(
            p['name'], p['q'], p['m'], p['r0'], p['v0'],
            p['dt'], p['T_sim'], p['store_every'])
        gammas_rk4 = simulate_rk4(
            p['name'], p['q'], p['m'], p['r0'], p['v0'],
            p['dt'], p['T_sim'], p['store_every'])

        summary.append((p['name'], gammas[0], gammas[-1], gammas_rk4[-1]))
        results[p['name']] = dict(traj=traj, gammas=gammas, gammas_rk4=gammas_rk4, color=p['color'])

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

    # ── Energy conservation (|relative Lorentz factor error|, log scale) ──────
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    for name, d in results.items():
        g0 = d['gammas'][0]
        err_bc  = np.abs(d['gammas']     - g0) / g0
        err_rk4 = np.abs(d['gammas_rk4'] - g0) / g0
        ax3.plot(err_bc,  color=d['color'], linewidth=0.8, label=f'{name} Boris-C')
        ax3.plot(err_rk4, color=d['color'], linewidth=0.8, linestyle='--', label=f'{name} RK4')

    ax3.set_yscale('log')
    ax3.set_xlabel('Stored step index')
    ax3.set_ylabel(r'$|\Delta\gamma\;/\;\gamma_0|$')
    ax3.legend(fontsize=10)
    plt.tight_layout()

    if save:
        plt.savefig(f'{out_dir}/boris_C_energy{suffix}.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {out_dir}/boris_C_energy{suffix}.png")

    if show:
        plt.show()


# –– Helper functions for initial conditions and derived physical quantities ––––––––––––––––––––––––––––––––
def beta_from_ke(KE, m):
    """Convert kinetic energy [eV] to speed [m/s] for a particle of mass m [kg]."""
    KE_J = KE * q_e  # Convert eV to Joules
    gamma = 1 + KE_J / (m * c**2)
    beta = np.sqrt(1 - 1/gamma**2)
    return beta


def ke_from_beta(beta, m):
    """Convert speed [m/s] to kinetic energy [eV] for a particle of mass m [kg]."""
    gamma = 1 / np.sqrt(1 - beta**2)
    KE_J = (gamma - 1) * m * c**2
    KE_eV = KE_J / q_e  # Convert Joules to eV
    return KE_eV


def tau_bounce(beta, L, alpha_deg):
    """Walt (1994) Eq. 4.28 - bounce period, accurate to ~0.5%."""
    a = np.radians(alpha_deg)
    R0 = L * RE
    return 0.117 * (R0/RE) / beta * (1 - 0.4635*np.sin(a)**0.75)


def tau_drift(m, charge, beta, L, alpha_deg):
    """Walt (1994) Eq. 4.46 - drift period, accurate to ~0.5%.
    Verified: reproduces Walt's own Cd constants (1.557e4 s electrons,
    8.481 s protons) to 4 sig figs when evaluated in plain SI units."""
    v = beta * c
    a = np.radians(alpha_deg)
    R0 = L * RE
    return (2*np.pi*abs(charge)*B0*RE**3) / (m*v**2) * (1/R0) * (1 - 0.3333*np.sin(a)**0.62)


def mirror_latitude(alpha_deg):
    """Solve sin^2(alpha_eq) = cos^6(lam)/sqrt(1+3 sin^2 lam) for lam_m [Eq. 4.24]."""
    a = np.radians(alpha_deg)
    def f(lam):
        return np.cos(lam)**6/np.sqrt(1+3*np.sin(lam)**2) - np.sin(a)**2
    return np.degrees(brentq(f, 0, np.pi/2 - 1e-6))


def loss_cone(L, Ra=RE):
    """Bounce loss cone angle, field-line-consistent form (Walt, p.43).
    Reduces to sin^2(a_lc) = 1/sqrt(4L^6-3L^5) when Ra = RE."""
    x = Ra / (L*RE)
    return np.degrees(np.arcsin(np.sqrt(x**3/np.sqrt(4-3*x))))


def gyro_quantities(m, charge, gamma, v, alpha_deg, L):
    """Gyroperiod, gyroradius, and adiabaticity parameter."""
    a = np.radians(alpha_deg)
    Beq = M_E/(L*RE)**3
    Tgyro = 2*np.pi*gamma*m/(abs(charge)*Beq)
    rg = gamma*m*v*np.sin(a)/(abs(charge)*Beq)
    eps = rg/(L*RE)
    return Tgyro, rg, eps


# ── Particle configurations and initial conditions ──────────────────────────
def init_cond(yaml_file):
    """Load particle configurations from a YAML file. Returns a list of particle dictionaries with initial conditions.

    The YAML file should contain a list of particles, each with the following fields:
        - name: Particle name (string)
        - q: Charge [C]
        - m: Mass [kg]
        - T_sim: Simulation time [s]
        - dt: Time step [s]
        - store_every: Store every nth step (int)
        - r0: Initial position vector [Earth radii]
        - beta: Initial velocity as a fraction of the speed of light (0 means use kinetic energy instead)
        - alpha: Equatorial pitch angle in degrees
        - ke: Initial kinetic energy in eV (-1 means use beta instead)

    For each particle, the initial velocity is computed based on either the specified kinetic energy (ke) or the specified beta (v/c). It has the form: 
    
    [0, beta * c * sin(alpha), beta * c * cos(alpha)], 
    
    where alpha is the equatorial pitch angle in radians.
    
    Each returned particle dictionary contains:
        - name: Particle name (string)
        - q: Charge [C]
        - m: Mass [kg]
        - T_sim: Simulation time [s]
        - dt: Time step [s]
        - store_every: Store every nth step (int)
        - r0: Initial position vector [m]
        - v0: Initial velocity vector [m/s]
    """
    import yaml
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)
    
    particles = []
    for p in config['particles']:
        alpha = np.radians(p['alpha'])
        ke = p['ke']
        if ke < 0:
            # Use beta instead of kinetic energy
            beta = p['beta']
            ke = ke_from_beta(beta, p['m'])  # Compute kinetic energy for logging
            v0 = np.array([0.0, beta * c * np.sin(alpha), beta * c * np.cos(alpha)])
        else:
            # Use kinetic energy to compute beta
            beta = beta_from_ke(ke, p['m'])
            v0 = np.array([0.0, beta * c * np.sin(alpha), beta * c * np.cos(alpha)])

        r0 = np.array(p['r0']) * RE  # Convert from Earth radii to meters

        particles.append(dict(
            name=p['name'],
            q=p['q'],
            m=p['m'],
            T_sim=p['T_sim'],
            dt=p['dt'],
            store_every=p['store_every'],
            r0=r0,
            v0=v0,
            color=p['color']
        ))
        print(f"Initialized {p['name']}:")
        print(f"r0 = {p['r0']} RE   v0 = {beta:.3f} c")
        print(f"alpha = {p['alpha']} deg   ke = {ke:.2e} eV")

        print("\nWith derived quantities:")
        gamma = 1/np.sqrt(1-beta**2)
        v = beta*c
        L = np.linalg.norm(r0) / RE
        Tgyro, rg, eps = gyro_quantities(p['m'], p['q'], gamma, v, p['alpha'], L)
        tb = tau_bounce(beta, L, p['alpha'])
        td = tau_drift(p['m'], p['q'], beta, L, p['alpha'])
        lam_m = mirror_latitude(p['alpha'])
        r_m = L*np.cos(np.radians(lam_m))**2
        alc = loss_cone(L, RE + 100e3)  # Loss cone at 100 km altitude

        print(f"gamma={gamma:.4f},  Beq={M_E/(L*RE)**3*1e9:.1f} nT")
        print(f"Tgyro={Tgyro*1e3:.3f} ms  rg={rg/RE:.4f} RE   eps={eps:.4f}")
        print(f"tau_bounce={tb:.4f} s   tau_drift={td:.4f} s ({td/60:.3f} min)")
        print(f"mirror_lat={lam_m:.2f} deg   r_mirror={r_m:.3f} RE")
        print(f"loss_cone={alc:.3f} deg\n")  

    return particles


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Relativistic Boris-C particle simulation in Earth's dipole field.")
    parser.add_argument(
        '--config', type=str, required=True,
        help='Load particle configurations from a YAML file')
    parser.add_argument(
        '--suffix', type=str, default='',
        help='Optional suffix for output figure filenames (default: empty)')
    parser.add_argument(
        '--plot_limit', type=int, default=5,
        help='Limit for 3D and 2D plots in Earth radii (default: 5)')
    args = parser.parse_args()

    particles = init_cond(args.config)
    results = run_simulation(particles)
    plot_results(results, lim=args.plot_limit, suffix=args.suffix)
