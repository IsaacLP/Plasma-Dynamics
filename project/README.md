Numerical integration of relativistic charged-particle motion using the **Boris-C algorithm** (Zenitani & Umeda, *Physics of Plasmas* **25**, 112110, 2018) in Earth's tilted dipole magnetic field (11.7° tilt). Each particle is also integrated with **RK4** for energy-conservation comparison.

## Running

**CLI:**
```bash
python rel_boris_c_dipole.py --config particles.yaml
python rel_boris_c_dipole.py --config particles.yaml --plot_limit 6 --suffix _run1
```

**Interactive notebook:** `Boris_C_Simulation.ipynb`

## Particle configuration

Particles are defined in a YAML file (see `particles.yaml`):

```yaml
particles:
  - name: Proton
    color: royalblue
    q:  1.6021766210e-19   # charge [C]
    m:  1.6726219e-27      # mass [kg]
    T_sim: 6               # simulation time [s]
    dt: 1.0e-4             # time step [s]
    store_every: 10        # store every nth step
    r0: [2.5, 0.0, 0.0]    # initial position [Earth radii]
    beta: 0.616            # v/c  (used when ke: -1)
    alpha: 60.0            # equatorial pitch angle [deg]
    ke: -1                 # kinetic energy [eV]  (-1 → use beta instead)
```

Set `ke: -1` to specify speed via `beta`, or set `ke>0` to ignore `beta`and specify speed via `ke`. In any case the initial velocity will be set to

$$v_0 = [0, \beta c\sin(\alpha), \beta c\cos(\alpha)]$$

## Outputs

- 3D trajectory plot
- Equatorial (*xy*) and meridional (*xz*) projections
- Energy conservation plot: $|\Delta\gamma/\gamma_0|$ for Boris-C (solid) and RK4 (dashed) on a log scale
- Terminal table of final gamma errors for both integrators

Figures are saved to `figures/` by default.