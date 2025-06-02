#!/usr/bin/env python3
"""
Compute longitudinal (ρ_xx) and Hall (ρ_xy) resistivities vs magnetic flux
for a six-terminal Hall bar at fixed 2D carrier density using Landau quantization.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
import kwant
from kwant.digest import uniform
from cmath import exp
try:
    from scipy.constants import h, e
except ImportError:
    h = 6.62607015e-34
    e = 1.602176634e-19

def make_hall_bar(W=30, L=50, t=1, U0=0, salt=0):
    lat = kwant.lattice.square(a=1, norbs=1)
    syst = kwant.Builder()
    def onsite(site, U0, t, salt):
        return U0 * (uniform(repr(site), repr(salt)) - 0.5) + 4 * t
    def hop_horiz(si, sj, phi, t):
        xi, yi = si.pos
        xj, yj = sj.pos
        return -t * exp(-0.5j * phi * (xi - xj) * (yi + yj))
    syst[(lat(x, y) for x in range(L) for y in range(W))] = onsite
    syst[lat.neighbors()] = hop_horiz
    # Left/right leads
    lead_left = kwant.Builder(kwant.TranslationalSymmetry([-1, 0]))
    lead_left[(lat(0, y) for y in range(W))] = 4 * t
    lead_left[lat.neighbors()] = hop_horiz
    syst.attach_lead(lead_left)
    syst.attach_lead(lead_left.reversed())
    # Vertical gauge for top/bottom probes
    def hop_vert(si, sj, phi, t):
        xi, yi = si.pos
        xj, yj = sj.pos
        if xi == xj:
            return -t * exp(0.5j * phi * (yj - yi) * (xi + xj))
        return -t
    x_mid = L // 2
    # Top-left
    lead_tl = kwant.Builder(kwant.TranslationalSymmetry([0, 1]))
    lead_tl[(lat(x, W - 1) for x in range(0, x_mid))] = 4 * t
    lead_tl[lat.neighbors()] = hop_vert
    syst.attach_lead(lead_tl)
    # Top-right
    lead_tr = kwant.Builder(kwant.TranslationalSymmetry([0, 1]))
    lead_tr[(lat(x, W - 1) for x in range(x_mid, L))] = 4 * t
    lead_tr[lat.neighbors()] = hop_vert
    syst.attach_lead(lead_tr)
    # Bottom-left
    lead_bl = kwant.Builder(kwant.TranslationalSymmetry([0, -1]))
    lead_bl[(lat(x, 0) for x in range(0, x_mid))] = 4 * t
    lead_bl[lat.neighbors()] = hop_vert
    syst.attach_lead(lead_bl)
    # Bottom-right
    lead_br = kwant.Builder(kwant.TranslationalSymmetry([0, -1]))
    lead_br[(lat(x, 0) for x in range(x_mid, L))] = 4 * t
    lead_br[lat.neighbors()] = hop_vert
    syst.attach_lead(lead_br)
    return syst.finalized()

def main():
    parser = argparse.ArgumentParser(
        description='Compute rho_xx and rho_xy vs B for fixed 2D density'
    )
    parser.add_argument('--W', type=int, default=50, help='Width of the bar')
    parser.add_argument('--L', type=int, default=90, help='Length of the bar')
    parser.add_argument('--t', type=float, default=1.0, help='Hopping amplitude')
    parser.add_argument('--U0', type=float, default=0.1, help='Disorder strength')
    parser.add_argument('--salt', type=int, default=13, help='Disorder seed')
    parser.add_argument('--phi-min', type=float, default=0.01, help='Min flux phi')
    parser.add_argument('--phi-max', type=float, default=0.1, help='Max flux phi')
    parser.add_argument('--nphis', type=int, default=51, help='Number of phi points')
    parser.add_argument('--n2d', type=float, required=True,
                        help='2D carrier density (m^{-2}) for fixed density')
    parser.add_argument('--mu', type=float, default=None,
                        help='Carrier mobility for classical overlay')
    parser.add_argument('--a', type=float, default=1e-9, help='Lattice constant a (m)')
    parser.add_argument('--outfile', type=str, default=None,
                        help='Output plot filename')
    args = parser.parse_args()

    print(f"Stage 1/3: Building system geometry W={args.W}, L={args.L}, U0={args.U0}, salt={args.salt}...")
    system = make_hall_bar(W=args.W, L=args.L, t=args.t,
                           U0=args.U0, salt=args.salt)
    print("Stage 1/3 complete.")
    phis = np.linspace(args.phi_min, args.phi_max, args.nphis)
    # Convert phi to physical B (Tesla)
    B = phis * h / (2 * np.pi * e * args.a**2)
    # Classical overlays
    rho_xx_cl = None
    rho_xy_cl = None
    if args.mu is not None:
        rho_xx_cl = (1.0/(args.n2d * e * args.mu)) * (e**2 / h)
        rho_xy_cl = (B/(args.n2d * e)) * (e**2 / h)
    # Measurement setup
    idx = [0, 2, 3, 4, 5]
    N = max(idx) + 1
    I = np.zeros(N); I[0] = 1; I[1] = -1
    dx = max((3*args.L//4 - args.L//4), 1)
    W = args.W
    rho_xx = np.zeros_like(phis)
    rho_xy = np.zeros_like(phis)

    print(f"Stage 2/3: Computing resistivities at fixed n2d={args.n2d:g} over {args.nphis} flux points...")
    for i, phi in enumerate(phis):
        if phi <= 0:
            rho_xx[i] = np.nan; rho_xy[i] = np.nan
            continue
        Bi = B[i]
        # Filling factor (integer part)
        nu = int(np.floor(args.n2d * h / (e * Bi)))
        # Landau level energy (continuum approx): E = 2*t*phi*(nu+1/2)
        Ephi = 2 * args.t * phi * (nu + 0.5)
        smat = kwant.smatrix(system, Ephi,
                             params=dict(phi=phi, U0=args.U0,
                                         salt=args.salt, t=args.t))
        G = smat.conductance_matrix()
        Gsub = G[np.ix_(idx, idx)]
        try:
            Vsub = np.linalg.solve(Gsub, I[idx])
        except np.linalg.LinAlgError:
            Vsub = np.linalg.lstsq(Gsub, I[idx], rcond=None)[0]
        V = np.zeros(N); V[1] = 0; V[idx] = Vsub
        Rxx = (V[2] - V[3]) * (W / dx)
        Vtop = (V[2] + V[3]) / 2
        Vbot = (V[4] + V[5]) / 2
        Rxy = Vbot - Vtop
        rho_xx[i] = Rxx
        rho_xy[i] = Rxy
        sys.stderr.write(f"\r flux {i+1}/{len(phis)}, nu={nu}, E={Ephi:.3g}")
        sys.stderr.flush()
    sys.stderr.write("\n")
    print("Stage 2/3 complete.")

    print("Stage 3/3: Plotting results...")
    plt.figure()
    plt.plot(B, rho_xx, label=r'$\rho_{xx}$')
    plt.plot(B, rho_xy, label=r'$\rho_{xy}$')
    if rho_xx_cl is not None:
        plt.plot(B, np.full_like(B, rho_xx_cl),
                 '--', color='gray', label='classical')
    # Plateau lines
    for n in [1, 2, 3, 4, 5]:
        plateau = 1.0 / n
        plt.axhline(plateau, color='black', linestyle=':', linewidth=0.8)
        plt.text(B[-1], plateau, f'1/{n}', color='black',
                 ha='right', va='bottom')
    plt.xlabel('B (T)')
    plt.ylabel(r'Resistivity ($h/e^2$)')
    plt.legend()
    plt.tight_layout()
    if args.outfile:
        filename = args.outfile
    else:
        filename = (f'hall_fixed_n2d{args.n2d:g}_W{args.W}_L{args.L}'
                    f'_phi{args.phi_min:g}-{args.phi_max:g}'
                    f'_nphis{args.nphis}.png')
    plt.savefig(filename)
    print(f"Saved plot to {filename}")
    print("Done.")

if __name__ == '__main__':
    main()