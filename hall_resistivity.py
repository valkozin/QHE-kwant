#!/usr/bin/env python3
"""
Compute longitudinal (rho_xx) and Hall (rho_xy) resistivities vs magnetic flux
for a six-terminal quantum-Hall bar using Kwant and the Landauer–Büttiker formalism.
"""
import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import kwant
from kwant.digest import uniform
from cmath import exp
from pydantic import BaseModel

@dataclass(frozen=True, init=False)
class Constants:
    """
    Common constants
    """
    # static properties
    h = 6.62607015e-34
    """Planck's constant"""

    # TODO: Rename to q_e in order to not confuse it with Euler's number
    e = 1.602176634e-19
    """Elementary electric charge"""


# TODO: It should be
# type npListFloat = np.ndarray[tuple[int], np.dtype[np.floating[Any]]]
# but many functions only return it with tuple[int,...]
type npListFloat = np.ndarray#np.ndarray[tuple[int,...], np.dtype[np.floating[Any]]]


class LeadParameters(BaseModel):
    """
    The parameters of the leads
    """

    @classmethod
    def defaultRelativeTo(cls, *, hallBarW: int, hallBarL: int):
        """
        Returns the default arguments of `LeadParameters` for a Hall bar of dimensions [W, L], where
        - The left and right lead have width W
        - The top and bottom leads have with and distance to corners and themselves L//5.

        Parameters
            hallBarDimensions: The dimensions of the Hall bar [W, L]
        """
        topBottomLeadDimensions = hallBarL//5
        return LeadParameters(
            BinLeads=False,
            leadLeftRightDistanceToCorner=0,
            leadTopBottomWidth=topBottomLeadDimensions,
            leadTopBottomDistanceToCorner=topBottomLeadDimensions
        )

    BinLeads: bool
    """Has B field in the leads (i.e., Peierls phase)."""

    leadLeftRightDistanceToCorner: int
    """Distance of left and right leads to the corners of the rectangular system in lattice constants."""

    leadTopBottomWidth: int
    """The width of the lead in lattice constants."""

    leadTopBottomDistanceToCorner: int
    """Distance of top and bottom leads to the corners of the rectangular system in lattice constants."""

    def fileNameComponents(self) -> list[tuple[str, str]]:
        return [
            ('BiL', f"{self.BinLeads}"),
            ('LLRdC', f"{self.leadLeftRightDistanceToCorner}"),
            ('LTBW', f"{self.leadTopBottomWidth}"),
            ('LTBdC', f"{self.leadTopBottomDistanceToCorner}"),
        ]


class StaticHallBarParameters(BaseModel):
    """
    The parameters of the Hall bar that cannot change once the system is built.
    """

    @classmethod
    def default(cls):
        """Generate the default Hall bar parameters"""
        W = 50
        L = 90
        leadParameters = LeadParameters.defaultRelativeTo(hallBarW=W, hallBarL=L)
        return StaticHallBarParameters(
            W=W,
            L=L,
            a=1e-9,
            leadParameters=leadParameters,
        )

    W: int
    """Width of the bar."""

    L: int
    """Length of the bar."""

    a: float
    """Lattice constant a (in meters) for B-field conversion."""

    leadParameters: LeadParameters | None
    """Parameters of the leads. If it is 'None', there are no leads."""

    def hasLead(self):
        return self.leadParameters is not None

    def fileNameComponents(self) -> list[tuple[str, str]]:
        return [
            ('W', f"{self.W}"),
            ('L', f"{self.L}"),
            ('a', f"{self.a:g}"),
        ] + (
            self.leadParameters.fileNameComponents()
            if self.leadParameters is not None else [])
    
    class Config:
        frozen = True


class DynamicHallBarParameters(BaseModel):
    """
    The parameters of the Hall bar that can change
    """

    @classmethod
    def default(cls):
        """
        Default value of `DynamicHallBarParameters
        """
        return DynamicHallBarParameters(
            t=1,
            phi=0,
            U0=0.0,
            salt=13,
        )

    t: float
    """Nearest neighbor hopping"""

    phi: float
    """Peierl's phase which is proportional to the magnetic field"""

    U0: float
    """Disorder strength."""

    salt: int
    """Random seed for disorder."""

    def fileNameComponents(self) -> list[tuple[str, str]]:
        # TODO: Add 'DynamicHallBarParameterGrid' where all parameters have type npListFloat.
        # phi is not added here since the 'phis' come from 'ResistivityCalculationParameters'
        return [
            ('t', f"{self.t}"),
            ('U0', f"{self.U0}"),
            ('slt', f"{self.salt}"),
        ]

class HallBarParameters(BaseModel):

    static: StaticHallBarParameters
    dynamic: DynamicHallBarParameters

    def fileNameComponents(self) -> list[tuple[str, str]]:
        return self.static.fileNameComponents() + self.dynamic.fileNameComponents()


class ResistivityCalculationParameters(BaseModel):

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def default(cls):
        """
        Default resistivity calculation parameters
        """
        return ResistivityCalculationParameters(
            energy=0.2,
            phis=np.linspace(0, 0.1, 51),
            nseeds=1,
            n2d=None,
            mu=None,
            temp=0.0,
            nE=21,
        )

    energy: float
    """Fermi energy."""

    phis: npListFloat
    """List of Peierls phases (B-field)"""

    nseeds: int
    """Number of disorder seeds for averaging (default: 1)."""

    n2d: float | None
    """2D carrier density (m^-2) for classical overlay."""

    mu: float | None
    """Carrier mobility (m^2/Vs) for classical overlay."""

    temp: float
    """Finite-temperature smearing (same units as 'energy')."""

    nE: int
    """Number of energy points for thermal averaging."""

    def fileNameComponents(self) -> list[tuple[str, str]]:
        return [
            ('E', f"{self.energy:g}"),
            ('T', f"{self.temp:g}"),
            ('phi', f"{np.min(self.phis):g}-{np.max(self.phis):g}"),
            ('nphis', f"{len(self.phis)}"),
            ('nseeds', f"{self.nseeds}"),
        ]

class ClassicalResistivityCalculationParameters(BaseModel):

    class Config:
        arbitrary_types_allowed = True

    phis: npListFloat
    a: float
    n2d: float
    mu: float


class Resistivities(BaseModel):

    class Config:
        arbitrary_types_allowed = True

    phis: npListFloat
    rho_xx: npListFloat
    rho_xy: npListFloat
    rho_xx_classical: npListFloat | None
    rho_xy_classical: npListFloat | None

    hallBarParameters: HallBarParameters
    calculationParameters: ResistivityCalculationParameters

    def magneticFields(self) -> npListFloat:
        """Calculated the magnetic field from the phis"""
        e = Constants.e
        h = Constants.h
        a = self.hallBarParameters.static.a

        # Convert dimensionless flux phi → physical B-field in Tesla
        # phi = (e/ħ) B a^2 ⇒ B = phi ħ/(e a^2) = phi * h/(2π e a^2)
        return self.phis * h / (2 * np.pi * e * a**2)

    def parameterFileNamePart(self):
        """"Part of the file name which just includes the parameters"""
        fileNameComponents = self.hallBarParameters.fileNameComponents() + self.calculationParameters.fileNameComponents()
        fileName = ""
        for label, formattedValue in fileNameComponents:
            fileName += "_" + label + "_" + formattedValue
        return fileName

    def makeBaseFileName(self):
        """Base name of the file plotting the resistivities as a function of mangetic filed without extension"""
        return "resistivities" + self.parameterFileNamePart()

class HallBar(BaseModel):

    parameters: HallBarParameters
    """Parameters of the Hall bar"""

    system: kwant.system.FiniteSystem
    """The kwant system"""

    class Config:
        arbitrary_types_allowed = True

    def __init__(
            self,
            parameters: HallBarParameters):


        W = parameters.static.W
        L = parameters.static.L
        # FIXME!!!!! t is not fully dynamic since the leads use a constant 't'
        t = parameters.dynamic.t
        
        
        # Use a square lattice with one orbital per site
        lat = kwant.lattice.square(a=1, norbs=1)
        sys = kwant.Builder()

        # Scattering region: disorder and horizontal gauge
        def onsite(site, U0, t, salt):
            # On-site disorder plus constant offset
            return U0 * (uniform(repr(site), repr(salt)) - 0.5) + 4 * t

        def hop_horiz(site_i, site_j, phi, t):
            # Hopping with Peierls phase in horizontal gauge
            xi, yi = site_i.pos
            xj, yj = site_j.pos
            return -t * exp(-0.5j * phi * (xi - xj) * (yi + yj))

        sys[(lat(x, y) for x in range(L) for y in range(W))] = onsite
        sys[lat.neighbors()] = hop_horiz

        leadParameters = parameters.static.leadParameters
        if leadParameters is not None:
            B_in_Lead = leadParameters.BinLeads
            leadLeftRightDistanceToCorner = leadParameters.leadLeftRightDistanceToCorner
            leadTopBottomWidth = leadParameters.leadTopBottomWidth
            leadTopBottomDistanceToCorner = leadParameters.leadTopBottomDistanceToCorner

            # Left lead (0) and right lead (1): horizontal gauge, clean
            lead_left = kwant.Builder(kwant.TranslationalSymmetry([-1, 0]))
            lead_left[(lat(0, y) for y in range(leadLeftRightDistanceToCorner, W - leadLeftRightDistanceToCorner))] = 4 * t
            lead_left[lat.neighbors()] = lambda i, j, phi, t: hop_horiz(i, j, phi if B_in_Lead else 0, t)
            sys.attach_lead(lead_left)
            sys.attach_lead(lead_left.reversed())

            # Vertical gauge for voltage probes on top and bottom edges
            def hop_vert(site_i, site_j, phi, t):
                xi, yi = site_i.pos
                xj, yj = site_j.pos
                # Phase on vertical hoppings only
                if xi == xj and B_in_Lead:
                    return -t * exp(0.5j * phi * (yj - yi) * (xi + xj))
                return -t

            # Voltage probes: attach four wide leads on top/bottom

            # Top-left (lead 2): covers x in [0, x_mid)
            lead_tl = kwant.Builder(kwant.TranslationalSymmetry([0, 1]))
            lead_tl[(lat(x, W - 1) for x in range(leadTopBottomDistanceToCorner, leadTopBottomDistanceToCorner + leadTopBottomWidth))] = 4 * t
            lead_tl[lat.neighbors()] = hop_vert
            sys.attach_lead(lead_tl)
            # Top-right (lead 3): covers x in [x_mid, L)
            lead_tr = kwant.Builder(kwant.TranslationalSymmetry([0, 1]))
            lead_tr[(lat(x, W - 1) for x in range(L-leadTopBottomDistanceToCorner-leadTopBottomWidth, L-leadTopBottomDistanceToCorner))] = 4 * t
            lead_tr[lat.neighbors()] = hop_vert
            sys.attach_lead(lead_tr)
            # Bottom-left (lead 4): covers x in [0, x_mid)
            lead_bl = kwant.Builder(kwant.TranslationalSymmetry([0, -1]))
            lead_bl[(lat(x, 0) for x in range(leadTopBottomDistanceToCorner, leadTopBottomDistanceToCorner + leadTopBottomWidth))] = 4 * t
            lead_bl[lat.neighbors()] = hop_vert
            sys.attach_lead(lead_bl)
            # Bottom-right (lead 5): covers x in [x_mid, L)
            lead_br = kwant.Builder(kwant.TranslationalSymmetry([0, -1]))
            lead_br[(lat(x, 0) for x in range(L-leadTopBottomDistanceToCorner-leadTopBottomWidth, L-leadTopBottomDistanceToCorner))] = 4 * t
            lead_br[lat.neighbors()] = hop_vert
            sys.attach_lead(lead_br)

        system: kwant.system.FiniteSystem = sys.finalized() # type: ignore

        super().__init__(parameters=parameters, system=system)


    def hamiltonian_matrix(self, params: DynamicHallBarParameters) -> np.ndarray:
        """Returns the matrix representation of the finite system (without leads)"""
        # Note: hamiltonian_submatrix is not added in the __init__ in kwant
        print("param dict: ", params.__dict__)
        return self.system.hamiltonian_submatrix(params=params.__dict__) # type: ignore


    def compute_classical_resistivities(self,
            args: ClassicalResistivityCalculationParameters
    ):
        e = Constants.e
        h = Constants.h

        # Convert dimensionless flux phi → physical B-field in Tesla
        # phi = (e/ħ) B a^2 ⇒ B = phi ħ/(e a^2) = phi * h/(2π e a^2)
        B = args.phis * h / (2 * np.pi * e * args.a**2)
        # Prepare classical resistivity curves if density and mobility are given
        rho_xx_cl = None
        rho_xy_cl = None
        if args.n2d is not None and args.mu is not None:
            # classical longitudinal resistivity: rho_xx = 1/(n e μ)
            rho_xx_cl = (1.0/(args.n2d * e * args.mu)) * (e**2 / h) * np.ones_like(B)
            # classical Hall resistivity: rho_xy = B/(n e)
            rho_xy_cl = (B/(args.n2d * e)) * (e**2 / h)
        return rho_xx_cl, rho_xy_cl

    def compute_resistivities(
            self,
            param: ResistivityCalculationParameters,
            usePrint=True) -> Resistivities:
        """
        Compute ρ_xx and ρ_xy for a six-terminal Hall bar.
        If temp>0, performs thermal averaging over energy range ±5·temp with nE points.
        temp and energy share the same units.
        Uses two current leads (0/1) and four voltage probes (2-5).
        """
        L = self.parameters.static.L
        W = self.parameters.static.W
        U0 = self.parameters.dynamic.U0
        salt = self.parameters.dynamic.salt
        t = self.parameters.dynamic.t

        temp = param.temp
        energy = param.energy
        nE = param.nE
        phis = param.phis

        rho_xx = []
        rho_xy = []
        # FIXME: Probe spacing is dependent on parameters of leads
        # Probe spacing along x-direction
        x_tl = L // 4
        x_tr = 3 * L // 4
        dx = x_tr - x_tl if x_tr > x_tl else 1
        # Probes and leads
        idx = [0, 2, 3, 4, 5]
        N = max(idx) + 1
        # Current injection: I0=+1, I1=-1
        I = np.zeros(N)
        I[0] = 1
        I[1] = -1
        # Thermal averaging setup
        if temp > 0 and nE > 1:
            nwidth = 5
            E_list = np.linspace(energy - nwidth * temp,
                                energy + nwidth * temp, nE)
            arg = (E_list - energy) / (2 * temp)
            weights = 1.0 / (4 * temp * np.cosh(arg)**2)
            weights = weights / np.sum(weights)
        else:
            E_list = np.array([energy])
            weights = np.array([1.0])
        # Loop over magnetic flux values with progress indicator
        nphis = len(phis)
        if usePrint:
            sys.stderr.write(f"Sweeping {nphis} flux points...\n")
        for iphi, phi in enumerate(phis, start=1):
            if usePrint:
                # Progress update
                sys.stderr.write(f"\r  flux step {iphi}/{nphis}")
                sys.stderr.flush()
            G_sub_T = np.zeros((len(idx), len(idx)))
            for Ei, wi in zip(E_list, weights):
                # Disorder averaging
                G_sub = np.zeros([len(idx), len(idx)], dtype=float)
                # FIXME: Discuss where disorder averaging should be done: In the conductivity or the resistivity?
                for i in range(param.nseeds):
                    params = dict(phi=phi, U0=U0, salt=salt + i, t=t)
                    smat = kwant.smatrix(self.system, Ei, params=params)
                    G = smat.conductance_matrix()
                    G_sub += G[np.ix_(idx, idx)]
                G_sub /= param.nseeds
                G_sub_T += wi * G_sub
            I_sub = I[idx]
            try:
                V_sub = np.linalg.solve(G_sub_T, I_sub)
            except np.linalg.LinAlgError:
                V_sub = np.linalg.lstsq(G_sub_T, I_sub, rcond=None)[0]
            V = np.zeros(N)
            V[1] = 0
            V[idx] = V_sub
            # Longitudinal resistivity
            R_xx = V[2] - V[3]
            rho_xx.append(R_xx * (W / dx))
            # Hall resistivity (bottom minus top)
            V_top = (V[2] + V[3]) / 2
            V_bot = (V[4] + V[5]) / 2
            R_xy = V_bot - V_top
            rho_xy.append(R_xy)
        # End of flux sweep
        if usePrint:
            sys.stderr.write("\n")

        rho_xx_cl = None
        rho_xy_cl = None
        if param.mu is not None and param.n2d is not None:
            rho_xx_cl, rho_xy_cl = self.compute_classical_resistivities(
                args=ClassicalResistivityCalculationParameters(
                    phis=param.phis,
                    a=self.parameters.static.a,
                    mu=param.mu,
                    n2d=param.n2d
                )
            )

        return Resistivities(
            phis=phis,
            rho_xx=np.array(rho_xx),
            rho_xy=np.array(rho_xy),
            rho_xx_classical=rho_xx_cl,
            rho_xy_classical=rho_xy_cl,
            calculationParameters=param.model_copy(deep=True),
            hallBarParameters=self.parameters.model_copy(deep=True)
        )


    def plotResistivity(self,
            resistivities: Resistivities
    ):
        B = resistivities.magneticFields()
        rho_xx = resistivities.rho_xx
        rho_xy = resistivities.rho_xy
        rho_xx_cl = resistivities.rho_xx_classical
        rho_xy_cl = resistivities.rho_xy_classical
        args = resistivities.calculationParameters

        plt.figure()
        plt.plot(B, rho_xx, "-", label=r'$\rho_{xx}$')
        plt.plot(B, rho_xy, "--", label=r'$\rho_{xy}$')
        # Overlay classical curves if provided
        if rho_xx_cl is not None:
            plt.plot(B, np.full_like(B, rho_xx_cl),
                    '--', color='gray', label=r'classical $\rho_{xx}$')
        if rho_xy_cl is not None:
            plt.plot(B, rho_xy_cl,
                    '--', color='gray', label=r'classical $\rho_{xy}$')
        # Add quantum Hall plateau lines at 1/n (h/e^2)
        for n in range(1, 8+1):
            plateau = 1.0 / n
            plt.axhline(plateau, color='black', linestyle=':', linewidth=0.8)
            # annotate plateau
            plt.text(B[-1], plateau, f'1/{n}', color='black', ha='right', va='bottom')
        plt.xlabel('B (T)')
        plt.ylabel(r'Resistivity ($h/e^2$)')
        plt.legend()
        plt.title(f'W={self.parameters.static.W}, L={self.parameters.static.L}, E={args.energy}, U0={self.parameters.dynamic.U0}')
        plt.tight_layout()


    # TODO: This function and 'plotPlateauExtractedCarrierDensityFromHallResistivity' calculate the exact same quantity
    def plotCarrierDensityFromResistivity(self,
            resistivities: Resistivities
    ):
        e = Constants.e
        h = Constants.h
        
        B = resistivities.magneticFields()
        rho_xy = resistivities.rho_xy
        args = resistivities.calculationParameters

        plt.figure()
        # Measured density from Hall resistivity: n = B*e/(h * rho_xy)
        # Avoid division by zero
        n2d_meas = np.where(rho_xy != 0, B * e / (h * rho_xy), np.nan)
        plt.plot(B, n2d_meas, label=r'$n_{\mathrm{2D}}^{\mathrm{meas}}$')
        plt.xlabel('B (T)')
        plt.ylabel(r'2D carrier density (m$^{-2}$)')
        plt.title(f'Density vs B (W={self.parameters.static.W}, L={self.parameters.static.L}, E={args.energy}, U0={self.parameters.dynamic.U0})')
        plt.legend()
        plt.tight_layout()

    def plotPlateauExtractedCarrierDensityFromHallResistivity(
            self,
            resistivities: Resistivities
    ):
        e = Constants.e
        h = Constants.h

        B = resistivities.magneticFields()
        rho_xy = resistivities.rho_xy

        plt.figure()
        # Estimate integer filling factors ν from quantized ρ_xy plateaus: ν ≈ 1/ρ_xy
        # (ρ_xy is in units of h/e², so 1/ρ_xy ≈ ν when on a plateau)
        nu_est = np.round(1.0 / rho_xy)
        # Compute density: n = ν·e·B/h
        n2d_plateau = nu_est * e * B / h
        plt.plot(B, n2d_plateau, marker='o', linestyle='-',
                label=r'$n_{\mathrm{2D}}^{\mathrm{plateau}}$')
        plt.xlabel('B (T)')
        plt.ylabel(r'2D carrier density (m$^{-2}$)')
        plt.title(
            f'Plateau-extracted density vs B (W={resistivities.hallBarParameters.static.W}, L={resistivities.hallBarParameters.static.L}, E={resistivities.calculationParameters.energy}, U0={resistivities.hallBarParameters.dynamic.U0})'
        )
        plt.legend()
        plt.tight_layout()


    def plot(self, file=None, dpi=None, fig_size=None):
        kwant.plot(
            self.system,
            file=file,
            dpi=dpi,
            fig_size=fig_size)


def main():
    parser = argparse.ArgumentParser(
        description='Compute rho_xx and rho_xy vs magnetic flux phi for a six-terminal Hall bar'
    )
    parser.add_argument('--W', type=int, default=50, help='Width of the bar')
    parser.add_argument('--L', type=int, default=90, help='Length of the bar')
    parser.add_argument('--energy', type=float, default=0.2, help='Fermi energy')
    parser.add_argument('--U0', type=float, default=0.0, help='Disorder strength')
    parser.add_argument('--salt', type=int, default=13, help='Random seed for disorder')
    parser.add_argument('--phi-min', type=float, default=0, help='Minimum phi')
    parser.add_argument('--phi-max', type=float, default=0.1, help='Maximum phi')
    parser.add_argument('--nphis', type=int, default=51, help='Number of phi points')
    parser.add_argument('--nseeds', type=int, default=1,
                        help='Number of disorder seeds for averaging (default: 1)')
    parser.add_argument('--outfile', type=str, default=None,
                        help='Output plot filename (png, pdf). If unset, show interactively.')
    parser.add_argument('--a', type=float, default=1e-9,
                        help='Lattice constant a (in meters) for B-field conversion')
    parser.add_argument('--n2d', type=float, default=None,
                        help='2D carrier density (m^-2) for classical overlay')
    parser.add_argument('--mu', type=float, default=None,
                        help='Carrier mobility (m^2/Vs) for classical overlay')
    parser.add_argument('--temp', type=float, default=0.0,
                        help='Finite-temperature smearing (same units as --energy)')
    parser.add_argument('--nE', type=int, default=21,
                        help='Number of energy points for thermal averaging')
    parser.add_argument('--BinLeads', type=bool, default=False,
                        help='Has B field in the leads (i.e., Peierls phase)')
    parser.add_argument('--leadLeftRightDistanceToCorner', type=int, default=0,
                        help='Distance of left and right leads to the corners of the rectangular system in lattice constants.')
    parser.add_argument('--leadTopBottomWidth', type=int, default=0,
                        help='The width of the lead in lattice constants.')
    parser.add_argument('--leadTopBottomDistanceToCorner', type=int, default=0,
                        help='Distance of top and bottom leads to the corners of the rectangular system in lattice constants.')
    args = parser.parse_args()

    # Stage 1/3: Build and finalize the 6-terminal Hall bar geometry
    print(f"Stage 1/3: Building system geometry (W={args.W}, L={args.L}, U0={args.U0}, salt={args.salt})...")

    args.leadTopBottomWidth = args.L // 5 # args.L // 2
    args.leadTopBottomDistanceToCorner = args.L // 5 # 0

    hallBar = HallBar(
        HallBarParameters(
            static=StaticHallBarParameters(
                W=args.W,
                L=args.L,
                a=args.a,
                leadParameters=LeadParameters(
                    BinLeads=args.BinLeads,
                    leadLeftRightDistanceToCorner=args.leadLeftRightDistanceToCorner,
                    leadTopBottomWidth=args.leadTopBottomWidth,
                    leadTopBottomDistanceToCorner=args.leadTopBottomDistanceToCorner,
                )
            ),
            dynamic=DynamicHallBarParameters(
                t=1,
                phi=0,
                U0=args.U0,
                salt=args.salt
            )
        )
    )

    # save system image to "system.pdf"
    hallBar.plot(file="system.pdf")

    print("Stage 1/3 complete: system geometry built.")

    # Stage 2/3: Prepare for resistivity computations
    nE_actual = args.nE if args.temp > 0 and args.nE > 1 else 1
    print(f"Stage 2/3: Computing resistivities ({args.nseeds} seed(s), {args.nphis} flux point(s), {nE_actual} energy point(s) per flux)")

    # Stage 2/3 complete: resistivities computed.
    # Stage 3/3: Plotting resistivities vs. B (Tesla)

    resistivities = hallBar.compute_resistivities(
        ResistivityCalculationParameters.default()
    )
    hallBar.plotResistivity(
        resistivities=resistivities
    )

    tempFiguresPath = Path("temp_figures")

    # Determine output filename based on parameters or use provided
    if args.outfile:
        filename = args.outfile
    else:
        filename = tempFiguresPath/Path("resistivities")/Path(resistivities.makeBaseFileName() + ".pdf")
    plt.savefig(filename)
    # use this PDF for showing the latest resistivity plot
    plt.savefig("resistivity_latest.pdf")
    print(f'Saved plot to {filename}')
    # Stage 3/3 complete: plot generated and saved.

    # Additional plot: 2D carrier density vs magnetic field
    hallBar.plotCarrierDensityFromResistivity(resistivities)

    # Determine output filename for density plot
    if args.outfile:
        base, ext = os.path.splitext(filename)
        dens_filename = f'{base}_density_vs_B{ext}'
    else:
        mainFileName = "density_from_rho_xy"
        dens_filename = tempFiguresPath/Path(mainFileName)/Path(mainFileName + resistivities.parameterFileNamePart() + ".pdf")
    plt.savefig(dens_filename)
    print(f'Saved density plot to {dens_filename}')

    # Extract actual 2D carrier density from quantum Hall plateaus
    hallBar.plotPlateauExtractedCarrierDensityFromHallResistivity(
        resistivities=resistivities
    )
    # Determine output filename for plateau-based density plot
    if args.outfile:
        base, ext = os.path.splitext(filename)
        plateau_filename = f'{base}_plateau_density_vs_B{ext}'
    else:
        mainFileName = "density_from_rho_xy_plateau"
        plateau_filename = tempFiguresPath/Path(mainFileName)/Path(mainFileName + resistivities.parameterFileNamePart() + ".pdf")

    plt.savefig(plateau_filename)
    print(f'Saved plateau-extracted density plot to {plateau_filename}')

if __name__ == '__main__':
    main()