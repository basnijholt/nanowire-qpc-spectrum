
# 1. Standard library imports
from copy import copy, deepcopy
import cmath
from types import SimpleNamespace

# 2. External package imports
import kwant
from kwant.continuum.discretizer import discretize
from kwant.digest import uniform
import numpy as np
import scipy.constants
from scipy.constants import physical_constants

# 3. Internal imports
import common
import peierls


# (Fundamental) constants definitions in nm and meV.
constants = {
    'm_eff': 0.015 * scipy.constants.m_e / scipy.constants.eV / (1e9)**2 * 1e3,
    'phi_0': 2 * physical_constants['mag. flux quantum'][0] * (1e9)**2,
    'mu_B': physical_constants['Bohr magneton in eV/T'][0] * 1e3,
    'hbar': scipy.constants.hbar / scipy.constants.eV * 1e3,
    'exp': cmath.exp,
    'V': lambda *x: 0,
}


# Hamiltonian and system definition

@common.memoize
def get_sympy_hamiltonian(subs, dim=3):
    ham = ("(0.5 * hbar**2 * (k_x**2 + k_y**2 + k_z**2) / m_eff - mu + V) * sigma_0 + "
           "alpha * (k_y * sigma_x - k_x * sigma_y) + "
           "0.5 * g * mu_B * (B_x * sigma_x + B_y * sigma_y + B_z * sigma_z)")

    if dim == 2:
        ham = ham.replace('+ k_z**2', '')

    ham = kwant.continuum.sympify(ham)
    return ham.subs(fix_sympy_substititions(subs))


def fix_sympy_substititions(subs):
    return {kwant.continuum.sympify(k): kwant.continuum.sympify(v)
            for k, v in subs.items()}


def get_template(a, subs=None, dim=3):
    ham = get_sympy_hamiltonian(subs if subs is not None else {}, dim)
    tb_ham, coords = kwant.continuum.discretize_symbolic(ham)
    if dim == 2:
        vector_potential = '[-B_z * y, 0]'
    elif dim == 3:
        vector_potential = '[B_y * z - B_z * y, 0, B_x * y]'
    tb_ham = peierls.apply(tb_ham, coords, A=vector_potential)
    template = kwant.continuum.build_discretized(
        tb_ham, grid_spacing=a, coords=coords)
    return template


def get_shape(R, L0=0, L1=None, shape='hexagon', dim=3):
    if L1 is None:
        start_coords = (0, 0, 0)[:dim]
    else:
        start_coords = ((L0 + L1) / 2, 0, 0)[:dim]

    def _shape(site):
        if dim == 2:
            (x, y) = site.pos
            is_in_shape = abs(y) < R
        else:
            (x, y, z) = site.pos
            if shape == 'hexagon':
                is_in_shape = (
                    y > -R and y < R and y > -2 * (R - z) and y < -2 *
                    (z - R) and y < 2 * (z + R) and y > -2 * (z + R)
                )
            elif shape == 'square':
                is_in_shape = abs(z) < R and abs(y) < R
            else:
                raise ValueError('Only `hexagon` and `square` shape allowed.')
        return is_in_shape and ((L1 is None) or (x >= 0 and x < L1))

    return _shape, start_coords


@common.memoize
def make_wire(a, r, shape='hexagon', dim=3, left_lead=True, right_lead=True):
    """Create a 3D wire.

    Parameters
    ----------
    a : int
        Discretization constant in nm.
    r : int
        Radius of the wire in nm.
    shape : str
        Either `hexagon` or `square` shaped cross section.

    Returns
    -------
    syst : kwant.builder.FiniteSystem
        The finilized kwant system.

    Examples
    --------
    This doesn't use default parameters because the variables need to be saved,
    to a file. So I create a dictionary that is passed to the function.

    >>> syst_params = dict(a=10, ...)
    >>> syst, hopping = make_3d_wire(**syst_params)

    """
    subs = {'V': ('V(x, y, z)' if dim == 3 else 'V(x, y)')}
    shape_lead = get_shape(r, shape=shape, dim=dim)
    symmetry = kwant.TranslationalSymmetry((a, 0, 0)[:dim])
    lead = kwant.Builder(symmetry)
    template = get_template(a, subs={'V': '0'}, dim=dim)
    lead.fill(template, *shape_lead)
    return lead.finalized()


def bands(lead, params, ks=None):
    if ks is None:
        ks = np.linspace(-3, 3)

    bands = kwant.physics.Bands(lead, params=params)

    if isinstance(ks, (float, int)):
        return bands(ks)
    else:
        return np.array([bands(k) for k in ks])


def get_h_k(lead, params):
    h, t = cell_mats(lead, params)
    h_k = lambda k: h + t * np.exp(1j * k) + t.T.conj() * np.exp(-1j * k)
    return h_k

