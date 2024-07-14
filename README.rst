ElasticConstants Crystal Genome Test Driver
===========================================

Computes the `bulk-modulus-isothermal-npt` and `elastic-constants-isothermal-npt` OpenKIM Properties: https://openkim.org/properties.

Computes the elastic constants of an arbitrary crystal using numerical differentiation. 

The code also computes the distance from the obtained elasticity tensor to
the nearest isotropic tensor using. This provides a measure of the anisotropy
of the crystal. Note that this calculation can fail if the elasticity tensor
is not positive definite.

The following methods for computing the elastic constants are supported:

# energy-condensed : Compute elastic constants from the hessian of the
condensed strain energy density, W_eff(eps) = min_d W(eps,d), where d are
the displacements of the internal atoms (aside from preventing rigid-body
translation) and eps is the strain.
# stress-condensed : Compute elastic constants from the jacobian of the
condensed stress, sig_eff(eps) = sig(eps,dmin), where dmin = arg min_d
W(eps,d). 
# energy-full : Compute elastic constants from the hessian of the full
strain energy density, W(eps,d). This involves an algebraic manipulation
to account for the effect of atom relaxation; see eqn (27), in Tadmor et
al, Phys. Rev. B, 59:235-245, 1999.

All three methods give similar results with differences beyond the first or
second digit due to the numerical differentiation. The `energy-condensed` and
`energy-full` approaches have comparable accuracy, but full Hessian is *much*
slower. The `stress-condensed` approach is significantly faster than
`energy-condensed`, but is less accurate. The default apporach is
`energy-condensed`. The other methods are retained in the code for testing
purposes or, for very large systems, the `stress-condensed` method is an
option.

If the error reported by the numdifftools is too large, the code automatically
escalates to the next more accurate (slower) method. This is controlled by the 
ELASTIC_CONSTANTS_ERROR_TOLERANCE constant in this file. Additionally,
you can provide a space group to check whether or not the elastic constants
conform to the expected material symmetry. To use this, the unit cell
must be in standard IUCr orientation.
