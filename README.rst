ElasticConstants Crystal Genome Test Driver
===========================================

Computes the elastic constants of an arbitrary crystal using numerical differentiation.
Reports the results as  the ``bulk-modulus-isothermal-npt`` and 
``elastic-constants-isothermal-npt`` OpenKIM Properties: https://openkim.org/properties.

The code also computes the distance from the obtained elasticity tensor to
the nearest isotropic tensor. This provides a measure of the anisotropy
of the crystal. Note that this calculation can fail if the elasticity tensor
is not positive definite.

As with any Crystal Genome Test Driver written using `kim-tools <https://kim-tools.readthedocs.io>`_, 
this Test Driver can be run directly. See how to do so here: 
https://kim-tools.readthedocs.io/en/latest/auto_examples/CrystalGenomeASEExample__TD_000000654321_000/run.html

Three methods are available, selected using the ``method`` argument when calling the Test Driver:

#. ``energy-condensed`` : Compute elastic constants from the hessian of the 
   condensed strain energy density, W_eff(eps) = min_d W(eps,d), where d are
   the displacements of the internal atoms (aside from preventing rigid-body
   translation) and eps is the strain.
#. ``stress-condensed`` : Compute elastic constants from the jacobian of the
   condensed stress, sig_eff(eps) = sig(eps,dmin), where dmin = arg min_d
   W(eps,d). 
#. ``energy-full`` : Compute elastic constants from the hessian of the full
   strain energy density, W(eps,d). This involves an algebraic manipulation
   to account for the effect of atom relaxation; see eqn (27), in Tadmor et
   al, Phys. Rev. B, 59:235-245, 1999.

For well-behaved potentials, all three methods give similar results with 
differences beyond the first or second digit due to the numerical differentiation. 
The ``energy-condensed`` and ``energy-full`` approaches have comparable accuracy, 
but full Hessian is *much* slower. The ``stress-condensed`` approach is 
significantly faster than ``energy-condensed``, but is less accurate. 

For potentials with a rough energy landscape due to sharp cut-offs or
electrostatics, ``stress-condensed`` is sometimes better behaved.

The default approach is ``energy-condensed``. 

If the numdifftools interpolation error, or the deviation from material
symmetry is too large, the Driver will first try increasingly coarse steps 
for numerical differentiation. Failing that, if ``stress-condensed`` is not 
being used already, the code will switch to it. Finally, if no run had
sufficiently low error, the code will return the results with the lowest error.

The choice of steps, as well as the method-switching behavior can be 
overridden by the arguments ``sg_override`` and ``escalate``.

