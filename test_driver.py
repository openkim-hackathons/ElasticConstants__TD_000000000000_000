#!/usr/bin/python

from kim_test_utils.test_driver import CrystalGenomeTestDriver, query_crystal_genome_structures
from typing import Optional, Union, List
from numdifftools import MaxStepGenerator
from ase.build import bulk
from ase.units import GPa
from ase.atoms import Atoms
from ase.cell import Cell
from elastic import ElasticConstants, calc_bulk, find_nearest_isotropy, get_unique_components_and_reconstruct_matrix
import numpy as np

class TestDriver(CrystalGenomeTestDriver):
    def _calculate(self, optimize:bool=False, method: str="energy-condensed", escalate: bool=False, 
                   sg_override: Optional[Union[List[Union[MaxStepGenerator, float]],Union[MaxStepGenerator, float]]] = None, **kwargs):
        """
        Example calculate method. Just recalculates the binding-energy-crystal property.

        You may add arbitrary arguments, which will be passed to this method when the test driver is invoked.

        You must include **kwargs in the argument list, but you don't have to do anything with it

        Args:
            optimize:
                whether to optimize provided simulation cell
            method:
                method for calculating elasticity matrix. 
            escalate:
                whether to automatically escalate calculation to a higher method if requested fails
            sg_override:
                override the default step sizes in the calculation
        """

        ####################################################
        # ACTUAL CALCULATION BEGINS 
        ####################################################
                
        print('\nE L A S T I C  C O N S T A N T  C A L C U L A T I O N S\n')
        print()

        self.atoms.write("log.poscar",format='vasp')

        space_group = int(self.prototype_label.split('_')[2])

        moduli = ElasticConstants(self.atoms)
        elastic_constants, error_estimate, elastic_constants_names, elastic_constants_values, maximum_deviation = \
            moduli.results(optimize=optimize, method=method, escalate=escalate, space_group=space_group, sg_override=sg_override)
        bulk = calc_bulk(elastic_constants)

        # Apply unit conversion
        elastic_constants /= GPa
        error_estimate /= GPa
        elastic_constants_values = [elconst/GPa for elconst in elastic_constants_values]
        bulk /= GPa
        maximum_deviation /= GPa
        units = 'GPa'        

        # Compute nearest isotropic constants and distance
        try:
            d_iso, bulk_iso, shear_iso = find_nearest_isotropy(elastic_constants)
            got_iso = True
        except:
            got_iso = False  # Failure can occur if elastic constants are
                             # not positive definite

        # Echo output
        print('\nR E S U L T S\n')
        print('Elastic constants [{}]:'.format(units))
        print(np.array_str(elastic_constants, precision=5, max_line_width=100, suppress_small=True))
        print()
        print('Error estimate [{}]:'.format(units))
        print(np.array_str(error_estimate, precision=5, max_line_width=100, suppress_small=True))
        print()
        print('Maximum deviation from material symmetry [{}] = {:.5f}'.format(units,maximum_deviation))
        print()
        print('Bulk modulus [{}] = {:.5f}'.format(units,bulk))
        print()
        print('Unique elastic constants for space group {} [{}]'.format(space_group,units))
        print(elastic_constants_names)
        print(elastic_constants_values)
        print()
        if got_iso:
            print('Nearest matrix of isotropic elastic constants:')
            print('Distance to isotropic state [-]  = {:.5f}'.format(d_iso))
            print('Isotropic bulk modulus      [{}] = {:.5f}'.format(units,bulk_iso))
            print('Isotropic shear modulus     [{}] = {:.5f}'.format(units,shear_iso))
        else:
            print('WARNING: Nearest isotropic state not computed.')

        ####################################################
        # ACTUAL CALCULATION ENDS 
        ####################################################

        ####################################################
        # PROPERTY WRITING
        ####################################################
        self._add_property_instance_and_common_crystal_genome_keys("elastic-constants-isothermal-npt",write_stress=True,write_temp=True)
        self._add_key_to_current_property_instance("elastic-constants-names",elastic_constants_names)
        self._add_key_to_current_property_instance("elastic-constants-values",elastic_constants_values,"GPa")
        self._add_key_to_current_property_instance("elasticity-matrix",elastic_constants,"GPa")
        if got_iso:
            self._add_key_to_current_property_instance("distance-to-isotropy",d_iso)
        self._add_property_instance_and_common_crystal_genome_keys("bulk-modulus-isothermal-npt",write_stress=True,write_temp=True)
        self._add_key_to_current_property_instance("isothermal-bulk-modulus",bulk,"GPa")


if __name__ == "__main__":        
    ####################################################
    # if called directly, do some debugging examples
    ####################################################
    kim_model_name = "Sim_LAMMPS_AIREBO_LJ_StuartTuteinHarrison_2000_CH__SM_069621990420_000"

    # For initialization, only pass a KIM model name or an ASE calculator
    test_driver = TestDriver(kim_model_name)

    test_driver(stoichiometric_species=["C"],prototype_label="A_hP4_194_bc",parameter_values_angstrom=[2.4175,20.],optimize=True,method="energy-condensed")

    # To do a calculation, you can pass an ASE.Atoms object or a Crystal Genome prototype designation.
    # Atoms object example:
    # atoms = bulk('Fe','bcc',a=2.863,cubic=True)
    # test_driver(atoms=atoms,optimize=True,method="stress-condensed-fast",escalate=True)

    # You can get a list of dictionaries of the results like this:
    # print(test_driver.get_property_instances())

    # Or write it to a file (by default `output/results.edn`) like this:
    #test_driver.write_property_instances_to_file()

    # Alternatively, you can pass a Crystal Genome designation. You can automatically query for all equilibrium structures for a given 
    # species and prototype label like this:
    # cg_des_list = query_crystal_genome_structures(kim_model_name, ["O","Si"], "A2B_hP9_154_c_a")

    # IMPORTANT: cg_des is a LIST. Pass only one element of it to the test, as keywords (i.e. using **):
    #for cg_des in cg_des_list:
    #   test_driver(**cg_des,optimize=False,method="energy-condensed",escalate=True,sg_override=[1e-4,
    #                1e-3,
    #                1e-2,])

    # Now both results are in the property instances:
    # print(test_driver.get_property_instances())

    # test_driver.write_property_instances_to_file()

