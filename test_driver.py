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
    def _calculate(self, method: str="energy-condensed", escalate: bool=True, 
                   sg_override: Optional[Union[List[Union[MaxStepGenerator, float]],Union[MaxStepGenerator, float]]] = None, **kwargs):
        """
        Example calculate method. Just recalculates the binding-energy-crystal property.

        You may add arbitrary arguments, which will be passed to this method when the test driver is invoked.

        You must include **kwargs in the argument list, but you don't have to do anything with it

        Args:
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
        elastic_constants_raw, elastic_constants_raw_error_estimate, \
            elastic_constants_names, elastic_constants_values, elastic_constants_values_error_estimate, \
                elastic_constants, message = \
            moduli.results(method=method, escalate=escalate, space_group=space_group, sg_override=sg_override)
        max_sym_dev = np.max(np.abs(elastic_constants-elastic_constants_raw))
        bulk = calc_bulk(elastic_constants_raw)

        if 'WARNING' in message:
            disclaimer = 'Elastic constants calculation had an uncertainty or deviation from material symmetry greater than 1%.\n'+\
            'See pipeline.stdout for calculation details.'
        else:
            disclaimer = None

        # Apply unit conversion
        elastic_constants_raw /= GPa
        elastic_constants_raw_error_estimate /= GPa
        elastic_constants_values = [elconst/GPa for elconst in elastic_constants_values]
        elastic_constants_values_error_estimate = [elconst/GPa for elconst in elastic_constants_values_error_estimate]
        bulk /= GPa
        max_sym_dev /= GPa
        units = 'GPa'        

        # Compute nearest isotropic constants and distance
        try:
            d_iso, bulk_iso, shear_iso = find_nearest_isotropy(elastic_constants_raw)
            got_iso = True
        except:
            got_iso = False  # Failure can occur if elastic constants are
                             # not positive definite

        # Echo output
        print('\nR E S U L T S\n')
        print('Elastic constants [{}]:'.format(units))
        print(np.array_str(elastic_constants_raw, precision=5, max_line_width=100, suppress_small=True))
        print()
        print('95 %% Error estimate [{}]:'.format(units))
        print(np.array_str(elastic_constants_raw_error_estimate, precision=5, max_line_width=100, suppress_small=True))
        print()
        print('Maximum deviation from material symmetry [{}] = {:.5f}'.format(units,max_sym_dev))
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
        self._add_property_instance_and_common_crystal_genome_keys("elastic-constants-isothermal-npt",write_stress=True,write_temp=True,disclaimer=disclaimer)
        self._add_key_to_current_property_instance("elastic-constants-names",elastic_constants_names)

        self._add_key_to_current_property_instance(
            "elastic-constants-values",
            elastic_constants_values,
            "GPa",
            {
                "source-expand-uncert-value":elastic_constants_values_error_estimate,
                "uncert-lev-of-confid":95                
            }
        )
        
        self._add_key_to_current_property_instance(
            "elasticity-matrix-raw",
            elastic_constants_raw,
            "GPa",
            {
                "source-expand-uncert-value":elastic_constants_raw_error_estimate,
                "uncert-lev-of-confid":95
            }
        )

        self._add_key_to_current_property_instance("elasticity-matrix",elastic_constants,"GPa")
        if got_iso:
            self._add_key_to_current_property_instance("distance-to-isotropy",d_iso)
        self._add_property_instance_and_common_crystal_genome_keys("bulk-modulus-isothermal-npt",write_stress=True,write_temp=True,disclaimer=disclaimer)
        self._add_key_to_current_property_instance("isothermal-bulk-modulus",bulk,"GPa")


if __name__ == "__main__":        
    ####################################################
    # if called directly, do some debugging examples
    ####################################################
    kim_model_name = "MEAM_LAMMPS_LeeShimBaskes_2003_Pt__MO_534993486058_001"

    # For initialization, only pass a KIM model name or an ASE calculator
    test_driver = TestDriver(kim_model_name)

    # To do a calculation, you can pass an ASE.Atoms object or a Crystal Genome prototype designation.
    # Atoms object example:
    #atoms = bulk('Al','fcc',a=4.0,cubic=True)
    #test_driver(atoms=atoms,optimize=True,method="stress-condensed",escalate=True)

    # You can get a list of dictionaries of the results like this:
    # print(test_driver.get_property_instances())

    # Or write it to a file (by default `output/results.edn`) like this:
    #test_driver.write_property_instances_to_file()

    # Alternatively, you can pass a Crystal Genome designation. You can automatically query for all equilibrium structures for a given 
    # species and prototype label like this:
    cg_des_list = query_crystal_genome_structures(kim_model_name, ['Pt'], 'A_cF4_225_a')	

    # IMPORTANT: cg_des is a LIST. Pass only one element of it to the test, as keywords (i.e. using **):
    for cg_des in cg_des_list:
       test_driver(**cg_des,)

    # Now both results are in the property instances:
    # print(test_driver.get_property_instances())

    test_driver.write_property_instances_to_file()

