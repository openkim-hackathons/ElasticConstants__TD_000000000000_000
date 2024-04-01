#!/usr/bin/python

from kim_test_utils.test_driver import CrystalGenomeTestDriver, query_crystal_genome_structures
from ase.build import bulk
from ase.units import GPa
from elastic import ElasticConstants, calc_bulk, find_nearest_isotropy, get_unique_components_and_reconstruct_matrix
import numpy as np

class TestDriver(CrystalGenomeTestDriver):
    def _calculate(self, method: str="energy-condensed", **kwargs):
        """
        Example calculate method. Just recalculates the binding-energy-crystal property.

        You may add arbitrary arguments, which will be passed to this method when the test driver is invoked.

        You must include **kwargs in the argument list, but you don't have to do anything with it

        Args:
            method:
                method for calculating elasticity matrix. 
        """

        ####################################################
        # ACTUAL CALCULATION BEGINS 
        ####################################################
                
        print('\nE L A S T I C  C O N S T A N T  C A L C U L A T I O N S\n')
        print()

        moduli = ElasticConstants(self.atoms, condensed_minimization_method='bfgs')
        elastic_constants, error_estimate = \
            moduli.results(optimize=False, method=method)
        bulk = calc_bulk(elastic_constants)

        # Apply unit conversion
        elastic_constants /= GPa
        error_estimate /= GPa
        bulk /= GPa
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
        print('Bulk modulus [{}] = {:.5f}'.format(units,bulk))
        print()
        if got_iso:
            print('Nearest matrix of isotropic elastic constants:')
            print('Distance to isotropic state [-]  = {:.5f}'.format(d_iso))
            print('Isotropic bulk modulus      [{}] = {:.5f}'.format(units,bulk_iso))
            print('Isotropic shear modulus     [{}] = {:.5f}'.format(units,shear_iso))
        else:
            print('WARNING: Nearest isotropic state not computed.')

        elastic_constants_names,elastic_constants_values,reconstructed_matrix = get_unique_components_and_reconstruct_matrix(
            elastic_constants,int(self.prototype_label.split('_')[2]))
        ####################################################
        # ACTUAL CALCULATION ENDS 
        ####################################################

        ####################################################
        # PROPERTY WRITING
        ####################################################
        self._add_property_instance_and_common_crystal_genome_keys("elastic-constants-isothermal-npt",write_stress=True,write_temp=True)
        self._add_key_to_current_property_instance("elastic-constants-names",elastic_constants_names)
        self._add_key_to_current_property_instance("elastic-constants-values",elastic_constants_values,"GPa")
        self._add_key_to_current_property_instance("elasticity-matrix",reconstructed_matrix,"GPa")
        self._add_key_to_current_property_instance("distance-to-isotropy",d_iso)
        self._add_property_instance_and_common_crystal_genome_keys("bulk-modulus-isothermal-npt",write_stress=True,write_temp=True)
        self._add_key_to_current_property_instance("isothermal-bulk-modulus",bulk,"GPa")


if __name__ == "__main__":        
    ####################################################
    # if called directly, do some debugging examples
    ####################################################
    kim_model_name = "MEAM_LAMMPS_KoJimLee_2012_FeP__MO_179420363944_002"

    # For initialization, only pass a KIM model name or an ASE calculator
    test_driver = TestDriver(kim_model_name)

    # To do a calculation, you can pass an ASE.Atoms object or a Crystal Genome prototype designation.
    # Atoms object example:
    # atoms = bulk('Fe','bcc',a=2.863,cubic=True)
    # test_driver(atoms)

    # You can get a list of dictionaries of the results like this:
    # print(test_driver.get_property_instances())

    # Or write it to a file (by default `output/results.edn`) like this:
    #test_driver.write_property_instances_to_file()

    # Alternatively, you can pass a Crystal Genome designation. You can automatically query for all equilibrium structures for a given 
    # species and prototype label like this:
    cg_des_list = query_crystal_genome_structures(kim_model_name, ["Fe", "P"], "A2B_hP9_189_fg_ad")

    # IMPORTANT: cg_des is a LIST. Pass only one element of it to the test, as keywords (i.e. using **):
    for cg_des in cg_des_list:
        test_driver(**cg_des,method="energy-full")

    # Now both results are in the property instances:
    print(test_driver.get_property_instances())

    test_driver.write_property_instances_to_file()

    # Here are some other crystal prototypes supported by the current model you can try:
    # ["Fe", "P"], "A2B_hP9_189_fg_ad"
    # ["Fe", "P"], "A3B_tI32_82_3g_g"
    # ["Fe", "P"], "AB_oP8_62_c_c"
    # ["Fe", "P"], "AB2_oP6_58_a_g"
    # ["Fe", "P"], "AB4_mC40_15_ae_4f"
    # ["Fe", "P"], "AB4_mP30_14_ae_6e"
    # ["Fe", "P"], "AB4_oC20_20_a_2c"
    # ["Fe"], "A_cF4_225_a"
    # ["Fe"], "A_cI2_229_a"
    # ["Fe"], "A_hP2_194_c"
    # ["Fe"], "A_tP28_136_f2ij"
    # ["P"], "A_aP24_2_12i"
    # ["P"], "A_cP1_221_a"
    # ["P"], "A_mC16_12_2ij"
    # ["P"], "A_oC8_64_f"
    # ["P"], "A_tI4_139_e"
