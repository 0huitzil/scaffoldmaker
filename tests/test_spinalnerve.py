import unittest

from cmlibs.utils.zinc.finiteelement import evaluateFieldNodesetRange
from cmlibs.utils.zinc.general import ChangeManager
from cmlibs.zinc.context import Context
from cmlibs.zinc.element import Element
from cmlibs.zinc.field import Field
from cmlibs.zinc.result import RESULT_OK
from scaffoldmaker.annotation.annotationgroup import getAnnotationGroupForTerm
from scaffoldmaker.annotation.spinal_nerve_terms import get_spinal_nerve_term, spinal_nerve_terms
from scaffoldmaker.utils.zinc_utils import createFaceMeshGroupExteriorOnFace

from testutils import assertAlmostEqualList, check_annotation_term_ids


class SpinalNerveScaffoldTestCase(unittest.TestCase):

    def test_spinal_nerve_annotations(self):
        """
        Test nomenclature of the spinal nerve terms. 
        """
        for term_ids in spinal_nerve_terms:
            self.assertTrue(check_annotation_term_ids(term_ids), "Invalid primary term id or order not UBERON < ILX < FMA for spinal nerve annotation term ids " + str(term_ids))


if __name__ == "__main__":
    unittest.main()
