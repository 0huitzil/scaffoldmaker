import unittest

from cmlibs.utils.zinc.finiteelement import evaluateFieldNodesetRange
from cmlibs.utils.zinc.general import ChangeManager
from cmlibs.zinc.context import Context
from cmlibs.zinc.element import Element
from cmlibs.zinc.field import Field
from cmlibs.zinc.result import RESULT_OK
from scaffoldmaker.annotation.annotationgroup import getAnnotationGroupForTerm
from scaffoldmaker.annotation.trigeminal_nerve_terms import get_trigeminal_nerve_term, trigeminal_nerve_terms
from scaffoldmaker.meshtypes.meshtype_3d_trigeminalnerve1 import MeshType_1d_human_trigeminal_nerve_network_layout1
from scaffoldmaker.utils.zinc_utils import createFaceMeshGroupExteriorOnFace

from testutils import check_annotation_term_ids


class TrigeminalNerveScaffoldTestCase(unittest.TestCase):

    def test_trigeminal_nerve_annotations(self):
        """
        Test nomenclature of the trigeminal nerve terms. 
        """
        for term_ids in trigeminal_nerve_terms:
            self.assertTrue(check_annotation_term_ids(term_ids), "Invalid primary term id or order not UBERON < ILX < FMA for trigeminal nerve annotation term ids " + str(term_ids)) 


if __name__ == "__main__":
    unittest.main()
