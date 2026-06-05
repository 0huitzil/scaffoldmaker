"""
Scaffold abstract base class.
Describes methods each scaffold must or may override.
"""
import copy

from cmlibs.utils.zinc.general import ChangeManager
from cmlibs.utils.zinc.scene import scene_get_selection_group
from cmlibs.zinc.field import Field
from scaffoldmaker.utils.derivativemoothing import DerivativeSmoothing
from scaffoldmaker.utils.interpolation import DerivativeScalingMode
from scaffoldmaker.utils.meshrefinement import MeshRefinement
from scaffoldmaker.utils.zinc_utils import get_nodeset_field_parameters, print_node_field_parameters


class Scaffold_base:
    '''
    Base class for scaffolds / mesh generator scripts.
    Not intended to be instantiated. Most methods must be overridden by actual scaffolds.
    '''

    @classmethod
    def getName(cls):
        '''
        Must override.
        :return: Unique type name for scaffold, for display in user interface.
        '''
        return None

    @classmethod
    def getParameterSetNames(cls):
        '''
        Optionally override to return additional default parameter set names supported by getDefaultOptions().
        Always have first set name 'Default'. Do not use name 'Custom' as clients may use internally.
        '''
        return ['Default']
 
    @classmethod
    def getDefaultOptions(cls, parameterSetName='Default'):
        '''
        Must override to get valid initial default or other named parameter set. Must support 'Default' parameter set.
        :param parameterSetName: Name of parameter set to get, from list returned by getParameterSetNames().
        :return: Dictionary of parameter name value pairs, of integer, real or boolean type.
        '''
        return {
            'Integer option' : 1,
            'Real option' : 1.0,
            'Boolean option' : True
        }

    @classmethod
    def getOrderedOptionNames(cls):
        '''
        Must override to get list of parameters in order for display and editing in user interface.
        Note can omit parameter names to remove from interface.
        :return: List of parameter names in display order.
        '''
        return []

    @classmethod
    def getOptionValidScaffoldTypes(cls, optionName):
        '''
        Override in derived types with ScaffoldPackage options, to return list of
        valid scaffold types for ScaffoldPackage option optionName.
        '''
        return []

    @classmethod
    def getOptionScaffoldTypeParameterSetNames(cls, optionName, scaffoldType):
        '''
        Override in derived types with ScaffoldPackage options to return
        custom list of parameter set names for ScaffoldPackage option optionName with the
        specified scaffoldType.
        Override may cut and paste this code to handle any types using their standard
        parameter set names.
        '''
        assert scaffoldType in cls.getOptionValidScaffoldTypes(optionName), cls.__name__ + '.getOptionScaffoldTypeParameterSetNames.  Invalid option ' + optionName + ' scaffold type ' + scaffoldType.getName()
        return scaffoldType.getParameterSetNames()

    @classmethod
    def getOptionScaffoldPackage(cls, optionName, scaffoldType, parameterSetName=None):
        '''
        Override in derived types with ScaffoldPackage options to create a ScaffoldPackage
        object for option optionName with the specified scaffoldType and parameterSetName.
        :param parameterSetName:  Name of valid parameter set for option Scaffold, or None for default.
        '''
        assert False, 'getOptionScaffoldPackage() not implemented for scaffold type.  Attempted to get option ' + optionName + ' scaffold type ' + scaffoldType.getName()

    @classmethod
    def checkOptions(cls, options):
        '''
        Must override to keep options within limits to prevent nonsense or errors.
        '''
        if options['Integer option'] < 1:
            options['Integer option'] = 1
        if options['Real option'] < 0.0:
            options['Real option'] = 0.0

    @classmethod
    def generateBaseMesh(cls, region, options):
        """
        Override to generate scaffold mesh in region using Zinc API with options.
        If function makes a "construction object" needed by the caller, this is
        returned as the second return value.
        :param region: Zinc region to define model in. Must be empty.
        :param options: Dict containing options. See getDefaultOptions().
        :return: list of AnnotationGroup, ConstructionObject-derived object (or None)
        """
        return [], None

    @classmethod
    def refineMesh(cls, meshrefinement, options):
        """
        Override to refine/resample mesh if supported.
        :param meshrefinement: MeshRefinement, which knows source and target region.
        :param options: Dict containing options. See getDefaultOptions().
        """
        pass

    @classmethod
    def defineFaceAnnotations(cls, region, options, annotationGroups):
        """
        Override in classes with face annotation groups.
        Add face annotation groups from the highest dimension mesh.
        Must have defined faces and added subelements for highest dimension groups.
        :param region: Zinc region containing model.
        :param options: Dict containing options. See getDefaultOptions().
        :param annotationGroups: List of annotation groups for elements created in generateBaseMesh().
        New face/line annotation groups are appended to this list.
        """
        pass

    @classmethod
    def generateMesh(cls, region, options):
        """
        Generate base or refined mesh.
        If function makes a "construction object" needed by the caller, this is
        returned as the second return value.
        Note that no construction object is returned when using 'Refine'.
        :param region: Zinc region to create mesh in. Must be empty.
        :param options: Dict containing options. See getDefaultOptions().
        :return: list of AnnotationGroup, construction object (or None)
        """
        fieldmodule = region.getFieldmodule()
        with ChangeManager(fieldmodule):
            constructionObject = None
            if options.get('Refine'):
                baseRegion = region.createRegion()
                annotationGroups = cls.generateBaseMesh(baseRegion, options)[0]
                # need faces to determine shared or boundary nodes during mesh refinement
                baseRegion.getFieldmodule().defineAllFaces()
                meshrefinement = MeshRefinement(baseRegion, region, annotationGroups)
                cls.refineMesh(meshrefinement, options)
                annotationGroups = meshrefinement.getAnnotationGroups()
            else:
                annotationGroups, constructionObject = cls.generateBaseMesh(region, options)
            fieldmodule.defineAllFaces()
            oldAnnotationGroups = copy.copy(annotationGroups)
            for annotationGroup in annotationGroups:
                annotationGroup.addSubelements()
            cls.defineFaceAnnotations(region, options, annotationGroups)
            for annotationGroup in annotationGroups:
                if annotationGroup not in oldAnnotationGroups:
                    annotationGroup.addSubelements()
        return annotationGroups, constructionObject

    @classmethod
    def printNodeFieldParameters(cls, region, options, constructionObject, functionOptions, editGroupName):
        """
        Interactive function for printing node field parameters for pasting into code.
        Optionally supports choice of field with functionOptions["Field"] a dict mapping field name to True/False.
        """
        fieldmodule = region.getFieldmodule()
        nodeset = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        selectionGroup = scene_get_selection_group(region.getScene(), inherit_root_region=region.getRoot())
        if selectionGroup:
            nodeset = selectionGroup.getNodesetGroup(nodeset)
            if not nodeset.isValid():
                print('Print node field parameters: No nodes selected')
                return False, False
        fieldName = "coordinates"
        fieldOption = functionOptions.get("Field")
        if fieldOption:
            for key in fieldOption:
                if fieldOption[key]:
                    fieldName = key
                    break
        coordinates = fieldmodule.findFieldByName(fieldName).castFiniteElement()
        if not coordinates.isValid():
            print("Print node parameters:  " + fieldName + " field not defined")
            return False, False
        valueLabels, fieldParameters = get_nodeset_field_parameters(nodeset, coordinates)
        numberFormat = '{:' + functionOptions['Number format (e.g. 8.3f)'] + '}'
        print_node_field_parameters(valueLabels, fieldParameters, numberFormat)
        return False, False  # no change to settings, nor node parameters

    @classmethod
    def smoothDerivatives(cls, region, options, constructionObject, functionOptions, editGroupName):
        """
        Interactive function for smoothing first derivatives over the mesh, or limited to selected nodes.
        Optionally supports choice of field with functionOptions["Field"] a dict mapping field name to True/False.
        """
        fieldmodule = region.getFieldmodule()
        fieldName = "coordinates"
        fieldOption = functionOptions.get("Field")
        if fieldOption:
            for key in fieldOption:
                if fieldOption[key]:
                    fieldName = key
                    break
        coordinates = fieldmodule.findFieldByName(fieldName).castFiniteElement()
        if not coordinates.isValid():
            print("Smooth derivatives:  " + fieldName + " field not defined")
            return False, False
        selectionGroup = scene_get_selection_group(region.getScene(), inherit_root_region=region.getRoot())
        groupName = selectionGroup.getName() if selectionGroup else None
        updateDirections = functionOptions['Update directions']
        scalingMode = DerivativeScalingMode.ARITHMETIC_MEAN if functionOptions['Scaling mode']['Arithmetic mean'] else DerivativeScalingMode.HARMONIC_MEAN
        smoothing = DerivativeSmoothing(region, coordinates, groupName, scalingMode, editGroupName)
        smoothing.smooth(updateDirections)
        del smoothing
        return False, True  # settings not changed, nodes changed

    @classmethod
    def getInteractiveFunctions(cls):
        """
        Override to return list of named interactive functions that client
        can invoke to modify mesh parameters with a push button control.
        Functions must take 5 arguments:
        Zinc region, scaffold options, relevant Construction Object (can be None),
        function options, editGroupName (can be None) for optional name of Zinc
        group to create or modify so changed nodes etc. are put in it.
        Functions return 2 boolean values: optionsChanged, nodesChanged.
        These tell the client whether to redisplay the options or process
        the effects of node edits (which will be recorded in edit group if
        its name is supplied).
        :return: list(tuples), (name, functionOptions, callable(region, options, functionOptions, editGroupName)).
        """
        return [
            ("Print node parameters...",
                { 'Number format (e.g. 8.3f)': ' 11e' },
                lambda region, options, constructionObject, functionOptions, editGroupName:
                    cls.printNodeFieldParameters(region, options, constructionObject, functionOptions, editGroupName)),
            ("Smooth derivatives...",
                { 'Update directions': False,
                  'Scaling mode': { 'Arithmetic mean': True, 'Harmonic mean': False } },
                lambda region, options, constructionObject, functionOptions, editGroupName:
                    cls.smoothDerivatives(region, options, constructionObject, functionOptions, editGroupName))
            ]
