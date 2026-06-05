"""
Common resource for vagus annotation terms.
"""
from scaffoldmaker.annotation.annotation_utils import annotation_term_id_to_url
import logging

logger = logging.getLogger(__name__)


# Note standard for vagus preferred annotation ID is UBERON > ILX > FMA

# convention: preferred name, preferred id, followed by any other ids and alternative names
vagus_marker_terms = [
    # anatomical landmarks
    ("level of superior border of jugular foramen on the vagus nerve", "ILX:0794617"),
    ("right level of superior border of jugular foramen on the vagus nerve", "ILX:0794618"),
    ("left level of superior border of jugular foramen on the vagus nerve", "ILX:0794619"),
    ("level of inferior border of jugular foramen on the vagus nerve", "ILX:0794620"),
    ("right level of inferior border of jugular foramen on the vagus nerve", "ILX:0794621"),
    ("left level of inferior border of jugular foramen on the vagus nerve", "ILX:0794622"),
    ("level of inferior border of cranium on the vagus nerve", "ILX:0794623"),
    ("right level of inferior border of cranium on the vagus nerve", "ILX:0794624"),
    ("left level of inferior border of cranium on the vagus nerve", "ILX:0794625"),
    ("level of C1 transverse process on the vagus nerve", "ILX:0794626"),
    ("right level of C1 transverse process on the vagus nerve", "ILX:0794627"),
    ("left level of C1 transverse process on the vagus nerve", "ILX:0794628"),
    ("level of greater horn of hyoid on the vagus nerve", "ILX:0794629"),
    ("right level of greater horn of hyoid on the vagus nerve", "ILX:0794630"),
    ("left level of greater horn of hyoid on the vagus nerve", "ILX:0794631"),
    ("level of laryngeal prominence on the vagus nerve", "ILX:0794632"),
    ("right level of laryngeal prominence on the vagus nerve", "ILX:0794633"),
    ("left level of laryngeal prominence on the vagus nerve", "ILX:0794634"),
    ("level of angle of the mandible on the vagus nerve", "ILX:0794635"),
    ("right level of angle of the mandible on the vagus nerve", "ILX:0794636"),
    ("left level of angle of the mandible on the vagus nerve", "ILX:0794637"),
    ("level of carotid bifurcation on the vagus nerve", "ILX:0794638"),
    ("right level of carotid bifurcation on the vagus nerve", "ILX:0794639"),
    ("left level of carotid bifurcation on the vagus nerve", "ILX:0794640"),
    ("level of superior border of the clavicle on the vagus nerve", "ILX:0794641"),
    ("right level of superior border of the clavicle on the vagus nerve", "ILX:0794642"),
    ("left level of superior border of the clavicle on the vagus nerve", "ILX:0794643"),
    ("level of jugular notch on the vagus nerve", "ILX:0794644"),
    ("right level of jugular notch on the vagus nerve", "ILX:0794645"),
    ("left level of jugular notch on the vagus nerve", "ILX:0794646"),
    # ("level of carina", ""),
    # ("right level of carina", ""),
    # ("left level of carina", ""),
    ("level of sternal angle on the vagus nerve", "ILX:0794647"),
    ("right level of sternal angle on the vagus nerve", "ILX:0794648"),
    ("left level of sternal angle on the vagus nerve", "ILX:0794649"),
    ("level of 1 cm superior to start of esophageal plexus on the vagus nerve", "ILX:0794650"),
    ("right level of 1 cm superior to start of esophageal plexus on the vagus nerve", "ILX:0794651"),
    ("left level of 1 cm superior to start of esophageal plexus on the vagus nerve", "ILX:0794652"),
    ("level of esophageal hiatus on the vagus nerve", "ILX:0794653"),
    ("right level of esophageal hiatus on the vagus nerve", "ILX:0794654"),
    ("left level of esophageal hiatus on the vagus nerve", "ILX:0794655"),
    ("level of aortic hiatus on the vagus nerve", "ILX:0794656"),
    ("right level of aortic hiatus on the vagus nerve", "ILX:0794657"),
    ("left level of aortic hiatus on the vagus nerve", "ILX:0794658"),
]

vagus_branch_terms = [
    # branch names, based on lit.review
    # branch names with no indication of the side
    ("cervical trunk", "ILX:0789914"),
    ("thoracic trunk", "ILX:0784729"),
    ("meningeal branch of vagus nerve", "ILX:0793828", "FMA:6231"),
    # following listed as "branch between vagus nerve and glossopharyngeal nerve"
    ("communicating branch of vagus nerve with glossopharyngeal nerve", "ILX:0793830", "FMA:6233"),
    ("auricular branch of vagus nerve", "ILX:0793829", "FMA:6232"),
    ("pharyngeal branch of vagus nerve", "UBERON:0000929", "ILX:0725793"),
    ("lingual branch of vagus nerve", "ILX:0791434", "FMA:6235"),
    ("branch of vagus nerve to carotid body", "ILX:0786059", "FMA:6237"),
    # following not found on interlex
    # ("communicating branch of superior cervical ganglion with vagus nerve", "FMA:6901"),
    ("superior laryngeal nerve", "UBERON:0011326", "ILX:0731053"),
    ("internal branch of superior laryngeal nerve", "ILX:0793561", "internal laryngeal nerve", "internal superior laryngeal nerve", "FMA:6240"),
    ("external branch of superior laryngeal nerve", "ILX:0793560", "FMA:6243"),
    ("communicating branch of internal laryngeal nerve with recurrent laryngeal nerve", "ILX:0787299", "FMA:53544"),
    ("communicating branch of external laryngeal nerve with superior cardiac nerve", "ILX:0786237", "FMA:6708"),
    ("pulmonary branch of vagus nerve", "ILX:0789095", "FMA:65515"),
    ("bronchial branch of vagus nerve", "ILX:0786619", "FMA:6247"),
    ("superior cervical cardiac branch of vagus nerve", "ILX:0784874", "FMA:6244"),
    ("recurrent laryngeal nerve", "UBERON:0003716", "ILX:0486534", "inferior laryngeal nerve"),
    ("extra-laryngeal branch of recurrent laryngeal nerve to larynx", "ILX:0792328", "FMA:6710"),
    ("esophageal branch of recurrent laryngeal nerve", "ILX:0787303", "FMA:6248"),
    ("tracheal branch of recurrent laryngeal nerve", "ILX:0793070", "FMA:6249"),
    ("communicating branch of recurrent laryngeal nerve with internal laryngeal nerve", "ILX:0786150", "FMA:53526"),
    ("inferior cervical cardiac branch of vagus nerve", "ILX:0788259", "FMA:6245"),
    ("thoracic cardiac branch of vagus nerve", "ILX:0794849"),
    ("esophageal vagus trunk", "ILX:0794853", "esophageal trunk"),
    ("cardiac branch of the esophageal plexus of vagus nerve", "ILX:0794917", "cardiac branch of esophageal plexus of vagus nerve"),
    ("esophageal branch of esophageal plexus of vagus nerve", "ILX:0794344"),
    ("esophageal branches from esophageal plexus of vagus nerve", "ILX:0796116"),
    ("esophageal plexus cardiopulmonary branches", "ILX:0796105"),

    # right vagus branches
    ("right vagus nerve", "ILX:0789705", "FMA:6219"),
    ("right vagus X nerve trunk", "UBERON:0035021", "ILX:0730515"),
    ("right cervical vagus nerve", "ILX:0794141"),
    ("right thoracic vagus nerve", "ILX:0786664"),
    ("right meningeal branch of right vagus nerve", "ILX:0785804", "FMA:53541"),
    ("right branch between vagus nerve and glossopharyngeal nerve", "ILX:0790506", "FMA:53559"),
    ("right auricular branch of right vagus nerve", "ILX:0785879", "FMA:53534"),
    ("communicating branch of auricular branch of right vagus nerve with right facial nerve", "ILX:0791102", "FMA:53587"),
    # following listed as "right branch between auricular branch and facial nerve"
    ("communicating branch of auricular branch of right vagus nerve with right posterior auricular nerve", "ILX:0786954", "FMA:53589"),
    ("right pharyngeal branch of right vagus nerve to pharyngeal nerve plexus", "ILX:0792814", "FMA:53635"),
    ("lingual branch of right vagus nerve", "ILX:0787083", "FMA:53633"),
    ("right pharyngeal branch of right vagus nerve to superior cervical ganglion", "ILX:0795066"),
    # following listed as "right branch of vagus nerve to ipsilateral carotid body"
    ("branch of right vagus nerve to carotid body", "ILX:0790433", "FMA:53606"),
    ("right branch between vagus nerve and superior cervical ganglion", "ILX:0794055"),
    ("right superior laryngeal nerve", "ILX:0787738", "FMA:53530"),
    # following listed as "Right internal branch of superior laryngeal nerve"
    ("right internal laryngeal nerve", "ILX:0788164", "FMA:53539"),
    # following listed as "Right external branch of superior laryngeal nerve"
    ("right external laryngeal nerve", "ILX:0792879", "FMA:53537"),
    # following listed as "Upper branch of right internal laryngeal nerve to laryngeal vestibule"
    ("superior branch of right internal laryngeal nerve", "ILX:0784524", "FMA:53575"),
    # nearest to following: "middle branch of internal branch of right superior laryngeal nerve", "ILX:0795865"
    # ("middle branch of right internal laryngeal nerve", ""),  # NA pending
    # following listed as "Lower branch of right internal laryngeal nerve to right aryepiglottic fold"
    ("inferior branch of right internal laryngeal nerve", "ILX:0789165", "FMA:53581"),
    ("communicating branch of right internal laryngeal nerve with right recurrent laryngeal nerve", "ILX:0791006", "FMA:53571"),
    ("communicating branch of right external laryngeal nerve with right superior cardiac nerve", "ILX:0787405", "FMA:53561"),
    ("right branch of cervical vagus nerve to sympathetic chain, cardiovascular, and pulmonary structures", "ILX:0796411"),
    ("right cardiovascular branch of cranial nerve bundle", "ILX:0795539"),
    ("right cardiovascular branch of cervical vagus nerve", "ILX:0794846"),
    ("right cardiovascular branch of thoracic vagus nerve", "ILX:0794851"),
    ("right cervical cardiopulmonary branch of vagus nerve", "ILX:0794153"),
    ("right thoracic cardiopulmonary branch of vagus nerve", "ILX:0794180", "right thoracic cardiopulmonary branch"),
    ("right pulmonary branch of vagus nerve", "ILX:0796017", "right pulmonary branch"),
    ("right lateral pulmonary branch of vagus nerve", "ILX:0794277"),
    ("right medial pulmonary branch of vagus nerve", "ILX:0794254"),
    ("bronchial branch of right vagus nerve", "ILX:0791364", "FMA:53613",),
    ("superior cervical cardiac branch of right vagus nerve", "ILX:0786396", "FMA:53598"),
    ("right recurrent laryngeal nerve", "UBERON:0011767", "ILX:0728322"),
    ("extra laryngeal branch of right recurrent laryngeal nerve to larynx", "ILX:0795068"),
    ("branch of right recurrent laryngeal nerve to muscle of larynx", "ILX:0795084"),
    ("esophageal branch of right recurrent laryngeal nerve", "ILX:0785794", "FMA:53608"),
    ("tracheal branch of right recurrent laryngeal nerve", "ILX:0792888", "FMA:53610"),
    # following listed as "right recurrent laryngeal nerve"
    ("right inferior laryngeal nerve", "UBERON:0011767"),  # same code as for left recurrent laryngeal nerve
    # ("external branch of right inferior laryngeal nerve", ""),  # NA removed
    ("anterior branch of right recurrent laryngeal nerve", "ILX:0795070"),
    ("posterior branch of right recurrent laryngeal nerve", "ILX:0795072"),
    # ("communicating branch of right recurrent laryngeal nerve with superior cervical ganglion", ""),  # NA rejected
    ("communicating branch of right recurrent laryngeal nerve with right internal laryngeal nerve", "ILX:0790555", "FMA:53528"),
    # following listed as "right cardiac branch of recurrent laryngeal nerve"
    ("inferior cervical cardiac branch of right recurrent laryngeal nerve", "ILX:0794235"),
    # following listed as "Inferior cervical cardiac branch of right vagus nerve to deep cardiac plexus"
    ("inferior cervical cardiac branch of right vagus nerve", "ILX:0789456", "FMA:6713"),
    ("thoracic cardiac branch of right vagus nerve", "ILX:0790057", "FMA:53604"),
    ("cardiac branch of right vagus to deep cardiac plexus", "ILX:0791784", "FMA:6711"),
    ("right branch of thoracic vagus nerve to esophagus", "ILX:0794299"),
    ("right branch between cranial nerve bundle and superior cervical ganglion", "ILX:0796398"),
    ("right branch between thoracic vagus nerve and 'branch between thoracic vagus nerve and thoracic vagus nerve'", "ILX:0794837"),
    ("right branch between of thoracic vagus nerve and esophagus plexus", "ILX:0796071", "right branch of thoracic vagus nerve to esophageal plexus"),
    ("right branch between vagus nerve and C2 spinal nerve", "ILX:0793977"),
    ("right branch between vagus nerve and cervical spinal nerves", "ILX:0795644"),
    ("right branch between vagus nerve and cervical sympathetic trunk", "ILX:0794775", "right branch to cervical sympathetic trunk"),
    ("right branch between cervical vagus nerve and a 'branch between cervical vagus nerve and cervical vagus nerve'", "ILX:0794821"),
    ("right branch between cervical vagus and inferior root of ansa cervicalis", "ILX:0795597"),
    ("right branch of inferior root of ansa cervicalis", "ILX:0796313"),
    ("right branch of vagus nerve to ipsilateral carotid bifurcation", "ILX:0794089", "right branch to carotid bifurcation"),
    ("right branch of vagus nerve to ipsilateral carotid sinus", "ILX:0794108"),
    ("right branch of vagus nerve to ipsilateral common carotid artery", "ILX:0794068", "right branch to common carotid artery"),
    ("right branch of vagus nerve to ipsilateral internal carotid artery", "ILX:0794086"),
    ("right branch of vagus nerve to ipsilateral internal jugular vein", "ILX:0794077"),
    ("right branch of vagus nerve to trachea", "ILX:0794756"),
    ("cardiovascular and cardiopulmonary branch of right cervical vagus nerve", "ILX:0796422", "right cardiovascular and cardiopulmonary branch of cervical vagus nerve"),
    ("right cardiovascular branch group of cervical vagus nerve", "ILX:0795379"),
    ("right cardiovascular branch group of thoracic vagus nerve", "ILX:0795412"),
    ("right cervical cardiac branch of vagus nerve", "ILX:0794165", "right cervical cardiac branch"),
    ("right medial pulmonary branch of the esophageal plexus", "ILX:0796554"),
    ("right pharyngeal branch of cranial nerve bundle", "ILX:0795526"),
    ("right pharyngeal branch of vagus nerve", "ILX:0793987"),
    ("right unknown branch of vagus nerve", "ILX:0793923"),
    ("branch between right thoracic vagus trunk and tracheo-esophageal plexus", "ILX:0796658"),

    # posterior vagus, a continuation of right vagus
    ("posterior esophageal vagus trunk", "ILX:0794858"),
    ("celiac branch of posterior vagal trunk", "ILX:0789580", "FMA:6667"),
    ("greater posterior gastric nerve", "ILX:0788809", "FMA:6689"),
    # following not on interlex:
    # ("pyloric branch of greater posterior gastric nerve", "FMA:6677"),
    ("left lateral pulmonary branch of the esophageal plexus", "ILX:0797435"),
    ("posterior esophageal communicating branch", "ILX:0789876", "FMA:53632"),

    # left vagus branches
    ("left vagus nerve", "ILX:0785628", "FMA:6220"),
    ("left vagus X nerve trunk", "UBERON:0035020", "ILX:0736691"),
    ("left cervical vagus nerve", "ILX:0794142"),
    ("left thoracic vagus nerve", "ILX:0787543", "FMA:18174"),
    ("left thoracic cardiac branch of vagus nerve", "ILX:0794213", "left thoracic cardiac branch"),
    ("left thoracic cardiopulmonary branch of vagus nerve", "ILX:0794191", "left thoracic cardiopulmonary branch"),
    ("left meningeal branch of left vagus nerve", "ILX:0792358", "FMA:53542"),
    ("left branch between vagus nerve and glossopharyngeal nerve", "ILX:0790685", "FMA:53560"),
    ("left auricular branch of left vagus nerve", "ILX:0789344", "FMA:53535"),
    ("communicating branch of auricular branch of left vagus nerve with left facial nerve", "ILX:0787673", "FMA:53588"),
    ("left branch between cranial nerve bundle and superior cervical ganglion", "ILX:0796404"),
    ("left cranial nerve bundle to cervical spinal nerve", "ILX:0795553"),
    # following listed as "left branch between auricular branch and facial nerve"
    ("communicating branch of auricular branch of left vagus nerve with left posterior auricular nerve", "ILX:0786219", "FMA:53590"),
    ("left pharyngeal branch of left vagus nerve to pharyngeal nerve plexus", "ILX:0789210", "FMA:53636"),
    ("lingual branch of left vagus nerve", "ILX:0791548", "FMA:53634"),
    ("left pharyngeal branch of left vagus nerve to superior cervical ganglion", "ILX:0795067"),
    # following listed as "left branch of vagus nerve to ipsilateral carotid body"
    ("branch of left vagus nerve to carotid body", "ILX:0791085", "FMA:53607"),
    ("left branch between vagus nerve and superior cervical ganglion", "ILX:0794061"),
    ("left superior laryngeal nerve", "ILX:0788780", "FMA:53536"),
    # following listed as "Left internal branch of superior laryngeal nerve"
    ("left internal laryngeal nerve", "ILX:0791167", "FMA:53540"),
    # following listed as "Left external branch of superior laryngeal nerve"
    ("left external laryngeal nerve", "ILX:0789760", "FMA:53538"),
    ("left branch of superior laryngeal nerve", "ILX:0795822"),
    # following listed as "Upper branch of left internal laryngeal nerve to laryngeal vestibule"
    ("superior branch of left internal laryngeal nerve", "ILX:0785786", "FMA:53576"),
    # ("middle branch of left internal laryngeal nerve", ""),  # NA pending
    # following listed as "Lower branch of left internal laryngeal nerve to left aryepiglottic fold"
    ("inferior branch of left internal laryngeal nerve", "ILX:0785467", "FMA:53582"),
    ("communicating branch of left internal laryngeal nerve with left recurrent laryngeal nerve", "ILX:0789900", "FMA:53572"),
    ("communicating branch of left external laryngeal nerve with left superior cardiac nerve", "ILX:0787107", "FMA:53562"),
    ("left cardiovascular branch of cranial nerve bundle", "ILX:0795540"),
    ("left cardiovascular branch of cervical vagus nerve", "ILX:0794847"),
    ("left cardiovascular branch of thoracic vagus nerve", "ILX:0794852"),
    ("left cervical cardiopulmonary branch of vagus nerve", "ILX:0794159"),
    ("left lateral pulmonary branch of vagus nerve", "ILX:0794288"),
    ("left medial pulmonary branch of vagus nerve", "ILX:0794265"),
    ("bronchial branch of left vagus nerve", "ILX:0791685", "FMA:53558"),
    ("superior cervical cardiac branch of left vagus nerve", "ILX:0791998", "FMA:53599"),
    ("left recurrent laryngeal nerve", "UBERON:0011766", "ILX:0724431"),
    ("extra laryngeal branch of left recurrent laryngeal nerve to larynx", "ILX:0795069"),
    ("branch of left recurrent laryngeal nerve to muscle of larynx", "ILX:0795085"),
    ("esophageal branch of left recurrent laryngeal nerve", "ILX:0791513", "FMA:53609"),
    ("tracheal branch of left recurrent laryngeal nerve", "ILX:0785091", "FMA:53611"),
    ("left inferior laryngeal nerve", "UBERON:0011766"),
    # ("external branch of left inferior laryngeal nerve", ""),  # NA removed
    ("anterior branch of left recurrent laryngeal nerve", "ILX:0795071"),
    ("posterior branch of left recurrent laryngeal nerve", "ILX:0795073"),
    # ("communicating branch of left recurrent laryngeal nerve with superior cervical ganglion", ""),  # NA rejected
    ("communicating branch of left recurrent laryngeal nerve with left internal laryngeal nerve", "ILX:0790440", "FMA:53529"),
    ("inferior cervical cardiac branch of left recurrent laryngeal nerve", "ILX:0794244"),
    # following listed as "Inferior cervical cardiac branch of left vagus nerve to superficial cardiac plexus"
    ("inferior cervical cardiac branch of left vagus nerve", "ILX:0786047", "FMA:6714"),
    ("thoracic cardiac branch of left vagus nerve", "ILX:0791489", "FMA:53605"),
    ("cardiac branch of left vagus to deep cardiac plexus", "ILX:0795087"),
    ("left branch between thoracic vagus nerve and esophagus plexus", "ILX:0796087", "left branch of thoracic vagus nerve to esophageal plexus"),
    ("left branch between recurrent laryngeal nerve and vagal trunk", "ILX:0794231"),
    ("left branch between vagus nerve and carotid sinus nerve", "ILX:0794124"),
    ("left branch between vagus nerve and cervical sympathetic trunk", "ILX:0794776"),
    ("left branch between vagus nerve and cervical sympathetic trunk and or superior cervical ganglion", "ILX:0794048"),
    ("left branch between vagus nerve and hypoglossal nerve", "ILX:0793914"),
    ("left branch between vagus nerve and spinal accessory nerve", "ILX:0793960"),
    ("left branch between vagus nerve and superior root of ansa cervicalis", "ILX:0793951"),
    ("left branch of thoracic vagus nerve to aorta", "ILX:0796495"),
    ("left branch of thoracic vagus nerve to esophagus", "ILX:0794310"),
    ("left branch of vagus nerve to ipsilateral carotid bifurcation", "ILX:0794094"),
    ("left branch of vagus nerve to ipsilateral common carotid artery", "ILX:0794072", "left branch to common carotid artery"),
    ("left branch of vagus nerve to ipsilateral internal carotid artery", "ILX:0794087"),
    ("left branch of vagus nerve to ipsilateral internal jugular vein", "ILX:0794081"),
    ("left cardiovascular branch group of cervical vagus nerve", "ILX:0795395"),
    ("left cardiovascular branch group of thoracic vagus nerve", "ILX:0795428"),
    ("left cervical cardiac branch of vagus nerve", "ILX:0794171", "left cervical cardiac branch"),
    ("left pulmonary branch of vagus nerve", "ILX:0796033", "left pulmonary branch"),

    # anterior vagus, a continuation of left vagus
    ("anterior esophageal vagus trunk", "ILX:0794854"),
    ("anterior gastric branch of vagus nerve", "ILX:0787035", "FMA:272326"),
    ("hepatic branch of anterior vagal trunk", "ILX:0784595", "FMA:6666"),
    ("greater anterior gastric nerve", "ILX:0793831", "FMA:6684"),
    # not on interlex:
    # ("branch of greater anterior gastric nerve to coeliac nerve plexus", "FMA:53675"),
    ("anterior esophageal communicating branch", "ILX:0792199", "FMA:53631"),

    # vagus built-in annotations
    ("vagus centroid", ""),
    ("epineurium", "ILX:0103892"),
    ("orientation anterior", "")  # line on the epineurium in anterior direction
]


def get_vagus_term(name):
    """
    Find a vagus term by matching name to any identifier held for a term.
    Note: Use separate get_vagus_marker_term for all level marker terms.
    :param name: Any name or ID to match against known terms.
    :return: ( preferred name, preferred id )
    """
    for term in vagus_branch_terms:
        if name in term:
            return annotation_term_id_to_url((term[0], term[1]))
    logger.warning("Unknown vagus term name or ID: '" + name + "'. Using as name without ID")
    return name, ""


def get_vagus_marker_term(name: str):
    """
    Find term by matching name to any identifier held for a term.
    Raise exception if name not found.
    return: ( preferred name, preferred id )
    """
    for term in vagus_marker_terms:
        if name in term:
            return annotation_term_id_to_url((term[0], term[1]))
    raise NameError("Vagus annotation term '" + name + "' not found.")


def marker_name_in_terms(name: str):
    """
    Check if term exists in approved marker terms
    """
    for term in vagus_marker_terms:
        if name in term:
            return True
    return False


def get_left_vagus_marker_locations_list():
    # vagus markers location in material coordinates between 0 to 1
    left_termNameVagusLengthList = {
        # cervical region
        "left level of superior border of jugular foramen on the vagus nerve": 0.044,
        "left level of inferior border of jugular foramen on the vagus nerve": 0.06, # standardized
        # "left level of inferior border of cranium on the vagus nerve": 0.072,
        # "left level of C1 transverse process on the vagus nerve": 0.111,
        "left level of angle of the mandible on the vagus nerve": 0.131,
        # "left level of greater horn of hyoid on the vagus nerve": 0.149,
        "left level of carotid bifurcation on the vagus nerve": 0.166, # calibrated to Feinstein REVA data
        "left level of laryngeal prominence on the vagus nerve": 0.214, # calibrated to Feinstein REVA data
        # thoracic region
        "left level of superior border of the clavicle on the vagus nerve": 0.320, # standardized
        "left level of jugular notch on the vagus nerve": 0.356,
        "left level of sternal angle on the vagus nerve": 0.447,
        # "left level of 1 cm superior to start of esophageal plexus on the vagus nerve": 0.487,
        # abdominal region
        # "left level of esophageal hiatus on the vagus nerve": 0.738,
        # "left level of aortic hiatus on the vagus nerve": 0.842,
        # "left level of end of trunk": 0.902, # note this term is also not on the list of annotations
    }
    return left_termNameVagusLengthList


def get_right_vagus_marker_locations_list():
    # vagus markers location in material coordinates between 0 to 1
    right_termNameVagusLengthList = {
        # cervical region
        "right level of superior border of jugular foramen on the vagus nerve": 0.044,
        "right level of inferior border of jugular foramen on the vagus nerve": 0.06, # standardized
        # "right level of inferior border of cranium on the vagus nerve": 0.072,
        # "right level of C1 transverse process on the vagus nerve": 0.111,
        "right level of angle of the mandible on the vagus nerve": 0.131,
        # "right level of greater horn of hyoid on the vagus nerve": 0.149,
        "right level of carotid bifurcation on the vagus nerve": 0.166, # calibrated to Feinstein REVA data
        "right level of laryngeal prominence on the vagus nerve": 0.214, # calibrated to Feinstein REVA data
        # thoracic region
        "right level of superior border of the clavicle on the vagus nerve": 0.320, # standardized
        "right level of jugular notch on the vagus nerve": 0.356,
        "right level of sternal angle on the vagus nerve": 0.447,
        # "right level of 1 cm superior to start of esophageal plexus on the vagus nerve": 0.487,
        # abdominal region
        # "right level of esophageal hiatus on the vagus nerve": 0.738,
        # "right level of aortic hiatus on the vagus nerve": 0.842,
    }
    return right_termNameVagusLengthList


def is_bony_landmark(marker_name):
    """
    Checks if supplied marker_name is a bony landmark.
    Not used currently.
    """
    bony_landmarks_names = [
        "level of superior border of jugular foramen on the vagus nerve",
        "level of inferior border of jugular foramen on the vagus nerve",
        "level of inferior border of cranium on the vagus nerve",
        "level of C1 transverse process on the vagus nerve",
        "level of angle of the mandible on the vagus nerve",
        "level of greater horn of hyoid on the vagus nerve",
        "level of superior border of the clavicle on the vagus nerve",
        "level of jugular notch on the vagus nerve",
        "level of sternal angle on the vagus nerve"
    ]
    if marker_name in bony_landmarks_names:
        return True
    return False
