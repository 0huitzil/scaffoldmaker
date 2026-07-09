"""
Common resource for body annotation terms.
"""

# convention: preferred name, preferred id, followed by any other ids and alternative names
body_terms = [
    ("abdomen", "UBERON:0000916", "ILX:0725977"),
    ("abdominal cavity", "UBERON:0003684"),
    ("abdominal cavity boundary surface", "ILX:0796509"),
    ("abdominopelvic cavity", "UBERON:0035819"),
    ("upper limb", "UBERON:0001460"),
    ("left upper limb", "UBERON:8300002", "FMA:7186"),
    ("left shoulder", ""),
    ("left brachium", ""),
    ("left elbow", ""),
    ("left antebrachium", ""),
    ("left hand", ""),
    ("left upper limb skin epidermis outer surface", "ILX:0796504"),
    ("right upper limb", "UBERON:8300001", "FMA:7185"),
    ("right shoulder", ""),
    ("right brachium", ""),
    ("right elbow", ""),
    ("right antebrachium", ""),
    ("right hand", ""),
    ("right upper limb skin epidermis outer surface", "ILX:0796503"),
    ("body", "UBERON:0000468", "ILX:0101370"),
    ("core", ""),
    ("core boundary", ""),
    ("dorsal", ""),
    ("head", "UBERON:0000033", "ILX:0104909"),
    ("head core", ""),
    ("diaphragm", "UBERON:0001103", "ILX:0103194"),
    ("hand", "ILX:0104885", "FMA:9712"),
    ("hip", ""),
    ("left", ""),
    ("lower limb", "UBERON:0000978"),
    ("left lower limb", "UBERON:8300004", "FMA:24981"),
    ("left upper leg", ""),
    ("left lower leg", ""),
    ("left foot", ""),
    ("left lower limb skin epidermis outer surface", "ILX:0796506"),
    ("right lower limb", "UBERON:8300003", "FMA:24980"),
    ("right upper leg", ""),
    ("right lower leg", ""),
    ("right foot", ""),
    ("right lower limb skin epidermis outer surface", "ILX:0796505"),
    ("foot", "ILX:0745450", "FMA:9664"),
    ("neck", "UBERON:0000974", "ILX:0733967"),
    ("neck core", ""),
    ("non core", ""),
    ("right", ""),
    ("shell", ""),
    ("skin epidermis outer surface", "UBERON:0001003", "ILX:0796507"),
    ("spinal cord", "UBERON:0002240", "ILX:0110909"),
    ("thoracic cavity", "UBERON:0002224"),
    ("thoracic cavity boundary surface", "ILX:0796508"),
    ("thorax", "ILX:0742178"),
    ("ventral", "")
    ]

def get_body_term(name : str):
    """
    Find term by matching name to any identifier held for a term.
    Raise exception if name not found.
    :return ( preferred name, preferred id )
    """
    for term in body_terms:
        if name in term:
            return ( term[0], term[1] )
    raise NameError("Body annotation term '" + name + "' not found.")
