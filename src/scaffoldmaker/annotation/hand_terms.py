"""
Common resource for hand annotation terms.
"""

# convention: preferred name, preferred id, followed by any other ids and alternative names
hand_terms = [
    ("hand", ""),
    ("skin of hand", ""),
    ("palm", ""),
    ("carpal", ""),
    ("metacarpal", ""),
    ("proximal phalanx", ""),
    ("middle phalanx", ""),
    ("distal phalanx", ""),
    ("thumb", ""),
    ("skin of thumb", ""),
    ("index finger", ""),
    ("skin of index finger", ""),
    ("middle finger", ""),
    ("skin of middle finger", ""),
    ("ring finger", ""),
    ("skin of ring finger", ""),
    ("little finger", ""),
    ("skin of little finger", ""),
    ]

def get_hand_term(name : str):
    """
    Find term by matching name to any identifier held for a term.
    Raise exception if name not found.
    :return ( preferred name, preferred id )
    """
    for term in hand_terms:
        if name in term:
            return ( term[0], term[1] )
    raise NameError("Hand annotation term '" + name + "' not found.")
