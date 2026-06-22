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
    # Finger parts
    ("proximal phalanx of little finger", ""),
    ("middle phalanx of little finger", ""),
    ("distal phalanx of little finger", ""),
    ("proximal phalanx of ring finger", ""),
    ("middle phalanx of ring finger", ""),
    ("distal phalanx of ring finger", ""),
    ("proximal phalanx of middle finger", ""),
    ("middle phalanx of middle finger", ""),
    ("distal phalanx of middle finger", ""),
    ("proximal phalanx of index finger", ""),
    ("middle phalanx of index finger", ""),
    ("distal phalanx of index finger", ""),
    ("proximal phalanx of thumb", ""),
    ("middle phalanx of thumb", ""),
    ("distal phalanx of thumb", ""),

    ("skin of proximal phalanx of little finger", ""),
    ("skin of middle phalanx of little finger", ""),
    ("skin of distal phalanx of little finger", ""),
    ("skin of proximal phalanx of ring finger", ""),
    ("skin of middle phalanx of ring finger", ""),
    ("skin of distal phalanx of ring finger", ""),
    ("skin of proximal phalanx of middle finger", ""),
    ("skin of middle phalanx of middle finger", ""),
    ("skin of distal phalanx of middle finger", ""),
    ("skin of proximal phalanx of index finger", ""),
    ("skin of middle phalanx of index finger", ""),
    ("skin of distal phalanx of index finger", ""),
    ("skin of proximal phalanx of thumb", ""),
    ("skin of middle phalanx of thumb", ""),
    ("skin of distal phalanx of thumb", ""),
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
