import regex

# Devanagari (Hindi, Sanskrit)
dev_cons = r'[\u0915-\u0939\u0958-\u095F][\u093C]?'
dev_halant = r'\u094D'
dev_matra = r'[\u093E-\u094C\u0962-\u0963]'
dev_mod = r'[\u0901-\u0903]'
dev_ind_vowel = r'[\u0904-\u0914\u0960-\u0961]'

dev_akshara = f'(?:(?:{dev_cons}{dev_halant})*{dev_cons}{dev_matra}?{dev_mod}?|{dev_ind_vowel}{dev_mod}?)'

# Telugu
tel_cons = r'[\u0C15-\u0C39\u0C58-\u0C59]'
tel_halant = r'\u0C4D'
tel_matra = r'[\u0C3E-\u0C4C\u0C55-\u0C56\u0C62-\u0C63]'
tel_mod = r'[\u0C01-\u0C03]'
tel_ind_vowel = r'[\u0C05-\u0C14\u0C60-\u0C61]'

tel_akshara = f'(?:(?:{tel_cons}{tel_halant})*{tel_cons}{tel_matra}?{tel_mod}?|{tel_ind_vowel}{tel_mod}?)'

# Combined pattern for Aksharas
# We also want to match any single character that is not an Akshara (like punctuation, spaces, Latin chars)
# so that the whole string is fully partitioned.
akshara_pattern = regex.compile(f'({dev_akshara}|{tel_akshara}|.)', regex.DOTALL)

def split_into_aksharas(text: str) -> tuple:
    """
    Splits a string into a sequence of Aksharas (for Indic scripts) and individual characters (for others).
    This implements Constrained BPE (CBPE) initialization.
    """
    return tuple(akshara_pattern.findall(text))

if __name__ == "__main__":
    print(split_into_aksharas("नमस्ते दुनिया"))
    print(split_into_aksharas("హలో ప్రపంచం"))
