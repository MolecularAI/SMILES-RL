import re


class SMILESTokenizer:
    """Deals with the tokenization and untokenization of SMILES."""

    REGEXPS = {
        "brackets": re.compile(r"(\[[^\]]*\])"),
        "2_ring_nums": re.compile(r"(%\d{2})"),
        "brcl": re.compile(r"(Br|Cl)"),
    }
    REGEXP_ORDER = ["brackets", "2_ring_nums", "brcl"]

    def tokenize(self, data, with_begin_and_end=True):
        """Tokenizes a SMILES string."""

        def split_by(data, regexps):
            if not regexps:
                return list(data)
            regexp = self.REGEXPS[regexps[0]]
            splitted = regexp.split(data)
            tokens = []
            for i, split in enumerate(splitted):
                if i % 2 == 0:
                    tokens += split_by(split, regexps[1:])
                else:
                    tokens.append(split)
            return tokens

        tokens = split_by(data, self.REGEXP_ORDER)
        if with_begin_and_end:
            tokens = ["^"] + tokens + ["$"]
        return tokens

    def untokenize(self, tokens):
        """Untokenizes a SMILES string."""
        smi = ""
        for token in tokens:
            if token == "$":
                break
            if token != "^":
                smi += token
        return smi
