from __future__ import annotations

import scriptconfig as scfg


class DraftCompositionCLI(scfg.DataConfig):
    """Build K-draft mixture contexts (T correct, K - T wrong) in several orders."""
    __command__ = 'draft-composition'

    input_dir = scfg.Value(None, required=True,
                           help='Processed clean-slate pool (.ds).')
    output_dir = scfg.Value(None, required=True, help='Output directory.')
    K = scfg.Value(4, type=int, help='Drafts per context.')
    n_orders = scfg.Value(8, type=int, help='Random orders per draft set.')
    min_correct = scfg.Value(4, type=int,
                             help='Keep a problem only with >= this many correct pool answers.')
    min_incorrect = scfg.Value(4, type=int,
                               help='Keep a problem only with >= this many incorrect pool answers.')
    seed = scfg.Value(20260916, type=int, help='Composition seed.')
    max_draft_chars = scfg.Value(0, type=int,
                                 help='Truncate each draft to this many characters (0 = never).')

    @classmethod
    def main(cls, argv=True, **kwargs):
        args = cls.cli(argv=argv, data=kwargs, strict=True, special_options=False)
        from contextual_drag.data import draft_composition
        return draft_composition.main(args)
