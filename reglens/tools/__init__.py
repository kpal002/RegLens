"""Deterministic tool layer for RegLens.

Every tool in this subpackage computes numbers — never opinions. Agents added
later reason *over* these outputs but must never invent a score. For the
Wednesday milestone only :mod:`reglens.tools.chrombpnet_score` exists; the
motif-effect, regulatory-context, gene-target, trait-link and literature tools
land on Thursday (see ``RegLens_spec.md`` §3).
"""
