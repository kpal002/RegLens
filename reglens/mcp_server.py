"""MCP stdio server exposing RegLens's deterministic tool layer.

This wraps the existing :mod:`reglens.tools` functions as `Model Context Protocol
<https://modelcontextprotocol.io>`_ tools so any MCP host (Claude Desktop, etc.) can
call the deterministic layer directly. Every tool here is a **thin wrapper** — it does
no interpretation and computes no new numbers; it forwards to the same
``reglens.tools.*`` functions the CLI and orchestrator use, and serializes results
through the canonical :meth:`reglens.report.schema.EvidenceBundle.to_dict` schema.

The primary, documented interface is :func:`get_evidence_bundle` — one call that
gathers every signal for a ``(variant, celltype)`` into the same bundle the reasoning
layer consumes. The six single-tool wrappers (``score_variant``, ``motif_effect``,
``regulatory_context``, ``gene_target``, ``trait_link``, ``literature``) are exposed for
targeted use.

Configuration (environment variables, read at call time):
    ``REGLENS_GENOME``: path to an hg38 FASTA. Required for ``score_variant`` and
        ``motif_effect`` (they need the sequence windows). The annotation tools
        (regulatory / gene / trait / literature) work without it.
    ``REGLENS_MODEL``: path to a pretrained ChromBPNet model file or fold directory.
        If unset, ``score_variant`` falls back to the offline stub backend, and the
        result's ``model`` field is labelled ``"stub(offline)"`` so it is never
        mistaken for a real biological score.

The server always starts and always serves the annotation tools; ``score_variant`` and
``motif_effect`` return a clear, actionable error object when ``REGLENS_GENOME`` is
unset rather than raising a cryptic genome-loading error.

Run it with ``python -m reglens.mcp_server`` (or the ``reglens-mcp`` console script).
The MCP SDK is an optional dependency — ``pip install -e ".[mcp]"``; it is imported
lazily inside :func:`build_server` / :func:`main` so importing this module (and the
tool registry) never requires it.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reglens.genome import (
    DEFAULT_WINDOW_LENGTH,
    GENOME_ENV_VAR,
    Variant,
    build_sequence_windows,
)
from reglens.orchestrator import analyze_variant
from reglens.report.schema import EvidenceBundle
from reglens.tools._http import HttpClient
from reglens.tools.chrombpnet_score import ChromBPNetScorer, load_backend
from reglens.tools.gene_target import gene_target
from reglens.tools.literature import build_query, search_literature
from reglens.tools.motif_effect import Motif, motif_effect
from reglens.tools.regulatory_context import regulatory_context
from reglens.tools.trait_link import trait_link

#: Environment variable naming the ChromBPNet model (file or fold directory).
MODEL_ENV_VAR = "REGLENS_MODEL"

#: MCP server name advertised to hosts.
SERVER_NAME = "reglens"


# --------------------------------------------------------------------------------------
# Configuration helpers
# --------------------------------------------------------------------------------------
def _resolve_genome(genome_path: str | os.PathLike[str] | None) -> str | None:
    """Return the genome FASTA path, falling back to ``$REGLENS_GENOME``."""
    if genome_path is not None:
        return str(genome_path)
    return os.environ.get(GENOME_ENV_VAR)


def _make_scorer(window_length: int) -> ChromBPNetScorer:
    """Build a scorer from ``$REGLENS_MODEL`` (offline stub if it is unset)."""
    model_path = os.environ.get(MODEL_ENV_VAR)
    model_name = Path(model_path).name if model_path else "stub(offline)"
    return ChromBPNetScorer(
        load_backend(model_path), window_length=window_length, model_name=model_name
    )


def _genome_required_error(tool: str) -> dict[str, Any]:
    """A clear, actionable error object for genome-dependent tools when unconfigured."""
    return {
        "error": f"{tool} needs a reference genome, but ${GENOME_ENV_VAR} is not set.",
        "hint": (
            f"Set {GENOME_ENV_VAR} to an hg38 FASTA path (and optionally {MODEL_ENV_VAR} "
            "to a pretrained ChromBPNet model file or fold directory; without it an "
            "offline stub score is returned) and restart the MCP server. The annotation "
            "tools (regulatory_context, gene_target, trait_link, literature) work "
            "without a genome."
        ),
    }


# --------------------------------------------------------------------------------------
# Tool wrappers — thin adapters over reglens.tools.*, serialized via EvidenceBundle.
#
# Each accepts injectable dependencies (genome_path, scorer, client, window_length,
# motifs) so it can be driven offline in tests; production calls resolve them from the
# environment. None of these compute or interpret anything the underlying tool doesn't.
# --------------------------------------------------------------------------------------
def score_variant(
    variant: str,
    celltype: str | None = None,
    *,
    genome_path: str | os.PathLike[str] | None = None,
    scorer: ChromBPNetScorer | None = None,
    window_length: int = DEFAULT_WINDOW_LENGTH,
) -> dict[str, Any]:
    """Score a variant's ChromBPNet Δ log-counts (ref vs alt accessibility).

    Args:
        variant: Variant as ``chr:pos:ref>alt`` (hg38), e.g. ``"chr2:60490908:T>G"``.
        celltype: Cell-type / model context label (recorded on the result).
        genome_path: hg38 FASTA path; defaults to ``$REGLENS_GENOME``.
        scorer: A pre-built scorer (for tests); defaults to one from ``$REGLENS_MODEL``.
        window_length: Sequence window width in bp.

    Returns:
        The serialized ``chrombpnet`` sub-dict, or an actionable ``error`` object if
        no genome is configured.
    """
    genome = _resolve_genome(genome_path)
    if genome is None:
        return _genome_required_error("score_variant")
    parsed = Variant.parse(variant)
    scorer = scorer or _make_scorer(window_length)
    score = scorer.score_variant(parsed, genome_path=genome, celltype=celltype)
    return EvidenceBundle(variant=parsed, celltype=celltype, chrombpnet=score).to_dict()[
        "chrombpnet"
    ]


def motif_effect_tool(
    variant: str,
    *,
    genome_path: str | os.PathLike[str] | None = None,
    window_length: int = DEFAULT_WINDOW_LENGTH,
    motifs: list[Motif] | None = None,
) -> dict[str, Any]:
    """Identify the TF motif most disrupted or created by a variant (in-silico ISM).

    Args:
        variant: Variant as ``chr:pos:ref>alt`` (hg38).
        genome_path: hg38 FASTA path; defaults to ``$REGLENS_GENOME``.
        window_length: Sequence window width in bp for the scan.
        motifs: Motif library; defaults to the bundled JASPAR subset.

    Returns:
        The serialized ``motif`` sub-dict, ``{"motif": None}`` if no motif clears the
        binding threshold, or an actionable ``error`` object if no genome is configured.
    """
    genome = _resolve_genome(genome_path)
    if genome is None:
        return _genome_required_error("motif_effect")
    parsed = Variant.parse(variant)
    window = build_sequence_windows(parsed, genome_path=genome, window_length=window_length)
    result = motif_effect(window, parsed, motifs=motifs)
    return EvidenceBundle(variant=parsed, motif=result).to_dict().get("motif", {"motif": None})


def regulatory_context_tool(
    variant: str, *, client: HttpClient | None = None
) -> dict[str, Any]:
    """Assemble ENCODE cCRE / Ensembl regulatory-element context for a variant.

    Args:
        variant: Variant as ``chr:pos:ref>alt`` (hg38).
        client: HTTP client (for tests); defaults to the shared urllib client.

    Returns:
        The serialized ``regulatory`` sub-dict.
    """
    parsed = Variant.parse(variant)
    result = regulatory_context(parsed, client)
    return EvidenceBundle(variant=parsed, regulatory=result).to_dict()["regulatory"]


def gene_target_tool(
    variant: str, rsid: str | None = None, *, client: HttpClient | None = None
) -> dict[str, Any]:
    """Assemble nearest-gene and GTEx-eQTL evidence for a variant.

    Args:
        variant: Variant as ``chr:pos:ref>alt`` (hg38).
        rsid: dbSNP rsID; required for the GTEx eQTL lookup (skipped if absent).
        client: HTTP client (for tests); defaults to the shared urllib client.

    Returns:
        The serialized ``gene`` sub-dict.
    """
    parsed = Variant.parse(variant)
    result = gene_target(parsed, rsid=rsid, client=client)
    return EvidenceBundle(variant=parsed, gene=result).to_dict()["gene"]


def trait_link_tool(
    rsid: str, variant: str | None = None, *, client: HttpClient | None = None
) -> dict[str, Any]:
    """Retrieve GWAS Catalog trait associations for a variant by rsID.

    Args:
        rsid: dbSNP rsID (e.g. ``"rs1427407"``) — the GWAS lookup key.
        variant: Optional ``chr:pos:ref>alt`` (hg38); the lookup is rsID-keyed and does
            not need it — it is only carried into the bundle for report context if given.
        client: HTTP client (for tests); defaults to the shared urllib client.

    Returns:
        The serialized ``trait`` sub-dict.
    """
    result = trait_link(rsid, client=client)
    parsed = Variant.parse(variant) if variant else None
    return EvidenceBundle(variant=parsed, rsid=rsid, trait=result).to_dict()["trait"]


def literature_tool(
    variant: str,
    rsid: str | None = None,
    gene: str | None = None,
    trait: str | None = None,
    *,
    client: HttpClient | None = None,
) -> dict[str, Any]:
    """Search Europe PMC for real citations about the variant / gene / trait.

    Args:
        variant: Variant as ``chr:pos:ref>alt`` (hg38).
        rsid: dbSNP rsID to anchor the query.
        gene: Target gene symbol to include in the query.
        trait: Trait term to include in the query.
        client: HTTP client (for tests); defaults to the shared urllib client.

    Returns:
        The serialized ``literature`` sub-dict (query, hit count, structured citations).
    """
    parsed = Variant.parse(variant)
    query = build_query(variant=parsed, rsid=rsid, gene=gene, trait=trait)
    result = search_literature(query, client=client)
    return EvidenceBundle(variant=parsed, rsid=rsid, literature=result).to_dict()["literature"]


def get_evidence_bundle(
    variant: str,
    celltype: str | None = None,
    rsid: str | None = None,
    *,
    genome_path: str | os.PathLike[str] | None = None,
    scorer: ChromBPNetScorer | None = None,
    client: HttpClient | None = None,
    window_length: int = DEFAULT_WINDOW_LENGTH,
) -> dict[str, Any]:
    """Gather **all** deterministic evidence for a variant into one bundle.

    This is the primary interface: it runs the full deterministic layer (via
    :func:`reglens.orchestrator.analyze_variant`) and returns the same serialized
    bundle the reasoning layer consumes. Genome/model-dependent signals are included
    when ``$REGLENS_GENOME`` (and optionally ``$REGLENS_MODEL``) are set and skipped
    otherwise — per-tool failures are reported under the bundle's ``errors`` key rather
    than aborting the call.

    Args:
        variant: Variant as ``chr:pos:ref>alt`` (hg38).
        celltype: Cell-type / model context label.
        rsid: dbSNP rsID; enables eQTL, GWAS, and rsID-anchored literature lookups.
        genome_path: hg38 FASTA path; defaults to ``$REGLENS_GENOME``.
        scorer: A pre-built scorer (for tests); defaults to one from ``$REGLENS_MODEL``
            when a genome is available.
        client: HTTP client (for tests); defaults to the shared urllib client.
        window_length: Sequence window width in bp.

    Returns:
        The full serialized :class:`EvidenceBundle` as a JSON-able dict.
    """
    parsed = Variant.parse(variant)
    genome = _resolve_genome(genome_path)
    # Only build a scorer when there is a genome to score against; without one the
    # ChromBPNet signal is skipped and recorded in the bundle's errors, same as the CLI.
    if scorer is None and genome is not None:
        scorer = _make_scorer(window_length)
    bundle = analyze_variant(
        parsed,
        rsid=rsid,
        celltype=celltype,
        genome_path=genome,
        scorer=scorer,
        client=client,
        window_length=window_length,
    )
    return bundle.to_dict()


# --------------------------------------------------------------------------------------
# Tool registry — declarative specs, decoupled from the MCP SDK so they are testable
# without it. build_server() turns these into MCP Tool objects + a dispatcher.
# --------------------------------------------------------------------------------------
_VARIANT_PROP = {
    "type": "string",
    "description": "Variant as chr:pos:ref>alt on hg38, e.g. chr2:60490908:T>G",
}
_RSID_PROP = {"type": "string", "description": "dbSNP rsID, e.g. rs1427407"}
_CELLTYPE_PROP = {"type": "string", "description": "Cell-type / model context label"}


@dataclass(frozen=True)
class ToolSpec:
    """A declarative MCP tool: identity, JSON input schema, and a dict->dict handler.

    Attributes:
        name: MCP tool name.
        description: Human-readable description surfaced to the host.
        input_schema: JSON Schema for the tool's arguments.
        handler: Callable mapping an ``arguments`` dict to a JSON-able result dict.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], dict[str, Any]]


def _require(arguments: dict[str, Any], key: str) -> Any:
    """Fetch a required argument or raise a clear ``ValueError``."""
    if key not in arguments or arguments[key] in (None, ""):
        raise ValueError(f"Missing required argument: {key!r}")
    return arguments[key]


def _schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    """Build a JSON Schema object for a tool's arguments."""
    return {"type": "object", "properties": properties, "required": required}


#: The deterministic tools exposed over MCP. ``get_evidence_bundle`` is listed first as
#: the primary interface.
TOOL_SPECS: list[ToolSpec] = [
    ToolSpec(
        name="get_evidence_bundle",
        description=(
            "PRIMARY INTERFACE. Gather all deterministic evidence for a variant — "
            "ChromBPNet accessibility, disrupted/created TF motif, regulatory-element "
            "context, target gene + eQTL, GWAS traits, and real Europe PMC citations — "
            "into one bundle. Genome/model-dependent signals are included when the "
            "server is configured and skipped otherwise (noted under 'errors')."
        ),
        input_schema=_schema(
            {"variant": _VARIANT_PROP, "celltype": _CELLTYPE_PROP, "rsid": _RSID_PROP},
            ["variant"],
        ),
        handler=lambda a: get_evidence_bundle(
            _require(a, "variant"), celltype=a.get("celltype"), rsid=a.get("rsid")
        ),
    ),
    ToolSpec(
        name="score_variant",
        description=(
            "ChromBPNet Δ log-counts: the change in predicted chromatin accessibility "
            "between the ref and alt alleles (effect size + direction). Requires "
            "REGLENS_GENOME; returns an actionable error if unset."
        ),
        input_schema=_schema(
            {"variant": _VARIANT_PROP, "celltype": _CELLTYPE_PROP}, ["variant"]
        ),
        handler=lambda a: score_variant(_require(a, "variant"), celltype=a.get("celltype")),
    ),
    ToolSpec(
        name="motif_effect",
        description=(
            "In-silico mutagenesis around the variant to find the TF motif (JASPAR) most "
            "disrupted or created by the alt allele. Requires REGLENS_GENOME; returns an "
            "actionable error if unset."
        ),
        input_schema=_schema({"variant": _VARIANT_PROP}, ["variant"]),
        handler=lambda a: motif_effect_tool(_require(a, "variant")),
    ),
    ToolSpec(
        name="regulatory_context",
        description=(
            "ENCODE cCRE / Ensembl Regulatory overlap for the variant: is it inside a "
            "candidate regulatory element, and the nearest one."
        ),
        input_schema=_schema({"variant": _VARIANT_PROP}, ["variant"]),
        handler=lambda a: regulatory_context_tool(_require(a, "variant")),
    ),
    ToolSpec(
        name="gene_target",
        description=(
            "Nearest gene and GTEx eQTL evidence for the variant (which gene's "
            "expression it plausibly regulates). Provide rsid to enable the eQTL lookup."
        ),
        input_schema=_schema({"variant": _VARIANT_PROP, "rsid": _RSID_PROP}, ["variant"]),
        handler=lambda a: gene_target_tool(_require(a, "variant"), rsid=a.get("rsid")),
    ),
    ToolSpec(
        name="trait_link",
        description="GWAS Catalog trait associations for a variant, looked up by rsID.",
        input_schema=_schema(
            {"rsid": _RSID_PROP, "variant": _VARIANT_PROP}, ["rsid"]
        ),
        handler=lambda a: trait_link_tool(_require(a, "rsid"), variant=a.get("variant")),
    ),
    ToolSpec(
        name="literature",
        description=(
            "Search Europe PMC for real citations (with PMIDs) about the variant, gene, "
            "or trait. Pass the 2-3 most specific terms (e.g. rsid + gene) for sharp hits."
        ),
        input_schema=_schema(
            {
                "variant": _VARIANT_PROP,
                "rsid": _RSID_PROP,
                "gene": {"type": "string", "description": "Target gene symbol"},
                "trait": {"type": "string", "description": "Trait term"},
            },
            ["variant"],
        ),
        handler=lambda a: literature_tool(
            _require(a, "variant"),
            rsid=a.get("rsid"),
            gene=a.get("gene"),
            trait=a.get("trait"),
        ),
    ),
]


def dispatch(name: str, arguments: dict[str, Any] | None) -> dict[str, Any]:
    """Run a registered tool by name over its arguments (SDK-independent).

    Args:
        name: A tool name from :data:`TOOL_SPECS`.
        arguments: The tool's arguments (may be ``None``).

    Returns:
        The tool's JSON-able result dict.

    Raises:
        ValueError: If ``name`` is not a registered tool.
    """
    for spec in TOOL_SPECS:
        if spec.name == name:
            return spec.handler(arguments or {})
    raise ValueError(f"Unknown tool: {name!r}")


# --------------------------------------------------------------------------------------
# MCP server wiring — the only code that touches the MCP SDK, imported lazily.
# --------------------------------------------------------------------------------------
def build_server() -> Any:
    """Build the MCP ``Server`` with all tools registered.

    Returns:
        A configured :class:`mcp.server.Server` instance.

    Raises:
        ImportError: If the MCP SDK is not installed (``pip install -e ".[mcp]"``).
    """
    try:
        from mcp.server import Server
        from mcp.types import TextContent, Tool
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ImportError(
            "The MCP SDK is required to run the RegLens MCP server. "
            'Install it with: pip install -e ".[mcp]"'
        ) from exc

    server: Any = Server(SERVER_NAME)

    @server.list_tools()
    async def _list_tools() -> list[Any]:
        return [
            Tool(name=s.name, description=s.description, inputSchema=s.input_schema)
            for s in TOOL_SPECS
        ]

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any] | None) -> list[Any]:
        result = dispatch(name, arguments)
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    return server


def main() -> None:
    """Run the RegLens MCP server over stdio (blocking).

    Raises:
        ImportError: If the MCP SDK is not installed.
    """
    import anyio
    from mcp.server.stdio import stdio_server

    server = build_server()

    async def _serve() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    anyio.run(_serve)


if __name__ == "__main__":  # pragma: no cover - process entry point
    main()
