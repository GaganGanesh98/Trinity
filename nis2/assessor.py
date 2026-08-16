"""
NIS2 readiness assessment.

Takes a customer document (security policy, incident response plan, supplier
policy) and reports, per ENISA requirement domain, whether the document
addresses the obligation — quoting the customer's own words as evidence.

Two design choices are load-bearing:

1. **The uploaded document is never persisted.** It is chunked and embedded into
   an in-memory index that is discarded when the request ends. Nothing is
   written to disk or sent to a third party: embeddings and generation both run
   on the local Ollama server. For a customer being asked to hand over their
   internal security policy, that property is the product.

2. **Every quoted excerpt is verified against the source.** The model is asked
   for a verbatim quote; if what comes back is not actually in the document, the
   excerpt is dropped and the finding is flagged rather than shown. See
   nis2/schema.py.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from llama_index.core import Document, Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever

from config import (
    NIS2_LLM_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_LLM_REQUEST_TIMEOUT,
)
from nis2.checkpoints import Checkpoint, get_checkpoints
from nis2.schema import Assessment, Finding, Report, Severity, Status, verify_excerpt
from rag.indexer import OLLAMA_LLM_CONTEXT_WINDOW, _configure_settings

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}

# Smaller than the corpus chunk size: policy documents are dense and clause-like,
# and a tighter chunk keeps a retrieved passage close to a quotable sentence.
DOC_CHUNK_SIZE = 384
DOC_CHUNK_OVERLAP = 48

DOC_TOP_K = 4       # passages from the customer's document per checkpoint
GUIDANCE_TOP_K = 2  # supporting passages from the ENISA corpus

ASSESS_PROMPT = """You are a NIS2 compliance assessor. Judge ONLY what the \
document below states. Never assume a control exists because it would be normal \
to have one.

REQUIREMENT
Domain: {domain} ({article})
Obligation: {obligation}
Evidence that would satisfy it:
{evidence}

REGULATORY GUIDANCE (background; do NOT quote this as the customer's text)
{guidance}

CUSTOMER DOCUMENT EXCERPTS
{passages}

Decide:
- status: ADDRESSED if the document clearly meets the obligation; PARTIAL if it \
covers some but not all of it; NOT_ADDRESSED if the document is silent; UNCLEAR \
if it mentions the topic too vaguely to judge.
- severity: how serious the gap is (HIGH/MEDIUM/LOW). Use NONE when status is \
ADDRESSED.
- rationale: at most two sentences, naming what the document does or omits.
- excerpt: copy a supporting sentence from the CUSTOMER DOCUMENT EXCERPTS \
word-for-word. Do not reword it. If nothing relevant is present, return an empty \
string.
"""


def _assessment_llm(model: str | None = None):
    """
    LLM used for assessment, kept separate from Settings.llm.

    /query and the assessor have different needs: chat answers favour speed,
    while a finding someone acts on favours judgement. Keeping them separable
    lets the assessor run a larger model without slowing the chat path.
    """
    from llama_index.llms.ollama import Ollama

    return Ollama(
        model=model or NIS2_LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        request_timeout=OLLAMA_LLM_REQUEST_TIMEOUT,
        context_window=OLLAMA_LLM_CONTEXT_WINDOW,
        temperature=0.0,
    )


def load_document(path: Path) -> tuple[list[Document], str]:
    """Read a PDF/TXT/MD file into Documents plus its full raw text."""
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported file type {path.suffix!r}. "
            f"Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )
    docs = SimpleDirectoryReader(input_files=[str(path)], exclude_hidden=False).load_data()
    if not docs:
        raise ValueError(f"No readable text in {path.name}")
    return docs, "\n".join(d.text for d in docs)


def load_document_bytes(data: bytes, filename: str) -> tuple[list[Document], str]:
    """
    Same as load_document() for an uploaded file.

    The bytes touch disk only inside a TemporaryDirectory that is removed before
    this returns — the PDF readers need a real path, but the upload does not
    outlive the call.
    """
    suffix = Path(filename).suffix.lower()
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / f"upload{suffix}"
        p.write_bytes(data)
        return load_document(p)


def _build_ephemeral_index(docs: list[Document]) -> VectorStoreIndex:
    """In-memory index over the uploaded document. Never persisted."""
    splitter = SentenceSplitter(chunk_size=DOC_CHUNK_SIZE, chunk_overlap=DOC_CHUNK_OVERLAP)
    return VectorStoreIndex.from_documents(docs, transformations=[splitter])


def _guidance_retriever() -> VectorIndexRetriever | None:
    """
    Retriever over the ENISA corpus, used to ground each checkpoint in published
    guidance. Optional: if the corpus index is missing the assessment still runs,
    just without the regulatory background.
    """
    try:
        from rag.indexer import load_index

        return VectorIndexRetriever(index=load_index(), similarity_top_k=GUIDANCE_TOP_K)
    except Exception as e:
        logger.warning("ENISA corpus unavailable (%s) — assessing without guidance", e)
        return None


def _assess_one(
    cp: Checkpoint,
    doc_retriever: VectorIndexRetriever,
    guidance_retriever: VectorIndexRetriever | None,
    document_text: str,
    llm,
) -> Finding:
    passages = [n.get_content().strip() for n in doc_retriever.retrieve(cp.document_query)]
    guidance = ""
    if guidance_retriever is not None:
        guidance = "\n---\n".join(
            n.get_content().strip()[:600]
            for n in guidance_retriever.retrieve(cp.guidance_query)
        )

    common = dict(
        checkpoint_id=cp.id,
        domain=cp.domain,
        article=cp.article,
        obligation=cp.obligation,
        evidence_expected=list(cp.evidence),
    )

    # Nothing retrieved at all — the document has no text near this topic. Say so
    # without spending an LLM call.
    if not passages:
        return Finding(
            **common,
            status=Status.NOT_ADDRESSED,
            severity=Severity.MEDIUM,
            rationale="No passage in the document relates to this obligation.",
            excerpt="",
        )

    prompt = ASSESS_PROMPT.format(
        domain=cp.domain,
        article=cp.article,
        obligation=cp.obligation,
        evidence="\n".join(f"- {e}" for e in cp.evidence),
        guidance=guidance or "(none available)",
        passages="\n---\n".join(passages),
    )

    try:
        result: Assessment = llm.structured_predict(Assessment, prompt=_as_template(prompt))
    except Exception as e:
        # One malformed structured response must not lose the other twelve
        # findings, so degrade this checkpoint rather than failing the report.
        logger.warning("Assessment failed for %s (%s)", cp.id, e)
        return Finding(
            **common,
            status=Status.UNCLEAR,
            severity=Severity.LOW,
            rationale=f"Automated assessment did not complete for this checkpoint ({type(e).__name__}).",
            excerpt="",
        )

    verified = verify_excerpt(result.excerpt, document_text)
    if result.excerpt and not verified:
        logger.info("%s: excerpt not found verbatim in document — dropped", cp.id)

    status = result.status
    rationale = result.rationale.strip()

    # A claim that the document covers an obligation has to be quotable. Models
    # assert coverage that isn't there — observed on a policy with no cryptography
    # section at all, returned as PARTIAL "covers key management".
    #
    # Demote to NOT_ADDRESSED rather than UNCLEAR. When a model cannot quote a
    # single supporting sentence, the likeliest explanation is that the document
    # does not contain the control, not that it is worded ambiguously; measured
    # against nis2/ground_truth.json this lifted status accuracy from 38% to 69%
    # (llama3.2) and 23% to 62% (qwen2.5:7b). It also errs in the safe direction:
    # over-reporting a gap costs review time, while under-reporting one hands out
    # a false clean bill of health.
    #
    # excerpt_verified stays False so a reviewer can see the finding rests on the
    # absence of evidence rather than on quoted text.
    if status in (Status.ADDRESSED, Status.PARTIAL) and not verified:
        logger.info("%s: %s claimed without a verifiable quote — demoted to NOT_ADDRESSED",
                    cp.id, status.value)
        rationale = (
            f"{rationale} (Reported as {status.value}, but no supporting text could be "
            "quoted from the document — treated as not addressed. Worth confirming "
            "manually if you believe this control exists.)"
        ).strip()
        status = Status.NOT_ADDRESSED

    return Finding(
        **common,
        status=status,
        severity=result.severity if status != Status.ADDRESSED else Severity.NONE,
        rationale=rationale,
        excerpt=result.excerpt.strip() if verified else "",
        excerpt_verified=verified or not result.excerpt,
    )


def _as_template(text: str):
    """Wrap a formatted string as a PromptTemplate (structured_predict wants one)."""
    from llama_index.core.prompts import PromptTemplate

    # The prompt is already fully formatted; escape braces so the template engine
    # does not try to substitute anything inside the customer's own text.
    return PromptTemplate(text.replace("{", "{{").replace("}", "}}"))


def assess(
    data: bytes,
    filename: str,
    *,
    checkpoint_ids: list[str] | None = None,
    model: str | None = None,
) -> Report:
    """
    Assess an uploaded document against the NIS2 checkpoints.

    Args:
        data: raw file bytes (PDF, TXT or MD).
        filename: original name, used for the suffix and in the report.
        checkpoint_ids: optional subset, e.g. ["NIS2-03"] to assess one domain.
        model: override the Ollama model (defaults to config.NIS2_LLM_MODEL).
    """
    _configure_settings()

    docs, document_text = load_document_bytes(data, filename)
    logger.info("Assessing %s (%d chars)", filename, len(document_text))

    index = _build_ephemeral_index(docs)
    doc_retriever = VectorIndexRetriever(index=index, similarity_top_k=DOC_TOP_K)
    guidance = _guidance_retriever()

    llm = _assessment_llm(model)
    findings = [
        _assess_one(cp, doc_retriever, guidance, document_text, llm)
        for cp in get_checkpoints(checkpoint_ids)
    ]
    return Report(document_name=filename, findings=findings, model=llm.model).finalise()
