"""
NIS2 assessment checkpoints.

The taxonomy is not invented here: it mirrors the thirteen requirement domains
that ENISA's *Technical Implementation Guidance* (June 2025) uses to decompose
the risk-management measures of NIS2 Article 21(2). Keeping ENISA's own
structure means every finding can point at a published requirement rather than
at a checklist we made up — which is the difference between an assessment
someone can act on and one they have to take on faith.

Article references are to Directive (EU) 2022/2555.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Checkpoint:
    """One assessable obligation."""

    id: str
    domain: str
    article: str

    # What an assessor is looking for in the customer's document.
    obligation: str

    # Concrete artefacts that would satisfy it. Taken from the "examples of
    # evidence" sections of the ENISA guidance; used both in the assessment
    # prompt and shown to the user as remediation hints.
    evidence: tuple[str, ...]

    # Query used to pull the customer's own text for this checkpoint. Phrased in
    # the vocabulary a policy document would use, not the regulation's — the
    # uploaded document will not say "Article 21(2)(b)".
    document_query: str

    # Query used to pull supporting regulation/guidance from the ENISA corpus.
    guidance_query: str = ""

    def __post_init__(self) -> None:
        if not self.guidance_query:
            object.__setattr__(self, "guidance_query", f"NIS2 {self.domain} {self.obligation}")


CHECKPOINTS: tuple[Checkpoint, ...] = (
    Checkpoint(
        id="NIS2-01",
        domain="Information security policy",
        article="Art. 21(2)(a)",
        obligation=(
            "An approved information security policy exists, is endorsed by the "
            "management body, is communicated to staff, and is reviewed at planned "
            "intervals."
        ),
        evidence=(
            "A written security policy with a version and approval date",
            "Evidence of management-body approval",
            "A defined review cycle and the date of the last review",
        ),
        document_query="information security policy approval management review scope",
    ),
    Checkpoint(
        id="NIS2-02",
        domain="Risk management",
        article="Art. 21(2)(a)",
        obligation=(
            "A risk management framework defines how risks are identified, "
            "analysed, treated and accepted, with named risk owners and a "
            "documented risk register."
        ),
        evidence=(
            "A risk assessment methodology",
            "A risk register with owners and treatment decisions",
            "Records of risk acceptance by an accountable person",
        ),
        document_query="risk assessment methodology risk register treatment acceptance owner",
    ),
    Checkpoint(
        id="NIS2-03",
        domain="Incident handling",
        article="Art. 21(2)(b), Art. 23",
        obligation=(
            "Incidents are detected, classified, escalated and recorded under a "
            "documented procedure, and significant incidents are reported to the "
            "CSIRT within the Article 23 deadlines: early warning within 24 hours, "
            "incident notification within 72 hours, final report within one month."
        ),
        evidence=(
            "An incident response procedure with severity classification",
            "Defined escalation paths and on-call responsibilities",
            "Explicit reference to the 24h / 72h / 1-month reporting deadlines",
            "A record of incidents and post-incident reviews",
        ),
        document_query=(
            "incident response procedure detection classification escalation "
            "notification authority reporting deadline"
        ),
    ),
    Checkpoint(
        id="NIS2-04",
        domain="Business continuity and crisis management",
        article="Art. 21(2)(c)",
        obligation=(
            "Business continuity and disaster recovery plans exist with defined "
            "RTO/RPO, backups are taken and their restoration is tested, and crisis "
            "management responsibilities are assigned."
        ),
        evidence=(
            "A business continuity plan naming critical services",
            "Backup policy with retention and restoration testing records",
            "Stated recovery time and recovery point objectives",
        ),
        document_query="business continuity disaster recovery backup restore RTO RPO crisis",
    ),
    Checkpoint(
        id="NIS2-05",
        domain="Supply chain security",
        article="Art. 21(2)(d), Art. 21(3)",
        obligation=(
            "Security requirements are imposed on direct suppliers and service "
            "providers, supplier risk is assessed, and security obligations are "
            "reflected in contracts."
        ),
        evidence=(
            "A supplier security policy or vendor risk process",
            "Security clauses in supplier contracts",
            "A register of critical suppliers with assessed risk",
        ),
        document_query="supplier vendor third party contract security requirements assessment",
    ),
    Checkpoint(
        id="NIS2-06",
        domain="Secure acquisition, development and maintenance",
        article="Art. 21(2)(e)",
        obligation=(
            "Security is addressed across the system lifecycle, including secure "
            "development practices, change management, and vulnerability handling "
            "and disclosure."
        ),
        evidence=(
            "A secure development or SDLC policy",
            "A change management procedure",
            "A vulnerability management process with remediation timelines",
            "A coordinated vulnerability disclosure contact",
        ),
        document_query=(
            "secure development lifecycle change management patching vulnerability "
            "disclosure remediation"
        ),
    ),
    Checkpoint(
        id="NIS2-07",
        domain="Effectiveness assessment",
        article="Art. 21(2)(f)",
        obligation=(
            "The effectiveness of cybersecurity risk-management measures is assessed "
            "on a defined cycle, through audit, testing or metrics, with results "
            "reported to management."
        ),
        evidence=(
            "An internal audit or control-testing schedule",
            "Security metrics or KPIs reported to management",
            "Records of corrective actions arising from assessments",
        ),
        document_query="audit testing effectiveness review metrics KPI corrective action",
    ),
    Checkpoint(
        id="NIS2-08",
        domain="Cyber hygiene and training",
        article="Art. 21(2)(g)",
        obligation=(
            "Basic cyber hygiene practices are defined and security awareness "
            "training is delivered to staff, including the management body, with "
            "completion tracked."
        ),
        evidence=(
            "A security awareness training programme with frequency",
            "Training records or completion rates",
            "Evidence that management-body members also receive training",
        ),
        document_query="security awareness training staff onboarding phishing hygiene",
    ),
    Checkpoint(
        id="NIS2-09",
        domain="Cryptography",
        article="Art. 21(2)(h)",
        obligation=(
            "A policy governs the use of cryptography, covering encryption of data "
            "in transit and at rest, approved algorithms, and key management."
        ),
        evidence=(
            "A cryptography or encryption policy",
            "Named approved algorithms and minimum key lengths",
            "A key management procedure covering rotation and storage",
        ),
        document_query="encryption cryptography key management TLS data at rest algorithm",
    ),
    Checkpoint(
        id="NIS2-10",
        domain="Human resources security",
        article="Art. 21(2)(i)",
        obligation=(
            "Security responsibilities apply across the employment lifecycle, "
            "including screening where lawful, terms of employment, disciplinary "
            "process, and revocation of access on termination."
        ),
        evidence=(
            "Background screening procedure where legally permitted",
            "Confidentiality or NDA terms in employment contracts",
            "A joiners/movers/leavers process revoking access on exit",
        ),
        document_query="employment screening background check onboarding offboarding termination access revocation",
    ),
    Checkpoint(
        id="NIS2-11",
        domain="Access control",
        article="Art. 21(2)(i), Art. 21(2)(j)",
        obligation=(
            "Access is granted on least-privilege and need-to-know, privileged "
            "accounts are controlled and reviewed, and multi-factor authentication "
            "is used where appropriate."
        ),
        evidence=(
            "An access control policy stating least privilege",
            "Privileged access management and periodic access reviews",
            "MFA enforced for remote and administrative access",
        ),
        document_query=(
            "access control least privilege privileged account multi-factor "
            "authentication MFA review provisioning"
        ),
    ),
    Checkpoint(
        id="NIS2-12",
        domain="Asset management",
        article="Art. 21(2)(i)",
        obligation=(
            "Assets are inventoried with assigned owners, classified according to "
            "sensitivity, and handled and disposed of under defined rules."
        ),
        evidence=(
            "An asset inventory with named owners",
            "An information classification scheme",
            "Acceptable use and secure disposal rules",
        ),
        document_query="asset inventory register ownership classification acceptable use disposal",
    ),
    Checkpoint(
        id="NIS2-13",
        domain="Physical and environmental security",
        article="Art. 21(2)(c), Art. 21(2)(e)",
        obligation=(
            "Physical access to facilities hosting network and information systems "
            "is controlled, and environmental threats such as power loss and fire "
            "are mitigated."
        ),
        evidence=(
            "Physical access controls for offices and data centres",
            "Visitor management records",
            "Power, cooling and fire-suppression provisions",
        ),
        document_query="physical access data centre visitor badge power cooling fire environmental",
    ),
)

CHECKPOINTS_BY_ID: dict[str, Checkpoint] = {c.id: c for c in CHECKPOINTS}


def get_checkpoints(ids: list[str] | None = None) -> tuple[Checkpoint, ...]:
    """All checkpoints, or the named subset (preserving canonical order)."""
    if not ids:
        return CHECKPOINTS
    wanted = set(ids)
    unknown = wanted - CHECKPOINTS_BY_ID.keys()
    if unknown:
        raise ValueError(f"Unknown checkpoint ids: {sorted(unknown)}")
    return tuple(c for c in CHECKPOINTS if c.id in wanted)
