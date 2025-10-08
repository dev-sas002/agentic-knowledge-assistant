"""
The operator console.

A thin view. It renders state and collects a decision; it contains no
retrieval, no prompting, no parsing and no dispatch. Two rules keep it that
way, and both of them are fixes for real bugs:

**It never calls the model.** The graph composes an answer and the critic
checks it. This module replays that text. It used to stream a *second*
completion for the typing effect, which billed every question twice and put an
answer on screen that the critic had never inspected — the decision log could
report ``grounded=True`` about text that had been thrown away.

**It never holds the pending approval.** Streamlit re-runs this entire script
on every widget interaction, so a variable holding a proposed action is rebuilt
several times while a human reads it. The gate owns the request; the session
holds its id.
"""

from __future__ import annotations

from html import escape

import streamlit as st

from agents import config
from agents.actions import registry_rows
from agents.approval import ApprovalError, ApprovalStatus, get_gate
from agents.cost import breakdown
from agents.graph import invoke_graph
from apps.ui.components import highlight_citations, pill, replay, request_card, source_card

# Kitchen-pass palette, matching .streamlit/config.toml. Every colour here is
# used as text on the paper background as well as a tint, so each one clears
# 4.5:1 contrast against #FBF7F1.
HERB = "#2F6B3A"
SAFFRON = "#8A5A00"
PAPRIKA = "#B4441E"
CHILLI = "#A4161A"
SLATE = "#5B534C"
RISK_COLOUR = {"low": HERB, "medium": SAFFRON, "high": CHILLI}
STATUS_COLOUR = {
    "pending": SAFFRON,
    "executed": HERB,
    "rejected": SLATE,
    "expired": "#6F5536",
    "failed": CHILLI,
    "approving": PAPRIKA,
}


# --------------------------------------------------------------------------
# Presentation helpers
# --------------------------------------------------------------------------
def meter(label: str, value: float, caption: str = "") -> None:
    colour = HERB if value >= 0.7 else SAFFRON if value >= 0.45 else CHILLI
    st.markdown(
        f"<div style='margin-bottom:6px;'>"
        f"<div style='display:flex;justify-content:space-between;font-size:0.8rem;color:{SLATE};'>"
        f"<span>{escape(label)}</span><span style='color:{colour};font-weight:600;'>{value:.0%}</span></div>"
        f"<div role='meter' aria-label='{escape(label)}' aria-valuemin='0' aria-valuemax='100' "
        f"aria-valuenow='{value * 100:.0f}' "
        f"style='background:#2B262214;border-radius:999px;height:6px;overflow:hidden;'>"
        f"<div style='background:{colour};width:{max(0.0, min(1.0, value)) * 100:.1f}%;height:6px;'></div>"
        f"</div></div>",
        unsafe_allow_html=True,
    )
    if caption:
        st.caption(caption)


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------
st.set_page_config(page_title="Ops Copilot", layout="wide", page_icon=":material/restaurant:")

st.markdown(
    """
    <style>
      .block-container {padding-top: 3.4rem; padding-bottom: 3rem;}
      div[data-testid="stMetricValue"] {font-size: 1.35rem;}
      .masthead {border-bottom: 2px solid #2B2622; padding-bottom: .6rem; margin-bottom: 1rem;}
      .masthead .eyebrow {font-size:.72rem; letter-spacing:.14em; text-transform:uppercase;
                          color:#B4441E; font-weight:700;}
      .masthead h1 {font-size: 2rem; margin: .1rem 0 .2rem; padding: 0;}
      .masthead p {color:#5B534C; font-size:.92rem; margin:0; max-width: 62rem;}
      .panel {background:#FFFDF9; border:1px solid #E3D8C9; border-radius:8px; padding:14px 16px; margin-bottom:12px;}
      .panel-title {font-size:.74rem; text-transform:uppercase; letter-spacing:.1em;
                    color:#5B534C; font-weight:700; margin: 1.1rem 0 .5rem;
                    border-top: 1px dashed #D6C8B5; padding-top: .7rem;}
      .src {border-left:3px solid #D99A1E; padding:4px 0 4px 10px; margin-bottom:10px;}
      .src-head {font-weight:600; font-size:.86rem;}
      .src-body {font-size:.82rem; color:#5B534C; line-height:1.5;}
      .muted {color:#5B534C; font-size:.8rem; font-weight:400;}
      .request-summary {font-family:"Source Serif 4", Georgia, serif; font-size:1.05rem;
                        font-weight:600; margin-bottom:6px;}
      .cite {background:#B4441E14; color:#9A3412; border:1px solid #B4441E33; border-radius:4px;
             padding:0 5px; font-size:.85em; font-weight:600; white-space:nowrap;}
      .log {font-family:"IBM Plex Mono", ui-monospace, Menlo, monospace; font-size:.76rem;
            color:#3A332E; padding:3px 0;}
      @media (max-width: 640px) {
        .masthead h1 {font-size: 1.55rem;}
        .block-container {padding-left: 1rem; padding-right: 1rem;}
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<header class='masthead'><div class='eyebrow'>Restaurant operations</div>"
    "<h1>Ops Copilot</h1>"
    "<p>Answers from the internal operations manuals, with citations. Menu and hours changes "
    "are proposed, never made: each one waits for a person to approve it, and text from the "
    "manuals can never choose an action.</p></header>",
    unsafe_allow_html=True,
)

gate = get_gate()

if "history" not in st.session_state:
    st.session_state.history: list[dict] = []
if "last_result" not in st.session_state:
    st.session_state.last_result: dict | None = None
if "approval_id" not in st.session_state:
    st.session_state.approval_id: str | None = None

# ---- Sidebar ----
with st.sidebar:
    st.markdown("### Retrieval")
    top_k = st.slider("Passages retrieved (top-k)", 2, min(8, config.MAX_TOP_K), 4)
    style = st.selectbox("Answer style", ["detailed", "concise"], index=0)
    temperature = st.slider("Temperature", 0.0, 0.8, 0.2, 0.05)

    st.markdown("### Registered actions")
    st.caption("A closed allow-list. Anything not listed here cannot be proposed or approved.")
    for row in registry_rows():
        st.markdown(
            f"{pill(row['risk'], RISK_COLOUR.get(row['risk'], SLATE))} "
            f"**{escape(row['action'])}** &nbsp;<code>{escape(row['endpoint'])}</code><br>"
            f"<span class='muted'>{escape(row['description'])}</span>",
            unsafe_allow_html=True,
        )

    st.markdown("### Environment")
    st.caption(
        f"Ops API: `{config.OPS_API_URL}`  \n"
        f"Approval TTL: `{config.APPROVAL_TTL_SECONDS:.0f}s`  \n"
        f"Context cap: `{config.MAX_CONTEXT_CHARS}` chars  \n"
        f"LLM: `{'groq' if config.GROQ_API_KEY else 'extractive (no key set)'}`"
    )

# ---- Input ----
prompt = st.chat_input("Ask about finance, menu, onboarding or platforms — or issue a command such as: pause 'Chicken Biryani'")
if prompt:
    with st.spinner("Searching the operations manuals…"):
        result = invoke_graph(
            query=prompt,
            top_k=top_k,
            style=style,
            temperature=temperature,
            operator=config.OPERATOR_NAME,
        )
    st.session_state.history.append({"role": "user", "content": prompt})
    st.session_state.history.append({"role": "assistant", "content": result.get("answer", "")})
    st.session_state.last_result = result
    st.session_state.approval_id = result.get("approval_id")
    # Reveal progressively on this run only. A Streamlit rerun (clicking
    # Approve, moving a slider) must re-render the transcript instantly rather
    # than replaying every answer from the top.
    st.session_state.stream_next = True

result = st.session_state.last_result or {}
contexts = result.get("contexts", []) or []
report = result.get("grounding", {}) or {}
meta = result.get("answer_meta", {}) or {}
turn_cost = meta.get("cost", {}) or {}

left, right = st.columns([1.65, 1], gap="large")

# ---- Conversation ----
with left:
    if not st.session_state.history:
        st.info(
            "**Try:** *When are partner payments issued and how are statements summarised?* — "
            "or a command: *pause 'Chicken Biryani'*, which stops at the human-approval gate."
        )
    for index, message in enumerate(st.session_state.history):
        with st.chat_message(message["role"]):
            is_last = index == len(st.session_state.history) - 1
            reveal = is_last and message["role"] == "assistant" and st.session_state.get("stream_next")
            if reveal:
                st.session_state.stream_next = False
                placeholder = st.empty()
                shown = ""
                for fragment in replay(message["content"]):
                    shown += fragment
                    placeholder.markdown(shown)
                placeholder.markdown(highlight_citations(message["content"]), unsafe_allow_html=True)
            elif message["role"] == "assistant":
                st.markdown(highlight_citations(message["content"]), unsafe_allow_html=True)
            else:
                st.markdown(message["content"])

    retrieval_error = (result.get("retrieval") or {}).get("error")
    if retrieval_error:
        st.warning(
            f"The document index could not be searched ({retrieval_error}), so this answer has no sources. "
            "Build it with `python -m ingestion.build_index`, or restart the compose stack.",
        )

    if contexts:
        st.markdown("<div class='panel-title'>Retrieved sources</div>", unsafe_allow_html=True)
        cited = set(report.get("cited_sources", []))
        for position, context in enumerate(contexts, start=1):
            source = context.get("source", "unknown")
            similarity = context.get("similarity")
            similarity_text = f"{similarity:.0%} similar" if isinstance(similarity, int | float) else "n/a"
            badge = pill("cited", HERB) if source.lower() in cited else pill("retrieved", SLATE)
            body = (context.get("document") or "")[:340]
            st.markdown(source_card(position, source, badge, similarity_text, body), unsafe_allow_html=True)

# ---- Approval gate, grounding, telemetry ----
with right:
    st.markdown("<div class='panel-title'>Human-approval gate</div>", unsafe_allow_html=True)
    request = gate.get(st.session_state.approval_id) if st.session_state.approval_id else None

    if request is None:
        st.markdown(
            "<div class='panel'><span style='color:#5B534C;font-size:.85rem;'>"
            "No action proposed. Knowledge questions never reach this gate — only a command "
            "the operator typed can, and it stops here."
            "</span></div>",
            unsafe_allow_html=True,
        )
    else:
        colour = STATUS_COLOUR.get(request.status.value, SLATE)
        remaining = request.seconds_remaining()
        # Past tense once the decision is made: "will be sent" on an executed
        # action reads as though something is still pending.
        if request.status is ApprovalStatus.EXECUTED:
            destination = "Was sent to"
        elif request.status is ApprovalStatus.PENDING:
            destination = "Will be sent to"
        else:
            destination = "Would have been sent to"
        st.markdown(
            request_card(
                status_pill=pill(request.status.value.upper(), colour),
                risk_pill=pill(f"{request.risk.value} risk", RISK_COLOUR.get(request.risk.value, SLATE)),
                request_id=request.id,
                summary=request.summary,
                requested_by=request.requested_by,
                original_query=request.original_query,
                destination=destination,
                endpoint=request.spec.endpoint if request.spec else "—",
                colour=colour,
            ),
            unsafe_allow_html=True,
        )

        if request.status is ApprovalStatus.PENDING:
            st.progress(
                min(1.0, remaining / max(1.0, gate.ttl_seconds)),
                text=f"Expires in {remaining:.0f}s if no decision is recorded",
            )
            approve_col, reject_col = st.columns(2)
            with approve_col:
                approve = st.button("Approve and execute", type="primary", width="stretch")
            with reject_col:
                reject = st.button("Reject", width="stretch")
            if approve or reject:
                try:
                    if approve:
                        gate.approve(request.id, decided_by=config.OPERATOR_NAME)
                    else:
                        gate.reject(request.id, decided_by=config.OPERATOR_NAME)
                except ApprovalError as exc:
                    st.error(str(exc))
                st.rerun()
        elif request.status is ApprovalStatus.EXECUTED:
            st.success(f"Executed by {request.decided_by} and written to the ops audit log.")
            if request.result:
                st.caption("Response from the ops API:")
                st.json(request.result, expanded=True)
        elif request.status is ApprovalStatus.REJECTED:
            st.info(f"Rejected by {request.decided_by}. Nothing was sent to the ops API.")
        elif request.status is ApprovalStatus.EXPIRED:
            st.warning(f"{request.reason} The proposal can no longer be approved; ask again to re-propose.")
        elif request.status is ApprovalStatus.FAILED:
            st.error(f"Approved, but the ops API refused: {request.error}")

    recent = [r for r in gate.recent(6) if r.status.is_terminal]
    if recent:
        with st.expander(f"Decision history ({len(recent)})"):
            for item in recent:
                st.markdown(
                    f"{pill(item.status.value, STATUS_COLOUR.get(item.status.value, SLATE))} "
                    f"<span style='font-size:.8rem;'>{escape(item.summary)}</span>",
                    unsafe_allow_html=True,
                )

    # ---- Grounding ----
    if report:
        st.markdown("<div class='panel-title'>Grounding check</div>", unsafe_allow_html=True)
        verdict = "GROUNDED" if report.get("grounded") else "NOT GROUNDED"
        st.markdown(
            pill(verdict, HERB if report.get("grounded") else CHILLI)
            + f" <span style='font-size:.8rem;color:#5B534C;'>&nbsp;{escape(report.get('reason', ''))}</span>",
            unsafe_allow_html=True,
        )
        meter("Citation validity", report.get("validity", 0.0), "Citations resolving to a document actually retrieved.")
        meter("Citation coverage", report.get("coverage", 0.0), "Substantive sentences carrying a citation.")
        meter("Passage support", report.get("support", 0.0), "Vocabulary shared with the cited passage.")
        if report.get("unknown_citations"):
            st.error("Fabricated citations: " + ", ".join(report["unknown_citations"]))

    # ---- Cost and latency ----
    if meta:
        st.markdown("<div class='panel-title'>Cost and latency</div>", unsafe_allow_html=True)
        retrieval = result.get("retrieval", {}) or {}
        one, two, three = st.columns(3)
        one.metric(
            "LLM calls",
            turn_cost.get("calls", 0),
            help="One hosted call per question — zero on the keyless extractive path. It used to be two.",
        )
        two.metric("Tokens", turn_cost.get("total_tokens", 0))
        three.metric("Retrieval", f"{retrieval.get('embed_ms', 0) + retrieval.get('query_ms', 0):.0f} ms")
        st.caption(
            f"Provider `{meta.get('provider', 'unknown')}` · "
            f"${turn_cost.get('usd', 0.0):.6f} this question · "
            f"embed {retrieval.get('embed_ms', 0):.0f}ms + query {retrieval.get('query_ms', 0):.0f}ms"
        )
        if contexts:
            with st.expander("Where the prompt tokens go"):
                st.caption("Retrieved context dominates, which is why top_k is a cost control and not only a quality dial.")
                for row in breakdown(result.get("query", ""), contexts, style):
                    # Flex rather than padded monospace: HTML collapses runs of
                    # spaces, so a column aligned with str.ljust does not stay aligned.
                    st.markdown(
                        f"<div class='log' style='display:flex;justify-content:space-between;'>"
                        f"<span>{escape(row['part'])}</span><span>{row['tokens']} tokens</span></div>",
                        unsafe_allow_html=True,
                    )

    # ---- Decision log ----
    if result.get("log"):
        st.markdown("<div class='panel-title'>Decision log</div>", unsafe_allow_html=True)
        for step in result["log"]:
            st.markdown(f"<div class='log'>{escape(step)}</div>", unsafe_allow_html=True)
