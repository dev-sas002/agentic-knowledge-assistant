import os
import textwrap
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

MARGIN = 50
LINE_HEIGHT = 16
MAX_CHARS = 100


def draw_wrapped_text(c: canvas.Canvas, text: str, x: int, y: int) -> int:
    for line in text.splitlines():
        wrapped = textwrap.wrap(line, width=MAX_CHARS) or [""]
        for w in wrapped:
            c.drawString(x, y, w)
            y -= LINE_HEIGHT
    return y


def create_pdf(file_path: Path, title: str, sections: list[dict[str, str]]):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(file_path), pagesize=A4)
    width, height = A4

    y = height - MARGIN
    c.setFont("Helvetica-Bold", 18)
    c.drawString(MARGIN, y, title)
    y -= LINE_HEIGHT * 2

    for sec in sections:
        heading = sec.get("heading", "")
        body = sec.get("body", "")

        if y < MARGIN + 4 * LINE_HEIGHT:
            c.showPage()
            y = height - MARGIN

        c.setFont("Helvetica-Bold", 14)
        c.drawString(MARGIN, y, heading)
        y -= LINE_HEIGHT * 1.5

        c.setFont("Helvetica", 11)
        y = draw_wrapped_text(c, body, MARGIN, y)
        y -= LINE_HEIGHT

    c.showPage()
    c.save()


def main():
    # Honour RAW_DOCS_DIR, which is what ingestion/build_index.py reads.
    # Hardcoding "data/raw" here meant the generator and the indexer
    # could point at different directories.
    raw_dir = Path(os.environ.get("RAW_DOCS_DIR", "data/raw"))

    create_pdf(
        raw_dir / "Finance_Processes_Guide.pdf",
        title="Finance Processes: Payments, Invoicing, Reconciliation",
        sections=[
            {
                "heading": "Payment Cycles & Statements",
                "body": (
                    "Payments are issued weekly on Fridays. Statements summarize orders, fees, and adjustments. "
                    "Disputes must be raised within 7 days of receipt."
                ),
            },
            {
                "heading": "Invoice Re-Send Procedure",
                "body": (
                    "To re-send an invoice: 1) Identify the partner and billing period; 2) Confirm contact email; "
                    "3) Trigger invoice re-send from Finance Portal; 4) Log the action with timestamp and operator; "
                    "5) Confirm delivery with partner."
                ),
            },
            {
                "heading": "Reconciliation Basics",
                "body": (
                    "Reconciliation aligns platform order data with payouts. Typical mismatches include fees, refunds, and "
                    "chargebacks. Use the Reconciliation Report to compare by order ID, then raise adjustments as needed."
                ),
            },
        ],
    )

    create_pdf(
        raw_dir / "Menu_and_Operations_FAQ.pdf",
        title="Menu & Operations FAQ",
        sections=[
            {
                "heading": "Pausing / Unpausing Menu Items",
                "body": (
                    "Items can be paused for stock-outs or quality issues. Suggested policy: if stock-out > 30 minutes, pause the item. "
                    "To unpause, verify availability and quality checks. Log all changes for audit."
                ),
            },
            {
                "heading": "Updating Opening Hours",
                "body": (
                    "Adjust hours for holidays or exceptions. Ensure third-party listings reflect the same hours to avoid customer friction. "
                    "Overnight operations should be split by platform guidelines."
                ),
            },
            {
                "heading": "Promotions & Price Changes",
                "body": (
                    "Promotions should have a clear objective (conversion, basket size, off-peak fill). Price changes require approval and should be synchronized across platforms."
                ),
            },
        ],
    )

    create_pdf(
        raw_dir / "System_Onboarding_and_Usage.pdf",
        title="System Onboarding & Usage",
        sections=[
            {
                "heading": "Partner Portal: Getting Started",
                "body": (
                    "Log in using the invite email. Set up 2FA. Complete profile details (banking, tax, contacts). Access training resources via the Help Center."
                ),
            },
            {
                "heading": "Reporting Tools",
                "body": (
                    "Use the Performance dashboard for orders, sales, and cancellations. Export CSVs for deeper analysis. "
                    "Use tags to segment by site or brand."
                ),
            },
            {
                "heading": "Support & Escalations",
                "body": (
                    "Use live chat for urgent issues. Finance tickets for payout concerns. Operations tickets for menu/item problems. Escalate service-impacting issues immediately."
                ),
            },
        ],
    )

    create_pdf(
        raw_dir / "Reporting_and_Insights_Guide.pdf",
        title="Reporting & Insights Guide",
        sections=[
            {
                "heading": "Core KPIs",
                "body": (
                    "Track Orders, Sales, AOV, Cancellation Rate, and On-Time Rate. Review item-level sales mix weekly to identify hero and underperforming items."
                ),
            },
            {
                "heading": "Recommendations Framework",
                "body": (
                    "Recommendations should be specific, measurable, and time-bounded. Examples: extend hours on Fri/Sat by 1 hour; promote top-margin items; pause items with high cancellation rates."
                ),
            },
            {
                "heading": "Data Freshness",
                "body": (
                    "Ensure data recency before decisions. Use date filters and verify that platform exports have been ingested for the current period."
                ),
            },
        ],
    )

    # New: Ops portal and Deliverect task guides with field names, SOPs, and edge cases
    create_pdf(
        raw_dir / "Ops_Portal_and_Deliverect_Tasks.pdf",
        title="Ops Portal & Deliverect: Operational Task SOPs",
        sections=[
            {
                "heading": "Pause / Unpause Menu Items (Deliverect)",
                "body": (
                    "Navigation: Menus > Items. Fields: Item Name, Availability (toggle), Reason (optional note).\n"
                    "Pause SOP: 1) Search item; 2) Toggle Availability=Off; 3) Add note with stock-out cause and ETA; 4) Save; 5) Verify sync status (should be 'Synced').\n"
                    "Unpause SOP: 1) Confirm stock; 2) Toggle Availability=On; 3) Remove outdated note; 4) Save; 5) Verify sync.\n"
                    "Edge cases: If sync pending >10m, check integration status; if multiple brands, ensure correct brand/site context."
                ),
            },
            {
                "heading": "Update Opening Hours (Ops Portal)",
                "body": (
                    "Navigation: Sites > Operating Hours. Fields: Day, Open Time, Close Time, Breaks, Exceptions.\n"
                    "SOP: 1) Select site; 2) Adjust Day row(s); 3) For overnight, split into 2 blocks (e.g., 18:00-23:59 and 00:00-02:00); 4) Add Exceptions for holidays; 5) Save and publish.\n"
                    "Validation: Ensure 3P platforms reflect changes (check monitoring report).\n"
                    "Edge cases: DST changes, bank holidays, split brands at the same site."
                ),
            },
            {
                "heading": "Promotion Creation Checklist",
                "body": (
                    "Fields: Promo Name, Start/End, Items, Discount Type/Value, Channels.\n"
                    "SOP: Align with objective (conversion, AOV, off-peak fill); avoid promo stacking; confirm margin impact; sync across platforms."
                ),
            },
            {
                "heading": "Audit Logging",
                "body": (
                    "Log each change with timestamp, operator, site/brand, before/after values. Store in immutable log for compliance."
                ),
            },
            {
                "heading": "FAQs",
                "body": (
                    "Q: How long do menu changes take to sync? A: Typically <5 minutes; investigate if >10 minutes.\n"
                    "Q: Can I pause multiple items at once? A: Yes, bulk actions are available in Deliverect under Menus > Bulk Edit.\n"
                    "Q: How to handle temporary closures? A: Use Exceptions and add a banner message if supported."
                ),
            },
        ],
    )

    # New: Third-party platform nuances and remediation checklist
    create_pdf(
        raw_dir / "Third_Party_Platform_Guidelines.pdf",
        title="Third-Party Platforms: Listing Consistency & Remediation",
        sections=[
            {
                "heading": "Platform Nuances (Deliveroo / Just Eat / Uber Eats)",
                "body": (
                    "Hours formatting: Some platforms require day-by-day entries; overnight hours may need split blocks.\n"
                    "Menu sync: Expect propagation delays (2-10 minutes). Price changes may require approval.\n"
                    "Availability: Some platforms cache availability; force refresh may be needed."
                ),
            },
            {
                "heading": "Validation Checklist",
                "body": (
                    "1) Hours match Ops Portal; 2) Menu items present and prices correct; 3) Paused items hidden; 4) Promotions visible; 5) Brand assets and descriptions correct."
                ),
            },
            {
                "heading": "Discrepancy Types & Actions",
                "body": (
                    "Hours mismatch: Re-publish hours, confirm timezone.\n"
                    "Missing item: Validate menu mapping and category; trigger re-sync.\n"
                    "Wrong price: Re-publish price list; if still wrong, open platform ticket.\n"
                    "Stale availability: Toggle item availability Off→On to refresh cache."
                ),
            },
            {
                "heading": "SLAs & Escalation",
                "body": (
                    "Target remediation: <2h for critical items; <24h for non-critical.\n"
                    "Escalate to platform support if unresolved after two re-sync attempts."
                ),
            },
            {
                "heading": "FAQs",
                "body": (
                    "Q: Why do hours look different by platform? A: Formatting rules differ; use split blocks for overnights.\n"
                    "Q: How to verify successful sync? A: Check 'Synced' status in Deliverect and confirm live listing."
                ),
            },
        ],
    )

    print(f"Generated sample PDFs in {raw_dir}/")


if __name__ == "__main__":
    main()
