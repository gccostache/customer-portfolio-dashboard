import json
import sqlite3
from contextlib import closing
from sentence_transformers import SentenceTransformer
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Customer Portfolio Dashboard",
    page_icon="📋",
    layout="wide"
)

DATA_PATH = Path(__file__).resolve().parent / "fictional_accounts.csv"

st.title("Customer Portfolio Review Dashboard")
st.caption(
    "Fictional accounts and CSMs. Flags use illustrative review rules."
)

if not DATA_PATH.exists():
    st.error("Place fictional_accounts.csv beside app.py.")
    st.stop()

accounts = pd.read_csv(DATA_PATH).fillna("")

for column in ["renewal_date", "last_engagement", "action_due_date"]:
    accounts[column] = pd.to_datetime(
        accounts[column], errors="coerce"
    )

review_date = st.sidebar.date_input(
    "Review date",
    value=date(2026, 10, 8)
)
review_timestamp = pd.Timestamp(review_date)

selected_csm = st.sidebar.selectbox(
    "CSM",
    ["All"] + sorted(accounts["csm"].unique().tolist())
)

only_review = st.sidebar.checkbox(
    "Show only accounts requiring review"
)


def assess_account(row):
    reasons = []
    missing = []

    renewal = row["renewal_date"]
    engagement = row["last_engagement"]
    action_due = row["action_due_date"]

    if pd.isna(renewal):
        missing.append("Renewal date missing")
        renewal_days = None
    else:
        renewal_days = (renewal - review_timestamp).days
        if renewal_days < 0:
            reasons.append("Renewal date passed; confirm outcome")

    renewal_soon = (
        renewal_days is not None and 0 <= renewal_days <= 90
    )

    if row["adoption_trend"] == "Declining":
        reasons.append("Declining adoption")
    elif row["adoption_trend"] not in ["Stable", "Growing"]:
        missing.append("Adoption trend unknown")

    if row["open_escalation"] == "Yes":
        reasons.append("Open escalation")
    elif row["open_escalation"] != "No":
        missing.append("Escalation status unknown")

    if pd.isna(engagement):
        missing.append("Last engagement missing")
    elif engagement > review_timestamp:
        missing.append("Engagement date is after review date")
    elif (review_timestamp - engagement).days > 30:
        reasons.append("No meaningful engagement in over 30 days")

    if not row["next_action"].strip():
        missing.append("Next action missing")

    if pd.isna(action_due):
        missing.append("Action due date missing")
    elif action_due < review_timestamp:
        reasons.append("Action overdue")

    if reasons:
        status = "Needs attention"
    elif missing:
        status = "Needs update"
    elif renewal_soon:
        status = "Renewal planning"
    else:
        status = "No current flags"

    if renewal_soon:
        reasons.append("Renewal within 90 days")

    return pd.Series({
        "Review status": status,
        "Days to renewal": renewal_days,
        "Renewal within 90 days": renewal_soon,
        "Missing information": bool(missing),
        "Reasons": "; ".join(reasons + missing) or "No current rule triggered"
    })


portfolio = pd.concat(
    [accounts, accounts.apply(assess_account, axis=1)],
    axis=1
)

# Metrics and CSM summary follow the selected CSM.
scoped = portfolio.copy()

if selected_csm != "All":
    scoped = scoped[scoped["csm"] == selected_csm]

columns = st.columns(4)
columns[0].metric("Accounts", len(scoped))
columns[1].metric(
    "Needs attention",
    int((scoped["Review status"] == "Needs attention").sum())
)
columns[2].metric(
    "Renewals within 90 days",
    int(scoped["Renewal within 90 days"].sum())
)
columns[3].metric(
    "Missing information",
    int(scoped["Missing information"].sum())
)

st.caption(
    "Metrics can overlap. They follow the CSM filter; "
    "the review-only checkbox filters the account table."
)

st.subheader("Accounts by CSM and review status")
summary = pd.crosstab(scoped["csm"], scoped["Review status"])
st.dataframe(summary)

visible = scoped.copy()

if only_review:
    visible = visible[
        visible["Review status"] != "No current flags"
    ]

status_order = {
    "Needs attention": 0,
    "Needs update": 1,
    "Renewal planning": 2,
    "No current flags": 3
}

visible = visible.assign(
    sort_order=visible["Review status"].map(status_order)
).sort_values(["sort_order", "renewal_date"])

display_columns = [
    "account", "csm", "Review status", "Reasons",
    "renewal_date", "Days to renewal", "adoption_trend",
    "last_engagement", "open_escalation",
    "next_action", "action_due_date"
]

st.subheader("Account review")
st.dataframe(
    visible[display_columns],
    hide_index=True,
    height=550
)

with st.expander("How the review rules work"):
    st.write(
        "Needs attention: declining adoption, an open escalation, "
        "engagement older than 30 days, an overdue action, "
        "or a renewal date that has passed."
    )
    st.write(
        "Needs update: missing information, when no attention rule applies."
    )
    st.write(
        "Renewal planning: renewal within 90 days, "
        "when no attention or missing-information rule applies."
    )
    st.write(
        "No current flags means no configured rule triggered. "
        "It does not establish that an account is healthy."
    )
    st.write(
        "For this demo, action dates represent pending actions. "
        "A date due today is not yet overdue."
    )
NOTE_THEMES = {
    "Adoption": (
        "Customer product usage, feature adoption, onboarding, "
        "user training, low engagement with the application, "
        "and barriers to achieving value."
    ),
    "Stakeholder engagement": (
        "Relationships with customer stakeholders, executive sponsors, "
        "decision makers, champion departures, sponsor changes, "
        "and difficulty arranging customer meetings."
    ),
    "Commercial concerns": (
        "Renewal negotiations, pricing, budget reductions, "
        "contract terms, procurement, subscription costs, "
        "and cancellation discussions."
    ),
    "Support issues": (
        "Technical problems, product defects, outages, "
        "unresolved support tickets, escalations, "
        "and troubleshooting with engineering."
    )
}


@st.cache_resource
def prepare_note_model(descriptions):
    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cpu"
    )
    embeddings = model.encode_document(descriptions)
    return model, embeddings


st.divider()
st.subheader("AI-assisted CSM note categorization")
st.caption(
    "Suggests a theme from the note's meaning. "
    "Account status remains governed by the dashboard rules."
)

account_options = scoped["account"].tolist()

with st.form("note_analysis_form"):
    note_account = st.selectbox(
        "Account for this note",
        account_options
    )
    note_text = st.text_area(
        "Fictional CSM note",
        placeholder=(
            "The executive sponsor has left and we have not "
            "connected with the replacement."
        ),
        height=130
    )
    analyze_note = st.form_submit_button("Analyze note")

if analyze_note:
    note_text = note_text.strip()

    if not note_text:
        st.warning("Please enter a note.")
    else:
        theme_names = list(NOTE_THEMES)
        descriptions = list(NOTE_THEMES.values())

        with st.spinner("Comparing note themes..."):
            model, theme_embeddings = prepare_note_model(descriptions)
            note_embedding = model.encode_query([note_text])
            scores = model.similarity(
                note_embedding, theme_embeddings
            )[0]

        results = sorted(
            [
                {
                    "Theme": name,
                    "Similarity": float(scores[index].item())
                }
                for index, name in enumerate(theme_names)
            ],
            key=lambda result: result["Similarity"],
            reverse=True
        )

        best_theme = results[0]["Theme"]
        best_score = results[0]["Similarity"]
        score_gap = best_score - results[1]["Similarity"]
        st.session_state.pop("confirmed_note_themes", None)
        st.session_state["note_result"] = {
            "account": note_account,
            "text": note_text,
            "suggested_theme": best_theme,
            "scores": results,
            "needs_review": best_score < 0.30 or score_gap < 0.05,
            "review_date": review_date.isoformat()
        }

        st.write(f"Account: {note_account}")

        if best_score < 0.30:
            st.warning("Needs review: no strong match to the defined themes.")
        elif score_gap < 0.05:
            st.warning(
                "Needs review: the two leading themes have similar scores. "
                "The note may cover multiple topics."
            )
        else:
            st.info(f"Suggested theme: {best_theme}")

        st.write("Closest theme description:")
        st.write(NOTE_THEMES[best_theme])

        st.dataframe(
            pd.DataFrame(results).round({"Similarity": 3}),
            hide_index=True
        )

        st.caption(
            "Similarity is not confidence. The 0.30 minimum and 0.05 "
            "score-gap checks are provisional demo settings."
        )
NOTES_DB = Path(__file__).resolve().parent / "portfolio_notes.db"

with closing(sqlite3.connect(NOTES_DB)) as connection:
    connection.execute("""
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            account TEXT NOT NULL,
            note_text TEXT NOT NULL,
            suggested_theme TEXT NOT NULL,
            confirmed_themes TEXT NOT NULL,
            scores TEXT NOT NULL,
            needs_review INTEGER NOT NULL,
            review_date TEXT NOT NULL,
            saved_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    connection.commit()

if "note_saved_message" in st.session_state:
    st.success(st.session_state.pop("note_saved_message"))

pending_note = st.session_state.get("note_result")

if pending_note:
    st.subheader("Confirm and save the analyzed note")
    st.write(f"Account: {pending_note['account']}")
    st.write(pending_note["text"])
    st.caption(
        "You are saving the analyzed text shown above. "
        "To change it, edit the original note and analyze again."
    )

    with st.form("confirm_note_form"):
        confirmed_themes = st.multiselect(
            "Select the themes you confirm",
            list(NOTE_THEMES) + ["Other"],
            key="confirmed_note_themes"
        )
        save_note = st.form_submit_button("Save confirmed note")

    if save_note:
        if not confirmed_themes:
            st.warning("Confirm at least one theme, or select Other.")
        else:
            with closing(sqlite3.connect(NOTES_DB)) as connection:
                connection.execute(
                    """
                    INSERT INTO notes (
                        account, note_text, suggested_theme,
                        confirmed_themes, scores, needs_review, review_date
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        pending_note["account"],
                        pending_note["text"],
                        pending_note["suggested_theme"],
                        json.dumps(confirmed_themes),
                        json.dumps(pending_note["scores"]),
                        int(pending_note["needs_review"]),
                        pending_note["review_date"]
                    )
                )
                connection.commit()

            st.session_state.pop("note_result", None)
            st.session_state["note_saved_message"] = "Confirmed note saved."
            st.rerun()

st.subheader("Saved notes for the selected CSM portfolio")

with closing(sqlite3.connect(NOTES_DB)) as connection:
    saved_notes = pd.read_sql_query(
        """
        SELECT account, note_text, confirmed_themes, saved_at
        FROM notes
        ORDER BY id DESC
        """,
        connection
    )

saved_notes = saved_notes[
    saved_notes["account"].isin(scoped["account"])
].copy()

if saved_notes.empty:
    st.info("No saved notes for these accounts yet.")
else:
    saved_notes["confirmed_themes"] = saved_notes[
        "confirmed_themes"
    ].apply(lambda value: ", ".join(json.loads(value)))

    st.dataframe(saved_notes, hide_index=True)

st.caption(
    "Saved notes do not change account flags automatically. "
    "Saved timestamps are in UTC."
)
        