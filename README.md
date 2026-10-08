\# Customer Portfolio Review Dashboard



A Customer Success leadership prototype using 15 fictional accounts

assigned to three fictional CSMs.



\## Features

\- Filter the portfolio by CSM and review date

\- Flag declining adoption, open escalations, stale engagement,

&#x20; overdue actions, and passed renewal dates

\- Highlight renewals within 90 days

\- Identify missing information

\- Show account actions, due dates, and reasons behind flags

\- Suggest CSM note themes using a local pretrained embedding model

\- Request review for weak or closely competing theme matches

\- Let users confirm multiple themes and save notes in SQLite



\## Architecture

Streamlit provides the interface and pandas handles account data.



Account flags use explicit Python rules.



Sentence Transformers compares notes with four theme descriptions:

Adoption, Stakeholder engagement, Commercial concerns, and Support issues.



SQLite stores notes, AI suggestions, scores, and user-confirmed themes.

Saved notes do not automatically change account flags.



\## Run on Windows

Requires Python 3.13.



Create an environment:

`py -3.13 -m venv .venv`



Install dependencies:

`.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt`



Start the app:

`.\\.venv\\Scripts\\python.exe -m streamlit run app.py`



The fictional\_accounts.csv file must be beside app.py.

The first note analysis downloads the pretrained model.

No paid AI API or API key is required.



\## Demo checks

At the review date 2026-10-08, the full portfolio shows:

\- 15 accounts

\- 7 needing attention

\- 9 renewals within 90 days

\- 2 accounts with missing information



CSM filtering and review-only filtering were manually checked.

Each of the four note themes matched its corresponding example.

An unrelated cafeteria note triggered review.

A mixed adoption/commercial note triggered ambiguity review.

Confirmed multi-theme notes persisted after browser refresh.



\## Limitations

Rules and thresholds are illustrative, not validated business policy.

Similarity scores are not confidence percentages.

The note checks use a small manual evaluation set.

No current flags does not establish that an account is healthy.

CSV action dates represent pending actions.

The account table is read-only in this version.

This local prototype has no authentication or CRM integration.



\## Local data

Confirmed notes are stored in portfolio\_notes.db.

The database is excluded from version control.



\## Development

Built with AI coding assistance and manually tested.



\## Next steps

Expand evaluation, validate uploaded CSVs, and add reviewed account updates.

