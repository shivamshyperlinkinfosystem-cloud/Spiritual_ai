import os
import json
import re
import smtplib
import streamlit as st
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

st.set_page_config(
    page_title="Dora — your profile companion",
    page_icon=":material/favorite:",
    layout="wide",
)

# ── Minimal structural CSS (config.toml handles colors/fonts/radius) ──────────
st.html("""
<style>
/* Remove default page top padding so hero touches the top */
[data-testid="stAppViewContainer"] > .main > .block-container {
    padding-top: 0 !important;
}

/* Avatar circle */
.dora-avatar {
    width: 72px; height: 72px; border-radius: 50%;
    background: linear-gradient(135deg, #5b21b6 0%, #a855f7 100%);
    color: #fff; font-size: 28px; font-weight: 700;
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 6px 20px rgba(124,58,237,.35);
    letter-spacing: -1px; flex-shrink: 0;
}
/* Tag pills in profile */
.tag {
    display: inline-block;
    background: #ede9fe; color: #5b21b6;
    border-radius: 999px; padding: 4px 13px;
    font-size: 12.5px; margin: 3px 3px 3px 0;
    font-weight: 500; border: 1px solid #ddd6fe;
    font-family: Poppins, sans-serif;
}
.tag.green  { background:#d1fae5; color:#065f46; border-color:#a7f3d0; }
.tag.blue   { background:#dbeafe; color:#1e40af; border-color:#bfdbfe; }
.tag.orange { background:#fef3c7; color:#92400e; border-color:#fde68a; }

/* Gradient hero banner */
.dora-hero {
    background: linear-gradient(135deg, #4c1d95 0%, #6d28d9 45%, #a855f7 100%);
    border-radius: 20px; padding: 28px 36px;
    display: flex; align-items: center; gap: 20px;
    margin-bottom: 24px; position: relative; overflow: hidden;
}
.dora-hero::after {
    content: "";
    position: absolute; top: -40px; right: -40px;
    width: 200px; height: 200px; border-radius: 50%;
    background: rgba(255,255,255,.07);
}
.dora-hero-text h1 {
    margin: 0 0 4px; font-size: 28px; font-weight: 700; color: #fff;
    font-family: Poppins, sans-serif;
}
.dora-hero-text p {
    margin: 0; font-size: 14px; color: rgba(255,255,255,.75);
    font-family: Poppins, sans-serif;
}
.dora-hero-icon {
    width: 56px; height: 56px; border-radius: 16px;
    background: rgba(255,255,255,.18); backdrop-filter: blur(8px);
    display: flex; align-items: center; justify-content: center;
    font-size: 28px; flex-shrink: 0;
}

/* Profile card header strip */
.card-header {
    background: linear-gradient(135deg, #5b21b6 0%, #7c3aed 100%);
    border-radius: 16px 16px 0 0; padding: 24px 20px 20px;
    display: flex; align-items: center; gap: 16px; margin: -20px -20px 20px;
}
.card-header-text h3 {
    margin: 0 0 3px; font-size: 18px; font-weight: 600; color: #fff;
    font-family: Poppins, sans-serif;
}
.card-header-text p {
    margin: 0; font-size: 12.5px; color: rgba(255,255,255,.7);
    font-family: Poppins, sans-serif; font-style: italic;
}

/* Completeness ring (visual stat row) */
.stat-row {
    display: flex; gap: 10px; margin: 12px 0;
}
.stat-box {
    flex: 1; background: #f5f3ff; border-radius: 12px;
    padding: 10px 12px; text-align: center;
    border: 1px solid #ede9fe;
}
.stat-box .stat-num {
    font-size: 22px; font-weight: 700; color: #6d28d9;
    font-family: Poppins, sans-serif; line-height: 1;
}
.stat-box .stat-label {
    font-size: 11px; color: #7c3aed; margin-top: 3px;
    font-family: Poppins, sans-serif; font-weight: 500;
    text-transform: uppercase; letter-spacing: .5px;
}

/* Section label */
.sec-label {
    font-size: 11px; font-weight: 600; color: #9ca3af;
    text-transform: uppercase; letter-spacing: .7px;
    margin: 14px 0 6px; font-family: Poppins, sans-serif;
}

/* Connection goal row */
.goal-row {
    display: flex; align-items: flex-start; gap: 8px;
    margin-bottom: 7px; font-size: 13.5px; color: #374151;
    font-family: Poppins, sans-serif;
}
.goal-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: #7c3aed; margin-top: 6px; flex-shrink: 0;
}

/* Empty state */
.empty-state {
    text-align: center; padding: 28px 16px;
    color: #a78bfa; font-family: Poppins, sans-serif;
}
.empty-state .icon { font-size: 40px; margin-bottom: 10px; }
.empty-state p { margin: 0; font-size: 13px; line-height: 1.6; }

/* Saved banner */
.saved-banner {
    background: linear-gradient(135deg, #065f46, #059669);
    border-radius: 14px; padding: 16px 20px;
    display: flex; align-items: center; gap: 12px; margin-top: 12px;
}
.saved-banner p { margin: 0; color: #fff; font-size: 13.5px;
                   font-family: Poppins, sans-serif; }
.saved-banner .sb-icon { font-size: 22px; }
</style>
""")

# ── Secrets ───────────────────────────────────────────────────────────────────
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

def get_secret(name):
    value = os.getenv(name)

    if value is None:
        value = st.secrets.get(name)

    if not value:
        raise RuntimeError(f"Missing required secret: {name}")

    return str(value)

SENDER_EMAIL = get_secret("SENDER_EMAIL")
SENDER_PASSWORD = get_secret("SENDER_PASSWORD").replace(" ", "")
GROQ_API_KEY = get_secret("GROQ_API_KEY")
# ── Email ─────────────────────────────────────────────────────────────────────
def send_profile_email(to_email: str, profile: dict) -> tuple[bool, str]:
    if not SENDER_EMAIL or not SENDER_PASSWORD:
        return False, "Email credentials not set in .env"
    try:
        name = profile.get("name") or "Member"

        def _pills(items: list, color: str = "#f3f0ff", text: str = "#5b21b6", border: str = "#ddd6fe") -> str:
            if not items:
                return "<em style='color:#9ca3af'>—</em>"
            return "".join(
                f'<span style="background:{color};color:{text};border:1px solid {border};'
                f'border-radius:999px;padding:4px 12px;font-size:13px;'
                f'margin:3px 3px 3px 0;display:inline-block;font-family:Poppins,sans-serif'
                f'">{i}</span>'
                for i in items
            )

        def _goals(items: list) -> str:
            if not items:
                return "<em style='color:#9ca3af'>—</em>"
            return "".join(
                f'<div style="display:flex;gap:8px;margin-bottom:6px;align-items:flex-start">'
                f'<span style="width:7px;height:7px;border-radius:50%;background:#7c3aed;'
                f'margin-top:6px;flex-shrink:0;display:inline-block"></span>'
                f'<span style="font-size:13.5px;color:#374151;font-family:Poppins,sans-serif">{g}</span>'
                f'</div>'
                for g in items
            )

        def _row(label: str, content: str) -> str:
            return f"""
            <tr>
              <td style="padding:14px 0 6px;vertical-align:top;
                         font-size:11px;font-weight:600;color:#9ca3af;
                         text-transform:uppercase;letter-spacing:.6px;
                         font-family:Poppins,sans-serif;width:120px">{label}</td>
              <td style="padding:14px 0 6px">{content}</td>
            </tr>"""

        html = f"""<!DOCTYPE html>
<html><body style="margin:0;padding:20px;background:#f5f3ff;font-family:Poppins,sans-serif">
<div style="max-width:580px;margin:auto;background:#fff;border-radius:20px;
            overflow:hidden;box-shadow:0 4px 30px rgba(109,40,217,.12)">

  <div style="background:linear-gradient(135deg,#4c1d95 0%,#6d28d9 50%,#a855f7 100%);
              padding:32px 36px 28px">
    <p style="margin:0 0 6px;font-size:11px;color:rgba(255,255,255,.65);
              text-transform:uppercase;letter-spacing:1.2px">Profile by Dora</p>
    <h1 style="margin:0;font-size:30px;font-weight:700;color:#fff">{name}</h1>
    {f'<p style="margin:10px 0 0;color:rgba(255,255,255,.8);font-size:14px;line-height:1.6">{profile.get("bio","")}</p>' if profile.get("bio") else ""}
  </div>

  <div style="padding:28px 36px">
    <table style="width:100%;border-collapse:collapse">
      {_row("Interests",        _pills(profile.get("interests",[])))}
      {_row("Hobbies",          _pills(profile.get("hobbies",[]),   "#d1fae5","#065f46","#a7f3d0"))}
      {_row("Activities",       _pills(profile.get("activities",[]),"#dbeafe","#1e40af","#bfdbfe"))}
      {_row("Looking to connect with", _goals(profile.get("connection_goals",[])))}
    </table>

    <details style="margin-top:24px;border:1px solid #ede9fe;border-radius:12px;overflow:hidden">
      <summary style="padding:12px 16px;cursor:pointer;font-size:13px;font-weight:600;
                      color:#7c3aed;background:#faf8ff">Raw JSON</summary>
      <pre style="background:#f5f3ff;margin:0;padding:16px;font-size:12px;
                  color:#4c1d95;overflow:auto">{json.dumps(profile, indent=2)}</pre>
    </details>
  </div>

  <div style="background:#faf8ff;padding:16px 36px;border-top:1px solid #ede9fe;
              font-size:12px;color:#a78bfa;text-align:center">
    Sent with ♡ by Dora — your profile companion
  </div>
</div>
</body></html>"""

        msg = MIMEMultipart("alternative")
        msg["From"]    = SENDER_EMAIL
        msg["To"]      = to_email
        msg["Subject"] = f"✨ Your Dora profile — {name}"
        msg.attach(MIMEText(
            f"Hi {name},\n\nHere's your profile:\n\n{json.dumps(profile, indent=2)}\n\nSent by Dora.",
            "plain",
        ))
        msg.attach(MIMEText(html, "html"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as srv:
            srv.login(SENDER_EMAIL, SENDER_PASSWORD)
            srv.sendmail(SENDER_EMAIL, to_email, msg.as_string())
        return True, ""
    except Exception as e:
        return False, str(e)

# ── LLMs ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_llm() -> ChatGroq:
    return ChatGroq(model="openai/gpt-oss-120b", temperature=0.7,
                    api_key=GROQ_API_KEY or None)

@st.cache_resource
def get_extractor() -> ChatGroq:
    return ChatGroq(model="openai/gpt-oss-120b", temperature=0,
                    api_key=GROQ_API_KEY or None)

# ── Prompts ───────────────────────────────────────────────────────────────────
DORA_SYSTEM = """\
You are Dora — a warm, fun companion on a dating app, helping members build their profile.
This is a DATING APP. People are here to find romantic connections and meaningful relationships.

GOLDEN RULE: Read the room. Short answers = keep it brief. Wrap up in 4–5 exchanges max.

WHAT YOU NEED TO COLLECT (weave naturally — never fire multiple questions at once):
1. Full name
2. Personality / vibe / lifestyle
3. Hobbies and interests
4. What they're looking for romantically (casual, serious, open, etc.)
5. Their email address — ask naturally near the end: "Mind sharing your email so we can send you a copy of your profile?"
6. Their phone number — ask right after email: "And a phone number? Totally optional but helps with verification."

DATING-APPROPRIATE QUESTIONS:
✓ "What do you usually do on a good weekend?"
✓ "Are you looking for something serious, casual, or just seeing what happens?"
✓ "What kind of person do you usually vibe with?"
✓ "What's something someone would discover about you on a first date?"
✓ "More of a homebody or always out somewhere?"

NEVER ASK:
✗ "What community are you looking for?"
✗ "Who are you hoping to connect with — workout buddies or trail-blazers?"
✗ Anything that sounds like a LinkedIn or job-application question

REPLY RULES:
- Max 2 sentences per turn. Warm and playful.
- Acknowledge what they shared, then ask ONE natural follow-up if still needed.
- Don't follow up on the same topic twice — move on.

WRAPPING UP:
Once you have name + vibe + interests + relationship goal + email + phone → end warmly and append: [PROFILE_READY]
If they skip phone, that's fine — still wrap up once you have the rest.
"""

EXTRACTION_SYSTEM = """\
Extract profile data from this conversation. Return ONLY valid JSON, no fences, no explanation.
Use null for missing fields, [] for empty lists.
{
  "name": "string or null",
  "bio": "1–2 sentence warm third-person summary or null",
  "interests": ["concise labels"],
  "hobbies": ["concise labels"],
  "activities": ["specific activities"],
  "looking_for": "what they want romantically (e.g. serious relationship, casual dating, open) or null",
  "email": "their email address or null",
  "phone": "their phone number or null"
}
"""

# ── Session state ─────────────────────────────────────────────────────────────
def _init():
    st.session_state.setdefault("chat", [])
    st.session_state.setdefault("profile", {
        "name": None, "bio": None,
        "interests": [], "hobbies": [], "activities": [],
        "looking_for": None, "email": None, "phone": None,
    })
    st.session_state.setdefault("profile_ready", False)
    st.session_state.setdefault("profile_saved", False)
    st.session_state.setdefault("awaiting_email", False)
    st.session_state.setdefault("email_sent", False)
    st.session_state.setdefault("manual_email", "")
    st.session_state.setdefault("seeded", False)

_init()

# ── Helpers ───────────────────────────────────────────────────────────────────
def history_to_lc(chat):
    return [HumanMessage(m["content"]) if m["role"] == "user"
            else AIMessage(m["content"]) for m in chat]

def extract_profile(chat):
    if not chat:
        return st.session_state.profile
    conv = "\n".join(
        f"{'Member' if m['role']=='user' else 'Dora'}: {m['content']}" for m in chat
    )
    try:
        raw = get_extractor().invoke([
            SystemMessage(EXTRACTION_SYSTEM),
            HumanMessage(f"Conversation:\n\n{conv}"),
        ]).content.strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
        d = json.loads(raw)
        return {
            "name":        d.get("name"),
            "bio":         d.get("bio"),
            "interests":   d.get("interests") or [],
            "hobbies":     d.get("hobbies") or [],
            "activities":  d.get("activities") or [],
            "looking_for": d.get("looking_for"),
            "email":       d.get("email"),
            "phone":       d.get("phone"),
        }
    except Exception:
        return st.session_state.profile

def completeness(p):
    checks = [
        bool(p.get("name")),
        bool(p.get("bio")),
        bool(p.get("interests")),
        bool(p.get("hobbies")),
        bool(p.get("looking_for")),
        bool(p.get("email")),
    ]
    return int(sum(checks) / len(checks) * 100)

def call_dora(user_msg):
    msgs = [SystemMessage(DORA_SYSTEM)] + history_to_lc(st.session_state.chat) + [HumanMessage(user_msg)]
    return get_llm().invoke(msgs).content

def scan_chat_for_email(chat: list) -> str | None:
    """Reliably find an email address typed anywhere in the conversation."""
    pattern = r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
    for msg in reversed(chat):       # most recent first
        if msg["role"] == "user":
            m = re.search(pattern, msg["content"])
            if m:
                return m.group(0)
    return None

def initials(name):
    if not name:
        return "✦"
    parts = name.strip().split()
    return (parts[0][0] + (parts[-1][0] if len(parts) > 1 else "")).upper()

def _tags_html(items, css_class=""):
    return "".join(f'<span class="tag {css_class}">{i}</span>' for i in items)

# ── Seed opening message ──────────────────────────────────────────────────────
if not st.session_state.seeded:
    st.session_state.chat.append({"role": "assistant", "content": (
        "Hey! I'm **Dora** — think of me as your profile wingperson. 😊\n\n"
        "I'm going to ask you a few things, but I promise it won't feel like a form. "
        "Just a chill chat to help you show up as *you* to the people you want to connect with.\n\n"
        "So — what's your name, and what's been taking up your time lately?"
    )})
    st.session_state.seeded = True

# ══════════════════════════════════════════════════════════════
# PAGE LAYOUT
# ══════════════════════════════════════════════════════════════
chat_col, card_col = st.columns([13, 7], gap="large")

# ────────────────────────────────────────────────────────────
# LEFT — Chat
# ────────────────────────────────────────────────────────────
with chat_col:
    # Hero banner
    st.html("""
    <div class="dora-hero">
      <div class="dora-hero-icon">💜</div>
      <div class="dora-hero-text">
        <h1>Dora</h1>
        <p>Your profile companion — a real conversation, not a form</p>
      </div>
    </div>
    """)

    # Chat history
    for msg in st.session_state.chat:
        avatar = ":material/favorite:" if msg["role"] == "assistant" else None
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # Quick-start chips (before first user message)
    if not any(m["role"] == "user" for m in st.session_state.chat):
        opts = {
            ":material/waving_hand: I'm new here":       "Hey! I'm new here and want to set up my profile.",
            ":material/edit: Update my profile":          "I'd like to update and improve my existing profile.",
            ":material/rocket_launch: Just exploring":    "I'm just checking things out for now.",
        }
        picked = st.pills("Choose a starting point", list(opts.keys()), label_visibility="collapsed")
        if picked:
            st.session_state.chat.append({"role": "user", "content": opts[picked]})
            st.rerun()

    # Input logic
    if not st.session_state.profile_saved:
        user_input = st.chat_input("Talk to Dora…", submit_mode="disable")
        if user_input:
            st.session_state.chat.append({"role": "user", "content": user_input})
            with st.chat_message("assistant", avatar=":material/favorite:"):
                with st.spinner(""):
                    reply = call_dora(user_input)
            done = "[PROFILE_READY]" in reply
            clean = reply.replace("[PROFILE_READY]", "").strip()
            if done:
                st.session_state.profile_ready = True
            st.session_state.chat.append({"role": "assistant", "content": clean})
            st.session_state.profile = extract_profile(st.session_state.chat)
            st.rerun()
    else:
        st.caption(":material/check_circle: Profile saved — come back anytime to update it.")

# ────────────────────────────────────────────────────────────
# RIGHT — Profile card
# ────────────────────────────────────────────────────────────
with card_col:
    p    = st.session_state.profile
    pct  = completeness(p)
    name = p.get("name")
    bio  = p.get("bio")

    # Stat counts
    n_interests  = len(p.get("interests", []))
    n_hobbies    = len(p.get("hobbies", []))
    n_activities = len(p.get("activities", []))

    # ── Card ──────────────────────────────────────────────────
    with st.container(border=True):

        # Gradient card header
        badge_color = "green" if pct == 100 else ("orange" if pct >= 60 else "violet")
        avatar_html = f'<div class="dora-avatar">{initials(name)}</div>'
        card_name   = name or "Your name"
        card_bio    = f'<p class="card-header-text-bio" style="margin:3px 0 0;font-size:12.5px;color:rgba(255,255,255,.65);font-style:italic">{bio}</p>' if bio else ""

        st.html(f"""
        <div class="card-header">
          {avatar_html}
          <div class="card-header-text">
            <h3>{card_name}</h3>
            {card_bio}
          </div>
        </div>
        """)

        # Completeness badge + bar
        col_pct, col_badge = st.columns([3, 1])
        with col_pct:
            st.progress(pct / 100)
        with col_badge:
            st.badge(f"{pct}%", color=badge_color)

        # Stat mini-boxes
        if n_interests + n_hobbies + n_activities > 0:
            st.html(f"""
            <div class="stat-row">
              <div class="stat-box">
                <div class="stat-num">{n_interests}</div>
                <div class="stat-label">Interests</div>
              </div>
              <div class="stat-box">
                <div class="stat-num">{n_hobbies}</div>
                <div class="stat-label">Hobbies</div>
              </div>
              <div class="stat-box">
                <div class="stat-num">{n_activities}</div>
                <div class="stat-label">Activities</div>
              </div>
            </div>
            """)

        # Interests
        if p.get("interests"):
            st.html('<div class="sec-label">Interests</div>'
                    + '<div>' + _tags_html(p["interests"]) + '</div>')

        # Hobbies
        if p.get("hobbies"):
            st.html('<div class="sec-label">Hobbies</div>'
                    + '<div>' + _tags_html(p["hobbies"], "green") + '</div>')

        # Activities
        if p.get("activities"):
            st.html('<div class="sec-label">Activities</div>'
                    + '<div>' + _tags_html(p["activities"], "blue") + '</div>')

        # Looking for (dating goal)
        if p.get("looking_for"):
            st.html(
                '<div class="sec-label">Looking for</div>'
                f'<div><span class="tag orange">{p["looking_for"]}</span></div>'
            )

        # Contact info
        contact_html = ""
        if p.get("email"):
            contact_html += f'<div class="goal-row"><div class="goal-dot"></div>📧 {p["email"]}</div>'
        if p.get("phone"):
            contact_html += f'<div class="goal-row"><div class="goal-dot"></div>📱 {p["phone"]}</div>'
        if contact_html:
            st.html('<div class="sec-label">Contact</div>' + contact_html)

        # Empty state
        if pct == 0:
            st.html("""
            <div class="empty-state">
              <div class="icon">✦</div>
              <p>Your profile will come alive<br>as you chat with Dora.</p>
            </div>
            """)

        # ── Save block ─────────────────────────────────────────
        if st.session_state.profile_ready and not st.session_state.profile_saved:
            st.space("small")
            st.success("Profile looks great! Ready to save?", icon=":material/auto_awesome:")
            c1, c2 = st.columns(2)
            with c1:
                if st.button(":material/save: Save it", type="primary", use_container_width=True):
                    st.session_state.profile_saved = True
                    # Try LLM extraction first, then regex scan as reliable backup
                    to_email = (
                        st.session_state.profile.get("email")
                        or scan_chat_for_email(st.session_state.chat)
                    )
                    if to_email:
                        # Patch profile so card shows it too
                        st.session_state.profile["email"] = to_email
                        with st.spinner("Sending your profile…"):
                            ok, err = send_profile_email(to_email, st.session_state.profile)
                        if ok:
                            st.session_state.email_sent = True
                            st.session_state.chat.append({"role": "assistant", "content":
                                f"Saved and sent to **{to_email}** 📬  \n"
                                "Hope you find someone amazing here! ✨"})
                            st.toast("Saved & emailed!", icon=":material/mark_email_read:")
                        else:
                            st.session_state.chat.append({"role": "assistant", "content":
                                f"Profile saved! 🎉  \n_(Email failed: {err})_"})
                    else:
                        # No email found — ask inline (shown below)
                        st.session_state.awaiting_email = True
                        st.session_state.chat.append({"role": "assistant", "content":
                            "Profile saved! 🎉  \nWhat's your email so I can send you a copy?"})
                    st.rerun()
            with c2:
                if st.button("Keep chatting", use_container_width=True):
                    st.session_state.profile_ready = False
                    st.session_state.chat.append({"role": "assistant", "content":
                        "Of course! What would you like to add or change?"})
                    st.rerun()

        # ── Email input fallback (profile saved but no email found) ──
        if st.session_state.profile_saved and st.session_state.awaiting_email and not st.session_state.email_sent:
            st.space("small")
            manual = st.text_input(
                "Your email address",
                placeholder="you@example.com",
                key="manual_email_input",
                label_visibility="visible",
            )
            if st.button(":material/send: Send my profile", type="primary", use_container_width=True):
                if re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", manual.strip()):
                    st.session_state.profile["email"] = manual.strip()
                    with st.spinner("Sending…"):
                        ok, err = send_profile_email(manual.strip(), st.session_state.profile)
                    if ok:
                        st.session_state.email_sent = True
                        st.session_state.awaiting_email = False
                        st.session_state.chat.append({"role": "assistant", "content":
                            f"Sent to **{manual.strip()}** 📬 Check your inbox!"})
                        st.toast("Emailed!", icon=":material/mark_email_read:")
                        st.rerun()
                    else:
                        st.error(f"Send failed: {err}")
                else:
                    st.warning("Please enter a valid email address.")

        # Saved state
        if st.session_state.profile_saved:
            st.space("small")
            if st.session_state.email_sent:
                st.html("""
                <div class="saved-banner">
                  <span class="sb-icon">📬</span>
                  <p><strong>Saved & emailed!</strong><br>
                  Check your inbox for your profile.</p>
                </div>
                """)
            else:
                st.success("Profile saved!", icon=":material/check_circle:")
