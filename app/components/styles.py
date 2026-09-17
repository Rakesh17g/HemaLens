"""
HemaLens AI — Premium Design System
=====================================
Inject via inject_css() at the top of every page.
Exports helper HTML functions for shared components.
"""

# ─── Lucide icon SVGs (inline, monochrome, stroke-based) ────────────────────────
ICONS = {
    "home": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>',
    "upload": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" x2="12" y1="3" y2="15"/></svg>',
    "microscope": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M6 18h8"/><path d="M3 22h18"/><path d="M14 22a7 7 0 1 0 0-14h-1"/><path d="M9 14h2"/><path d="M9 12a2 2 0 0 1-2-2V6h6v4a2 2 0 0 1-2 2Z"/><path d="M12 6V3a1 1 0 0 0-1-1H9a1 1 0 0 0-1 1v3"/></svg>',
    "flame": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.153.433-2.294 1-3a2.5 2.5 0 0 0 2.5 2.5z"/></svg>',
    "chart": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><line x1="18" x2="18" y1="20" y2="10"/><line x1="12" x2="12" y1="20" y2="4"/><line x1="6" x2="6" y1="20" y2="14"/></svg>',
    "info": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>',
    "sparkles": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/><path d="M5 3v4"/><path d="M3 5h4"/><path d="M19 17v4"/><path d="M17 19h4"/></svg>',
    "settings": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/></svg>',
    "cpu": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/><path d="M15 2v2"/><path d="M15 20v2"/><path d="M2 15h2"/><path d="M2 9h2"/><path d="M20 15h2"/><path d="M20 9h2"/><path d="M9 2v2"/><path d="M9 20v2"/></svg>',
    "target": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>',
    "file-text": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/><polyline points="14 2 14 8 20 8"/><line x1="16" x2="8" y1="13" y2="13"/><line x1="16" x2="8" y1="17" y2="17"/><line x1="10" x2="8" y1="9" y2="9"/></svg>',
    "alert": '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><path d="M12 9v4"/><path d="M12 17h.01"/></svg>',
    "check": '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>',
    "x": '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>',
    "arrow-right": '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14"/><path d="m12 5 7 7-7 7"/></svg>',
    "scan": '<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" stroke-linecap="round" stroke-linejoin="round"><path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/><path d="M21 17v2a2 2 0 0 1-2 2h-2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/><line x1="7" x2="7.01" y1="12" y2="12"/><line x1="12" x2="12.01" y1="12" y2="12"/><line x1="17" x2="17.01" y1="12" y2="12"/></svg>',
    "image-plus": '<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h7"/><line x1="16" x2="22" y1="5" y2="5"/><line x1="19" x2="19" y1="2" y2="8"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/></svg>',
    "logo": '<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 28 28" fill="none"><circle cx="14" cy="14" r="13" stroke="#a5f9ef" stroke-width="1.2"/><circle cx="14" cy="14" r="7" stroke="#a5f9ef" stroke-width="1.2" stroke-dasharray="2 2"/><circle cx="14" cy="14" r="3" fill="#a5f9ef" opacity="0.9"/><line x1="14" y1="1" x2="14" y2="5" stroke="#a5f9ef" stroke-width="1"/><line x1="14" y1="23" x2="14" y2="27" stroke="#a5f9ef" stroke-width="1"/><line x1="1" y1="14" x2="5" y2="14" stroke="#a5f9ef" stroke-width="1"/><line x1="23" y1="14" x2="27" y2="14" stroke="#a5f9ef" stroke-width="1"/></svg>',
}

HEMALENS_CSS = """
<style>
/* ── Google Fonts ──────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Design tokens ─────────────────────────────────────────── */
:root {
  --bg:         #050505;
  --card:       #0e0e0e;
  --surface:    #141414;
  --border:     rgba(255,255,255,0.07);
  --border-hover: rgba(255,255,255,0.14);
  --accent:     #a5f9ef;
  --accent-dim: rgba(165,249,239,0.10);
  --accent-glow:rgba(165,249,239,0.06);
  --red:        #dc443c;
  --amber:      #9d7250;
  --green:      #8fcdb7;
  --text:       #f0f0f0;
  --subtext:    #888888;
  --muted:      #555555;
  --font:       'Inter', -apple-system, sans-serif;
  --mono:       'JetBrains Mono', monospace;
  --radius:     8px;
  --radius-lg:  14px;
}

/* ── Global reset ──────────────────────────────────────────── */
html, body, [class*="css"], .stApp {
  font-family: var(--font) !important;
  background:  var(--bg) !important;
  color:       var(--text) !important;
}
* { box-sizing: border-box; }

/* ── Hide Streamlit chrome ─────────────────────────────────── */
#MainMenu       { visibility: hidden !important; }
footer          { visibility: hidden !important; }
header          { visibility: hidden !important; }
.stDeployButton { display: none !important; }

/* Ensure no top spacing gap left behind */
section[data-testid="stSidebar"] > div:first-child {
  padding-top: 0 !important;
}

/* ── Scrollbar ─────────────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 4px; }

/* Sidebar configuration */
section[data-testid="stSidebar"] {
  background: #080808 !important;
  border-right: 1px solid var(--border) !important;
  width: 220px !important;
}

/* Hide the native Streamlit sidebar header (which contains the broken keyboard_doub text) */
[data-testid="stSidebarHeader"] {
  display: none !important;
}


/* ── Native Streamlit Navigation Styling (st.page_link) ── */
[data-testid="stPageLink-NavLink"] {
  padding: 8px 18px !important;
  border-radius: 0 !important;
  transition: all 0.15s ease !important;
  text-decoration: none !important;
}
[data-testid="stPageLink-NavLink"]:hover {
  background: rgba(255,255,255,0.03) !important;
}
[data-testid="stPageLink-NavLink"][data-active="true"],
[data-testid="stPageLink-NavLink"][aria-current="page"] {
  background: rgba(165,249,239,0.04) !important;
  border-right: 3px solid var(--accent) !important;
}
[data-testid="stPageLink-NavLink"] p { 
  font-size: 0.8rem !important;
  font-weight: 500 !important;
  color: var(--subtext) !important;
  font-family: var(--font) !important;
  margin: 0 !important;
}
[data-testid="stPageLink-NavLink"][data-active="true"] p,
[data-testid="stPageLink-NavLink"][aria-current="page"] p {
  color: var(--accent) !important;
}
[data-testid="stPageLink-NavLink"] .material-symbols-rounded,
[data-testid="stPageLink-NavLink"] .stIcon-material {
  color: var(--muted) !important;
  font-size: 1.1rem !important;
  margin-right: 10px !important;
  font-family: "Material Symbols Rounded", sans-serif !important;
}
[data-testid="stPageLink-NavLink"][data-active="true"] .material-symbols-rounded,
[data-testid="stPageLink-NavLink"][data-active="true"] .stIcon-material,
[data-testid="stPageLink-NavLink"][aria-current="page"] .material-symbols-rounded,
[data-testid="stPageLink-NavLink"][aria-current="page"] .stIcon-material {
  color: var(--accent) !important;
}

[data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div {
  padding: 0 !important;
}

.hl-nav-divider { border: none; border-top: 1px solid var(--border); margin: 8px 18px; }
.hl-nav-section {
  font-size: 0.58rem;
  font-weight: 600;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--muted) !important;
  padding: 12px 18px 4px;
}

/* Hide default Streamlit sidebar label */
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
  display: none !important;
}

/* ── Main content area ─────────────────────────────────────── */
.main .block-container {
  padding: 2rem 2.5rem 4rem !important;
  max-width: 1400px !important;
}

/* ── Page header ───────────────────────────────────────────── */
.hl-page-header {
  margin-bottom: 2rem;
  padding-bottom: 1.5rem;
  border-bottom: 1px solid var(--border);
}
.hl-eyebrow {
  font-size: 0.65rem;
  font-weight: 600;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--accent);
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 0.6rem;
}
.hl-page-title {
  font-size: 1.75rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: var(--text);
  line-height: 1.15;
  margin: 0 0 0.4rem 0;
}
.hl-page-desc {
  font-size: 0.875rem;
  color: var(--subtext);
  line-height: 1.6;
  margin: 0;
  max-width: 680px;
}

/* ── Section label ─────────────────────────────────────────── */
.hl-section-label {
  font-size: 0.62rem;
  font-weight: 600;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 0.75rem;
  padding-bottom: 0.5rem;
  border-bottom: 1px solid var(--border);
}

/* ── Card ──────────────────────────────────────────────────── */
.hl-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 1.25rem 1.5rem;
  margin-bottom: 1rem;
  transition: border-color 0.2s;
}
.hl-card:hover { border-color: var(--border-hover); }

/* ── KPI card ──────────────────────────────────────────────── */
.hl-kpi {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 1.1rem 0.75rem;
  text-align: center;
  transition: border-color 0.2s;
}
.hl-kpi:hover { border-color: var(--border-hover); }
.hl-kpi-label {
  font-size: 0.6rem;
  font-weight: 600;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 0.5rem;
}
.hl-kpi-value {
  font-size: 2rem;
  font-weight: 700;
  line-height: 1;
  letter-spacing: -0.02em;
}
.hl-kpi-sub {
  font-size: 0.68rem;
  color: var(--muted);
  margin-top: 0.3rem;
}

/* ── Stat row ──────────────────────────────────────────────── */
.hl-stat-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.4rem 0;
  border-bottom: 1px solid var(--border);
  font-size: 0.82rem;
}
.hl-stat-row:last-child { border-bottom: none; }
.hl-stat-key   { color: var(--subtext); }
.hl-stat-value { color: var(--text); font-weight: 500; font-family: var(--mono); font-size: 0.78rem; }

/* ── Progress bar ──────────────────────────────────────────── */
.hl-bar-wrap {
  margin: 0.55rem 0;
}
.hl-bar-header {
  display: flex;
  justify-content: space-between;
  font-size: 0.78rem;
  color: var(--subtext);
  margin-bottom: 0.25rem;
}
.hl-bar-pct { font-weight: 600; color: var(--text); font-family: var(--mono); }
.hl-bar-track {
  background: rgba(255,255,255,0.06);
  border-radius: 9999px;
  height: 3px;
  overflow: hidden;
}
.hl-bar-fill {
  height: 100%;
  border-radius: 9999px;
  background: var(--accent);
  transition: width 0.7s cubic-bezier(.4,0,.2,1);
}

/* ── Status pill ───────────────────────────────────────────── */
.hl-pill {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 0.28rem 0.75rem;
  border-radius: 9999px;
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.04em;
}
.hl-pill-pos  { background: rgba(220,68,60,0.12);  color: #dc443c; border: 1px solid rgba(220,68,60,0.25); }
.hl-pill-neg  { background: rgba(143,205,183,0.10); color: #8fcdb7; border: 1px solid rgba(143,205,183,0.25); }
.hl-pill-low  { background: rgba(143,205,183,0.10); color: #8fcdb7; border: 1px solid rgba(143,205,183,0.25); }
.hl-pill-med  { background: rgba(157,114,80,0.12);  color: #c4956a; border: 1px solid rgba(157,114,80,0.25); }
.hl-pill-high { background: rgba(220,68,60,0.12);   color: #dc443c; border: 1px solid rgba(220,68,60,0.25); }

/* ── Divider ───────────────────────────────────────────────── */
.hl-divider {
  border: none;
  height: 1px;
  background: var(--border);
  margin: 1.5rem 0;
}

/* ── Disclaimer ────────────────────────────────────────────── */
.hl-disclaimer {
  background: rgba(157,114,80,0.07);
  border: 1px solid rgba(157,114,80,0.2);
  border-radius: var(--radius);
  padding: 0.9rem 1.1rem;
  font-size: 0.78rem;
  color: var(--subtext);
  line-height: 1.65;
  display: flex;
  gap: 0.65rem;
  align-items: flex-start;
}
.hl-disclaimer strong { color: #c4956a; }

/* ── Image caption ─────────────────────────────────────────── */
.hl-img-caption {
  text-align: center;
  font-size: 0.7rem;
  color: var(--muted);
  margin-top: 0.3rem;
  letter-spacing: 0.05em;
}

/* ── Uploader ──────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
  border: 1px dashed rgba(255,255,255,0.12) !important;
  border-radius: var(--radius-lg) !important;
  padding: 2.5rem 2rem !important;
  background: rgba(255,255,255,0.015) !important;
  transition: border-color 0.2s, background 0.2s !important;
}
[data-testid="stFileUploader"]:hover {
  border-color: rgba(165,249,239,0.3) !important;
  background: var(--accent-glow) !important;
}

/* ── Streamlit metric overrides ─────────────────────────────── */
[data-testid="metric-container"] {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  padding: 0.75rem 1rem !important;
}

/* ── Buttons ────────────────────────────────────────────────── */
.stButton > button {
  background: var(--surface) !important;
  color: var(--text) !important;
  border: 1px solid var(--border-hover) !important;
  border-radius: var(--radius) !important;
  font-family: var(--font) !important;
  font-weight: 500 !important;
  font-size: 0.825rem !important;
  padding: 0.5rem 1.4rem !important;
  letter-spacing: 0.02em !important;
  transition: background 0.2s, border-color 0.2s, color 0.2s !important;
}
.stButton > button:hover {
  background: var(--text) !important;
  color: var(--bg) !important;
  border-color: var(--text) !important;
}
.stButton > button[kind="primary"] {
  background: var(--accent-dim) !important;
  color: var(--accent) !important;
  border-color: rgba(165,249,239,0.3) !important;
}
.stButton > button[kind="primary"]:hover {
  background: var(--accent) !important;
  color: var(--bg) !important;
  border-color: var(--accent) !important;
}

/* ── Tabs ────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
  background: transparent !important;
  gap: 0 !important;
  border-bottom: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  color: var(--subtext) !important;
  font-family: var(--font) !important;
  font-size: 0.8rem !important;
  font-weight: 500 !important;
  padding: 0.55rem 1.1rem !important;
  border-radius: 0 !important;
  border-bottom: 2px solid transparent !important;
  transition: color 0.15s, border-color 0.15s !important;
}
.stTabs [aria-selected="true"] {
  color: var(--text) !important;
  background: transparent !important;
  border-bottom: 2px solid var(--accent) !important;
}

/* ── Selectbox / inputs ─────────────────────────────────────── */
.stSelectbox > div > div,
.stTextInput > div > div > input {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  color: var(--text) !important;
  font-size: 0.82rem !important;
}

/* ── Slider ─────────────────────────────────────────────────── */
.stSlider [data-baseweb="slider"] {
  padding: 0 !important;
}

/* ── Alerts ─────────────────────────────────────────────────── */
.stAlert {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  font-size: 0.82rem !important;
}

/* ── Code / mono ─────────────────────────────────────────────── */
code { font-family: var(--mono) !important; font-size: 0.82em; }

/* ── Hero (landing page only) ────────────────────────────────── */
.hl-hero {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  justify-content: center;
  padding: 6rem 0 4rem;
  position: relative;
  overflow: hidden;
}
.hl-hero-eyebrow {
  font-size: 0.65rem;
  font-weight: 600;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: var(--accent);
  margin-bottom: 1.2rem;
  display: flex;
  align-items: center;
  gap: 8px;
}
.hl-hero-eyebrow::before {
  content: '';
  display: inline-block;
  width: 24px;
  height: 1px;
  background: var(--accent);
}
.hl-hero-title {
  font-size: clamp(2.8rem, 6vw, 5.5rem);
  font-weight: 700;
  letter-spacing: -0.03em;
  line-height: 1.04;
  color: var(--text);
  margin: 0 0 1.5rem;
}
.hl-hero-title em {
  font-style: normal;
  color: var(--accent);
}
.hl-hero-desc {
  font-size: clamp(0.9rem, 1.5vw, 1.05rem);
  color: var(--subtext);
  line-height: 1.7;
  max-width: 560px;
  margin-bottom: 2.5rem;
}
.hl-hero-cta {
  display: flex;
  gap: 1rem;
  flex-wrap: wrap;
  align-items: center;
}
.hl-btn-primary {
  background: var(--accent) !important;
  color: var(--bg) !important;
  border: 1px solid var(--accent) !important;
  border-radius: var(--radius) !important;
  font-weight: 600 !important;
  font-size: 0.85rem !important;
  padding: 0.65rem 1.7rem !important;
  letter-spacing: 0.02em !important;
  cursor: pointer;
  transition: opacity 0.2s !important;
  font-family: var(--font) !important;
}
.hl-btn-primary:hover { opacity: 0.88 !important; }
.hl-btn-secondary {
  background: transparent !important;
  color: var(--subtext) !important;
  border: 1px solid var(--border-hover) !important;
  border-radius: var(--radius) !important;
  font-weight: 500 !important;
  font-size: 0.82rem !important;
  padding: 0.63rem 1.5rem !important;
  cursor: pointer;
  transition: color 0.2s, border-color 0.2s !important;
  font-family: var(--font) !important;
}
.hl-btn-secondary:hover { color: var(--text) !important; border-color: var(--border-hover) !important; }

/* ── Feature strip ───────────────────────────────────────────── */
.hl-strip {
  border-top: 1px solid var(--border);
  border-bottom: 1px solid var(--border);
  padding: 1rem 0;
  display: flex;
  gap: 2.5rem;
  flex-wrap: wrap;
  align-items: center;
}
.hl-strip-item {
  font-size: 0.7rem;
  font-weight: 500;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--muted);
  display: flex;
  align-items: center;
  gap: 6px;
}

/* ── Workflow steps ──────────────────────────────────────────── */
.hl-step-num {
  font-size: 0.62rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  color: var(--accent);
  font-family: var(--mono);
  margin-bottom: 0.5rem;
}
.hl-step-title {
  font-size: 1rem;
  font-weight: 600;
  color: var(--text);
  margin-bottom: 0.3rem;
  letter-spacing: -0.01em;
}
.hl-step-desc {
  font-size: 0.8rem;
  color: var(--subtext);
  line-height: 1.55;
}

/* ── Animations ──────────────────────────────────────────────── */
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(16px); }
  to   { opacity: 1; transform: translateY(0); }
}
.fade-up { animation: fadeUp 0.5s ease both; }

@keyframes hl-pulse {
  0%,100% { box-shadow: 0 0 0 0 rgba(165,249,239,0); }
  50%      { box-shadow: 0 0 0 6px rgba(165,249,239,0.05); }
}
.hl-pulse { animation: hl-pulse 3s ease infinite; }

/* ── Backwards compat aliases ────────────────────────────────── */
.medical-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 1.1rem 1.4rem;
  margin-bottom: 0.75rem;
  transition: border-color 0.2s;
}
.medical-card:hover { border-color: var(--border-hover); }
.section-title {
  font-size: 0.62rem;
  font-weight: 600;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 0.75rem;
  padding-bottom: 0.5rem;
  border-bottom: 1px solid var(--border);
}
.stat-row { display:flex; justify-content:space-between; align-items:center; padding:0.38rem 0; border-bottom:1px solid var(--border); font-size:0.82rem; }
.stat-row:last-child { border-bottom:none; }
.stat-key   { color: var(--subtext); }
.stat-value { color: var(--text); font-weight:500; font-family:var(--mono); font-size:0.78rem; }
.conf-bar-track { background:rgba(255,255,255,0.06); border-radius:9999px; height:3px; overflow:hidden; margin-top:0.3rem; }
.conf-bar-fill  { height:100%; border-radius:9999px; background:var(--accent); transition:width 0.6s ease; }
.risk-badge { display:inline-flex; align-items:center; gap:5px; padding:0.28rem 0.75rem; border-radius:9999px; font-weight:600; font-size:0.72rem; letter-spacing:0.04em; }
.risk-low    { background:rgba(143,205,183,0.10); color:#8fcdb7; border:1px solid rgba(143,205,183,0.25); }
.risk-medium { background:rgba(157,114,80,0.12);  color:#c4956a; border:1px solid rgba(157,114,80,0.25); }
.risk-high   { background:rgba(220,68,60,0.12);   color:#dc443c; border:1px solid rgba(220,68,60,0.25); }
.pred-pill   { display:inline-flex; align-items:center; gap:5px; padding:0.4rem 1.1rem; border-radius:9999px; font-size:1rem; font-weight:700; letter-spacing:0.03em; }
.pred-positive { background:rgba(220,68,60,0.12);   color:#dc443c; border:2px solid rgba(220,68,60,0.3); }
.pred-negative { background:rgba(143,205,183,0.10); color:#8fcdb7; border:2px solid rgba(143,205,183,0.3); }
.disclaimer { background:rgba(157,114,80,0.07); border:1px solid rgba(157,114,80,0.2); border-radius:var(--radius); padding:0.9rem 1.1rem; font-size:0.78rem; color:var(--subtext); line-height:1.65; }
.fancy-hr { border:none; height:1px; background:var(--border); margin:1.5rem 0; }
.img-caption { text-align:center; font-size:0.7rem; color:var(--muted); margin-top:0.3rem; letter-spacing:0.05em; }

</style>
"""

# ─── Sidebar HTML ────────────────────────────────────────────────────────────────
def _sidebar_nav(active: str = "home") -> str:
    items = [
        ("home",       "home",       "Home"),
        ("upload",     "upload",     "Upload"),
        ("prediction", "microscope", "Prediction"),
        ("gradcam",    "flame",      "Explainability"),
        ("metrics",    "chart",      "Metrics"),
        ("analysis",   "sparkles",   "Full Analysis"),
        ("about",      "info",       "About"),
    ]
    nav_html = f"""
<div class="hl-sidebar-brand">
  {ICONS["logo"]}
  <div>
    <div class="hl-brand-name">HEMALENS</div>
    <div class="hl-brand-sub">AI Screening</div>
  </div>
</div>
<div class="hl-nav-section">Navigation</div>
"""
    for key, icon, label in items:
        cls = "active" if key == active else ""
        nav_html += f"""<div class="hl-nav-item {cls}">
  <span class="hl-nav-icon">{ICONS[icon]}</span>{label}</div>"""
    nav_html += """
<hr class="hl-nav-divider">
<div class="hl-nav-section">System</div>
"""
    nav_html += f"""<div class="hl-nav-item"><span class="hl-nav-icon">{ICONS["settings"]}</span>Settings</div>"""
    return nav_html


def inject_css(active: str = "home") -> None:
    """Call at the top of every Streamlit page."""
    import streamlit as st
    st.markdown(HEMALENS_CSS, unsafe_allow_html=True)


# ─── Page header helper ──────────────────────────────────────────────────────────
def page_header(eyebrow: str, title: str, desc: str = "") -> str:
    desc_html = f'<p class="hl-page-desc">{desc}</p>' if desc else ""
    return f"""
<div class="hl-page-header fade-up">
  <div class="hl-eyebrow">{eyebrow}</div>
  <h1 class="hl-page-title">{title}</h1>
  {desc_html}
</div>"""


def section_label(text: str) -> str:
    return f'<div class="hl-section-label">{text}</div>'


# ─── Backwards-compat shims (used by existing pages) ──────────────────────────
def card(content_html: str, extra_class: str = "") -> str:
    return f'<div class="hl-card {extra_class}">{content_html}</div>'


def section_title(text: str) -> str:
    return f'<div class="section-title">{text}</div>'


def risk_badge(risk: str) -> str:
    cls = {"LOW": "risk-low", "MEDIUM": "risk-medium", "HIGH": "risk-high"}.get(risk.upper(), "risk-medium")
    dot = {"LOW": "●", "MEDIUM": "●", "HIGH": "●"}.get(risk.upper(), "●")
    return f'<span class="risk-badge {cls}">{dot} {risk}</span>'


def pred_pill(prediction: str) -> str:
    is_pos = "ALL" in prediction.upper()
    cls = "pred-positive" if is_pos else "pred-negative"
    return f'<span class="pred-pill {cls}">{prediction}</span>'


def confidence_bar(score: float, label: str = "Confidence") -> str:
    pct = int(score * 100)
    return f"""
<div class="hl-bar-wrap">
  <div class="hl-bar-header">
    <span>{label}</span>
    <span class="hl-bar-pct">{pct}%</span>
  </div>
  <div class="hl-bar-track">
    <div class="hl-bar-fill" style="width:{pct}%;"></div>
  </div>
</div>"""


def stat_row(key: str, value: str) -> str:
    return (f'<div class="stat-row">'
            f'<span class="stat-key">{key}</span>'
            f'<span class="stat-value">{value}</span>'
            f'</div>')


def fancy_hr() -> str:
    return '<hr class="fancy-hr">'
