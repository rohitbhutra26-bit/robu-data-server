"""
ROBU Discovery Engine
=====================

Self-contained package that turns the data server's stock universe into a ranked
feed of proactive investment ideas. Designed so main.py only needs ONE line:

    from discovery import init_discovery
    init_discovery(app, deps=DiscoveryDeps(...))

Pipeline (runs as a nightly batch, NOT on user request):

    universe ──▶ quant funnel ──▶ candidates (top N)
                                      │
                                      ├─ feeds (news, FII/DII)
                                      ├─ 7 Gemini agents
                                      └─ scoring engine ──▶ Discovery Store (SQLite)
                                                                    │
                                                       /discovery (read) ──▶ UI

See ROBU_Discovery_Engine_Architecture.docx for the full design.
"""

from .deps import DiscoveryDeps
from .integration import init_discovery

__all__ = ["DiscoveryDeps", "init_discovery"]
