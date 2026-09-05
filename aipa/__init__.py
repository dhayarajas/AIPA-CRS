"""AIPA-CRS: Adaptive Intent-Preference Arbitration for Conversational Recommendation."""

RELATIONSHIPS = ["Complement", "Consistent", "Conflict", "Override", "Uncertain"]
ACTIONS = ["Fuse", "Prioritize_LTP", "Prioritize_STI", "Ask_Clarification"]
REL2ID = {r: i for i, r in enumerate(RELATIONSHIPS)}
ACT2ID = {a: i for i, a in enumerate(ACTIONS)}
