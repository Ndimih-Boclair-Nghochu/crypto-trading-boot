from pathlib import Path
import sys
BASE = Path(__file__).resolve().parent
def patch(rel, reps):
    p = BASE / rel; src = p.read_text(encoding="utf-8")
    for old, new in reps:
        if new in src: continue
        if old not in src: sys.exit(f"FAILED anchor in {rel}: {old[:90]}")
        src = src.replace(old, new, 1)
    p.write_text(src, encoding="utf-8"); print("  patched", rel)
patch("config.py", [(
 '    testnet_trade_count: int = field(default_factory=lambda: _int("TESTNET_TRADE_COUNT", 0))',
 '    testnet_trade_count: int = field(default_factory=lambda: _int("TESTNET_TRADE_COUNT", 0))\n'
 '    lab_mode: bool = field(default_factory=lambda: _bool("LAB_MODE", False))')])
patch("models/confidence_gate.py", [
(' base_threshold = self._confidence_threshold()\n        threshold = base_threshold * 0.85 if extreme_regime else base_threshold'.lstrip(),
 'base_threshold = self._confidence_threshold()\n        threshold = base_threshold * 0.85 if extreme_regime else base_threshold\n'
 '\n        # LAB_MODE: testnet-only relaxation so the execution path can be\n'
 '        # exercised end to end. ANDed with use_testnet -- a loosened gate\n'
 '        # must never apply to real money.\n'
 '        lab = bool(getattr(self.settings, "lab_mode", False)) and self.settings.use_testnet\n'
 '        min_confluence = 35 if lab else 65\n'
 '        min_agreement = 2 if lab else 4'),
('        expected_action = "BUY" if direction == "LONG" else "SELL"\n'
 '        if rl_decision.action != expected_action:\n'
 '            reasons.append(f"RL action {rl_decision.action} does not match {expected_action}")',
 '        expected_action = "BUY" if direction == "LONG" else "SELL"\n'
 '        # A HOLD from an undertrained RL agent is tolerated in lab mode;\n'
 '        # an outright opposite call still vetoes.\n'
 '        rl_conflicts = (\n'
 '            rl_decision.action not in {expected_action, "HOLD"} if lab\n'
 '            else rl_decision.action != expected_action\n'
 '        )\n'
 '        if rl_conflicts:\n'
 '            reasons.append(f"RL action {rl_decision.action} does not match {expected_action}")'),
('        if float(confluence.get("score", 0) or 0) < 65:\n            reasons.append("multi-timeframe confluence below 65")',
 '        if float(confluence.get("score", 0) or 0) < min_confluence:\n            reasons.append(f"multi-timeframe confluence below {min_confluence}")'),
('        if agreement < 4:', '        if agreement < min_agreement:'),
])
print("patch done")
