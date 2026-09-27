#!/usr/bin/env bash
set -euo pipefail
cd ~/crypto-trading-boot
say() { echo -e "\n=== $* ==="; }

say "1/6 Patch LSTM (class weights + lab inference)"
python3 - <<'PYEOF'
from pathlib import Path
import sys
p = Path("models/lstm_model.py"); src = p.read_text(encoding="utf-8")
reps = [
('        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)\n        criterion = nn.CrossEntropyLoss()',
 '        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)\n'
 '        # generate_labels() marks a bar LONG/SHORT only on a >=1.5*ATR excursion, so\n'
 '        # the training set is overwhelmingly NO_TRADE. With an unweighted loss the model\n'
 '        # minimises error by predicting NO_TRADE unconditionally -- ~90% accurate and\n'
 '        # useless, which is what it learned. Inverse-frequency weights fix that.\n'
 '        _counts = np.bincount(y_arr[:split], minlength=3).astype(np.float64)\n'
 '        _counts[_counts == 0] = 1.0\n'
 '        _weights = _counts.sum() / (3.0 * _counts)\n'
 '        logger.info(\n'
 '            f"{symbol}: label counts LONG/SHORT/NO_TRADE={_counts.tolist()} "\n'
 '            f"class_weights={[round(w, 3) for w in _weights.tolist()]}"\n'
 '        )\n'
 '        criterion = nn.CrossEntropyLoss(weight=torch.tensor(_weights, dtype=torch.float32))'),
('            probabilities = {INDEX_TO_SIGNAL[i]: float(probs[i]) for i in range(3)}\n'
 '            idx = int(np.argmax(probs))\n'
 '            return LSTMSignal(INDEX_TO_SIGNAL[idx], float(probs[idx]), probabilities)',
 '            probabilities = {INDEX_TO_SIGNAL[i]: float(probs[i]) for i in range(3)}\n'
 '            idx = int(np.argmax(probs))\n\n'
 '            if _lab_mode():\n'
 '                logger.info(f"LSTM {symbol} probabilities: {probabilities}")\n'
 '                # Testnet only: re-ask as "given we trade, which way?" by renormalising\n'
 '                # over directional classes so confidence stays comparable to the gate.\n'
 '                if INDEX_TO_SIGNAL[idx] == "NO_TRADE":\n'
 '                    long_p = probabilities["LONG"]; short_p = probabilities["SHORT"]\n'
 '                    total = long_p + short_p\n'
 '                    if total > 0:\n'
 '                        direction = "LONG" if long_p >= short_p else "SHORT"\n'
 '                        return LSTMSignal(direction, max(long_p, short_p) / total, probabilities)\n\n'
 '            return LSTMSignal(INDEX_TO_SIGNAL[idx], float(probs[idx]), probabilities)'),
('@dataclass(frozen=True)\nclass LSTMSignal:',
 'def _lab_mode() -> bool:\n'
 '    """LAB_MODE relaxations apply on testnet only -- never against real money."""\n'
 '    try:\n'
 '        from config import settings as _s\n'
 '        return bool(getattr(_s, "lab_mode", False)) and bool(_s.use_testnet)\n'
 '    except Exception:\n'
 '        return False\n\n\n'
 '@dataclass(frozen=True)\nclass LSTMSignal:'),
]
for old, new in reps:
    if new in src: continue
    if old not in src: sys.exit("ANCHOR FAIL: " + old[:100])
    src = src.replace(old, new, 1)
p.write_text(src, encoding="utf-8")
import ast; ast.parse(src)
print("  patched models/lstm_model.py")
PYEOF

say "2/6 Archive old (degenerate) weights"
sudo mkdir -p models/weights_old_$(date +%s)
sudo mv models/weights/* models/weights_old_$(date +%s)/ 2>/dev/null || true
ls models/weights/ || true

say "3/6 Rebuild"
sudo docker compose up -d --build
sleep 20

say "4/6 Retrain with class weights (several minutes)"
sudo docker compose exec -T app python main.py --mode train_models

say "5/6 Label distribution actually seen"
sudo docker compose logs app 2>&1 | grep -i "label counts" | tail -10

say "6/6 Restart + verify"
ls -la models/weights/
sudo docker compose restart app
sleep 45
curl -s http://127.0.0.1/api/health; echo
