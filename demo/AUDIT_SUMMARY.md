# Audit Summary

## Checklist

- A) Demo audit output: PASS
- B) Deterministic primitive ordering: PASS
- C) README precision: PASS
- D) Tests enforce guarantees: PASS

## Commands Used

```bash
PATH="/Users/tetsuy/repo/stabilized-mhc/.venv/bin:$PATH" python demo/constrained_routing_demo.py --primitives ../synthetic-danger/demo_outputs/primitives.yaml
PATH="/Users/tetsuy/repo/stabilized-mhc/.venv/bin:$PATH" pytest -q
```

## Key Numeric Outputs

- primitives.yaml: `/Users/tetsuy/repo/synthetic-danger/demo_outputs/primitives.yaml`
- H.shape before mask: `(1, 7, 7)`
- H.shape after mask: `(1, 7, 7)`
- Forbidden columns: aggressive_accel=4, high_speed_follow=5, open_tool_call=6
- max(H[:, forbidden_cols]) before: `0.24362638592720032`
- max(H[:, forbidden_cols]) after: `0.0`
- row sum error before: `1.1920928955078125e-07`
- row sum error after: `0.0`
- col sum error after (vs target): `0.0`
