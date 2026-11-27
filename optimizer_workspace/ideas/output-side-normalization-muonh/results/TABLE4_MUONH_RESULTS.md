# Hyperball Results in Paper v4 Table 4 Format (`130M`, `20 TPP = 2.68B tokens`)

Generated: `2026-11-27T14:49:41.052531+00:00`

Conventions match Table 4 (`tab:main_scaling`): `Held-out` is final $\mathcal{L}_{\mathrm{sel}}$ in **nats**; all $\Delta$ columns report differences from same-regime `Muon` ($\eta^* = 0.01600$) in **millinats** (negative favors the row method).

## 1. Seed 0 Verified Optima (`130M`, `20 TPP`, `2.68B` tokens, Seed 0)

| Method | LR $\eta$ | Held-out (nats) | $\Delta$ Held-out | Late-11 $\Delta$ | $\Delta$ C4 | $\Delta$ Paloma | Wins |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Muon ($Q=W$) | `0.01600` | `3.2461` | `+0.00` | `+0.00` | `+0.00` | `+0.00` | — |
| MuonT ($Q=W\hat R^2$) | `0.01600` | `3.2342` | `-11.93` | `-23.36` | `-10.75` | *pending Stage 3* | — |
| Muon2-Diag ($Q=W\hat R^2$) | `0.01600` | `3.2318` | `-14.29` | `-26.92` | `-14.32` | *pending Stage 3* | — |
| **Muon2-ExpDiag** ($Q=W\hat R^2$) | `0.01600` | `3.2303` | `-15.77` | `-28.07` | `-14.61` | *pending Stage 3* | — |
| Muon2-ExpDiag ($Q=W$) | `0.01600` | `3.2389` | `-7.19` | `-7.47` | `-4.17` | *pending Stage 3* | — |
| MuonH (Hyperball, $\eta^*$) | `0.01131` | `3.2442` | `-1.90` | `-22.18` | `-0.36` | *pending Stage 3* | — |
| MuonTH (Hyperball, $\eta^*$) | `0.01131` | `3.2349` | `-11.19` | `-32.89` | `-8.77` | *pending Stage 3* | — |
| **Muon2-ExpDiag + MuonH** ($\eta^*$) | `0.01600` | `3.2357` | `-10.36` | `-9.00` | `-6.80` | *pending Stage 3* | — |
| Muon2-ExpDiag + MuonH (matched $0.01131$) | `0.01131` | `3.2359` | `-10.17` | `-32.83` | `-7.44` | *pending Stage 3* | — |
| MuonH (matched $0.01600$) | `0.01600` | `3.2505` | `+4.43` | `+8.60` | `+5.46` | *pending Stage 3* | — |
| MuonTH (matched $0.01600$) | `0.01600` | `3.2370` | `-9.13` | `-7.22` | `-7.84` | *pending Stage 3* | — |

## 2. Fresh Seeds 1–3 Confirmation (`130M`, `20 TPP`, `2.68B` tokens, paired mean ± SD)

| Method | LR $\eta$ | Held-out (nats) | $\Delta$ Held-out | Late-11 $\Delta$ | $\Delta$ C4 | $\Delta$ Paloma | Wins |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Muon ($Q=W$) | `0.01600` | `3.2035` | `+0.00` | `+0.00` | `+0.00` | `+0.00` | — |
| MuonT ($Q=W\hat R^2$) | `0.01600` | `3.1916` | `-11.91 ± 1.40` | `-23.26 ± 2.07` | `-11.79 ± 0.98` | `-11.49 ± 4.18` | `16/16` |
| Muon2-Diag ($Q=W\hat R^2$) | `0.01600` | `3.1919` | `-11.62 ± 1.09` | `-22.91 ± 1.63` | `-12.63 ± 1.22` | `-12.79 ± 1.92` | `16/16` |
| **Muon2-ExpDiag** ($Q=W\hat R^2$) | `0.01600` | `3.1881` | `-15.41 ± 2.18` | `-26.78 ± 2.30` | `-15.64 ± 1.80` | `-15.91 ± 1.68` | `16/16` |
| Muon2-ExpDiag ($Q=W$) | `0.01600` | `3.1956` | `-7.90 ± 3.28` | `-8.07 ± 3.21` | `-7.63 ± 2.29` | `-6.63 ± 2.70` | `15/16` |
| MuonH (Hyperball, $\eta^*$) | `0.01131` | `3.2025` | `-0.99 ± 1.19` | `-21.23 ± 0.53` | `-1.42 ± 1.36` | *pending Stage 3* | — |
| MuonTH (Hyperball, $\eta^*$) | `0.01131` | `3.1923` | `-11.27 ± 0.96` | `-32.82 ± 0.34` | `-9.85 ± 1.31` | *pending Stage 3* | — |
| **Muon2-ExpDiag + MuonH** ($\eta^*$) | — | *running (0/3 seeds)* | — | — | — | — | — |
| Muon2-ExpDiag + MuonH (matched $0.01131$) | — | *running (0/3 seeds)* | — | — | — | — | — |
