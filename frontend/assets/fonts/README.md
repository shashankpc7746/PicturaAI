# Fonts

The two typefaces are served from this folder instead of Google Fonts, so a page
load makes no third-party request and is not render-blocked by a cross-origin
stylesheet. The `@font-face` rules are in `frontend/assets/css/tokens.css`.

| Family | Files | Weights | Licence |
|---|---|---|---|
| Libre Caslon Display (Pablo Impallari) | `libre-caslon-display/libre-caslon-display-latin.woff2`, `…-latin-ext.woff2` | 400 | SIL Open Font License 1.1, `libre-caslon-display/OFL.txt` |
| Schibsted Grotesk (Bakken & Bæck for Schibsted) | `schibsted-grotesk/schibsted-grotesk-latin.woff2`, `…-latin-ext.woff2` | 400–700 (one variable file per subset) | SIL Open Font License 1.1, `schibsted-grotesk/OFL.txt` |

## Source

Downloaded on 16 November 2026 from the files Google Fonts serves to a current
Chrome for this stylesheet:

```
https://fonts.googleapis.com/css2?family=Libre+Caslon+Display&family=Schibsted+Grotesk:wght@400;500;600;700&display=swap
```

- Libre Caslon Display **v18**: `fonts.gstatic.com/s/librecaslondisplay/v18/…`
- Schibsted Grotesk **v7**: `fonts.gstatic.com/s/schibstedgrotesk/v7/…` (Google
  serves the same variable file for 400, 500, 600 and 700)

Only the `latin` and `latin-ext` subsets are kept. The `unicode-range` of each
`@font-face` is copied from that stylesheet, so a browser downloads `latin-ext`
only when a page uses one of its characters. The licence texts come from
`https://raw.githubusercontent.com/google/fonts/main/ofl/<family>/OFL.txt`.

SHA-256:

```
84267e937417f10dff37df818c5ef7f1e9f2e6a703e03c2dd286d3320af53269  libre-caslon-display/libre-caslon-display-latin.woff2
7a17a86fb7c3cd003bf073f0455f49c746424b37f3e1a931663eae80288a1e89  libre-caslon-display/libre-caslon-display-latin-ext.woff2
4c8b93f431d462c696e12b9d6a033feb3394d36e66e781357c496b95d8a75e05  schibsted-grotesk/schibsted-grotesk-latin.woff2
ee9eeb232cc1fe7979dad813a570f4b19f1eb4496daf56c94ec9de63abdb2936  schibsted-grotesk/schibsted-grotesk-latin-ext.woff2
```

## Updating

Fetch the stylesheet above with a current browser's User-Agent, download the
`latin` and `latin-ext` `.woff2` URLs it lists, replace the files here (keep the
names), copy any changed `unicode-range` into `tokens.css`, and update the
versions and hashes in this file.
