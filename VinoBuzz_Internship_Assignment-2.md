# VinoBuzz Internship Assignment
## Automated Wine Photo Sourcing — Accuracy Challenge

**Role:** Technical Intern (Part-time / Full-time)
**Deadline:** 48 hours from receipt
**Estimated Task Duration:** Track your actual time spent and report it in the write-up
**Submission:** Working demo + write-up (max 2 pages or 5 slides)

---

## Background

VinoBuzz (vinobuzz.ai) is Hong Kong's first AI wine marketplace with 4,000+ SKUs. Every listing requires a high-quality product photo. We currently source these photos using an automated pipeline — but accuracy sits at around **50%**. That's not good enough.

Your job: **design and build a better pipeline that hits 90% accuracy.**

---

## Photo Standard

Before starting, visit **vinobuzz.ai/marketplace** and study the product photos. The standard:
- Single bottle, upright
- Clean white or neutral grey background
- Label clearly readable, correct angle
- No blur, no glare, no lifestyle props, no watermarks
- No AI-generated images — must be real product photos

---

## The Core Problem

Finding a photo is easy. Finding the **right** photo is hard.

Our current system fails because it finds a photo of the right producer — but wrong appellation, wrong specific vineyard (climat), or wrong vintage label. In Burgundy especially, names look nearly identical:

- *Domaine Leflaive Puligny-Montrachet* vs *Domaine Leflaive Puligny-Montrachet Les Pucelles 1er Cru* — completely different wines, easy to confuse
- *Olivier Leflaive* vs *Domaine Leflaive* — different producers, similar name

A wrong photo on a wine listing destroys customer trust. We'd rather show "no image" than show the wrong bottle.

---

## The Challenge

Design a pipeline that:
1. Finds a photo for each wine SKU from the web
2. **Verifies** the photo is correct before accepting it
3. Handles cases where no good photo exists
4. Runs at scale — fast enough to process thousands of SKUs

Think about:
- How do you confirm the label text matches exactly? (Producer, appellation, specific vineyard/cru)
- How do you automatically filter out low quality, watermarked, or lifestyle images?
- What's your confidence scoring mechanism?
- What's your fallback when no verified photo can be found?
- How do you handle wines with near-zero online photo coverage?

**Target: 90% accuracy on the 10 test SKUs.**

---

## Sample SKUs (Reference Set)

These 10 wines are live on VinoBuzz. Visit the marketplace to study the approved photo style for each one.

| # | Wine Name | Vintage | Format | Region |
|---|-----------|---------|--------|--------|
| 1 | Caroline Morey Beaune 'Les Grèves' Premier Cru | 2022 | 750ml | Burgundy |
| 2 | Charles Lachaux Côte de Nuits Villages 'Aux Montagnes' | 2019 | 750ml | Burgundy |
| 3 | Château de Charodon Beaune 'Les Chouacheux' Premier Cru | 2023 | 750ml | Burgundy |
| 4 | Charles van Canneyt Griotte Chambertin Grand Cru | 2021 | 750ml | Burgundy |
| 5 | Comtesse de Cherisey Meursault Blagny 'La Genelotte' Monopole 1er Cru | 2019 | 750ml | Burgundy |
| 6 | Champagne André Clouet 'Chalky' Brut | NV | 750ml | Champagne |
| 7 | Château Rayas 'Les Tours' Réserve Grenache Blanc | 2013 | 750ml | Rhône |
| 8 | Colgin Cellars Napa Valley 'Cariad' | 2017 | 750ml | Napa |
| 9 | Castello di Querceto 'Cignale' Colli della Toscana | 1989 | 750ml | Tuscany |
| 10 | Champagne Virginie de Valandraud Blanc | 2019 | 750ml | Bordeaux |

---

## Test SKUs (Your Pipeline Must Find These)

These 10 wines are **not listed on VinoBuzz**. You cannot source photos from our site. Your pipeline must find and verify them independently.

| # | Wine Name | Vintage | Format | Region | Difficulty |
|---|-----------|---------|--------|--------|------------|
| 1 | Domaine Rossignol-Trapet Latricieres-Chambertin Grand Cru | 2017 | 750ml | Burgundy | Hard |
| 2 | Domaine Arlaud Morey-St-Denis 'Monts Luisants' 1er Cru | 2019 | 750ml | Burgundy | Hard |
| 3 | Domaine Taupenot-Merme Charmes-Chambertin Grand Cru | 2018 | 750ml | Burgundy | Hard |
| 4 | Château Fonroque Saint-Émilion Grand Cru Classé | 2016 | 750ml | Bordeaux | Medium |
| 5 | Eric Rodez Cuvée des Crayères Blanc de Noirs | NV | 750ml | Champagne | Medium |
| 6 | Domaine du Tunnel Cornas 'Vin Noir' | 2018 | 750ml | Northern Rhône | Hard |
| 7 | Poderi Colla Barolo 'Bussia Dardi Le Rose' | 2016 | 750ml | Piedmont | Medium |
| 8 | Arnot-Roberts Trousseau Gris Watson Ranch | 2020 | 750ml | Sonoma | Very Hard |
| 9 | Brokenwood Graveyard Vineyard Shiraz | 2015 | 750ml | Hunter Valley | Medium |
| 10 | Domaine Weinbach Riesling 'Clos des Capucins' Vendanges Tardives | 2017 | 750ml | Alsace | Hard |

---

## Evaluation Criteria

| Criteria | Weight |
|---|---|
| Accuracy — right wine, right producer, right appellation, right climat | 40% |
| Verification logic — how the pipeline confirms a photo is correct | 25% |
| Approach & architecture — quality of the overall design | 20% |
| Speed — time per SKU, total pipeline runtime | 15% |

---

## Tools

Use anything: Python, LangChain, GPT-4V, Google Custom Search API, SerpAPI, Playwright, vision models, OCR, etc. We care about results and thinking, not your stack.

---

## Deliverables

1. **Working demo** — run on all 10 test SKUs, output includes:
   - Photo URL found (or "No Image" if none verified)
   - Confidence score per SKU
   - Pass / Fail verdict per SKU

2. **Write-up (max 2 pages or 5 slides):**
   - Your pipeline design and verification approach
   - How you handle false positives (wrong photo that looks right)
   - Biggest failure modes and how you handled them
   - **Total time spent on this task** (design + build + test)
   - What you'd improve with more time or resources

---

*We're not just testing if you can find photos. We're testing if you can build something accurate enough to trust at scale.*
