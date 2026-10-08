# Practical Assignment I - Technical Report Template

This Overleaf project is the official report template for **Data Stream Mining - Practical Assignment I**.

## Start in 5 steps

1. Upload this ZIP to Overleaf and open `main.tex`.
2. Edit the **STUDENT METADATA** block near the top of `main.tex` (`PAxx`, student names/numbers, cities, business scenario, target, historical period, warm-up and seed).
3. Keep `\draftinstructionstrue` while writing. The grey guidance boxes explain what belongs in each section.
4. Add your figures to the `figures/` folder and either use the suggested filenames or update the `\reportfigure` calls.
5. Before final submission, change `\draftinstructionstrue` to `\draftinstructionsfalse`, remove unused placeholders/optional material, compile again, and verify that the main body is approximately **6-8 pages** (cover, references, and optional appendix excluded).

## Fixed report structure

Do **not** change the section order, margins, body font size, or page geometry.

1. Business Context and Objective
2. Data and Stream Construction
3. City A Analysis
4. City B Analysis
5. Comparative Data Analysis
6. Data Stream Mining Experiment
7. Results and Discussion
8. Business and Technical Recommendations
9. Limitations
10. AI Assistance Statement
11. References
12. Optional Appendix (only if needed)

## Important

- Each figure must support an argument; do not paste every output produced by the starter code.
- The report must contain at least **two analyses or visualisations designed by the group**, not direct reproductions of starter-code figures.
- Preserve chronology and the **PREDICT -> EVALUATE -> LEARN** order.
- Generative AI may assist with permitted programming tasks, but it may not perform the interpretation of results, discussion, recommendations, or evidence-based conclusions.
- The final PDF and audit ZIP must follow the naming/submission rules in the Practical Assignment I Handbook.

## Compilation

The project is compatible with Overleaf's default pdfLaTeX workflow. `latexmk` / BibTeX will process `references.bib` when citations are present.
