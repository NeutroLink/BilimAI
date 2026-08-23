"""Page-level Qwen prompts — the ONE source of truth for probing, training and serving. 2026-08-23.

Extracted verbatim from qwen_grounding_probe.py (which now imports them back) because the prompt is worth
2.4x on its own: the same model on the same pages found 55.5 % of lines with `complete` and 22.7 % with a
`fragments` wording written the day before. A prompt that drifts between training and serving is therefore a
silent, large regression — bilimai/guards.py fingerprints it for exactly this reason.

PAGE_PROMPT is the training/serving prompt: `complete`, the measured best.
"""

PROMPTS = {
 'base': (
    'Look at this page from a Russian school notebook.\n'
    'Find EVERY handwritten line of text written by the pupil.\n'
    'Return ONLY a JSON array. Each element: {"bbox_2d": [x1, y1, x2, y2], "text": "..."}\n'
    '- bbox_2d is the pixel box of that whole line in this image.\n'
    '- text is the exact transcription of that line, in Russian, copied as written.\n'
    "- Do NOT correct the pupil's spelling. Do NOT translate. Do not add commentary.\n"
 ),
 'complete': (
    'This is a page from a Russian school notebook, handwritten by a pupil.\n'
    'Transcribe EVERYTHING the pupil wrote. Miss nothing.\n'
    "Work top to bottom. For every line of the pupil's handwriting, however short or however faint,\n"
    'return one entry. That includes:\n'
    '  - lines written very lightly or in pale ink\n'
    '  - single words, single letters, single digits\n'
    '  - short notes squeezed above, below or between lines, and anything in the margin\n'
    '  - headings and dates\n'
    'Do NOT skip a line because it is hard to read — transcribe your best reading of it.\n'
    'Do NOT stop early. Continue until you reach the bottom of the page.\n'
    "IGNORE anything written in red — that is the teacher's marking, not the pupil's work.\n"
    'Return ONLY a JSON array, ordered top to bottom. Each element: {"bbox_2d": [x1, y1, x2, y2], "text": "..."}\n'
    '- bbox_2d is the pixel box of that line in this image.\n'
    '- text is the exact transcription, in Russian, copied as the pupil wrote it.\n'
    "- Do NOT correct the pupil's spelling. Do NOT translate. No commentary.\n"
 ),
 'fragments': (
    'Look at this page from a Russian school notebook.\n'
    'Find EVERY separate piece of handwriting by the pupil, however small.\n'
    'Include, as their own separate entries:\n'
    '  - very short lines: a single word, a single letter, a single digit\n'
    '  - grammar notes such as «м.р.», «ед.ч.», «прош. врем.», «сущ», «гл»\n'
    '  - anything squeezed ABOVE or BETWEEN the main lines, or written in the margin\n'
    'A small note written above a word is its OWN entry, not part of the line beneath it.\n'
    'Return ONLY a JSON array, ordered top to bottom. Each element: {"bbox_2d": [x1, y1, x2, y2], "text": "..."}\n'
    '- bbox_2d is the pixel box of that piece of writing in this image.\n'
    '- text is the exact transcription, in Russian, copied as written.\n'
    "- Do NOT correct the pupil's spelling. Do NOT translate. Do not add commentary.\n"
 ),
}

PAGE_PROMPT = PROMPTS["complete"]


# ---------------------------------------------------------------------------------------------------
# REGION_PROMPT — pass 1 of the two-pass architecture (plans/QWEN-ALL-DOMAINS-ROADMAP.md §1).
#
# Pass 1 asks WHERE and WHAT KIND, never WHAT IT SAYS. Three reasons that split is the whole design:
#   * a flat [{bbox_2d, text}] cannot express a table, a stacked fraction or a ticked option, and three
#     of the five assignment types need one of those. "Add a region type" beats "invent a new model".
#   * resolution: measured 2026-08-23, sending the page bigger makes the whole-page pass WORSE
#     (recall 0.658 -> 0.652 at max-side 2560, and the large-text controls lose 0.068). You cannot buy
#     the crop reader's 128 px per line by resizing the page — only by sending a crop.
#   * finding regions is LAYOUT work, so one model serves both languages and every paper type; reading
#     is language work and belongs in pass 2.
#
# The three types below are exactly what `annotations_train.json` already labels — pupil_text (244,529),
# pupil_comment (14,581) and teacher_comment (10,327) — so this trains on data that already exists.
# Separating the teacher's ink is not cosmetic: 2.9 % of boxes in the zero-shot run were the teacher's
# red pen read as if the pupil had written it.
REGION_PROMPT = (
    'This is a photograph of a page from a school exercise book.\n'
    'Find EVERY separate piece of handwriting and say WHAT KIND it is. Do not transcribe anything.\n'
    'Return ONLY a JSON array, ordered top to bottom, left column before right.\n'
    'Each element: {"bbox_2d": [x1, y1, x2, y2], "type": "..."}\n'
    'The three kinds are:\n'
    '  - "pupil_line"   a line of the pupil\'s own work\n'
    '  - "pupil_note"   something the pupil squeezed above, between or beside the lines: a short note,\n'
    '                   a grammar mark, a single letter or digit in the margin\n'
    '  - "teacher_mark" anything written by the teacher — corrections, ticks, crosses, a grade\n'
    'Miss nothing, however short or however faint. Do not stop early. Continue to the bottom of the page.\n'
    'No text, no transcription, no commentary — boxes and kinds only.\n'
)

REGION_TYPES = ("pupil_line", "pupil_note", "teacher_mark")
