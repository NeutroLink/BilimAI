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
