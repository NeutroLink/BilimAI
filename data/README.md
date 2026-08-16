# data/ — training vs exam, physically separated

| Folder | Contents | Rule |
|---|---|---|
| `data/train/` | school_notebooks_RU **train + val** splits — 1,707 pages + `annotations_train.json`, `annotations_val.json` | the *only* place training code may read from |
| `data/exam/` | school_notebooks_RU **test** split — 150 pages + `annotations_test.json` | **sealed** — scoring only, never training. The 20-page `eval/testset_v1/ru_pages` set is drawn from here |

Images are git-ignored (2.9 GB). Source: https://huggingface.co/datasets/ai-forever/school_notebooks_RU (MIT).
Later additions follow the same rule: Uzbek/math/MCQ pages go to `train/` or `exam/`, never both.
