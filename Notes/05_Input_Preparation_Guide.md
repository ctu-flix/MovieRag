# Input Preparation Guide (`Phase A`)

> Quick reference for preparing your movie video, character bank images, and Google Drive layout before running `PhaseA1_Visual_Pipeline.ipynb`.

---

## 1. Google Drive Folder Layout

Create this exact structure inside your Google Drive at `MyDrive/MovieRAG/`:

```text
MyDrive/MovieRAG/
├── videos/
│   ├── sintel.mp4                    # Movie file (.mp4 or .mkv)
│   └── sintel.srt                    # Optional (.srt or SDH .srt for Notebook 2)
└── char_bank/
    └── sintel/
        ├── character_glossary.json   # 1-sentence spoiler-free visual/role description per character
        └── raw_images/
            ├── Sintel/
            │   ├── Sintel_0.png      # Anchor image (clear front/3-quarter view)
            │   ├── Sintel_1.png      # 5-8 varied shots (full-body, medium, side, close-up)
            │   └── ...
            ├── Scales/
            │   ├── Scales_0.png
            │   └── ...
            └── Shaman/
                ├── Shaman_0.png
                └── ...
```

---

## 2. How to Choose Character Images (`raw_images/{Character}/`)

In Stage A2 & A3, `OWLv2` detects the **whole visible character bounding box** in each movie frame and `DINOv2` matches that box crop against your character bank. Therefore, **a mix of framing scales and angles works much better than face-only portraits**.

### Recommended 6–9 Images per Character

| File Name | Framing & Pose | Why It Matters |
| :--- | :--- | :--- |
| **`{Name}_0.png` (Anchor)** | **Clean Full-Body or 3/4 Medium Shot** (front or three-quarter angle, good lighting, **solo character only**) | Used as the reference anchor to score and filter all other exemplar images. Transparent PNGs or clean in-movie frames work best. |
| **`{Name}_1..3.png`** | **Full-Body & Medium Shots** (standing, walking, action pose, back/side angle) | Matches wide and medium camera shots where outfit, silhouette, and body proportions dominate. |
| **`{Name}_4..6.png`** | **Close-Up / Bust Shots** (face + shoulders, different expressions/lighting) | Matches dialogue close-ups and reaction shots where only the head and upper torso are inside the `OWLv2` box. |
| **`{Name}_7..8.png`** *(Optional)* | **Costume / Appearance Changes** (e.g., Sintel with hood/cloak vs. without cloak) | Prevents tracking drops when a character's outfit, hairstyle, or lighting changes mid-movie. |

### Do's and Don'ts
- **DO** include **only 1 character per image** whenever possible (especially for `{Name}_0.png`). `OWLv2` automatically crops the character out of the background, so you do **not** need to manually crop tight around the character.
- **DO** take screenshots directly from the movie (`VLC` $\rightarrow$ *Video* $\rightarrow$ *Take Snapshot*) so lighting and 3D shaders match the film's exact render style.
- **DON'T** use extreme eye-only close-ups, heavy motion blur, dark silhouettes, or 2D concept art that looks different from the 3D render.

---

## 3. `character_glossary.json` Rules

Keep descriptions to **1 concise sentence** describing **visual appearance + initial role** (never include late-movie plot twists or spoilers):

```json
{
  "Sintel": "Young woman warrior and protagonist searching for her companion dragon Scales; has short dark-brown hair, wears a weathered brown tunic and desert cloak, and carries a wooden staff.",
  "Scales": "Small, playful baby red dragon with large amber eyes and an injured wing bandaged and nursed back to health by Sintel.",
  "Shaman": "Elderly tribal healer and gatekeeper with long white dreadlocks and ceremonial robes who questions Sintel inside a desert hut."
}
```
