# Input Preparation Guide (`Phase A`)

> Quick reference for preparing your movie video, character bank images, timeline gates, and Google Drive layout before running `PhaseA0_CharBank_Bootstrap.ipynb` / `PhaseA1_Visual_Pipeline.ipynb`.

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
        ├── character_glossary.json   # Visual-first + timeline-gated + negative constraints
        ├── appearance_windows.json   # Optional {Character: [[start_s, end_s], ...]} gates
        ├── external_plot.txt         # Optional time-aligned (MM:SS - MM:SS) grounding synopsis
        └── raw_images/               # Auto-populated by PhaseA0 clustering or manual frames
            ├── Sintel/
            │   ├── Sintel_0.png
            │   └── ...
            ├── Scales/
            ├── Adult Dragon/
            ├── Shaman/
            └── Gatekeeper/
```

*(Tip: You can run **`PhaseA0_CharBank_Bootstrap.ipynb`** first on a T4 GPU to inspect the end-of-movie `CREDITS_START_SEC` frame grid and automatically harvest & cluster crops with `OWLv2` + `DINOv2` into `clusters_contact_sheet.jpg`, then assign cluster IDs directly into `raw_images/`.)*

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

## 3. `character_glossary.json` & `appearance_windows.json` Rules

- Write **visual-first descriptions** grouped by timeline state (outfit/hair/weapon changes), and put each timeline phase in its **own separate sentence** with its own `(MM:SS - MM:SS)` timestamp range. Phase B's `filter_glossary_text_for_timestamp()` splits glossary bios by sentence/clause and strips any sentence whose earliest timestamp is `> T_user` — keeping each phase in its own sentence ensures late-movie twists never leak into earlier timestamps while still distinguishing separate characters (e.g., two different adult dragons).
- Add **`appearance_windows.json`** (`{"Character": [[start_sec, end_sec], ...]}`) so Stage A5 gates out inactive characters from unrelated scenes and ignores low-confidence out-of-window visual tracker false positives (`score < 0.60`).

```json
{
  "Sintel": "Young woman protagonist. Visuals by timeline: in the Ishtar flashback (02:35 - 05:35), she is a teenager with short dark-brown hair, a sleeveless brown tunic, a dark metallic left shoulder guard, fingerless gloves, and no weapon; in the mountain/desert quest (00:25 - 02:35 and 05:35 - 08:15), she wears a dark fur-lined travel cloak and scarf, wielding a wooden staff until 00:55 and the Gatekeeper's curved spear-blade after 01:25; in the final volcanic cavern act (08:15 - 12:25), she has weathered shoulder-length hair with gray streaks and facial scars.",
  "Scales": "Small, playful baby light-brown dragon with large amber eyes, leathery light-brown wings, and a long tail. In the Ishtar flashback (02:58 - 05:05), his injured left wing is wrapped in a pale cloth bandage by Sintel. In the volcanic cavern (09:10 - 12:25), a separate, unbandaged fledgling baby light-brown dragon appears—this is a different baby dragon (Scales's own offspring), though Sintel initially mistakes it for young Scales.",
  "Adult Dragon": "Massive, fierce winged dark-brown dragon with horns, spiked spine, and large leathery wings. Over Ishtar (05:00 - 05:30), a giant wild adult dragon swoops down from the clouds and snatches baby Scales away. Inside the volcanic cavern (09:15 - 11:15), a different giant adult dragon lairs beside a fledgling baby dragon and battles Sintel. In the finale (11:15 - 12:35), a healed notch and bandage-scar on this cavern adult dragon's wing reveals it is actually grown-up Scales decades later.",
  "Shaman": "Elderly wearing a dark textured cap/headdress adorned with small curved horn ornaments, white ceremonial face paint across his eyes and nose, patterned robes, and a short weathered greyish-green beard, who shelters and questions Sintel inside a warmly lit desert hut.",
  "Gatekeeper": "Fierce, bald, unmasked tribal warrior with a dark mustache and goatee, wearing heavy robes and tall wooden banner poles strapped to his back (which create a spiky silhouette in the mountain mist), wielding a curved metallic spear-blade with a circular hilt ring (00:35 - 01:26)."
}
```
