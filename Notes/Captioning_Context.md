---
Date: 2026-04-27
Time: 14:17
Top Level:
  - "[[_tech_]]"
Topic:
  - "[[_VideoRAG_]]"
  - "[[_Chunking_]]"
  - "[[_Embbeding_]]"
---
- improve captioning
- improve chunking
- improve knowledge graph


movie/episode story summary
- manually time stamp the texts
- this act as current scene summary
- fetch the coresponding text to the VLM
world building overview:
 - pass this to every clip
Subtitle:
- priority SDH subtitles if possible
- if there are no subtitle, use the transcript model
[[Character_detection]]

I might need to restructure the output json format

run [[Character_detection]] on the entire video first, keep track on segments dictionary
use that dictionary along with world building overview and time stamped summaries to provide context for the VLM





