https://arxiv.org/html/2509.12204v1
This is another way of doing, character detection in movie scene.
This specifically focus on animated movies, but it may still work on live action.

first build a character visual and audio bank

extract visual and audio from the movie clip, then add it to the bank of highest match character

# Visual Character Recognition
- Detect and clip scene(clip without cut) and for each scene
- choose k  random frames as the anchor, detect  character, track them back and ford through the video, do this for every anchor frame
- then rank them by overlap score, choose the best bounding box
- for every box, extract n frames from each scene, compare to the visual bank, and assign the highest match character

# Audio Character recognition
- First use audio only approach, if a result has low confident, 
- combine it with result from the previous phase, to give it visual clue, on who's speaking.

All of the results from the previous phases, enhance these task
1. Character-Aware Subtitling
	- Use WhisperX to create a transcript along with percise timestamp
		- or just use available subtitle
	- use the [[#Audio Character recognition]] to assign the correct character name to the transcript
2. Audio Description Generation(This is called video captioning in [[Video_Rag]])
	 - use result from [[#Visual Character Recognition]] to draw bouding box over character, then pass it to VLM to generate description
	 - They took it a step further by using LLM to restructure the VLM results

These 2 tasks' results is exactly what I needed for MovieRAG.