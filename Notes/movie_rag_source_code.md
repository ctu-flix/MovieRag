how to mount google drive
`rclone mount gdrive:MovieRAG/MovieRAG ./gdrive --vfs-cache-mode writes`
use this if the mount is stuck
`fusermount -u /home/tris/TRIS_WORKS/ctu-flix/movie-rag/gdrive`

`git config core.filemode false`
- to untrack file permission, so google drive mount doesnt messes up git status



the video insert loop:
go through all video paths
	process the length, names, frame times,..
	use the data to get transcript
	use transcript and data and frames to get captions,
	save all info to a key value store
	embbed the video 
		use the video segments cache to embbed
	save database
into ainsert



# todo
to do now:
- make insert video sequential, load and use models one at a time, 
	- [[Video_Rag#sequential insert]]
- remove image bind from the process
- ~~update to newest version of dependencies for compatibilities (maybe)~~