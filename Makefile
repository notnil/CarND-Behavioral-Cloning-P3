run:
	docker run -it --rm -p 4567:4567 -v `pwd`:/src udacity/carnd-term1-starter-kit python drive.py model.h5
train:
	docker run -it --rm -p 4567:4567 -v `pwd`:/src udacity/carnd-term1-starter-kit python model.py
video:
	docker run -it --rm -p 4567:4567 -v `pwd`:/src udacity/carnd-term1-starter-kit python video.py run1	