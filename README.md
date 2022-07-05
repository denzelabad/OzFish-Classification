# Fish Video Classification
### Description
This project uses a Convolutional Neural Network to attempt to identify a variety of fish species in underwater research videos. 

### Background

Baited Underwater Video Stations (BRUVS) are a tool used to monitor and survey fish in a variety of underwater habitats, and involve lowering stationary video cameras rigged with bait to the seafloor to attract and record nearby fish. I would occasionally find online livestreams of these BRUVS where they would feature a number of amazing looking fish, but weren't supplemented with any commentary or labels so identifying any fish in the frame required extensive googling on my part. I thought that these recordings would be so much more interesting and educational if they automated the identification of fish that would show up on the camera. Using my knowledge of machine learning and computer vision I decided to try and provide a solution to that problem.

### Data

This project uses the Ozfish dataset for testing and training which can be found [here](https://github.com/open-AIMS/ozfish). It consists of over 80k image crops of hundreds of species of fish taken from a BRUVS program, including an associated metadata file that links each fish crop with its species, genus and family.

### Method

The idea behind this project was to train a Convolutional Neural Network using Keras to classify images of different fish, and then use that model to predict, in real-time, the species of fish in video clips taken from BRUVS. 

1. Wrangle and clean data, perform data augmentation, train and optimise the CNN. This was all done in a [Jupyter Notebook](https://github.com/denzelabad/OzFish-Classification/blob/main/Fish%20Image%20Classification.ipynb). 
2. Use OpenCV to extract each frame from a BRUVS video clip as an image, feed the image into the CNN, label the image with the class containing the highest probability, export the new labelled video clip. Uses the rolling average of predictions to prevent label flickering, ie. the class with the highest average probability in the last 90 frames was written onto the image to make the outputted label more stable in consecutive frames. This step was done in this [Python script](https://github.com/denzelabad/OzFish-Classification/blob/main/Ozfish%20Video%20Script.py).

### Results

The resulting CNN model achieved an accuracy of 83% on a test dataset containing 14,299 images with 151 classes. Its performance was limited by the large class imbalance present in the data, which resulted in the model performing much better in predictions for some classes over others.

The clips below highlight the model's ability to identify fish in BRUVS video clips.

#### Clip 1:

The fish featured below is a [Rubrioperculatus](https://fishesofaustralia.net.au/home/species/2465). In videos where a single fish is clearly seen and not interrupted by heavy fish activity the model performs quite well, as seen below. 

https://user-images.githubusercontent.com/69582949/169447809-1b285c0a-90ed-4353-8abc-1cc1fa3a4edd.mp4

#### Clip 2:

Things start to get a little messy once multiple fish of different types are swimming in the frame. The fish centered in the frame below is a [Stellatus](https://fishesofaustralia.net.au/home/species/759), which the model properly identifies. However once the fish moves into an unfavourable angle, it starts to produce incorrect predictions. The various silver-looking fish swimming around are a combination of [Gymnostethus](https://fishesofaustralia.net.au/home/species/4262) and [Rubrioperculatus](https://fishesofaustralia.net.au/home/species/2465), which the model corrently identifies at times, but it can be unclear what the model is predicting when there are so many fish in the frame. An object detection model would be better suited in this case.

https://user-images.githubusercontent.com/69582949/169447858-1b4fc145-9d63-43fe-a4ad-aeb0619267f4.mp4

#### Clip 3:

The model performs quite well in identifying schools of fish of the same species. Below features a school of [Fulvoguttatus](https://fishesofaustralia.net.au/home/species/4261) fish, where the model produces mostly correct predictions despite the large number of fish present.

https://user-images.githubusercontent.com/69582949/169447878-6b8984ea-dc6a-4ba3-8380-cefb093f0cd3.mp4

These clips were obtained from raw BRUVS footage found [here](https://data.jcu.edu.au/aims/Oceanic_Shoals2014_AIMS/BRUVS/)
