# Here is the machine learning model

Since the machine learning model is large in size it is not possible to push the model as a commit. Hence, we have linked the model below:

[Machine Learning Model](https://drive.google.com/drive/folders/1awecN5xYR2W1Bm6ymhSh4qvVgWL_q8qX?usp=sharing)

[Data Set](https://www.kaggle.com/c/facial-keypoints-detection/data?select=SampleSubmission.csv)

## For the triplet generation py file the following is applicable

__init__: The constructor takes the path to the dataset directory defined in the previous subsection. The constructor uses the list.txt to make a dictionary. This dictionary has the directory name as its key and a list of images in that directory as its value. It is here, and in the shuffling step, that the list.txt becomes an easy way for us to have a dataset overview, thus avoiding to load images for shuffling.

__getitem__: We get the names of the people from the above dictionary keys. For 1st batch, the first 32 (batch size) people images are used as anchors, and a different image, of the same person, is used as positives. A negative image, from any other directory, is selected for training. For all of the triplets, the anchors, the positive, and the negative images are chosen randomly. The next 32 people become the anchor for the next batch.

curate_dataset: Creates the dictionary explained in the __init__

on_epoch_end: On each epoch end, the order of people is shuffled, so that in the next epoch, the first 32 images are different than the one seen in the previous epoch.
get_image: The get image function uses the preprocess_input after resizing the image to (224 x 224) size.

__len__: This will return the number of batches that will define one epoch.
