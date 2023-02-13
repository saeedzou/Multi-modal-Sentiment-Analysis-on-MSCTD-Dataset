# Multi-modal Sentiment Analysis on MSCTD Dataset
This is a deep learning project that performs multi-modal sentiment analysis on the MSCTD dataset, which contains 30,370 English-German utterance pairs in 3,079 bilingual dialogues. Each utterance pair is annotated with a sentiment label.

## Dataset
The MSCTD dataset is a collection of bilingual dialogues, where each utterance pair is associated with a visual context that reflects the current conversational scene. The dataset is annotated with the following sentiment labels:

- Positive
- Negative
- Neutral

## Approach

### Zeroth Phase

In the zeroth phase, we created a dataset class and performed exploratory data analysis to understand the dataset better. We analyzed the sentiment distribution over different data splits, text length distribution, number of images per scene, and number of faces per image.

### First Phase

In the first phase, we experimented with using only the image to predict the sentiment. We first cropped faces in each image and trained a sentiment classifier on the cropped faces. We then used this sentiment classifier to detect how many faces in each image have each sentiment. We passed these results to a multi-layer perceptron (MLP) to predict the overall sentiment of the image. We also applied data augmentation techniques mentioned in the paper PRIME to improve the accuracy.

Afterwards, we fine-tuned a sentiment classifier on the entire image, and combined these results with the MLP results to improve the accuracy. However, we found that using only the image was not indicative enough for sentiment analysis, and our best accuracy in this phase was 42%.

### Second Phase

In the second phase, we experimented with using only the text to predict the sentiment. We first used the TF-IDF method to convert the text into features and trained an MLP on top of it. The model achieved an accuracy of 52%.

We then experimented with using word embeddings instead of the TF-IDF method. We trained an SVM on the TF-IDF features for each word in the dataset and used the average weights and biases for each word to obtain a new word embedding. We then averaged the word embeddings in a sentence to get the sentence embedding and used it to predict the sentiment label. This method achieved an accuracy of 48%. We then used pre-trained GloVe embeddings instead and achieved an accuracy of 50%.

Finally, we fine-tuned a pre-trained BERT model to predict the sentiment. We used the BERT model to generate text embeddings for the utterances in the dataset and trained a classifier on top of it. Our best accuracy in this phase was 60%.

Overall, our approach involved using various text processing techniques, including TF-IDF, word embeddings, and pre-trained BERT, to generate text features for sentiment analysis. We experimented with different methods and found that fine-tuning pre-trained BERT gave the best performance.

### Third Phase

In the third phase, we explored multi-modal sentiment analysis by combining both the text and image modalities.

First, we concatenated the output feature layers of our best models from phase 1 and phase 2, and trained an MLP on top of it. This approach gave an accuracy of 60% with an F1 score of 58%.

Next, we used MTCNN to extract faces from the images, and then passed the cropped images to InceptionResnetV1 with pretrained 'vggface2' weights, which embedded each face to a 512 dimensional vector. We then averaged these embeddings for images with multiple faces and used an all-zeros 512 dimensional vector for images with no detected faces. We concatenated these image embeddings with BERT output and passed them to an MLP. This method gave an accuracy of 60% with an F1 score of 59%.

Then, we used ViT as the backbone for images and BERT as the backbone for text. We concatenated the heads of each backbone and used max pooling on the features obtained from each backbone. We then trained an MLP on top of the concatenated features, which gave a 61% accuracy and 60% F1 score.

Finally, we used our image and text backbones from phase 1 and 2, and chose the COCO dataset for weakly supervised learning. We selected only the samples that are categorized as "people" and chose only the first caption. We then selected a fraction of this dataset and trained only the image FC layer on the pseudo labels obtained from the text backbone. However, because the COCO dataset mostly contained neutral examples, our model diverged from its objective.

Overall, our approach involved experimenting with different ways of combining the text and image modalities for multi-modal sentiment analysis, including concatenation, face detection, and using different backbones for images and text. We also attempted to use weakly supervised learning on the COCO dataset, but the results were not satisfactory due to the nature of the dataset.