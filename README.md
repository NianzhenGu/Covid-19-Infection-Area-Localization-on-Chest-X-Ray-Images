# Covid-19-Infection-Area-Localization-on-Chest-X-Ray-Images
Group project for class COMP4026 Computer Vision and Pattern Recognition 
Run *main.py* for training and *test_main.py* for testing. 

## Dataset
In this project, we use the COVID-QU-Ex dataset. It contains 33,920 CXR images including 11,956 Covid images, 11,263 non-Covid infections (viral or bacterial pneumonia), and 10,701 normal images. In each category, it has three kinds of images: original CXR images, ground-truth lung segmentation masks, and ground-truth infection area masks.
However, for the ground-truth infection area masks, we find that only the Covid-infected class has masks whereas the non- Covid-infected and normal classes have blank masks. Therefore, we only choose part images of the whole dataset and discard the non-Covid infections and normal images.
<img width="888" alt="Screenshot 2024-07-03 at 4 38 56 PM" src="https://github.com/NianzhenGu/Covid-19-Infection-Area-Localization-on-Chest-X-Ray-Images-/assets/145458678/8c042b25-357e-465c-a8a2-e3929ef8a763">

## Model

<img width="774" alt="Screenshot 2024-07-03 at 5 05 37 PM" src="https://github.com/NianzhenGu/Covid-19-Infection-Area-Localization-on-Chest-X-Ray-Images-/assets/145458678/f78b12dc-e781-4a40-b284-fb6f0ab00c63">

The basic formation of the U-Net architecture is as follows:
- Contracting Path: Reduces the spatial resolution of the image while increasing the number of feature channels to capture high-level information.
- Expanding Path: Gradually increases the spatial resolution of the feature maps while reducing the number of feature channels, a symmetrical reverse process of the contracting path. This enables the network to recover the details and localization from higher-level encoded information.
- Skip Connection: The skip connection design in the U-Net architecture provides more context information from the encoder to the decoder. With skip connection paths, the model can focus on the image with different scales, and make sure that critical information did not being filtered away.

## Results compared with Resnet101


<img width="528" alt="Screenshot 2024-07-03 at 5 06 52 PM" src="https://github.com/NianzhenGu/Covid-19-Infection-Area-Localization-on-Chest-X-Ray-Images-/assets/145458678/db98e96b-2989-4aba-9ff8-44bf7ded071f">


We find that Resnet101-pre-trained gives the best result. It makes sense because a pre-trained model can better encode the image. Our model achieves a better result than Resnet101 without pre-train, and the results are a little lower than the best result. Therefore, we believe our model would achieve better if pre-trained using ImageNet.

