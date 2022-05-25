# CNN Encrypted
This is code part of my work on **Privacy Preserving Malicious Nework Traffic Detection Using Fully Homomorphic CNN** for my MS Thesis. 

This work is prepared while keeping in mind requirements of thesis.

This is not very organized code and I will keep working to improve this code in much easy to understand form along with other necessary documents additions.

There is also a model that I specifically created for batch predictions and it very good at large number of prediction it can be found at branch [cnn_batch_encrypted](https://github.com/Arman001/CNN_Encrypted/tree/cnn_batch_encrypted)

Some mistakes will still be in code along with improvements chances, so this work will continue.

Kindly check it, improve it and report about the issues so we can all learn and make it better. 

Regards:
**Muhammad Saad**
***
### CNN Model
![image](https://user-images.githubusercontent.com/21517793/168409136-1856e8c5-e685-441f-b5f9-33cfc50ab30e.png)

***
### CKKS Scheme
Python wrapper of Microsoft SEAL from https://github.com/Huelse/SEAL-Python is used

![image](https://user-images.githubusercontent.com/21517793/168409379-9b600ae1-d475-4c42-b5ee-e032c3cf6eed.png)

***
### Dataset
Dataset is self prepared by extracting the TCP and UDP payloads from USTC-TFC2016 provided at https://github.com/yungshenglu/USTC-TFC2016

![image](https://user-images.githubusercontent.com/21517793/168409571-ce59cfaa-4a3d-46f4-8e80-79dfd5ed54c2.png)

***
### Results
**Plain Model**

![image](https://user-images.githubusercontent.com/21517793/168409622-bfa061d8-1591-45d7-8526-1cbdbebeb283.png)


**Fast Encrypted CNN**

This is a faster version of previous Encrypted CNN as it reduces the prediction time to near 1.4 seconds rather than 2.5 seconds

![image](https://user-images.githubusercontent.com/21517793/168408381-e1c2faf4-ef5d-4118-8de5-b4f9233b04c4.png)
***
