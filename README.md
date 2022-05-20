# CNN Encrypted
This is code part of my work on **Privacy Preserving Malicious Nework Traffic Detection Using Fully Homomorphic CNN** for my MS Thesis. 

This work is prepared while keeping in mind requirements of thesis.

This is not very organized code and I will keep working to improve this code in much easy to understand form along with other necessary documents additions.

An improved faster version is abailable at branch [cnn_encrypted_faster](https://github.com/Arman001/CNN_Encrypted/tree/encrypted_cnn_faster) 

There is also a model that I specifically created for batch predictions and it very good at large number of prediction it can be found at branch [cnn_batch_encrypted](https://github.com/Arman001/CNN_Encrypted/tree/encrypted_cnn_faster)

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

**Encrypted Model**

![image](https://user-images.githubusercontent.com/21517793/168409302-6c4ecba1-cba5-40d3-9b55-62a7ab9f6467.png)

***
A faster version as **CNN Encrypted Faster**  is at branch
https://github.com/Arman001/CNN_Encrypted/tree/encrypted_cnn_faster
***
