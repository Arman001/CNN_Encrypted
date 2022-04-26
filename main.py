import numpy as np
from parameters_generation import SEALWork
from convolution import Convolution
from pooling import Pooling
from output import Output
import time

# from data_preprocessing import load_data
import pickle


def main():
    seal_obj = SEALWork()
    seal_tuple = seal_obj.initialize()
    encryptor, evaluator, decryptor, slot_count, context, ckks_encoder, scale, galois_keys, relin_keys = seal_tuple
    print("Total slots available: ", slot_count)
    print(".........Loading the Data........")
    class_labels = ["cridex", "smb"]
    with open('./Data/Test_X2.pkl', 'rb') as file:
        X = pickle.load(file)
    with open('./Data/Test_Y2.pkl', 'rb') as file:
        Y = pickle.load(file)

    # X = X / 255.0
    print(f"Classes are: {class_labels}")
    print(f"Total Data is: {len(X)}")
    print("----------------------------------------")
    print("------Testing of HE CNN is started------")
    input_size = 28

    con = Convolution(8, seal_tuple, input_size)
    pool = Pooling(2, seal_tuple, 6272, 14, 8, 28)
    output = Output(392, 112, 0, seal_tuple)
    acc = 0
    print("----------------------------------------")
    print("Samples Used for Testing: 1000")
    print("----------------------------------------")
    start = time.time()
    for i in range(1000):
        checker = i
        # print("Real Prediction = ", Y[checker])
        plain_input = ckks_encoder.encode(X[i], scale)
        input_cipher = encryptor.encrypt(plain_input)
        out, out_size, total, output_gap = con.Convolve(input_cipher)
        out_size, rotations, out, output_gap = pool.Mean_Pool(out)
        out = output.Final_Calculations(out)
        if(out[0] == Y[checker][0]):
            acc = acc + 1
        if(i != 0 and (i + 1) % 200 == 0):
            print(f"{i + 1} items done!")
    end = time.time()
    print(f"Total time {end-start} seconds")
    print("----------------------------------------")
    print("Accurately Predicted items: ", acc)
    print(f"Accuracy is: {(acc / 1000) * 100}%")
    print("----------------------------------------")


if __name__ == "__main__":
    main()

# ........................ #
# Fast Model prediction increase more than 1 second #
# Completed On 26/04/2022 #
# At 11;23 AM #
