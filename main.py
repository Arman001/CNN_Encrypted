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

    input_size = 28
    batch_size = 10
    con = Convolution(8, seal_tuple, input_size, batch_size)
    pool = Pooling(2, seal_tuple, 14, 8, 28, batch_size)
    output = Output(392, 112, 0, seal_tuple, 8, batch_size)

    print("----------------------------------------")
    print("Samples Used for Testing: 1000")
    print("----------------------------------------")
    print(f"Creating batches of {batch_size}")
    input_batches = []
    label_batches = []
    count = 0
    for i in range(int(1000 / batch_size)):

        arr = X[count]
        ls = []
        ls.append(Y[count])

        for j in range(1, batch_size):
            count = count + 1
            arr = np.append(arr, X[count])
            ls.append(Y[count])

        input_batches.append(arr)
        label_batches.append(ls)
        count = count + 1
    print("----------------------------------------")
    print("------Testing of HE CNN is started------")
    print("----------------------------------------")
    acc = 0
    start = time.time()
    for i in range(len(input_batches)):
        plain_input = ckks_encoder.encode(input_batches[i], scale)
        input_cipher = encryptor.encrypt(plain_input)
        out, out_size, output_gap = con.Convolve(input_cipher)
        out_size, rotations, out, output_gap = pool.Mean_Pool(out)
        out = output.Final_Calculations(out)
        for j in range(batch_size):
            check = np.array_equal(out[j], label_batches[i][j])
            if (check == True):
                acc = acc + 1
        if(i != 0 and (i + 1) % 20 == 0):
            print(f"{(i + 1)*batch_size} items done!")

    end = time.time()
    print(f"Total time {end-start} seconds")
    print("----------------------------------------")
    print("Accurately Predicted items: ", acc)
    print(f"Accuracy is: {(acc / 1000) * 100}%")
    print("----------------------------------------")


if __name__ == "__main__":
    main()
