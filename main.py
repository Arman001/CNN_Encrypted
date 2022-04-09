import numpy as np
from parameters_generation import SEALWork
from convolution import Convolution
from pooling import Pooling
from output import Output
from data_preprocessing import load_data
import time


def main():
    seal_obj = SEALWork()
    seal_tuple = seal_obj.initialize()
    encryptor, evaluator, decryptor, slot_count, context, ckks_encoder, scale, galois_keys, relin_keys = seal_tuple
    print("Total slots available: ", slot_count)
    print(".........Loading the Data........")
    class_labels, X, Y = load_data()
    X = X / 255
    print(f"Classes are: {class_labels}")
    print(f"Total Data is: {X.shape}")

    print("...........Testing of HE CNN is started.........")
    # input = np.ones(784)
    # input = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0])
    # input = np.array([1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0, 6.0, 0.0, 7.0, 0.0, 8.0, 0.0, 9.0, 0.0, 10.0, 0.0, 11.0, 0.0, 12.0, 0.0, 13.0, 0.0, 14.0, 0.0, 15.0, 0.0, 16.0])

    print(Y[0])
    input_size = 28
    plain_input = ckks_encoder.encode(X[0], scale)
    input_cipher = encryptor.encrypt(plain_input)
    con = Convolution(8, seal_tuple, input_size)
    out, out_size = con.Convolve(input_cipher)
    print("Size of Con output = ", out_size * out_size)
    pool = Pooling(2, seal_tuple, out_size, len(out))
    out_size, rotations, out = pool.Mean_Pool(out)
    print("Size of Pool output = ", out_size)
    output = Output(out_size, rotations, 8, seal_tuple)
    output.Final_Calculations(out)

    # con = Convolution(1, seal/ple, input_size)
    # out_size, rotations, out = pool.Mean_Pool(input_cipher)
    # output = Output(out_size, rotations, seal_tuple)
    # output.Final_Calculations(out)


if __name__ == "__main__":
    main()

# This is CNN network is now completed on#
# 08/04/2022 at 3:25PM #
