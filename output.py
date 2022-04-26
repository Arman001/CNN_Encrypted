# ......................... #
# Created By Muhammad Saad #
# On 30/03/2022 #
# At 7:15 AM #
# .......... #

from convolution import Level_Checker
import numpy as np
import pickle


class Output:

    # Initilization of Output class
    def __init__(self, input_size, rotations, input_len, seal_tuple):
        self.encryptor, self.evaluator, self.decryptor, self.slot_count, self.context, self.ckks_encoder, self.scale, self.galois_keys, self.relin_keys = seal_tuple
        self.input_size = int(input_size)
        with open('./Data/Approximate88/fc_weights.pkl', 'rb') as file:
            fc_weights = pickle.load(file)
        with open('./Data/Approximate88/fc_biases.pkl', 'rb') as file:
            fc_biases = pickle.load(file)
        weights = np.zeros((2, self.slot_count))

        i = 0
        count = 0
        while i < 6185:
            for j in range(0, 28, 4):
                for item in range(2):
                    weights[item][i + j] = fc_weights[count][item]
                count = count + 1
            i = i + 112

        self.plain_weights1 = self.ckks_encoder.encode(weights[0], self.scale)
        self.plain_weights2 = self.ckks_encoder.encode(weights[1], self.scale)
        self.plain_biases1 = self.ckks_encoder.encode(fc_biases[0], self.scale)
        self.plain_biases2 = self.ckks_encoder.encode(fc_biases[1], self.scale)

    # Final calculations to get output
    def Final_Calculations(self, input_cipher):
        self.evaluator.mod_switch_to_inplace(self.plain_weights1, input_cipher.parms_id())
        self.evaluator.mod_switch_to_inplace(self.plain_weights2, input_cipher.parms_id())

        out_cipher1 = self.evaluator.multiply_plain(input_cipher, self.plain_weights1)
        out_cipher2 = self.evaluator.multiply_plain(input_cipher, self.plain_weights2)

        # result1 = self.ckks_encoder.decode(self.decryptor.decrypt(out_cipher1))
        # print("Result 1 is ", np.sum(result1))
        # #
        # result2 = self.ckks_encoder.decode(self.decryptor.decrypt(out_cipher2))
        # print("Result 2 is ", np.sum(result2))

        self.evaluator.relinearize_inplace(out_cipher1, self.relin_keys)
        self.evaluator.rescale_to_next_inplace(out_cipher1)
        out_cipher1.scale(2**50)

        self.evaluator.relinearize_inplace(out_cipher2, self.relin_keys)
        self.evaluator.rescale_to_next_inplace(out_cipher2)
        out_cipher2.scale(2**50)

        out_plain = self.ckks_encoder.encode(np.zeros(self.slot_count), self.scale)
        out_cipher1_final = self.encryptor.encrypt(out_plain)
        out_cipher2_final = self.encryptor.encrypt(out_plain)

        self.evaluator.mod_switch_to_inplace(out_cipher1_final, out_cipher1.parms_id())
        self.evaluator.mod_switch_to_inplace(out_cipher2_final, out_cipher2.parms_id())

        # self.evaluator.add_inplace(out_cipher1_final, self.evaluator.rotate_vector(out_cipher1, 0, self.galois_keys))
        self.evaluator.add_inplace(out_cipher1, self.evaluator.rotate_vector(out_cipher1, 3136, self.galois_keys))
        self.evaluator.add_inplace(out_cipher1, self.evaluator.rotate_vector(out_cipher1, 1568, self.galois_keys))
        self.evaluator.add_inplace(out_cipher1, self.evaluator.rotate_vector(out_cipher1, 784, self.galois_keys))

        self.evaluator.add_inplace(out_cipher2, self.evaluator.rotate_vector(out_cipher2, 3136, self.galois_keys))
        self.evaluator.add_inplace(out_cipher2, self.evaluator.rotate_vector(out_cipher2, 1568, self.galois_keys))
        self.evaluator.add_inplace(out_cipher2, self.evaluator.rotate_vector(out_cipher2, 784, self.galois_keys))

        i = 0
        while i < 784:
            self.evaluator.add_inplace(out_cipher1_final, self.evaluator.rotate_vector(out_cipher1, i, self.galois_keys))
            self.evaluator.add_inplace(out_cipher2_final, self.evaluator.rotate_vector(out_cipher2, i, self.galois_keys))
            i = i + 4
            if(i % 28 == 0):
                i = i + 84

        self.evaluator.mod_switch_to_inplace(self.plain_biases1, out_cipher1_final.parms_id())
        self.evaluator.mod_switch_to_inplace(self.plain_biases2, out_cipher2_final.parms_id())
        self.evaluator.add_plain_inplace(out_cipher1_final, self.plain_biases1)
        self.evaluator.add_plain_inplace(out_cipher2_final, self.plain_biases2)
        # Level_Checker(out_cipher1_final.parms_id(), self.context)
        # self.evaluator.square_inplace(out_cipher1_final)
        # self.evaluator.square_inplace(out_cipher2_final)

        # print("Final Outputs::::::")
        out1 = self.ckks_encoder.decode(self.decryptor.decrypt(out_cipher1_final))
        out2 = self.ckks_encoder.decode(self.decryptor.decrypt(out_cipher2_final))

        # print("Not Squared")

        out = np.zeros(2)
        if(out1[0] > out2[0]):
            out[0] = 1
        else:
            out[1] = 1

        return out

# After Huge Performance improvements and testing #
# Everything completerd on 02/04/2022 #
# At 8:00 PM #
# Fast Model prediction increase more than 1 second #
# Completed On 26/04/2022 #
# At 11;23 AM #
