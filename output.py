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
    def __init__(self, input_size, rotations, input_len, seal_tuple, no_of_filters, batch_size):
        self.encryptor, self.evaluator, self.decryptor, self.slot_count, self.context, self.ckks_encoder, self.scale, self.galois_keys, self.relin_keys = seal_tuple
        self.input_size = int(input_size)
        self.no_of_filters = no_of_filters

        self.index_list = list(range(0, (batch_size) * 784, 784))

        with open('./Data/Approximate88/fc_weights.pkl', 'rb') as file:
            fc_weights = pickle.load(file)
        with open('./Data/Approximate88/fc_biases.pkl', 'rb') as file:
            fc_biases = pickle.load(file)
        weights1 = np.zeros((no_of_filters, 784))
        weights2 = np.zeros((no_of_filters, 784))

        count = 0
        for item in range(no_of_filters):
            i = 0
            while i < 784:
                for j in range(0, 28, 4):
                    weights1[item][i + j] = fc_weights[count][0]
                    weights2[item][i + j] = fc_weights[count][1]
                    count = count + 1
                i = i + 112
        self.plain_weights1 = []
        self.plain_weights2 = []

        # print(weights1[0])
        # print(weights2[0][0])

        for i in range(no_of_filters):
            self.plain_weights1.append(self.ckks_encoder.encode(np.tile(weights1[i], (batch_size)), self.scale))
            self.plain_weights2.append(self.ckks_encoder.encode(np.tile(weights2[i], (batch_size)), self.scale))

        self.plain_biases1 = self.ckks_encoder.encode(fc_biases[0], self.scale)
        self.plain_biases2 = self.ckks_encoder.encode(fc_biases[1], self.scale)

    # Final calculations to get output
    def Final_Calculations(self, input_ciphers):
        out_cipher1 = []
        out_cipher2 = []
        out_plain = self.ckks_encoder.encode(np.zeros(self.slot_count), self.scale)
        out_cipher1_final = []
        out_cipher2_final = []
        final_out1 = self.encryptor.encrypt(out_plain)
        final_out2 = self.encryptor.encrypt(out_plain)
        for i in range(self.no_of_filters):
            out_cipher1_final.append(final_out1)
            out_cipher2_final.append(final_out2)
        final_output1 = self.encryptor.encrypt(out_plain)
        final_output2 = self.encryptor.encrypt(out_plain)

        for i in range(self.no_of_filters):
            self.evaluator.mod_switch_to_inplace(self.plain_weights1[i], input_ciphers[i].parms_id())
            self.evaluator.mod_switch_to_inplace(self.plain_weights2[i], input_ciphers[i].parms_id())

            out_cipher1.append(self.evaluator.multiply_plain(input_ciphers[i], self.plain_weights1[i]))
            out_cipher2.append(self.evaluator.multiply_plain(input_ciphers[i], self.plain_weights2[i]))

            self.evaluator.relinearize_inplace(out_cipher1[i], self.relin_keys)
            self.evaluator.rescale_to_next_inplace(out_cipher1[i])
            out_cipher1[i].scale(2**50)

            self.evaluator.relinearize_inplace(out_cipher2[i], self.relin_keys)
            self.evaluator.rescale_to_next_inplace(out_cipher2[i])
            out_cipher2[i].scale(2**50)

            self.evaluator.mod_switch_to_inplace(out_cipher1_final[i], out_cipher1[i].parms_id())
            self.evaluator.mod_switch_to_inplace(out_cipher2_final[i], out_cipher2[i].parms_id())

            # result1 = self.ckks_encoder.decode(self.decryptor.decrypt(out_cipher1[i]))
            # print("Result 1 is ", np.sum(result1[0:784]))
            # result2 = self.ckks_encoder.decode(self.decryptor.decrypt(out_cipher2[i]))
            # print("Result 2 is ", np.sum(result2[0:784]))

            rot = 0
            while rot < 784:
                self.evaluator.add_inplace(out_cipher1_final[i], self.evaluator.rotate_vector(out_cipher1[i], rot, self.galois_keys))
                self.evaluator.add_inplace(out_cipher2_final[i], self.evaluator.rotate_vector(out_cipher2[i], rot, self.galois_keys))
                rot = rot + 4
                if(rot % 28 == 0):
                    rot = rot + 84

            if(i == 0):
                self.evaluator.mod_switch_to_inplace(final_output1, out_cipher1_final[i].parms_id())
                self.evaluator.mod_switch_to_inplace(final_output2, out_cipher2_final[i].parms_id())

            self.evaluator.add_inplace(final_output1, out_cipher1_final[i])
            self.evaluator.add_inplace(final_output2, out_cipher2_final[i])

        # # Level_Checker(out_cipher1_final.parms_id(), self.context)
        # self.evaluator.square_inplace(out_cipher1_final)
        # self.evaluator.square_inplace(out_cipher2_final)
        self.evaluator.mod_switch_to_inplace(self.plain_biases1, final_output1.parms_id())
        self.evaluator.mod_switch_to_inplace(self.plain_biases2, final_output2.parms_id())
        self.evaluator.add_plain_inplace(final_output1, self.plain_biases1)
        self.evaluator.add_plain_inplace(final_output2, self.plain_biases2)
        # print("Final Outputs::::::")
        out1 = self.ckks_encoder.decode(self.decryptor.decrypt(final_output1))
        out2 = self.ckks_encoder.decode(self.decryptor.decrypt(final_output2))

        # print("Not Squared")
        # print(out1[0])
        # print(out2[7056])

        output = []

        for item in self.index_list:
            out = np.zeros(2)
            if(out1[item] > out2[item]):
                out[0] = 1
            else:
                out[1] = 1

            output.append(out)
        # print("Squared")
        # print(out1[0] * out1[0])
        # print(out2[0] * out2[0])

        # print("Output 1 is: ", out1[6184])
        # print("Output 2 is: ", out2[6184])
        return output

# After Huge Performance improvements and testing #
# Everything completerd on 02/04/2022 #
# At 8:00 PM #
