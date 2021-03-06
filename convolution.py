# Created By Muhammad Saad
# On 3/22/22 at 7:56 PM
# .........................
import numpy as np
import time
import pickle


# Checking the current level of a ciphertext
def Level_Checker(p_id, context):
    context_data = context.first_context_data()
    count = 0
    count2 = 0
    while(context_data):
        count = count + 1
        context_data = context_data.next_context_data()
    context_data = context.first_context_data()
    while (context_data):
        count2 = count2 + 1
        index = context_data.chain_index()
        if (p_id == context_data.parms_id()):
            print(f"Total Data Levels are {count-1} | Currently at {index} | Remaining {count-count2}")
            break
        context_data = context_data.next_context_data()


class Convolution:
    # Initiating the keys and kernels here when class object is created
    def __init__(self, no_of_filters, seal_tuple, input_size):
        with open('./Data/Approximate88/con_weights.pkl', 'rb') as file:
            weights = pickle.load(file)
        with open('./Data/Approximate88/con_biases.pkl', 'rb') as file:
            biases = pickle.load(file)
        weights = weights.reshape(8, 4)
        self.input_size = input_size
        self.no_of_filters = no_of_filters
        self.encryptor, self.evaluator, self.decryptor, self.slot_count, self.context, self.ckks_encoder, self.scale, self.galois_keys, self.relin_keys = seal_tuple
        self.k_s = 2
        self.output_size = ((input_size - self.k_s) // 2) + 1
        # print(self.output_size)
        self.interval = input_size - self.k_s
        self.kernel_size = input_size * input_size
        self.total = self.kernel_size * self.no_of_filters
        self.output_gap = self.input_size
        filters = np.zeros((self.no_of_filters, self.kernel_size))
        f_biases = np.zeros((8, self.kernel_size))
        switch = 0

        for i in range(0, self.kernel_size, 2):
            if(i != 0 and i % input_size == 0):
                if switch == 0:
                    switch = 1
                elif switch == 1:
                    switch = 0
            for item in range(self.no_of_filters):
                if(switch == 0):
                    filters[item][i] = weights[item][0]
                    filters[item][i + 1] = weights[item][1]
                if(switch == 1):
                    filters[item][i] = weights[item][2]
                    filters[item][i + 1] = weights[item][3]
        i = 0
        while i < 784:
            for j in range(0, self.input_size, 2):
                for item in range(self.no_of_filters):
                    f_biases[item][i + j] = biases[item]

            i = i + (self.input_size * 2)
        filter = filters[0]
        biase_t = f_biases[0]
        for i in range(1, self.no_of_filters):
            filter = np.append(filter, filters[i])
            biase_t = np.append(biase_t, f_biases[i])
        # For covering whole available space with extra zeros
        extra_bits = np.zeros(self.slot_count - len(filter))
        filter = np.append(filter, extra_bits)
        out = np.zeros(self.slot_count)
        out[0:784] = 1
        self.output = self.ckks_encoder.encode(out, self.scale)
        self.filter_plain = self.ckks_encoder.encode(filter, self.scale)
        self.biases_plain = self.ckks_encoder.encode(biase_t, self.scale)
        plain1 = np.zeros(self.slot_count)
        plain2 = np.zeros(self.slot_count)
        plain3 = np.zeros(self.slot_count)
        i = 0
        while i < self.total:
            for j in range(0, self.input_size, 2):
                plain1[i + j] = 0.009
                plain2[i + j] = 0.50
                plain3[i + j] = 0.47
            i = i + (self.input_size * 2)

        self.plain_mul1 = self.ckks_encoder.encode(plain1, self.scale)
        self.plain_mul2 = self.ckks_encoder.encode(plain2, self.scale)
        self.plain_mul3 = self.ckks_encoder.encode(plain3, self.scale)

    # Degree 2 relu approximation
    def Encrypted_ReLU(self, output_cipher):
        temp_plain = []
        temp_cipher = []

        for i in range(2):
            temp_plain.append(self.ckks_encoder.encode(np.zeros(self.slot_count, dtype=float), self.scale))
            temp_cipher.append(self.encryptor.encrypt(temp_plain[i]))
        # Squaring
        temp_cipher[0] = self.evaluator.square(output_cipher)
        self.evaluator.relinearize_inplace(temp_cipher[0], self.relin_keys)
        self.evaluator.rescale_to_next_inplace(temp_cipher[0])
        temp_cipher[0].scale(2**50)

        # Multiplying with plane
        self.evaluator.mod_switch_to_inplace(self.plain_mul1, temp_cipher[0].parms_id())
        self.evaluator.multiply_plain_inplace(temp_cipher[0], self.plain_mul1)

        # Multiplying  with second cipher
        self.evaluator.mod_switch_to_inplace(self.plain_mul2, output_cipher.parms_id())
        temp_cipher[1] = self.evaluator.multiply_plain(output_cipher, self.plain_mul2)

        # Adding both ciphers
        self.evaluator.mod_switch_to_inplace(temp_cipher[1], temp_cipher[0].parms_id())
        self.evaluator.add_inplace(temp_cipher[0], temp_cipher[1])
        self.evaluator.relinearize_inplace(temp_cipher[0], self.relin_keys)
        self.evaluator.rescale_to_next_inplace(temp_cipher[0])
        temp_cipher[0].scale(2**50)

        # Adding final plain text
        self.evaluator.mod_switch_to_inplace(self.plain_mul3, temp_cipher[0].parms_id())
        self.evaluator.add_plain_inplace(temp_cipher[0], self.plain_mul3)
        # print("Relued output: ")
        # print(self.ckks_encoder.decode(self.decryptor.decrypt(temp_cipher[0])))
        # print(self.ckks_encoder.decode(self.decryptor.decrypt((self.evaluator.rotate_vector(temp_cipher[0], 752, self.galois_keys)))))

        # Level_Checker(temp_cipher[0].parms_id(), self.context)
        return temp_cipher[0]

    # Main convolution operation
    def Convolve(self, input_cipher):
        # This part is also useful when there is input larger than 28x28
        output = self.evaluator.multiply_plain(input_cipher, self.output)
        self.evaluator.relinearize_inplace(output, self.relin_keys)
        self.evaluator.rescale_to_next_inplace(output)
        self.output.scale(2**50)
        self.evaluator.rotate_vector_inplace(output, 784, self.galois_keys)
        self.evaluator.add_inplace(output, self.evaluator.rotate_vector(output, -784, self.galois_keys))
        self.evaluator.rotate_vector_inplace(output, 784, self.galois_keys)
        self.evaluator.add_inplace(output, self.evaluator.rotate_vector(output, -(784 * 2), self.galois_keys))
        self.evaluator.rotate_vector_inplace(output, 784 * 2, self.galois_keys)
        self.evaluator.add_inplace(output, self.evaluator.rotate_vector(output, -(784 * 4), self.galois_keys))
        self.evaluator.rotate_vector_inplace(output, -(784 * 4), self.galois_keys)
        # Now 8 inputs for 8 filters are prepared to be multiplied only once

        # multiplying prepared plain kernel with input

        self.evaluator.mod_switch_to_inplace(self.filter_plain, output.parms_id())
        output_cipher = self.evaluator.multiply_plain(output, self.filter_plain)

        self.evaluator.relinearize_inplace(output_cipher, self.relin_keys)
        self.evaluator.rescale_to_next_inplace(output_cipher)
        output_cipher.scale(2**50)

        output_cipher = self.evaluator.add(output_cipher, self.evaluator.rotate_vector(output_cipher, self.input_size, self.galois_keys))
        output_cipher = self.evaluator.add(output_cipher, self.evaluator.rotate_vector(output_cipher, 1, self.galois_keys))
        self.evaluator.mod_switch_to_inplace(self.biases_plain, output_cipher.parms_id())
        output_cipher = self.evaluator.add_plain(output_cipher, self.biases_plain)
        output_cipher = self.Encrypted_ReLU(output_cipher)

        return output_cipher, self.total, self.output_size, self.output_gap


# Finished at 29/03/22 #
# at 11:11 AM #
# Currently Convolution is too much time taking process and I dont have any other choice yet
# ....................... #
# Finished with major changes on 31/03/22 #
# at 1:12 PM #
# Performance issue seems to be solved by using stride 2 with cleverly constructed
# ........................ #
# Fast Model prediction increase more than 1 second #
# Completed On 26/04/2022 #
# At 11;23 AM #
