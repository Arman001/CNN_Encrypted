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
        with open('con_weights.pkl', 'rb') as file:
            weights = pickle.load(file)
        weights = weights.reshape(8, 4)

        self.input_size = input_size
        self.no_of_filters = no_of_filters
        self.encryptor, self.evaluator, self.decryptor, self.slot_count, self.context, self.ckks_encoder, self.scale, self.galois_keys, self.relin_keys = seal_tuple
        self.k_s = 2
        self.output_size = ((input_size - self.k_s) // 2) + 1
        self.interval = input_size - self.k_s
        self.kernel_size = input_size * input_size

        filters = np.zeros((self.no_of_filters, self.kernel_size))
        magic_vec = np.zeros(self.slot_count)
        switch = 0
        for item in range(self.no_of_filters):
            for i in range(0, self.kernel_size, 2):
                if(i != 0 and i % input_size == 0):
                    if switch == 0:
                        switch = 1
                    elif switch == 1:
                        switch = 0
                if(switch == 0):
                    filters[item][i] = weights[item][0]
                    filters[item][i + 1] = weights[item][1]
                if(switch == 1):
                    filters[item][i] = weights[item][2]
                    filters[item][i + 1] = weights[item][3]
                if(i % 8 == 0):
                    magic_vec[i] = 1
                    magic_vec[i + 2] = 1
        self.filter_plain = []
        for i in range(self.no_of_filters):
            self.filter_plain.append(self.ckks_encoder.encode(filters[i], self.scale))
        self.magic_plain = self.ckks_encoder.encode(magic_vec, self.scale)

    # Degree 2 relu approximation
    def Encrypted_ReLU(self, output_cipher):
        temp_plain = []
        temp_cipher = []
        for i in range(2):
            temp_plain.append(self.ckks_encoder.encode(np.zeros(self.slot_count, dtype=float), self.scale))
            temp_cipher.append(self.encryptor.encrypt(temp_plain[i]))

        # All plain texts
        plain2 = self.ckks_encoder.encode(0.009, self.scale)
        plain1 = self.ckks_encoder.encode(0.50, self.scale)
        plain0 = self.ckks_encoder.encode(0.47, self.scale)

        # Squaring
        temp_cipher[0] = self.evaluator.square(output_cipher)
        self.evaluator.relinearize_inplace(temp_cipher[0], self.relin_keys)
        self.evaluator.rescale_to_next_inplace(temp_cipher[0])
        temp_cipher[0].scale(2**40)

        # Multiplying with plane
        self.evaluator.mod_switch_to_inplace(plain2, temp_cipher[0].parms_id())
        self.evaluator.multiply_plain_inplace(temp_cipher[0], plain2)

        # Multiplying  with second cipher
        self.evaluator.mod_switch_to_inplace(plain1, output_cipher.parms_id())
        temp_cipher[1] = self.evaluator.multiply_plain(output_cipher, plain1)

        # Adding both ciphers
        self.evaluator.mod_switch_to_inplace(temp_cipher[1], temp_cipher[0].parms_id())
        self.evaluator.add_inplace(temp_cipher[0], temp_cipher[1])
        self.evaluator.relinearize_inplace(temp_cipher[0], self.relin_keys)
        self.evaluator.rescale_to_next_inplace(temp_cipher[0])
        temp_cipher[0].scale(2**40)

        # Adding final plain text
        self.evaluator.mod_switch_to_inplace(plain0, temp_cipher[0].parms_id())
        self.evaluator.add_plain_inplace(temp_cipher[0], plain0)
        # print("Relued output: ")
        # print(self.ckks_encoder.decode(self.decryptor.decrypt(temp_cipher[0])))
        # print(self.ckks_encoder.decode(self.decryptor.decrypt((self.evaluator.rotate_vector(temp_cipher[0], 16, self.galois_keys)))))

        # Level_Checker(temp_cipher[0].parms_id(), self.context)
        return temp_cipher[0]

    # Main convolution operation
    def Convolve(self, input_cipher):
        start = time.time()
        # multiplying prepared plain kernel with input
        output_ciphers = []
        for i in range(self.no_of_filters):
            output_cipher = self.evaluator.multiply_plain(input_cipher, self.filter_plain[i])
            self.evaluator.relinearize_inplace(output_cipher, self.relin_keys)
            self.evaluator.rescale_to_next_inplace(output_cipher)
            output_cipher.scale(2**40)

            # print(self.ckks_encoder.decode(self.decryptor.decrypt(output_cipher)))
            # print(self.ckks_encoder.decode(self.decryptor.decrypt(self.evaluator.rotate_vector(output_cipher, 3, self.galois_keys))))
            # Addition 1
            self.evaluator.add_inplace(output_cipher, self.evaluator.rotate_vector(output_cipher, self.input_size, self.galois_keys))
            # print(self.ckks_encoder.decode(self.decryptor.decrypt(output_cipher)))
            # Final Addition
            self.evaluator.add_inplace(output_cipher, self.evaluator.rotate_vector(output_cipher, 1, self.galois_keys))
            # print(self.ckks_encoder.decode(self.decryptor.decrypt(output_cipher)))
            # print(self.ckks_encoder.decode(self.decryptor.decrypt(self.evaluator.rotate_vector(output_cipher, 8, self.galois_keys))))

            # Reducing the gap between results to 1
            self.evaluator.mod_switch_to_inplace(self.magic_plain, output_cipher.parms_id())
            self.evaluator.multiply_plain_inplace(output_cipher, self.magic_plain)
            # print(self.ckks_encoder.decode(self.decryptor.decrypt(output_cipher)))
            self.evaluator.add_inplace(output_cipher, self.evaluator.rotate_vector(output_cipher, self.input_size, self.galois_keys))
            self.evaluator.relinearize_inplace(output_cipher, self.relin_keys)
            self.evaluator.rescale_to_next_inplace(output_cipher)
            output_cipher.scale(2**40)
            if(i == 0):
                print("Level before Relued")
                Level_Checker(output_cipher.parms_id(), self.context)
            # print(self.ckks_encoder.decode(self.decryptor.decrypt(output_cipher)))
            output_cipher = self.Encrypted_ReLU(output_cipher)
            if(i == 0):
                print("Level After Relued")
                Level_Checker(output_cipher.parms_id(), self.context)
            output_ciphers.append(output_cipher)
            del output_cipher
        end = time.time()
        # print(f"Total Time Taken in Convolution is: {end-start} seconds")
        return output_ciphers, self.output_size
        # print(self.ckks_encoder.decode(self.decryptor.decrypt(self.evaluator.rotate_vector(output_cipher, 4, self.galois_keys))))


# Finished at 29/03/22 #
# at 11:11 AM #
# Currently Convolution is too much time taking process and I dont have any other choice yet
# ....................... #
# Finished with major changes on 31/03/22 #
# at 1:12 PM #
# Performance issue seems to be solved by using stride 2 with cleverly constructed
