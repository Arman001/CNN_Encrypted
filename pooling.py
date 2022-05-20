# Created By Muhammad Saad #
# On 29/03/2022 #
# At 12:33 PM #
from convolution import Level_Checker
import numpy as np


class Pooling:
    # Initiating necessary variables
    def __init__(self, window, seal_tuple, input_size, input_len, input_gap, batch_size):
        self.window = window
        self.input_len = input_len
        self.input_gap = input_gap
        self.windows_total = window * window
        self.out_size = int(input_size * input_size / self.windows_total)
        self.out_size = self.out_size * input_len
        self.encryptor, self.evaluator, self.decryptor, self.slot_count, self.context, self.ckks_encoder, self.scale, self.galois_keys, self.relin_keys = seal_tuple
        self.input_size = input_size
        self.output_gap = self.input_gap * 4
        self.rotations = 696 + 88
        mul = 1 / self.windows_total
        mul_vec = np.zeros(784)
        i = 0
        while i < 784:
            for j in range(0, self.input_gap, 4):
                mul_vec[i + j] = mul
            i = i + self.output_gap
        mul_vec1 = np.tile(mul_vec, (batch_size))
        print(len(mul_vec1))
        self.mul_plain = self.ckks_encoder.encode(mul_vec1, self.scale)

    # Applying mean pooling with standard 2x2 window structure
    def Mean_Pool(self, con_input):
        total = np.zeros(self.slot_count)
        total_plain = self.ckks_encoder.encode(total, self.scale)
        total_cipher = self.encryptor.encrypt(total_plain)
        output_ciphers = []

        for c in range(self.input_len):
            output_cipher = con_input[c]

            output_cipher = self.evaluator.add(output_cipher, (self.evaluator.rotate_vector(con_input[c], 56, self.galois_keys)))
            output_cipher = self.evaluator.add(output_cipher, (self.evaluator.rotate_vector(output_cipher, 2, self.galois_keys)))

            # print(self.ckks_encoder.decode(self.decryptor.decrypt(output_cipher)))
            # print(self.ckks_encoder.decode(self.decryptor.decrypt(self.evaluator.rotate_vector(output_cipher, 696, self.galois_keys))))

            self.evaluator.mod_switch_to_inplace(self.mul_plain, output_cipher.parms_id())
            self.evaluator.multiply_plain_inplace(output_cipher, self.mul_plain)

            self.evaluator.relinearize_inplace(output_cipher, self.relin_keys)
            self.evaluator.rescale_to_next_inplace(output_cipher)
            output_cipher.scale(2**50)

            # print(self.ckks_encoder.decode(self.decryptor.decrypt(output_cipher)))
            # print(self.ckks_encoder.decode(self.decryptor.decrypt(self.evaluator.rotate_vector(output_cipher, 696, self.galois_keys))))

            # if c == 0:
            #     # Level_Checker(output_cipher.parms_id(), self.context)
            #     self.evaluator.mod_switch_to_inplace(total_cipher, output_cipher.parms_id())
            #
            # else:
            #     self.evaluator.rotate_vector_inplace(total_cipher, self.rotations, self.galois_keys)
            # self.evaluator.add_inplace(total_cipher, output_cipher)
            output_ciphers.append(output_cipher)

            del output_cipher
        # self.evaluator.rotate_vector_inplace(total_cipher, - (self.rotations * (self.input_len - 1)), self.galois_keys)
        # print(self.ckks_encoder.decode(self.decryptor.decrypt(self.evaluator.rotate_vector(total_cipher, 6184, self.galois_keys))))
        # out = self.ckks_encoder.decode(self.decryptor.decrypt(total_cipher))
        # print(out)
        # print(out[6184])
        return self.out_size, self.windows_total * 2, output_ciphers, self.output_gap

    # Completed On 03/29/2022 #
    # At 8:04 PM #
    # Change complete on 31/03/2022 #
    # At 4:05 #
