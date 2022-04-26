# Created By Muhammad Saad #
# On 29/03/2022 #
# At 12:33 PM #
from convolution import Level_Checker
import numpy as np


class Pooling:
    # Initiating necessary variables
    def __init__(self, window, seal_tuple, total, input_size, input_len, input_gap):
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
        mul_vec = np.zeros(self.slot_count)
        i = 0
        while i < total:
            for j in range(0, self.input_gap, 4):
                mul_vec[i + j] = mul
            i = i + self.output_gap
        self.mul_plain = self.ckks_encoder.encode(mul_vec, self.scale)

    # Applying mean pooling with standard 2x2 window structure
    def Mean_Pool(self, con_input):

        output_cipher = con_input

        output_cipher = self.evaluator.add(output_cipher, (self.evaluator.rotate_vector(con_input, 56, self.galois_keys)))
        output_cipher = self.evaluator.add(output_cipher, (self.evaluator.rotate_vector(output_cipher, 2, self.galois_keys)))

        # print(self.ckks_encoder.decode(self.decryptor.decrypt(output_cipher)))

        self.evaluator.mod_switch_to_inplace(self.mul_plain, output_cipher.parms_id())
        self.evaluator.multiply_plain_inplace(output_cipher, self.mul_plain)

        self.evaluator.relinearize_inplace(output_cipher, self.relin_keys)
        self.evaluator.rescale_to_next_inplace(output_cipher)
        output_cipher.scale(2**50)
        # out = self.ckks_encoder.decode(self.decryptor.decrypt(total_cipher))
        # print(out)
        # print(out[6184])
        return self.out_size, self.windows_total * 2, output_cipher, self.output_gap

    # Completed On 03/29/2022 #
    # At 8:04 PM #
    # Change complete on 31/03/2022 #
    # At 4:05 #
    # ........................ #
    # Fast Model prediction increase more than 1 second #
    # Completed On 26/04/2022 #
    # At 11;23 AM #
