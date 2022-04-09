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
        self.input_calc = int(input_size * input_len)
        self.rotations = rotations
        with open('fc_weights.pkl', 'rb') as file:
            fc_weights = pickle.load(file)

        weights = np.zeros((2, self.slot_count))

        for item in range(2):
            i = 0
            count = 0
            while i < self.input_calc:
                weights[item][i] = fc_weights[count][item]
                i += rotations
                count = count + 1
        self.plain_weights1 = self.ckks_encoder.encode(weights[0], self.scale)
        self.plain_weights2 = self.ckks_encoder.encode(weights[1], self.scale)

    # Final calculations to get output
    def Final_Calculations(self, input_cipher):
        self.evaluator.mod_switch_to_inplace(self.plain_weights1, input_cipher.parms_id())
        self.evaluator.mod_switch_to_inplace(self.plain_weights2, input_cipher.parms_id())

        out_cipher1 = self.evaluator.multiply_plain(input_cipher, self.plain_weights1)
        out_cipher2 = self.evaluator.multiply_plain(input_cipher, self.plain_weights2)

        self.evaluator.relinearize_inplace(out_cipher1, self.relin_keys)
        self.evaluator.rescale_to_next_inplace(out_cipher1)
        out_cipher1.scale(2**40)

        self.evaluator.relinearize_inplace(out_cipher2, self.relin_keys)
        self.evaluator.rescale_to_next_inplace(out_cipher2)
        out_cipher2.scale(2**40)

        out_plain = self.ckks_encoder.encode(np.zeros(self.slot_count), self.scale)
        out_cipher1_final = self.encryptor.encrypt(out_plain)
        out_cipher2_final = self.encryptor.encrypt(out_plain)

        self.evaluator.mod_switch_to_inplace(out_cipher1_final, out_cipher1.parms_id())
        self.evaluator.mod_switch_to_inplace(out_cipher2_final, out_cipher2.parms_id())

        rot_val = self.input_size // 2
        for i in range(3):
            self.evaluator.add_inplace(out_cipher1, (self.evaluator.rotate_vector(out_cipher1, (self.rotations * rot_val), self.galois_keys)))
            self.evaluator.add_inplace(out_cipher2, (self.evaluator.rotate_vector(out_cipher2, (self.rotations * rot_val), self.galois_keys)))

            if(i != 2):
                rot_val = rot_val // 2

        # adding all weights multiplied with input_size
        for i in range(0, rot_val):
            self.evaluator.add_inplace(out_cipher1_final, (self.evaluator.rotate_vector(out_cipher1, (self.rotations * i), self.galois_keys)))
            self.evaluator.add_inplace(out_cipher2_final, (self.evaluator.rotate_vector(out_cipher2, (self.rotations * i), self.galois_keys)))

        print("Final Outputs::::::")
        print("Output 1 is: ", self.ckks_encoder.decode(self.decryptor.decrypt(out_cipher1_final)))
        print("Output 2 is: ", self.ckks_encoder.decode(self.decryptor.decrypt(out_cipher2_final)))


# After Huge Performance improvements and testing #
# Everything completerd on 02/04/2022 #
# At 8:00 PM #

# It is in working condition #
# 08/04/2022 #
# At 03:27 PM #

# 10 times faster improvement completed #
# 09/04/2022 #
# At 12:30 PM #
