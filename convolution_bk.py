# Created By Muhammad Saad
# On 3/22/22 at 7:56 PM
# .........................
import numpy as np


class Convolution:
    # Initiating the keys and kernels here when class object is created
    def __init__(self, no_of_filters, seal_tuple, input_size):
        self.input_size = input_size
        self.no_of_filters = no_of_filters
        self.encryptor, self.evaluator, self.decryptor, self.slot_count, self.context, self.ckks_encoder, self.scale, self.galois_keys, self.relin_keys = seal_tuple
        self.k_s = 2
        self.output = (input_size - self.k_s) + 1
        self.interval = input_size - self.k_s
        self.kernel_size = 7
        filter = np.array([1.0, 2.0, 0.0, 0.0, 0.0, 3.0, 4.0])
        self.filter_plain = self.ckks_encoder.encode(filter, self.scale)

    # Checking the current level of a ciphertext
    def Level_Checker(self, p_id):
        context_data = self.context.first_context_data()
        count = 0
        count2 = 0
        while(context_data):
            count = count + 1
            context_data = context_data.next_context_data()
        context_data = self.context.first_context_data()
        while (context_data):
            count2 = count2 + 1
            index = context_data.chain_index()
            if (p_id == context_data.parms_id()):
                print(f"Total Data Levels are {count} | Currently at {index} | Remaining {count-count2}")
                break
            context_data = context_data.next_context_data()

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
        print("Relued output: ")
        print(self.ckks_encoder.decode(self.decryptor.decrypt(temp_cipher[0])))

        self.Level_Checker(temp_cipher[0].parms_id())

    # Main convolution operation
    def Convolve(self, input_cipher):
        # Setting up the encrpted variables
        rot_add = 2 * self.kernel_size
        output_size = self.output * self.output
        magic_vector = np.zeros(self.slot_count)
        magic_vector[0] = 1.0
        result = np.zeros(10, dtype=float)
        plain_result = self.ckks_encoder.encode(result, self.scale)
        plain_magic = self.ckks_encoder.encode(magic_vector, self.scale)
        cipher_results = []
        output_cipher = self.encryptor.encrypt(plain_result)

        # Multiplying the filter with each input
        rotate = 0
        for i in range(output_size):
            cipher_results.append(input_cipher)
            cipher_results[i] = self.evaluator.rotate_vector(cipher_results[i], rotate, self.galois_keys)
            cipher_results[i] = self.evaluator.multiply_plain(cipher_results[i], self.filter_plain)
            if(i != 0 and ((i + 1) % self.output) == 0):
                rotate = rotate + 1
            rotate = rotate + 1

        # Adding the final windows results
        for item in range(output_size):
            output_cipher = self.evaluator.rotate_vector(output_cipher, 1, self.galois_keys)

            self.evaluator.rescale_to_next_inplace(cipher_results[item])
            cipher_results[item].scale(2**40)

            self.evaluator.rotate_vector_inplace(cipher_results[item], rot_add, self.galois_keys)

            for i in range(self.kernel_size):
                self.evaluator.add_inplace(cipher_results[item], self.evaluator.rotate_vector(cipher_results[item], -(rot_add - i), self.galois_keys))
            if(item == 0):
                self.evaluator.mod_switch_to_inplace(plain_magic, cipher_results[item].parms_id())
            self.evaluator.multiply_plain_inplace(cipher_results[item], plain_magic)
            self.evaluator.rescale_to_next_inplace(cipher_results[item])
            cipher_results[item].scale(2**40)
            if(item == 0):
                self.evaluator.mod_switch_to_inplace(output_cipher, cipher_results[item].parms_id())

            # print(self.ckks_encoder.decode(self.decryptor.decrypt(cipher_results[item])))
            output_cipher = self.evaluator.add(output_cipher, cipher_results[item])

        # self.evaluator.rotate_vector_inplace(output_cipher, -(output_size - 1), self.galois_keys)
        self.evaluator.rotate_vector_inplace(output_cipher, -(output_size - 1), self.galois_keys)
        print("Convolution Final output: ")
        print(self.ckks_encoder.decode(self.decryptor.decrypt(output_cipher)))

        self.Encrypted_ReLU(output_cipher)
