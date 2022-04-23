from seal import scheme_type, EncryptionParameters, CoeffModulus, SEALContext, CKKSEncoder, KeyGenerator, Encryptor, Evaluator, Decryptor
from seal_helper import print_parameters
import time


class SEALWork:

    def __init__(self):
        print("-------------------------------------------------------")
        print("--------Generating Parameters for HE Operations--------")
        print("-------------------------------------------------------")

    def initialize(self):
        start = time.time()
        parms = EncryptionParameters(scheme_type.ckks)
        poly_modulus_degree = 16384
        parms.set_poly_modulus_degree(poly_modulus_degree)
        # parms.set_coeff_modulus(CoeffModulus.Create(
        #     poly_modulus_degree, [60, 40, 40, 40, 40, 40, 40, 40, 60]))
        parms.set_coeff_modulus(CoeffModulus.Create(
            poly_modulus_degree, [60, 50, 50, 50, 50, 50, 60]))
        scale = 2.0 ** 50
        context = SEALContext(parms)
        print_parameters(context)

        ckks_encoder = CKKSEncoder(context)
        slot_count = ckks_encoder.slot_count()

        keygen = KeyGenerator(context)
        public_key = keygen.create_public_key()
        secret_key = keygen.secret_key()
        galois_keys = keygen.create_galois_keys()
        relin_keys = keygen.create_relin_keys()

        encryptor = Encryptor(context, public_key)
        evaluator = Evaluator(context)
        decryptor = Decryptor(context, secret_key)
        seal_tuple = encryptor, evaluator, decryptor, slot_count, context, ckks_encoder, scale, galois_keys, relin_keys
        end = time.time()
        print("-------------------------------------------------------")
        print(f"Parmaeters Generation Ended in {end-start} seconds")
        print("-------------------------------------------------------")

        return seal_tuple
