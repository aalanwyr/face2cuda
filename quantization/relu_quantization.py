import numpy as np


def quantization(x, s, z, alpha_q, beta_q):

    x_q = np.round(1 / s * x + z, decimals=0)
    x_q = np.clip(x_q, a_min=alpha_q, a_max=beta_q)

    return x_q


def quantization_int8(x, s, z):

    x_q = quantization(x, s, z, alpha_q=-128, beta_q=127)
    x_q = x_q.astype(np.int8)

    return x_q


def quantization_uint8(x, s, z):

    x_q = quantization(x, s, z, alpha_q=0, beta_q=255) #非对称量化 uint8
    x_q = x_q.astype(np.uint8)

    return x_q


def dequantization(x_q, s, z):

    # x_q - z might go outside the quantization range so should astype int32
    x_q = x_q.astype(np.int32)
    x = s * (x_q - z)
    x = x.astype(np.float32)

    return x


def generate_quantization_constants(alpha, beta, alpha_q, beta_q):

    # Affine quantization mapping
    s = (beta - alpha) / (beta_q - alpha_q)
    z = int((beta * alpha_q - alpha * beta_q) / (beta - alpha))

    return s, z


def generate_quantization_int8_constants(alpha, beta):

    b = 8
    alpha_q = -2**(b - 1)
    beta_q = 2**(b - 1) - 1

    s, z = generate_quantization_constants(alpha=alpha,
                                           beta=beta,
                                           alpha_q=alpha_q,
                                           beta_q=beta_q)

    return s, z


def generate_quantization_uint8_constants(alpha, beta):

    b = 8
    alpha_q = 0
    beta_q = 2**(b) - 1

    s, z = generate_quantization_constants(alpha=alpha,
                                           beta=beta,
                                           alpha_q=alpha_q,
                                           beta_q=beta_q)

    return s, z


def relu(x, z_x, z_y, k):

    x = np.clip(x, a_min=z_x, a_max=None)
    y = z_y + k * (x - z_x)

    return y


def quantization_relu_uint8(x_q, s_x, z_x, s_y, z_y):

    y = relu(x=x_q.astype(np.int32), z_x=z_x, z_y=z_y, k=s_x / s_y)
    y = np.round(y, decimals=0)
    y = np.clip(y, a_min=0, a_max=255)
    y = y.astype(np.uint8)

    return y


if __name__ == "__main__":

    # Set random seed for reproducibility
    random_seed = 0
    np.random.seed(random_seed)

    # Random matrices
    m = 2
    n = 4

    alpha_X = -60.0
    beta_X = 60.0
    s_X, z_X = generate_quantization_int8_constants(alpha=alpha_X, beta=beta_X)
    X = np.random.uniform(low=alpha_X, high=beta_X,
                          size=(m, n)).astype(np.float32)
    X_q = quantization_int8(x=X, s=s_X, z=z_X)
    X_q_dq = dequantization(x_q=X_q, s=s_X, z=z_X)

    alpha_Y = 0.0
    beta_Y = 200.0
    s_Y, z_Y = generate_quantization_uint8_constants(alpha=alpha_Y,
                                                     beta=beta_Y)
    Y_expected = relu(x=X, z_x=0, z_y=0, k=1)
    Y_q_expected = quantization_uint8(x=Y_expected, s=s_Y, z=z_Y)

    Y_expected_prime = relu(x=X_q_dq, z_x=0, z_y=0, k=1)
    Y_expected_prime_q = quantization_uint8(x=Y_expected_prime, s=s_Y, z=z_Y)
    Y_expected_prime_q_dq = dequantization(x_q=Y_expected_prime_q,
                                           s=s_Y,
                                           z=z_Y)
    
    #错误的计算方法 直接通过量化后的输入计算 Relu
    Y_error_q = relu(x=X_q, z_x=0, z_y=0, k=1)
    Y_error_q_dq = dequantization(x_q=Y_error_q,s=s_Y,z=z_Y)

    print("X:")
    print(X)
    print("X_q:")
    print(X_q)

    print("Expected FP32 Y:")
    print(Y_expected)
    print("Expected FP32 Y Quantized:")
    print(Y_q_expected)

    Y_q_simulated = quantization_relu_uint8(x_q=X_q,
                                            s_x=s_X,
                                            z_x=z_X,
                                            s_y=s_Y,
                                            z_y=z_Y)
    Y_simulated = dequantization(x_q=Y_q_simulated, s=s_Y, z=z_Y)

    print("Expected Quantized Y_q from Quantized ReLU:")
    print(Y_q_simulated)
    print("Expected Quantized Y_q from Quantized ReLU Dequantized:")
    print(Y_simulated)
    
    print("-------------------------------------------------------")
    print("Y_error_q 错误的计算方法，直接通过量化的输入矩阵乘:")
    print(Y_error_q)
    print("Y_error_q_dq 错误的计算方法，直接通过量化的输入矩阵乘:")
    print(Y_error_q_dq)

    # Ensure the algorithm implementation is correct
    assert (np.array_equal(Y_simulated, Y_expected_prime_q_dq))
    assert (np.array_equal(Y_q_simulated, Y_expected_prime_q))