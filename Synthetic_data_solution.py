import numpy as np
from sklearn.cluster import KMeans

def read_data_file(in_filename, out_filename):
    input_matrix = []
    output_vec = []

    lines_in = [line.rstrip('\n') for line in open(in_filename)]
    lines_out = [line.rstrip('\n') for line in open(out_filename)]

    print len(lines_in), len(lines_out)
    for line in lines_out:
        output_vec.append(int(line))

    for line in lines_in:
        token_list = line.split(',')
        input_vec = []
        for token in token_list:
            input_vec.append(float(token))
        input_matrix.append(input_vec)

    output_vec = np.array(output_vec)
    input_matrix = np.array(input_matrix)
    return input_matrix, output_vec

def split_training_data(input_matrix, output_vec, train_percent, validation_percent):
    training_data = []
    training_labels = []
    valid_data = []
    valid_labels = []
    test_data = []
    test_labels = []

    train_len = int(np.floor(float(train_percent) * len(input_matrix)))
    for i in range(train_len):
        training_data.append(input_matrix[i])
        training_labels.append(output_vec[i])

    validation_len = int(np.floor(validation_percent * len(input_matrix)))
    for i in range(train_len, train_len+validation_len):
        valid_data.append(input_matrix[i])
        valid_labels.append(output_vec[i])

    for i in range(train_len+validation_len, len(input_matrix)):
        test_data.append(input_matrix[i])
        test_labels.append(output_vec[i])

    print len(training_labels), len(valid_labels), len(test_labels)
    return np.array(training_data), np.array(training_labels), np.array(valid_data), np.array(valid_labels), np.array(test_data), np.array(test_labels)

def k_means_clusters(train_data, training_labels, num_basis):
    kmeans = KMeans(n_clusters=num_basis, random_state=0).fit(train_data)
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    clusters = {}
    for i in range(num_basis):
        clusters[i] = {'train_data':[], 'output' : [], 'center' : cluster_centers[i]}

        for j in range(len(train_data)):
            if labels[j] == i:
                clusters[i]['train_data'].append(train_data[j])
                clusters[i]['output'].append(training_labels[j])

    for i in range(num_basis):
        print i, len(clusters[i]['train_data'])

    return cluster_centers


def closed_form_solution_validation_phase(valid_data, valid_labels, weights, sigma_inv, rand_centers,num_basis):
    valid_design_matrix=np.zeros((len(valid_data),num_basis));
    for i in range(len(valid_data)):
        for j in range(num_basis):
            if j==0:
                valid_design_matrix[i][j] = 1;
            else:
                x_mu = valid_data[i]-rand_centers[j]
                x_mu_trans = x_mu.transpose()
                temp1_valid = np.dot(sigma_inv, x_mu_trans)
                temp2_valid = np.dot(x_mu, temp1_valid)
                valid_design_matrix[i][j] = np.exp(((-0.5)*temp2_valid))

    predicted_output_validation = np.dot(weights, valid_design_matrix.transpose())
    count = 0
    for i in range(len(predicted_output_validation)):
        if np.rint(predicted_output_validation[i]) != int(valid_labels[i]):
            count +=1

    print "Error: ", float(count)/len(predicted_output_validation)
    sq_error_sum_validation = np.sum(np.square(valid_labels - predicted_output_validation))

    # print sq_error_sum
    validation_error = np.sqrt(float(sq_error_sum_validation)/len(valid_data))
    return validation_error


def closed_form_solution_training_phase(train_data, training_labels,lamda, num_basis):
    variance = train_data.var(axis=0) 
    sigma = variance * np.identity(len(train_data[0]))
    sigma = sigma + lamda * np.identity(len(train_data[0]))
    sigma_inv = np.linalg.inv(sigma)

    # rand_indices = np.random.randint(0, len(train_data), size=(1,num_basis))

    # rand_centers = []
    rand_centers = k_means_clusters(train_data, training_labels, num_basis)
    # for i in range(len(rand_indices[0])):
    #     index = rand_indices[0][i]
    #     rand_centers.append(train_data[index])

    rand_centers = np.array(rand_centers)
    design_matrix=np.zeros((len(train_data),num_basis));

    for i in range(len(train_data)):
        for j in range(num_basis):
            if j==0:
                design_matrix[i][j] = 1;
            else:
                x_mu = train_data[i]-rand_centers[j]
                x_mu_trans = x_mu.transpose()
                temp1 = np.dot(sigma_inv, x_mu_trans)
                temp2 = np.dot(x_mu, temp1)
                design_matrix[i][j] = np.exp(((-0.5)*temp2))
    
    design_matrix_trans = design_matrix.transpose()
    regularisation_mat = lamda * np.identity(num_basis)
    pinv_temp = np.dot(design_matrix_trans, design_matrix) + regularisation_mat
    pinv = np.linalg.inv(pinv_temp)
    out_temp = np.dot(design_matrix_trans, training_labels)
    weights = np.dot(pinv, out_temp)

    predicted_output = np.dot(weights, design_matrix_trans)
    sq_error_sum = np.sum(np.square(training_labels - predicted_output))

    train_error = np.sqrt(float(sq_error_sum)/len(train_data))
    return design_matrix, rand_centers, sigma_inv, weights, train_error

def main():
    in_filename = 'input.csv'
    out_filename = 'output.csv'
    feature_mat, output_labels = read_data_file(in_filename, out_filename)
    training_data, training_labels, valid_data, valid_labels, test_data, test_labels = split_training_data(feature_mat, output_labels, 0.8, 0.1)
    print training_data.shape, valid_data.shape, test_data.shape
    for num_basis in range(2, 11):
        print "Num Basis Functions: ", num_basis
        design_matrix, rbf_centers, sigma_inv, weights, rmse_train = closed_form_solution_training_phase(training_data, training_labels, 0.3, num_basis)
        rmse_validation = closed_form_solution_validation_phase(valid_data, valid_labels, weights, sigma_inv, rbf_centers, num_basis)
        # print "Weights: ", weights
        print "RMSE Train: ", rmse_train
        print "RMSE Validation: ", rmse_validation
        print closed_form_solution_validation_phase(test_data, test_labels, weights, sigma_inv, rbf_centers, num_basis)


main()