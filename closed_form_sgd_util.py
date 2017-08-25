import numpy as np
import matplotlib.pyplot as plt


def calculate_error(input_matrix, weights, output_vec):
    predicted_output = np.dot(weights, input_matrix.transpose())
    sq_error_sum = np.sum(np.square(output_vec - predicted_output))
    error = np.sqrt(float(sq_error_sum)/len(output_vec))
    return error


def plot_data(y_values1, y_values2, lamda, label1, label2, axis_dim):
    plt.plot(y_values1, 'ro',label=label1)
    plt.plot(y_values2, 'b-', label = label2)
    plt.axis(axis_dim)
    plt.ylabel('RMSE')
    plt.xlabel('Model Complexity')
    plt.title('Lambda = ' + str(lamda))
    l = plt.legend()
    plt.show()


def random_shuffle_dataset(input_matrix, output_vec):
    complete_train_data = np.insert(input_matrix, 0, output_vec, axis=1)
    np.random.shuffle(complete_train_data)
    training_labels = complete_train_data[:,0]
    input_matrix = np.delete(complete_train_data,0,axis=1)
    return input_matrix, training_labels

def split_training_data(input_matrix, output_vec, train_percent, validation_percent):
    input_matrix, output_vec = random_shuffle_dataset(input_matrix, output_vec)
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

    return np.array(training_data), np.array(training_labels), np.array(valid_data), np.array(valid_labels), np.array(test_data), np.array(test_labels)

def k_means_clusters(train_data, training_labels, num_basis):
    try:
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=num_basis, random_state=0).fit(train_data)
        labels = kmeans.labels_
        cluster_centers = kmeans.cluster_centers_
    except Exception as e:
        print "Error: ", str(e)
        from kmeans_implement import kmeans
        cluster_centers = kmeans(train_data, k=num_basis)
    return cluster_centers

def create_design_matrix_train_data(train_data, training_labels, lamda, num_basis):
    variance = train_data.var(axis=0) 
    sigma = variance * np.identity(len(train_data[0]))
    sigma = sigma + 0.001 * np.identity(len(train_data[0])) # Add a small quantity to avoid 0 values in variance matrix.
    sigma_inv = np.linalg.inv(sigma)

    rand_centers = k_means_clusters(train_data, training_labels, num_basis)
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
                
    return design_matrix, sigma_inv, rand_centers


def create_design_matrix_data(data, sigma_inv, rand_centers, num_basis):
    design_matrix = np.zeros((len(data),num_basis))
    for i in range(len(data)):
        for j in range(num_basis):
            if j==0:
                design_matrix[i][j] = 1;
            else:
                x_mu = data[i] - rand_centers[j]
                x_mu_trans = x_mu.transpose()
                temp1_valid = np.dot(sigma_inv, x_mu_trans)
                temp2_valid = np.dot(x_mu, temp1_valid)
                design_matrix[i][j] = np.exp(((-0.5) * temp2_valid))
    return design_matrix


def closed_form_solution_training_phase(design_matrix, sigma_inv, training_labels, lamda, num_basis):
    design_matrix_trans = design_matrix.transpose()
    regularisation_mat = lamda * np.identity(num_basis)
    pinv_temp = np.dot(design_matrix_trans, design_matrix) + regularisation_mat
    pinv = np.linalg.inv(pinv_temp)
    out_temp = np.dot(design_matrix_trans, training_labels)
    weights = np.dot(pinv, out_temp)
    train_error = calculate_error(design_matrix, weights, training_labels)
    return weights, train_error


def stochastic_gradient_solution(design_matrix_train, training_labels, lamda, num_basis):
    n = 1
    boost_factor = 1.25
    degrade_factor = 0.8
    del_error = 100000
    weights = np.random.uniform(-1.0,1.0,size=(1,num_basis))[0]
    eta1 = []
    error_iteration = []
    num_iter = 0
    # We shall choose a small value for change in the relative error between two passes 
    # as the termination condition, but sometimes the model performs additional unnecessary passes 
    # before achieving this small error change. Hence we prune the number of passes and allow the model
    # to make constant passes, as additional passes don't lead to any significant gain.
    while del_error > 0.00001 and num_iter < 5:
        complete_train_data = np.insert(design_matrix_train, 0, training_labels, axis=1)
        np.random.shuffle(complete_train_data)
        training_labels = complete_train_data[:,0]
        design_matrix_train = np.delete(complete_train_data,0,axis=1)
        for i in range(len(training_labels)):
            error_iteration.append(calculate_error(design_matrix_train, weights, training_labels))
            temp1 = training_labels[i] - np.dot(weights, design_matrix_train[i,:].transpose())
            temp2 = -1 * temp1 * design_matrix_train[i,:]
            temp3 = temp2 + lamda * weights
            eta1.append(n)
            new_weights = weights - n * temp3
            new_weight_vec = np.sum(np.square(new_weights))
            old_weight_vec = np.sum(np.square(weights))
            if np.sqrt(np.abs(new_weight_vec - old_weight_vec)) < 0.0001:
                n = n * boost_factor
            else:
                n = n * degrade_factor
            weights = new_weights
        train_error = calculate_error(design_matrix_train, weights, training_labels)
        if num_iter == 0:
            init_error = train_error
            del_error = 100000
        else:
            del_error = init_error - train_error
            init_error = train_error
        num_iter +=1
    return weights, train_error, eta1, error_iteration