from closed_form_sgd_util import *

def read_data_file_letor(filename):
    lines = [line.rstrip('\n') for line in open(filename)]
    input_matrix = []
    output_vec = []
    for line in lines:
        token_list = line.split()
        if len(token_list) == 57:
            output_vec.append(int(token_list[0]))
            input_vector = []
            for i in range(2, 48):
                token = token_list[i]
                input_vector.append(float(token.split(':')[1]))
            input_matrix.append(input_vector)

    output_vec = np.array(output_vec)
    input_matrix = np.array(input_matrix)
    return input_matrix, output_vec


def read_data_file_synthetic(in_filename, out_filename):
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


def training_closed_form_for_multiple_lamda_basis_functions(lamda_values, error_matrix_letor, training_data_letor, training_labels_letor, valid_data_letor, valid_labels_letor, test_data_letor, test_labels_letor):
    for lamda in lamda_values:
        print "lamda = ", lamda
        error_matrix_letor['closed_form']['train'][lamda] = []
        error_matrix_letor['closed_form']['validation'][lamda] = []
        for num_basis in range(1, 31):
            design_matrix_train_letor, sigma_inv_letor, rbf_centers_letor = create_design_matrix_train_data(training_data_letor, training_labels_letor, lamda, num_basis)
            weights_letor, rmse_train_letor = closed_form_solution_training_phase(design_matrix_train_letor, sigma_inv_letor, training_labels_letor, lamda, num_basis)
            design_matrix_validation_letor = create_design_matrix_data(valid_data_letor, sigma_inv_letor, rbf_centers_letor, num_basis)
            rmse_validation_letor = calculate_error(design_matrix_validation_letor, weights_letor, valid_labels_letor)
            error_matrix_letor['closed_form']['train'][lamda].append(rmse_train_letor)
            error_matrix_letor['closed_form']['validation'][lamda].append(rmse_validation_letor)
            print num_basis, rmse_validation_letor
    return error_matrix_letor

def train_closed_form_letor(lamda_values, error_matrix_letor, training_data_letor, training_labels_letor, valid_data_letor, valid_labels_letor, test_data_letor, test_labels_letor):

    ''' Uncomment to train the closed form solution for different values of M and lamda.
        We print the RMSE validation for different values of M and lamda and choose the parameters that give least RMSE validation. '''
    # error_matrix_letor = training_closed_form_for_multiple_lamda_basis_functions(lamda_values, error_matrix_letor, training_data_letor, training_labels_letor, valid_data_letor, valid_labels_letor, test_data_letor, test_labels_letor)

    ''' The plots can be generated using the error matrix obtained from training. These plots help us choose the optimum hyperparameters M and lamda'''
    for lamda in sorted(error_matrix_letor['closed_form']['train']):
        print "Plot for RMSE Validation across different number of Gaussian Basis Functions, for lamda = ", lamda
        plot_data(error_matrix_letor['closed_form']['validation'][lamda],error_matrix_letor['closed_form']['train'][lamda], lamda, 'RMSE Validation', 'RMSE Training', [0, 30, 0.5, 0.7])


    ''' After choosing the best hyperparameters, we fix their values and compute the model performance on test set'''
    lamda = 0.01
    num_basis = 30
    design_matrix_train_cf, sigma_inv_cf, rbf_centers_cf = create_design_matrix_train_data(training_data_letor,training_labels_letor, lamda, num_basis)
    weights_cf, rmse_train_cf = closed_form_solution_training_phase(design_matrix_train_cf, sigma_inv_cf, training_labels_letor, lamda, num_basis)
    design_matrix_validation_cf = create_design_matrix_data(valid_data_letor, sigma_inv_cf, rbf_centers_cf, num_basis)
    rmse_validation_cf = calculate_error(design_matrix_validation_cf, weights_cf, valid_labels_letor)
    design_matrix_test_cf = create_design_matrix_data(test_data_letor, sigma_inv_cf, rbf_centers_cf, num_basis)
    rmse_test_cf = calculate_error(design_matrix_test_cf,weights_cf, test_labels_letor)

    print "Best fit Linear Regression Model using Closed Form Solution Performance: "
    print "\n"
    print "RMSE on Training Set: ", rmse_train_cf 
    print "\n"
    print "RMSE on Validation Set: ", rmse_validation_cf
    print "\n"
    print "RMSE on Test Set: ", rmse_train_cf
    print "\n"
    print "Weights Vector(w) for the trained Model:\n", weights_cf


def training_sgd_for_multiple_lamda_basis_functions(lamda_values, error_matrix_letor, training_data_letor, training_labels_letor, valid_data_letor, valid_labels_letor, test_data_letor, test_labels_letor):
    for lamda in lamda_values:
        print "lamda = ", lamda
        error_matrix_letor['sgd']['train'][lamda] = []
        error_matrix_letor['sgd']['validation'][lamda] = []
        for num_basis in range(1, 31):
            design_matrix_train_letor, sigma_inv_letor, rbf_centers_letor = create_design_matrix_train_data(training_data_letor, training_labels_letor, lamda, num_basis)
            weights_letor, rmse_train_letor, learning_rate_changes, error_iteration = stochastic_gradient_solution(design_matrix_train_letor, training_labels_letor, lamda, num_basis)
            design_matrix_validation_letor = create_design_matrix_data(valid_data_letor, sigma_inv_letor, rbf_centers_letor, num_basis)
            rmse_validation_letor = calculate_error(design_matrix_validation_letor, weights_letor, valid_labels_letor)
            error_matrix_letor['sgd']['train'][lamda].append(rmse_train_letor)
            error_matrix_letor['sgd']['validation'][lamda].append(rmse_validation_letor)
            print num_basis, rmse_validation_letor
    return error_matrix_letor


def train_sgd_letor(lamda_values, error_matrix_letor, training_data_letor, training_labels_letor, valid_data_letor, valid_labels_letor, test_data_letor, test_labels_letor):

    ''' Uncomment to train the closed form solution for different values of M and lamda.
        We print the RMSE validation for different values of M and lamda and choose the parameters that give least RMSE validation. '''
    # error_matrix_letor = training_sgd_for_multiple_lamda_basis_functions(lamda_values, error_matrix_letor, training_data_letor, training_labels_letor, valid_data_letor, valid_labels_letor, test_data_letor, test_labels_letor)

    ''' The plots can be generated using the error matrix obtained from training. These plots help us choose the optimum hyperparameters M and lamda'''
    for lamda in sorted(error_matrix_letor['sgd']['train']):
        print "Plot for RMSE Validation across different number of Gaussian Basis Functions, for lamda = ", lamda
        plot_data(error_matrix_letor['sgd']['validation'][lamda],error_matrix_letor['sgd']['train'][lamda], lamda, 'RMSE Validation', 'RMSE Training', [0, 30, 0.5, 0.7])


    ''' After choosing the best hyperparameters, we fix their values and compute the model performance on test set'''
    lamda = 0.01
    num_basis = 11
    design_matrix_train_sgd, sigma_inv_sgd, rbf_centers_sgd = create_design_matrix_train_data(training_data_letor, training_labels_letor, lamda, num_basis)
    weights_sgd, rmse_train_sgd, learning_rate_changes_sgd, error_iteration_letor_sgd = stochastic_gradient_solution(design_matrix_train_sgd, training_labels_letor, lamda, num_basis)
    design_matrix_validation_sgd = create_design_matrix_data(valid_data_letor, sigma_inv_sgd, rbf_centers_sgd, num_basis)
    rmse_validation_sgd = calculate_error(design_matrix_validation_sgd, weights_sgd, valid_labels_letor)
    design_matrix_test_sgd = create_design_matrix_data(test_data_letor, sigma_inv_sgd, rbf_centers_sgd, num_basis)
    rmse_test_sgd = calculate_error(design_matrix_test_sgd, weights_sgd, test_labels_letor)

    print "Best fit Linear Regression Model using Stochastic Gradient Descent(SGD) Performance Metrics: "
    print "\n"
    print "RMSE on Training Set: ", rmse_train_sgd 
    print "\n"
    print "RMSE on Validation Set: ", rmse_validation_sgd
    print "\n"
    print "RMSE on Test Set: ", rmse_test_sgd
    print "\n"
    print "Weights Vector(w) for the trained Model:\n", weights_sgd

    ''' Commented the plots, to handle error on timberlake server, uncomment to plot '''
    # plt.plot(error_iteration_letor_sgd, 'r-', label='Train Error')
    # plt.axis([0, 50, 0.4, 1])
    # plt.ylabel('RMSE Training')
    # plt.xlabel('SGD Iteration')
    # plt.title('Change in Training Error vs SGD Iterations')
    # l = plt.legend()
    # plt.show()

    # plt.plot(learning_rate_changes_sgd, 'g-', label='Learning Rate')
    # plt.axis([0, 100, 0, 1])
    # plt.ylabel('Learning Rate Eta Value')
    # plt.xlabel('SGD Iteration')
    # plt.title('Learning Rate Decay as Model converges')
    # l = plt.legend()
    # plt.show()

def training_closed_form_for_multiple_lamda_basis_functions_syn(lamda_values, error_matrix_syn, training_data_syn, training_labels_syn, valid_data_syn, valid_labels_syn, test_data_syn, test_labels_syn):
    for lamda in lamda_values:
        print "lamda = ", lamda
        error_matrix_syn['closed_form']['train'][lamda] = []
        error_matrix_syn['closed_form']['validation'][lamda] = []
        for num_basis in range(1, 11):
            design_matrix_train_syn, sigma_inv_syn, rbf_centers_syn = create_design_matrix_train_data(training_data_syn, training_labels_syn, lamda, num_basis)
            weights_syn, rmse_train_syn = closed_form_solution_training_phase(design_matrix_train_syn, sigma_inv_syn, training_labels_syn, lamda, num_basis)
            design_matrix_validation_syn = create_design_matrix_data(valid_data_syn, sigma_inv_syn, rbf_centers_syn, num_basis)
            rmse_validation_syn = calculate_error(design_matrix_validation_syn, weights_syn, valid_labels_syn)
            error_matrix_syn['closed_form']['train'][lamda].append(rmse_train_syn)
            error_matrix_syn['closed_form']['validation'][lamda].append(rmse_validation_syn)
            print num_basis, rmse_validation_syn
    return error_matrix_syn


def train_closed_form_synthetic_data(lamda_values, error_matrix_syn, training_data_syn, training_labels_syn, valid_data_syn, valid_labels_syn, test_data_syn, test_labels_syn):
    
    ''' Uncomment to train the closed form solution for different values of M and lamda.
        We print the RMSE validation for different values of M and lamda and choose the parameters that give least RMSE validation. '''
    # error_matrix_syn = training_closed_form_for_multiple_lamda_basis_functions_syn(lamda_values, error_matrix_syn, training_data_syn, training_labels_syn, valid_data_syn, valid_labels_syn, test_data_syn, test_labels_syn)

    ''' The plots can be generated using the error matrix obtained from training. These plots help us choose the optimum hyperparameters M and lamda'''
    for lamda in sorted(error_matrix_syn['closed_form']['train']):
        print "Plot for RMSE Validation across different number of Gaussian Basis Functions, for lamda = ", lamda
        plot_data(error_matrix_syn['closed_form']['validation'][lamda],error_matrix_syn['closed_form']['train'][lamda], lamda, 'RMSE Validation', 'RMSE Training', [1, 9, 0.6, 0.8])

    ''' After choosing the best hyperparameters, we fix their values and compute the model performance on test set'''
    lamda = 0.1
    num_basis = 8
    design_matrix_train_syn_closed, sigma_inv_syn_closed, rbf_centers_syn_closed = create_design_matrix_train_data(training_data_syn, training_labels_syn, lamda, num_basis)
    weights_syn_closed, rmse_train_syn_closed= closed_form_solution_training_phase(design_matrix_train_syn_closed, sigma_inv_syn_closed, training_labels_syn, lamda, num_basis)
    design_matrix_validation_syn_closed = create_design_matrix_data(valid_data_syn, sigma_inv_syn_closed, rbf_centers_syn_closed, num_basis)
    rmse_validation_syn_closed = calculate_error(design_matrix_validation_syn_closed, weights_syn_closed, valid_labels_syn)
    design_matrix_test_syn_closed = create_design_matrix_data(test_data_syn, sigma_inv_syn_closed, rbf_centers_syn_closed, num_basis)
    rmse_test_syn_closed = calculate_error(design_matrix_test_syn_closed, weights_syn_closed, test_labels_syn)

    print "Best fit Linear Regression Model using Closed Form Solution Performance: "
    print "\n"
    print "RMSE on Training Set: ", rmse_train_syn_closed 
    print "\n"
    print "RMSE on Validation Set: ", rmse_validation_syn_closed
    print "\n"
    print "RMSE on Test Set: ", rmse_test_syn_closed
    print "\n"
    print "Weights Vector(w) for the trained Model:\n", weights_syn_closed


def training_sgd_for_multiple_lamda_basis_functions_syn(lamda_values, error_matrix_syn, training_data_syn, training_labels_syn, valid_data_syn, valid_labels_syn, test_data_syn, test_labels_syn):
    for lamda in lamda_values:
        print "lamda = ", lamda
        error_matrix_syn['sgd']['train'][lamda] = []
        error_matrix_syn['sgd']['validation'][lamda] = []
        for num_basis in range(1, 11):
            design_matrix_train_syn_sgd, sigma_inv_syn_sgd, rbf_centers_syn_sgd = create_design_matrix_train_data(training_data_syn, training_labels_syn, lamda, num_basis)
            weights_syn_sgd, rmse_train_syn_sgd, learning_rate_changes_syn = stochastic_gradient_solution(design_matrix_train_syn_sgd, training_labels_syn, lamda, num_basis)
            design_matrix_validation_syn_sgd = create_design_matrix_data(valid_data_syn, sigma_inv_syn_sgd, rbf_centers_syn_sgd, num_basis)
            rmse_validation_syn_sgd = calculate_error(design_matrix_validation_syn_sgd, weights_syn_sgd, valid_labels_syn)
            error_matrix_syn['sgd']['train'][lamda].append(rmse_train_syn_sgd)
            error_matrix_syn['sgd']['validation'][lamda].append(rmse_validation_syn_sgd)
            print num_basis, rmse_validation_syn_sgd
    return error_matrix_syn


def train_sgd_synthetic_data(lamda_values, error_matrix_syn, training_data_syn, training_labels_syn, valid_data_syn, valid_labels_syn, test_data_syn, test_labels_syn):

    ''' Uncomment to train the closed form solution for different values of M and lamda.
        We print the RMSE validation for different values of M and lamda and choose the parameters that give least RMSE validation. '''
    # error_matrix_syn = training_sgd_for_multiple_lamda_basis_functions_syn(lamda_values, error_matrix_syn, training_data_syn, training_labels_syn, valid_data_syn, valid_labels_syn, test_data_syn, test_labels_syn)

    ''' The plots can be generated using the error matrix obtained from training. These plots help us choose the optimum hyperparameters M and lamda'''
    for lamda in sorted(error_matrix_syn['sgd']['train']):
        print "Plot for RMSE Validation across different number of Gaussian Basis Functions, for lamda = ", lamda
        plot_data(error_matrix_syn['sgd']['validation'][lamda],error_matrix_syn['sgd']['train'][lamda], lamda, 'RMSE Validation', 'RMSE Training', [0, 10, 0.7, 1.2])

    ''' After choosing the best hyperparameters, we fix their values and compute the model performance on test set'''
    lamda = 0.1
    num_basis = 5
    design_matrix_train_syn_sgd, sigma_inv_syn_sgd, rbf_centers_syn_sgd = create_design_matrix_train_data(training_data_syn, training_labels_syn, lamda, num_basis)
    weights_syn_sgd, rmse_train_syn_sgd, learning_rate_changes_syn_sgd, error_iteration_syn_sgd = stochastic_gradient_solution(design_matrix_train_syn_sgd, training_labels_syn, lamda, num_basis)
    design_matrix_validation_syn_sgd = create_design_matrix_data(valid_data_syn, sigma_inv_syn_sgd, rbf_centers_syn_sgd, num_basis)
    rmse_validation_syn_sgd = calculate_error(design_matrix_validation_syn_sgd, weights_syn_sgd, valid_labels_syn)
    design_matrix_test_syn_sgd = create_design_matrix_data(test_data_syn, sigma_inv_syn_sgd, rbf_centers_syn_sgd, num_basis)
    rmse_test_syn_sgd = calculate_error(design_matrix_test_syn_sgd, weights_syn_sgd, test_labels_syn)

    print "Best fit Linear Regression Model using Stochastic Gradient Descent(SGD) Performance Metrics: "
    print "\n"
    print "RMSE on Training Set: ", rmse_train_syn_sgd 
    print "\n"
    print "RMSE on Validation Set: ", rmse_validation_syn_sgd
    print "\n"
    print "RMSE on Test Set: ", rmse_test_syn_sgd
    print "\n"
    print "Weights Vector(w) for the trained Model:\n", weights_syn_sgd

    ''' Commented the plots, to handle error on timberlake server, uncomment to plot '''
    # plt.plot(error_iteration_syn_sgd, 'r-', label='Train Error')
    # plt.axis([0, 50, 0.7, 1.8])
    # plt.ylabel('RMSE Training')
    # plt.xlabel('SGD Iteration')
    # plt.title('Change in Training Error vs SGD Iterations')
    # l = plt.legend()
    # plt.show()

    # plt.plot(learning_rate_changes_syn_sgd, 'g-', label='Learning Rate')
    # plt.axis([0, 100, 0, 1])
    # plt.ylabel('Learning Rate Eta Value')
    # plt.xlabel('SGD Iteration')
    # plt.title('Learning Rate Decay as Model converges')
    # l = plt.legend()
    # plt.show()


def main():
    train_percent = 0.8
    validation_percent = 0.1
    lamda_values = [0.001, 0.01, 0.1, 0.5, 1]

    filename_letor = 'Querylevelnorm.txt'
    feature_mat_letor, output_labels_letor = read_data_file_letor(filename_letor)
    training_data_letor, training_labels_letor, valid_data_letor, valid_labels_letor, test_data_letor, test_labels_letor = split_training_data(feature_mat_letor, output_labels_letor, train_percent, validation_percent)
    print "Number of samples in Training Data: ", len(training_data_letor)
    print "Number of samples in Validation Data: ", len(valid_data_letor)
    print "Number of samples in Test Data: ", len(test_data_letor)

    error_matrix_letor = {'closed_form':{'train':{}, 'validation':{}}, 'sgd': {'train':{}, 'validation':{}}}
    
    train_closed_form_letor(lamda_values, error_matrix_letor, training_data_letor, training_labels_letor, valid_data_letor, valid_labels_letor, test_data_letor, test_labels_letor)

    train_sgd_letor(lamda_values, error_matrix_letor, training_data_letor, training_labels_letor, valid_data_letor, valid_labels_letor, test_data_letor, test_labels_letor)


    in_filename_syn = 'input.csv'
    out_filename_syn = 'output.csv'
    feature_mat_syn, output_labels_syn = read_data_file_synthetic(in_filename_syn, out_filename_syn)
    training_data_syn, training_labels_syn, valid_data_syn, valid_labels_syn, test_data_syn, test_labels_syn = split_training_data(feature_mat_syn, output_labels_syn, train_percent, validation_percent)
    print "Number of samples in Training Data: ", len(training_data_syn)
    print "Number of samples in Validation Data: ", len(valid_data_syn)
    print "Number of samples in Test Data: ", len(test_data_syn)

    error_matrix_syn = {'closed_form':{'train':{}, 'validation':{}}, 'sgd': {'train':{}, 'validation':{}}}

    train_closed_form_synthetic_data(lamda_values, error_matrix_syn, training_data_syn, training_labels_syn, valid_data_syn, valid_labels_syn, test_data_syn, test_labels_syn)

    train_sgd_synthetic_data(lamda_values, error_matrix_syn, training_data_syn, training_labels_syn, valid_data_syn, valid_labels_syn, test_data_syn, test_labels_syn)


main()