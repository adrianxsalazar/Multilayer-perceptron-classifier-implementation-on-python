class multi_layer_perceptron():
    ######    SET OF FUNTIONS FOR THE MULTICLASS PERCEPTRON CLASSIFIER    ######
    #Main concept
    #The classifier assigns to a given instance multiple class belonging values
    #by multiplying the instance attribute values by each class weights.
    #Then, the classifier assigns to the intance the class with highest class
    #belonging values.

    #####     FUNCTION TO GENERATE THE WEIGHTS MATRIX  FOR PERCEPTRON      #####
    #We use this function to generate the random weight values required for the
    #multiclass perceptron algortihm. Later we eill update the weights using
    #stochastic gradient descent.
    def creation_random_weights_perceptron(self,attributes, labels):
        #Basic information about the input.
        number_attributes=attributes.shape[1]

        #Get information about the labels: label unique values, frequencies.....
        #and number of labels unique values.
        labels_values_for_weights , frequency_labels_for_weights=\
        np.unique(labels, return_counts=True)
        number_different_labels=len(labels_values_for_weights)

        #number of unique label values
        number_different_labels_weights=len(labels_values_for_weights)

        #We create an empty matrix where we will store the perceptron weights..
        #in form of row vector (1 x number attributes).Matrix dimensions are:
        #(Number of different labels x number of attributes).
        #Where each row is a weight vector assiged to a class.
        weights_coordinates_matrix=\
        np.zeros((number_different_labels,number_attributes),dtype=float)

        #loop to create as many weights vectors as number of diiferent labels.
        #Each of the weight vectors corresponds to a class.
        for weight_vector_creation_index in range(number_different_labels_weights):

            #Vector to store the weight values of an individual weight vector.
            weight_vector_random_values=np.zeros((1,number_attributes),dtype=float)

            #loop that generates a random number between the maximum and minimum
            #value of the attribute domain assigned to the weight value.
            for random_weight_index in range(number_attributes):

                #Get the random value with the required characteristics.
                value_random_cluster_coordinate=\
                random.uniform(attributes[:,random_weight_index].min(),\
                               attributes[:,random_weight_index].max())

                #Place the random value into the vector that stores them.
                weight_vector_random_values[0,random_weight_index]=\
                value_random_cluster_coordinate

            #Place the vector we have created with the weight values into the
            #matrix that stores all of them.
            weights_coordinates_matrix[weight_vector_creation_index,:]=\
            weight_vector_random_values

        #return the random weights matrixmatrix
        return weights_coordinates_matrix
    ####################       END OF THE FUNCTION        ######################


    ###########  FUNCTION TO UPDATE THE WEIGHTS/PERCEPTRONS MATRIX    ##########
    #We use this function in the training set to update the perceptron weights.
    #We update the weights of two perceptrons. The set of weights values for the
    #perceptron that correspond with the class that has been misclassified and
    #the set of values that corresponds with the right class.
    def update_weights_process(self,weights_matrix_to_modify,instance_analysis,\
    alpha,right_label_index, wrong_label_index):
        #using the indexes, retrieve the weight instances we want to update
        wrong_weight=weights_matrix_to_modify[wrong_label_index,:]
        right_weight=weights_matrix_to_modify[right_label_index,:]

        #The weights update modifies the weights of the perceptrons that
        #corresponds with the classes that were predicted wrongly and the class
        #that should have been predicted. Moving the former away from the
        #coordinates of the misclassified instance and the later closer to the
        #label that correspond with the intance
        new_weight_wrong_weight=wrong_weight-alpha*instance_analysis
        new_weight_right_weight=right_weight+alpha*instance_analysis


        #Update the original weights matrix with the updated weights
        weights_matrix_to_modify[wrong_label_index,:]=new_weight_wrong_weight
        weights_matrix_to_modify[right_label_index,:]=new_weight_right_weight

        #return the weight matrix
        return weights_matrix_to_modify
    ######################        END OF THE FUNCTION        ###################


    #####   FUNCTION TO COUNT THE MISCLASSIFIED INTANCES PER ITERATION    ######
    #We use this function in the perceptron training to count the instances
    #that were misclassified in the training iterations.
    def counter_misclassified_instances(self,prediction, real_label):
        #rtransfomr inputs into arrays. This is a security check
        prediction=np.array(prediction)
        real_label=np.array(real_label)

        #Basic informantion about the inputs like the number of given instances.
        number_instances_comparison=len(real_label)

        #Set up a counter to count the number of elements that are the same
        counter=0

        #Loop to compare the real labels and the prediciton. We increment the
        #value of the counter each time we classified properly.
        for comparison_index in range(number_instances_comparison):
            if prediction[comparison_index] == real_label[comparison_index]:
                counter=counter+1

        #Return the number of wrongly predicted instances by substracting the
        #total number of instances by the number of instances well classified.
        wrong_predictions=float(number_instances_comparison-counter)

        #return the number of wrong predictions
        return wrong_predictions
    ######################        END OF THE FUNCTION        ###################

    ####  FUNCTION TO ADD THE GAMMAS VALUE TO THE ADVERSARIAL PERCEPTRONS   ####
    #We use this function during the training process. Adding gammas to the
    #wrong perceptron makes more difficult to choose the right label and
    #eventually it will increase the marging that must be between the instance
    #and the perceptron to classify it with the label assigned to that perceptron.
    def add_gamma_values_adversarial_intance(self,real_label_index,\
    vector_degree_of_attribute_belonging_class, gamma ):
        #First get basic information about the inputs
        lenght_vector_belonging_class=\
        vector_degree_of_attribute_belonging_class.shape[1]

        #Second, generate an index for each element of the input vector
        index_input_vector=np.arange(lenght_vector_belonging_class)

        #Third, remove from the created index vector the index of the real label.
        #We do not want to add gamma to the belonging degree of the right class.
        indexes_to_update=np.delete(index_input_vector,real_label_index)

        #We add gamma to the instances we want.
        new_vector_degree_of_attribute_belonging_class=\
        vector_degree_of_attribute_belonging_class[0,indexes_to_update]+gamma

        #We reasamble the vector
        new_vector_degree_of_attribute_belonging_class=\
        np.insert(new_vector_degree_of_attribute_belonging_class,[real_label_index],\
        [vector_degree_of_attribute_belonging_class[0,real_label_index]])

        #return the vector with the new class belonging values
        return new_vector_degree_of_attribute_belonging_class
    ######################        END OF THE FUNCTION        ###################

    #########    FUNCTION TO TRAIN THE MULTIVARIABLE PERCEPTRONs    ############
    #Training model. The training process updates the weight values of each..
    #perceptron through stochastic gradient descent until we reach a certain
    #number of iterations or when all the train intances are predicted properly
    #(if the data is linearly separabble).The function returns a weight matrix..
    #with the weights of each perceptron that best classify each instance.
    def perceptron_learning_algorithm_training(self,attributes, labels,\
     alpha=0.1, iterations=1,gamma=1 ,error_threeshold=0.05):
        #Make sure the inputs are in the desired format.
        attributes=np.array(attributes)
        labels=np.array(labels)

        #Acquire basic information about the inputs.
        number_attributes=attributes.shape[1]
        number_instances=attributes.shape[0]

        #Obtain information about the labels such as the values that....
        #labels can get, their frequencies and number of different labels
        labels_values , frequency_labels=np.unique(labels, return_counts=True)
        number_different_labels=len(labels_values)

        #Following the theory behind the perceptron algorithm we have to add a
        #bias to the perceptron. To do so, we add a column of ones at the
        #beginning of the attribute matrix.
        bias_vector=np.ones(number_instances)

        #We create a new attribute matrix where the bias is included. To do so,
        #we concatenate the transpose bias vector to the attribute matrix.
        attribute_matrix=\
        np.concatenate((bias_vector.reshape((len(bias_vector),1)),\
        attributes), axis=1)

        #Store in a variable the new attribute matrix with bias.
        number_attributes_new_matrix=attribute_matrix.shape[1]

        #Third, we generate a matrix with the initial weights for each class
        #We use the function 'creation_random_weights_perceptron we created.
        weights_matrix=\
        self.creation_random_weights_perceptron(attribute_matrix,labels)

        #Create an initial variable that counts the number of  iterations.
        #we will use it as an emergency stopping criteria of the while loop in
        #case of not convergence. One iteration is when we scan an entire epoch.
        #We also, create an initial variable for the error threeshold.
        number_iterations=0
        error_while=0.99

        #while loop that stops when the number of misclassified variables is....
        #under a threeshold or when we reach a specific number of iterations.
        while (error_while>error_threeshold) and (number_iterations<iterations):
            #To find the weight values that classifies the best we use
            #stochastic gradient descent. With stochastic gradient descent we go
            #throught all the intances of the dataset randomly. We generate a
            #random ordex index of the instances.
            stochastic_instance_index\
            =random.sample(range(number_instances), number_instances)

            #We create a vector to store the classes we predict in each
            #iteration to then calculate the number of proper classifications.
            prediction_vector=np.zeros((number_instances),dtype=int)

            #We loop the instances of the attribute matrix as indicates in the
            #random list generated by the random process.
            for instance_index in stochastic_instance_index:
                #Select the intance vector we are going to classify.
                instance_of_analysis=attribute_matrix[instance_index,:]

                #Get the right class assigned to the instance.
                label_of_analysis=int(labels[instance_index])

                #Matrix multiplication of the weight matrix and the attribute
                #values of the instance we retrieved. We multiply a weight matrix
                #dimensions(number of different classes x number of attributes+1)
                #by a column vector of dimensions (number of attributes+1 x 1).
                #resulting in a column vector (number of different classes x 1)
                #that represents the value that each perceptron assigns to the
                #attribute values given by the instance.
                vector_degree_of_attribute_belonging_class=\
                np.dot(weights_matrix,np.transpose(instance_of_analysis))

                #Transform  the vector degree to a row vector
                vector_degree_of_attribute_belonging_class=\
                vector_degree_of_attribute_belonging_class.reshape\
                ((1,number_different_labels))

                #Add the gamma to those values in the class belonging vector
                #that do not represent the real labels.
                vector_degree_of_attribute_belonging_class=\
                self.add_gamma_values_adversarial_intance(label_of_analysis,\
                vector_degree_of_attribute_belonging_class, gamma)

                #Chosee the maximum value in the belongin class vector.
                #The index of the max value corresponds with the label.
                predicted_label=\
                np.argmax(vector_degree_of_attribute_belonging_class)

                #Update the perceptron weights: If the predicted label and the
                #original label are the same we do not update the weight matrix.
                #If the perceptron did not classify properly, we update 2
                #weight vectors, the perceptron that of the misclassified class
                #and the perceptron of the right class.
                if label_of_analysis != predicted_label:
                    weights_matrix=self.update_weights_process(weights_matrix,\
                    instance_of_analysis,alpha,label_of_analysis, predicted_label)

                #put the prediction into the vector that stores the predictions
                prediction_vector[instance_index]=predicted_label

            #Calculate the number of misclassified intances.
            error_while=float(\
            self.counter_misclassified_instances\
            (prediction_vector, labels)/number_instances)

            #Update the number of iterations of the while loop
            number_iterations=number_iterations+1

        #return the weight matrix that better classifies the instances.
        return weights_matrix
    ######################        END OF THE FUNCTION        ###################

    #######    FUNCTION TO PREDICT USING THE MULTIVARIABLE PERCEPTRONs    ######
    #This function uses the weight matrix resulting from the training process
    #to predict the label of each intance using multilabels perceptron.
    def perceptron_learning_algortihm_prediction(self,weights_matrix,\
     testing_attributes):
        #Make sure the input is in a form of a 2D array.
        testing_attributes=\
        np.array(testing_attributes).reshape((1,len(testing_attributes)))

        #First we get basic information about the testing data we will predict
        #such as the number of given instances and number of given attributes.
        number_intances_testing=testing_attributes.shape[0]
        number_attributes_testing=testing_attributes.shape[1]

        #We create the bias vector full of one values that we will add to the
        #attribute testing set to produce the predictions.
        bias_vector_testing=\
        np.ones(number_intances_testing).reshape((1,number_intances_testing))

        #We create a new attribute matrix where the bias is included. We........
        #concatenate the transpose bias vector to the original attribute matrix.
        attribute_matrix_testing=\
        np.concatenate((bias_vector_testing.reshape((number_intances_testing,1)),\
        testing_attributes), axis=1)

        #create a vector to store the predictions. The dimensions are the same
        #as the number of given instances.
        predicitons_vector=np.zeros((number_intances_testing), dtype=int)

        #Loop all the intances/rows given in the testing set.
        for intance_predict_index in range(number_intances_testing):
            #Get the attribute values of the intance we are going to predict.
            instance_prediciton_analysis=\
            attribute_matrix_testing[intance_predict_index,:]

            #Using the perceptron weights we acquired during the training part.
            #We Multiply the attribute vector by all the perceptron weights and
            #we identify the assign the the instance the class with highest
            #belonging degree.
            vector_degree_of_attribute_belonging_class_prediction=\
            np.dot(weights_matrix,np.transpose(instance_prediciton_analysis))

            #Choose the label with the highest belonging degree.
            classified_label=\
            np.argmax(vector_degree_of_attribute_belonging_class_prediction)

            #Store the prediction into a vector
            predicitons_vector[intance_predict_index]=classified_label

        #return a vector that contains the predicted classes
        return predicitons_vector
    ######################        END OF THE FUNCTION        ###################
    ##############   END OF MULTICLASS PERCEPTRON CLASSIFIER   #################
