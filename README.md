# Machine Learning API
Deep Learning models and Machine Learning functions.

# Architecture

## Modules

**ml**

Machine Learning

    - Supervised Learning

    - Cost functions

    - Regularization

    - Unsupervised learning

**dl**

Deep Learning

    - Neural Networks

    - Layers

        - Convolutions

        - Dense

        - Max Pool

        - Flatten

        - Dropout

# Machine Learning functions

**root of M.L. functions: /ml/**

## Models

**models/**

create

    Creates a new Machine Learning model.

    POST REQUEST
        
        JSON data:

        "name": string. Required.
            Max length: 50 characters. Min length: 3 characters.
        
        "algorithm": string. Required.
            "regression" or "classification".
        
        "cost_function": string. Required if algorithm is supervised learning.
            if "algorithm" is "regression":
                "mean_squared_error".
            if it's "classification":
                "binary_crossentropy", "sparse_categorical_crossentropy".
        
        "epochs": int. Required.
            Steps/iterations.
        
        "lr": float. Optional.
            Learning rate.
            Defaults to 0.001
    
    RESPONSE
    
        JSON body:
        
        
