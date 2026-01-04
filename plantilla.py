class NombreClase:
    """
    Breve descripción de la clase.

    Explicación más detallada de qué representa la clase,
    qué problema resuelve y cómo se usa en el sistema.

    Attributes
    ----------
    atributo_1 : tipo
        Descripción del atributo.
    atributo_2 : tipo
        Descripción del atributo.
    """

    def compute_cost(x, y, w, b): 
    """
    Computes the cost function for linear regression.
    
    Args:
        x (ndarray): Shape (m,) Input to the model (Population of cities) 
        y (ndarray): Shape (m,) Label (Actual profits for the cities)
        w, b (scalar): Parameters of the model
    
    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    # number of training examples
    m = x.shape[0] 
    
    # You need to return this variable correctly
    total_cost = 0
    
    ### START CODE HERE ###
    cost_sum = 0
    for i in range(m):
        f_wb = w * x[i] + b
        
        cost = (f_wb-y[i])**2
        
        cost_sum = cost_sum + cost 
    total_cost = (1 / (2 * m)) * cost_sum
    ### END CODE HERE ### 

    return total_cost