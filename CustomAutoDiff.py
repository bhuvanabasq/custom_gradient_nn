import numpy as np
from numpy.core.fromnumeric import var
class Variable(object):
    def __init__(self, matrix):
        self.value = np.array(matrix,dtype=np.float64)
        if len(self.value.shape)==0:
            self.value = self.value.reshape([1,1])
        elif len(self.value.shape)!=2:
            raise Exception("Only 2D matrices or scalars supported.")
        
        self.fanout = 0
        self.gradient = 0
        
    
    def __add__(self,other):
        if not(isinstance(other, Variable)):
            other = Variable(other)
        register_operation(self,other)
        return MatrixAddition(self,other)
    
    def __truediv__(self,other):
        if not(isinstance(other, Variable)):
            other = Variable(other)
        register_operation(self,other)
        return MatrixDivision(self,other)
    
    def __mul__(self,other):
        if not(isinstance(other, Variable)):
            other = Variable(other)
        register_operation(self,other)
        return ElementwiseMultiplication(self,other)
    
    def __matmul__(self,other):
        if not(isinstance(other, Variable)):
            other = Variable(other)
        register_operation(self,other)
        return MatrixMultiplication(self,other)
    
    def exp(self):
        register_operation(self)
        return Exp(self)
    
    def log(self):
        register_operation(self)
        return Log(self)
    
    def sum(self,axis):
        register_operation(self)
        return Sum(self,axis=axis)

    def reset(self):
        self.gradient = 0
        self.fanout = 0

## HELPER FUNCTIONS
def propagate_gradients(*inputs):
    """
    This function checks if the variable is "ready" to backpropagate.
    """
    for variable in inputs:
        variable.fanout -= 1
        if variable.fanout == 0 and "backward" in dir(variable):
            variable.backward()

def register_operation(*inputs):
    """
    This function counts the number of times a variable is used.
    """
    for variable in inputs:
        variable.fanout += 1
        

def broadcast_gradients(gradient,variable):
    """
    In some cases, the variable gets broadcasted during an operation.
    Ex: In adding a [2,2] Matrix with [1,2] Vector, the Vector gets broadcasted.
    During backpropagation, we need to appropriately "broadcast" the gradients to the variable.
    Given the gradient and the variable, return the broadcasted version of the gradient.
    """
    if gradient.shape == variable.value.shape:
        return gradient
    else:
        axis = 0 if variable.value.shape[0] == 1 else 1
        return np.sum(gradient, axis=axis, keepdims=True)
    
class Log(Variable):
    """
    Usage: 
        v = Variable([[1,2,3]])
        log_v = v.log()
    """
    def __init__(self,v):
        super().__init__(np.log(v.value))
        self.v = v
    
    def backward(self):
        self.v.gradient += self.gradient/self.v.value
        
        propagate_gradients(self.v)
            
class Exp(Variable):
    """
    Usage: 
        v = Variable([[1,2,3]])
        exp_v = v.exp()
    """
    def __init__(self,v):
        super().__init__(np.exp(v.value))
        self.v = v
    
    def backward(self):
        self.v.gradient += self.gradient * np.exp(self.v.value)
        
        propagate_gradients(self.v)

class Sigmoid(Variable):
    """
    Usage: 
        v1 = Variable([[1,2],
                       [3,4]])
        v1_act = Sigmoid(v1)
    """ 
    def __init__(self,v):
        sigmoid = 1/(1 + np.exp(-v.value))
        super().__init__(sigmoid)
        self.v = v
        self.v.fanout += 1
        
    def backward(self):

        self.v.gradient += self.gradient* (1/(1 + np.exp(-self.v.value))*(1-(1/(1 + np.exp(-self.v.value)))))
        
        propagate_gradients(self.v)

class ReLU(Variable):
    """
    Usage: 
        v1 = Variable([[1,2],
                       [3,4]])
        v1_act = ReLU(v1)
    """     
    def __init__(self,v):
        relu = np.maximum(v.value, 0)
        super().__init__(relu)
        self.v = v
        self.v.fanout += 1
        
    def backward(self):
        b = np.copy(self.v.value)
        b[b<0] = 0
        b[b>0] = 1
        self.v.gradient += self.gradient* b
        
        propagate_gradients(self.v)

class Tanh(Variable):
    """
    Usage: 
        v1 = Variable([[1,2],
                       [3,4]])
        v1_act = Tanh(v1)
    """  
    def __init__(self,v):
        tanh = np.tanh(v.value)
        super().__init__(tanh)
        self.v = v
        self.v.fanout += 1
        
    def backward(self):
        self.v.gradient += self.gradient * (1- np.tanh(self.v.value)**2)
        
        propagate_gradients(self.v)
        
class MatrixMultiplication(Variable):
    """
    Usage: 
        v1 = Variable([[1,2],
                       [3,4]])
        v2 = Variable([[6],
                       [7]])
        v1_v2 = v1 @ v2
    """ 
    def __init__(self,v1,v2):
        super().__init__(np.matmul(v1.value, v2.value))
        self.v1 = v1
        self.v2 = v2
  
    def backward(self):
        self.v1.gradient += broadcast_gradients(self.gradient @ np.transpose(self.v2.value), self.v1)
        self.v2.gradient += broadcast_gradients(np.transpose(self.v1.value) @ self.gradient, self.v2)

        propagate_gradients(self.v1,self.v2)

class Sum(Variable):
    """
    Usage: 
        v = Variable([[1,2,3]])
        sum_v = v.sum(axis=1) 
    """
    def __init__(self,v,axis):
        super().__init__(np.sum(v.value,axis=axis,keepdims=True))
        self.v = v
        self.axis = axis
    
    def backward(self):
        self.v.gradient += self.gradient * np.ones_like(self.v.value)
        
        propagate_gradients(self.v)       

class MatrixAddition(Variable):
    """
    Usage: 
        v1 = Variable([[1,2],
                       [3,4]])
        v2 = Variable([[6],
                       [7]])
        v1_v2 = v1 + v2
    """ 
    def __init__(self,v1,v2):
        super().__init__(v1.value+v2.value)        
        
        self.v1 = v1
        self.v2 = v2
        
    def backward(self):
        # L X N
        self.v1.gradient += broadcast_gradients(self.gradient,self.v1)
        self.v2.gradient += broadcast_gradients(self.gradient,self.v2)

        propagate_gradients(self.v1,self.v2)

class ElementwiseMultiplication(Variable):
    """
    Usage: 
        v1 = Variable([[1,2,3]])
        v2 = Variable([[4,5,6]])
        v1_v2 = v1 * v2
    """
    def __init__(self,v1,v2):
        super().__init__(v1.value * v2.value)
        self.v1 = v1
        self.v2 = v2
  
    def backward(self):
        self.v1.gradient += self.gradient * self.v2.value
        self.v2.gradient += self.gradient * self.v1.value
        
        propagate_gradients(self.v1,self.v2)

class MatrixDivision(Variable):
    """
    Usage: 
        v1 = Variable([[1,2,3]])
        v2 = Variable([[4,5,6]])
        v1_v2 = v1 / v2
    """    
    def __init__(self,v1,v2):
        super().__init__(v1.value / v2.value)
        self.v1 = v1
        self.v2 = v2
  
    def backward(self):
        self.v1.gradient +=  broadcast_gradients(self.gradient * (1/self.v2.value), self.v1)
        self.v2.gradient +=  broadcast_gradients(-self.gradient * np.divide(self.v1.value, self.v2.value**2), self.v2)

        propagate_gradients(self.v1,self.v2)