# Practical Machine learning
# Deep learning - Autoencoder example 
# Chapter 11


autoencode <- function(X.train,X.test=NULL,nl=3,N.hidden,unit.type=c("logistic","tanh"),
                       lambda,beta,rho,epsilon,optim.method=c("BFGS","L-BFGS-B","CG"),
                       rel.tol=sqrt(.Machine$double.eps),max.iterations=2000,rescale.flag=c(F,T),rescaling.offset=0.001){
  # Autoencode function: trains a sparse autoencoder on a given dataset 
  
  # List of arguments:
  # X.train : a training set matrix (rows are the training examples, columns are the input channels)
  # X.test (optional) : a test set matrix
  # nl : number of layers (default nl=3); layer 1 is the input layer, layer nl is the output layer, the inner layers are the hidden layers (default - one hidden layer)
  # N.hidden[[]] : a list with numbers of hidden units in each hidden layer
  # unit.type : specifies the type of network units: "logistic" or "tanh"
  # lambda : weight decay parameter     
  # beta : weight of sparsity penalty term 
  # rho : desired sparsity parameter
  # epsilon : a (small) parameter for initialization of weight matrices as small gaussian random numbers sampled from N(0,epsilon^2)
  # optim.method : specifies the optimization method passed to optim() when searching for minimum of J(W,b)
  # rel.tol : relative convergence tolerance of the optim() optimizer used internally
  # max.iterations <- 2000 : maximum number of optimizer iterations
  # rescale (TRUE or FALSE) : a logical variable specifying whether to preprocess the training matrix to make sure the values of input channels are within the range of unit outputs ([0,1] for "logistic" units, [-1,1] for "tanh" units)
  # rescaling offset: a small value used in rescaling to [offset,1-offset] for "logistic" units, and to [-1+offset,1-offset] for "tanh" units
  
  # Output:
  # W,b : weights matrices and biases vectors of the trained autoencoder
  # unit.type : type of network units
  # rescale : a list containing logical flag indicating whether re-scaling back is required (should be just equal to the input value of rescale), and the minimum and maximum elements of the training matrix used for rescaling
  # nl : number of layers in the network
  # sl : list of numbers of units in each layer (i.e., network architecture)
  # N.input : number of input units
  # N.hidden : list of numbers of units in all hidden layers (for nl=3 we have one hidden layer)
  # mean.error.training.set : average, over all training matrix rows (training examples), sum of squares of (X.output-X.train)
  # mean.error.test.set : average, over all test matrix rows (test examples), sum of squares of (X.output-X.test)  

  if (is.matrix(X.train)) training.matrix <- X.train else stop("X.train must be a matrix!")
  Ntrain <- nrow(training.matrix)
  N.input <- ncol(training.matrix)  #this replaces Nx.patch*Ny.patch in the "old" autoencoder
  
  unit.type <- match.arg(unit.type)
  optimizer <- match.arg(optim.method)
  
  #   Setup the autoencoder's architecture:
  sl <- list()
  for (l in 1:nl){
    if (l==1 || l==nl) sl[[l]] <- N.input else sl[[l]] <- N.hidden[[l-1]]
  }
  W <- list()  #create an empty list of weight matrices
  b <- list()  #create an empty list of biases
  
  ### DEFINE INNER FUNCTIONS: ###
  
  activation <- function(z){#activation (output) of neurons
    if (unit.type=="logistic") return(1/(1+exp(-z)))
    if (unit.type=="tanh") return(tanh(z))
  }
  
  activation.prime <- function(z){# f'(z)
    if (unit.type=="logistic") {
      act <- activation(z)
      return(act*(1-act))  #f0or logistic f(z), f'(z)=f(z)*(1-f(z))
    }
    if (unit.type=="tanh") return(2/(1+cosh(2*z)))  #tanh'[z]=sech[z]^2=2/(1+cosh(2*z))
  }
  
  feedforward.pass.matrix <- function(W,b,X){#performs a feedforward pass in the network, given input matrix X
    #1. Perform a feedforward pass, computing the activations for layers L2,
    #L3, and so on up to the output layer Lnl, using Eqs (6)-(7):
    NrowX = nrow(X)
    z <- list()
    a <- list()
    a[[1]] <- X  #the first element of the activations list is the input matrix, with rows i=1,...,Ntrain counting the training examples, and columns j=1,...,sl[1] counting the input channels.
    for (l in 2:nl){
      #    z[[l]] <- t(W[[l-1]] %*% t(a[[l-1]]) + matrix(b[[l-1]],nrow=length(b[[l-1]]),ncol=Ntrain))  #matrix z[[l]][k,i], rows k=1,...,Ntrain, columns i=1,...,sl[l+1]
      z[[l]] <- t(W[[l-1]] %*% t(a[[l-1]])) + matrix(b[[l-1]],nrow=NrowX,ncol=length(b[[l-1]]),byrow=TRUE)  #matrix z[[l]][k,i], rows k=1,...,Ntrain, columns i=1,...,sl[l+1]
      a[[l]] <- activation(z[[l]])  #matrix a[[l]][k,i], rows k=1,...,Ntrain, columns i=1,...,sl[l+1]
    }
    
    return(list("z"=z,"a"=a))
  }
  
  J.example.matrix <- function(W,b,X,Y){
    #J(W,b,x,y) = 0.5*||h(W,b;X)-Y||^2 = 0.5*sum((h[k,i]-y[k,i])^2, i=1:sl[3]), 
    #where h(W,b;X) is the activation matrix of the nl-th layer, i.e., h(W,b;X)=a[[nl]]
    
    #Perform a feedforward pass, computing the activations for layers L2,
    #L3, and so on up to the output layer Lnl, using Eqs (6)-(7):
    a <- feedforward.pass.matrix(W,b,X)$a #matrices a[[l]][k,i], k=1,...,Ntrain, i=1,...,sl[l]
    
    return(0.5*rowSums((a[[nl]] - Y)^2))
  }
  
  J.matrix <- function(W,b){
    #function J(W,b) defined by Eq.(8) of Andrew Ng's notes.
    #W is a list of matrices, so W_{ij}^{(l)} is addressed as W[[l]][i,j]
    #b is a list of vectors, so b_i^{(l)} is addressed as b[[l]][i]
    rho.hat <- find.rho.hat.matrix(W,b)
    X <- training.matrix
    term1 <- mean(J.example.matrix(W,b,X,X))
    
    term2 <- 0
    for (l in 1:(nl-1)){
      term2 <- term2 + sum(W[[l]]^2)  #a significant speedup compared to two nested loops over i,j
    }
    term2 <- term2*lambda/2
    
    term3 <- beta*sum(rho*log(rho/rho.hat) + (1-rho)*log((1-rho)/(1-rho.hat)))  #penalty for rho.hat (average, over the training set, activation of j-th hidden unit) deviating from the sparsity parameter rho
    
    return(term1+term2+term3)
  }
  
  J.theta <- function(theta){#evaluate J(theta)
    tmp <- theta.to.W.b(theta)
    W.local <- tmp[[1]]
    b.local <- tmp[[2]]
    return(J.matrix(W.local,b.local))
  }
  
  grad.J.backpropagation.matrix <- function(W,b){#"analytical" evaluation of grad.W.J and grad.b.J using backpropagation algorithm
    #Evaluate grad.W.J and grad.b.J using backpropagation algorithm:
    
    rho.hat <- find.rho.hat.matrix(W,b)  #can't just use the old rho.hat from J(W,b) here, as W,b might have changed since the last call of J(W,b)
    tmp <- backpropagation.matrix(W,b,rho.hat,training.matrix)
    grad.W.J <- tmp$grad.W.J
    grad.b.J <- tmp$grad.b.J
    
    return(list("grad.W.J"=grad.W.J,"grad.b.J"=grad.b.J))
  }
  
  grad.theta.J <- function(theta){#evaluate grad.theta.J
    tmp <- theta.to.W.b(theta)
    W.local <- tmp[[1]]
    b.local <- tmp[[2]]
    tmp <- grad.J.backpropagation.matrix(W.local,b.local)
    grad.W.J <- tmp$grad.W.J
    grad.b.J <- tmp$grad.b.J
    result <- W.b.to.theta(grad.W.J,grad.b.J)
    return(result)
  }
  
  backpropagation.matrix <- function(W,b,rho.hat,X){#{X} is the training matrix containing ALL training examples as rows!
    #   This function is MEMORY-HUNGRY and is SLOWER than backpropagation()!
    #Backpropagation algorithm:
    #1. Perform a feedforward pass, computing the activations for layers L2,
    #L3, and so on up to the output layer Lnl, using Eqs (6)-(7):
    NrowX = nrow(X)
    tmp <- feedforward.pass.matrix(W,b,X)
    a <- tmp$a
    z <- tmp$z
    #2. For the output layer (layer nl), set
    delta <- list()
    delta[[nl]] <- -(X-a[[nl]]) * activation.prime(z[[nl]])  #recall that Y(output)=X(input) for autoencoder
    #3. For l=nl-1,nl-2,nl-3,...,2, set
    for (l in (nl-1):2){
      delta[[l]] <- (delta[[l+1]] %*% W[[l]] + matrix(beta*(-rho/rho.hat + (1-rho)/(1-rho.hat)),nrow=NrowX,ncol=sl[[l]],byrow=TRUE))*activation.prime(z[[l]])
    }
    #4. Compute the desired partial derivatives (summed up over training examples):
    
    grad.W.J <- list()
    grad.b.J <- list()
    for (l in 1:(nl-1)){
      grad.W.J[[l]] <- (t(delta[[l+1]]) %*% a[[l]])/NrowX + lambda*W[[l]]  #the summation over all training examples is done here!
      grad.b.J[[l]] <- colMeans(delta[[l+1]])      #the averaging over all training examples is done here!
    }
    
    return(list("grad.W.J"=grad.W.J,"grad.b.J"=grad.b.J))
  }
  
  
  W.b.to.theta <- function(W,b){#Convert W,b into a single long vector theta
    #pack W[[l]] and b[[l]] into theta, for all l:
    theta <- vector(mode="numeric", length=0)
    for (l in 1:(nl-1)){
      theta <- append(theta,as.vector(W[[l]]))
      theta <- append(theta,b[[l]])
    }
    return(theta)
  }
  
  theta.to.W.b <- function(theta){#Convert a vector theta into W,b format
    W.local <- list()
    b.local <- list()
    snip.start <- 1
    for (l in 1:(nl-1)){
      #snip the W[[l]] part of theta:
      snip.end <- snip.start-1 + sl[[l]]*sl[[l+1]]  #end index of snip from theta
      snip <- theta[snip.start:snip.end]
      W.local[[l]] <- matrix(snip,nrow=sl[[l+1]],ncol=sl[[l]])
      #snip the b[[l]] part of theta:
      snip.start <- snip.end+1
      snip.end <- snip.start-1+sl[[l+1]]
      snip <- theta[snip.start:snip.end]
      b.local[[l]] <- snip
      snip.start <- snip.end+1 #proceed to next l
    }
    return(list(W.local,b.local))
  }
  
  find.rho.hat.matrix <- function(W,b){
    #Do feedforward passes on all training examples to determine the average activation of 
    #hidden units in layer l=2 (j=1,..,sl[2]) (averaged over the training set), rho.hat (for the current W,b):
    a.2 <- feedforward.pass.matrix(W,b,training.matrix)$a[[2]]
    if (unit.type=="logistic") rho.hat <- colMeans(a.2)  #"logistic" neurons: "active" if output is close to 1, "inactive" if output is close to 0
    if (unit.type=="tanh") rho.hat <- colMeans(a.2+1)/2  #"tanh" neurons: "active" if output is close to 1, "inactive" if output is close to -1
    
    return(rho.hat)
  }
  
  rescale <- function(X.in,X.in.min=NULL,X.in.max=NULL,unit.type,offset){ #offset is a small number to avoid having exactly 0 or 1, e.g., offset=0.01
    if (!is.numeric(X.in.min)) X.in.min <- min(X.in)  #minimum of all elements of X.in
    if (!is.numeric(X.in.max)) X.in.max <- max(X.in)  #maximum of all elements of X.in
    if (unit.type=="logistic"){#rescale X.in in a uniform way, so that all input channels are within (0,1)=[offset,1-offset]
      #shift to [0,...]:
      X.in <- X.in - X.in.min
      #rescale uniformly to fit all rows within [0,1]:
      X.in <- X.in/(X.in.max-X.in.min)
      #rescale uniformly to fit all rows withing [offset,1-offset]:
      L <- 1-2*offset
      X.in <- X.in*L + offset
    }
    if (unit.type=="tanh"){#rescale X.in so that all input channels are within (-1,1)=[-1+offset,1-offset]
      #shift to [0,...]:
      X.in <- X.in - X.in.min
      #rescale uniformly to fit all rows within [0,2]:
      X.in <- X.in/(X.in.max-X.in.min)*2
      #rescale uniformly to fit all rows within [offset,2-offset], and then shift to [-1+offset,1-offset]:
      L <- 1-offset
      X.in <- X.in*L + offset - 1
    }
    
    #Return rescaled X.in, X.in.min, X.in.max, and unit.type, for rescaling back
    return(list("X.rescaled"=X.in,"X.min"=X.in.min,"X.max"=X.in.max,"unit.type"=unit.type,"offset"=offset))
  }
  
  rescale.back <- function(X.in,X.in.min,X.in.max,unit.type,offset){#revert the scaling done by rescale()
    if (unit.type=="logistic"){
      X.in <- (X.in-offset)/(1-2*offset)*(X.in.max-X.in.min) + X.in.min
    }
    if (unit.type=="tanh"){
      X.in <- (X.in-offset+1)/(1-offset)*(X.in.max-X.in.min)/2 + X.in.min
    }
    #Return rescaled back X.in
    return(list("X.rescaled"=X.in))
  }
  
  ### END of DEFINING INNER FUNCTIONS ###
  
  cat("autoencoding...\n")
  
  #Rescale the input matrix if rescale=TRUE:
  if (rescale.flag){
    tmp <- rescale(X.in=training.matrix,unit.type=unit.type,offset=rescaling.offset)
    training.matrix <- tmp$X.rescaled
    training.matrix.min <- tmp$X.min
    training.matrix.max <- tmp$X.max
  } else {
    training.matrix.min <- training.matrix.max <- NULL
  }
  
  #Initialize the weight matrices W_{ij}^{(l)} (weights between unit j in level l and unit i in level l+1)
  #and the bias vectors b_i^{(l)} (bias associated with unit i in the (l+1) layer)
  for (l in 1:(nl-1)){
    nj <- sl[[l]]  #number of units j in level l
    ni <- sl[[l+1]] #number of units i in level l+1
    W[[l]] <- matrix(data=rnorm(ni*nj,0,epsilon), nrow=ni, ncol=nj)
    b[[l]] <- vector(mode="numeric", length=sl[[l+1]])  #initialize biases as zeros (b_i^{(l)} is the bias associated with unit i in layer l+1)
  }
  
  ### START LEARNING ###
  
  if (optimizer=="BFGS" || optimizer=="L-BFGS-B" || optimizer=="CG"){
    W.init <- W
    b.init <- b
    J.init <- J.matrix(W,b)
    theta <- W.b.to.theta(W,b)
    optimized <- optim(par=theta,fn=J.theta,gr=grad.theta.J,method=optimizer,control=list(maxit=max.iterations,reltol=rel.tol))
    J.final <- optimized$value
    
    #W,b of the found minimum of J:
    tmp <- theta.to.W.b(optimized$par)
    W <- tmp[[1]]
    b <- tmp[[2]]
    
    cat("Optimizer counts:\n",sep="")
    print(optimized$counts)
    if (optimized$convergence == 0){
      cat("Optimizer: successful convergence.\n")
    }
    if (optimized$convergence == 1){
      cat("Optimizer: iterations limit max.iterations =",max.iterations,"has been reached without convergence. Consider increasing the iteration limit.\n")
    }
    cat("Optimizer: convergence = ",optimized$convergence,", message = ",optimized$message,"\n",sep="")
    cat("J.init = ",J.init,", J.final = ",J.final,", mean(rho.hat.final) = ",mean(find.rho.hat.matrix(W,b)),"\n",sep="")
  }
  
  #Calculate final training and test (if included) set errors:
  #Training set error = error between outputs and inputs:
  #N.B.: training set error must be measured on the SAME training matrix (rescaled, if rescale.flag=T, or not rescaled, if rescale.flag=F) 
  #that was used for training the network!
  X.output <- feedforward.pass.matrix(W,b,training.matrix)$a[[nl]]  #calculate output matrix (rows correspond to outputs calculated from training examples)
  training.set.error <- mean(rowSums((X.output - training.matrix)^2))  #average, over training examples, sum of squares of (X.output-training.matrix)
  
  #Test set error:
  if (is.matrix(X.test)){
    # If the network was trained on a rescaled matrix, we should feed rescaled X.test (with X.min and X.max from the TRAINING matrix!) into it to obtain X.output,
    # then rescale X.output back, and compare it to X.test
    if (rescale.flag) {
      X.test.rescaled <- rescale(X.in=X.test,X.in.min=training.matrix.min,X.in.max=training.matrix.max,
                     unit.type=unit.type,offset=rescaling.offset)$X.rescaled
      X.output <- feedforward.pass.matrix(W,b,X.test.rescaled)$a[[nl]]
      X.output <- rescale.back(X.in=X.output,X.in.min=training.matrix.min,X.in.max=training.matrix.max,unit.type,offset=rescaling.offset)$X.rescaled
    } else {
      X.output <- feedforward.pass.matrix(W,b,X.test)$a[[nl]]
    }
      test.set.error <- mean(rowSums((X.output - X.test)^2))    #average, over X.test rows, sum of squares of (X.output-X.test)
  } else test.set.error <- NULL
  
  
  # Output W,b,unit.type,network architecture,rescaling parameters:
  result <- list("W"=W,"b"=b,"unit.type"=unit.type,"rescaling"=list("rescale.flag"=rescale.flag,"rescaling.min"=training.matrix.min,"rescaling.max"=training.matrix.max,"rescaling.offset"=rescaling.offset),"nl"=nl,"sl"=sl,"N.input"=N.input,"N.hidden"=N.hidden,
                 "mean.error.training.set"=training.set.error,"mean.error.test.set"=test.set.error)
  class(result) <- "autoencoder"
  return(result)
}

predict.autoencoder <- function(object,X.input=NULL,hidden.output=c(F,T),...){
  # Predict function: takes in the parameters of a trained autoencoder and an input matrix (rows=input examples, columns=input channels), 
  # and predicts the corresponding output matrix.
  
  # Input parameters:
  # object : an object of class 'autoencoder' containing a list of a trained autoencoder parameters (result of autoencode() function)
  # X.input : matrix of new data for which predictions should be evaluated (rows = input examples, columns = input channels)
  # hidden.output : a flag telling whether to output hidden layer output matrix instead of the total output matrix (needed for stacked autoencoders)
  # hidden.output = FALSE : X.output from the output layer (rescaled if necessary)
  # hidden.output = TRUE : X.output from the hidden layer (not rescaled)
  
  # Output:
  # X.output : output matrix of the same dimensionality as X.input (rows = output examples, columns = output channels)
  # hidden.output: a logical flag indicating whether to output the "output layer's" (hidden.output=F) 
  # or the "hidden layer's" (hidden.output=T) outputs matrix (rows correspond to outputs calculated from input examples)
  # mean.error : average, over rows of X.output, sum of squares of (X.output - X.input)
  
  ### DEFINE INNER FUNCTIONS: ###
  
  activation <- function(z){#activation (output) of neurons
    if (unit.type=="logistic") return(1/(1+exp(-z)))
    if (unit.type=="tanh") return(tanh(z))
  }
  
  feedforward.pass.matrix <- function(W,b,X){#performs a feedforward pass in the network, given input matrix X
    #1. Perform a feedforward pass, computing the activations for layers L2,
    #L3, and so on up to the output layer Lnl, using Eqs (6)-(7):
    NrowX = nrow(X)
    z <- list()
    a <- list()
    a[[1]] <- X  #the first element of the activations list is the input matrix, with rows i=1,...,Ntrain counting the training examples, and columns j=1,...,sl[1] counting the input channels.
    for (l in 2:nl){
      #    z[[l]] <- t(W[[l-1]] %*% t(a[[l-1]]) + matrix(b[[l-1]],nrow=length(b[[l-1]]),ncol=Ntrain))  #matrix z[[l]][k,i], rows k=1,...,Ntrain, columns i=1,...,sl[l+1]
      z[[l]] <- t(W[[l-1]] %*% t(a[[l-1]])) + matrix(b[[l-1]],nrow=NrowX,ncol=length(b[[l-1]]),byrow=TRUE)  #matrix z[[l]][k,i], rows k=1,...,Ntrain, columns i=1,...,sl[l+1]
      a[[l]] <- activation(z[[l]])  #matrix a[[l]][k,i], rows k=1,...,Ntrain, columns i=1,...,sl[l+1]
    }
    
    return(list("z"=z,"a"=a))
  }
  
  rescale <- function(X.in,X.in.min=NULL,X.in.max=NULL,unit.type,offset){ #offset is a small number to avoid having exactly 0 or 1, e.g., offset=0.01
    if (!is.numeric(X.in.min)) X.in.min <- min(X.in)  #minimum of all elements of X.in
    if (!is.numeric(X.in.max)) X.in.max <- max(X.in)  #maximum of all elements of X.in
    if (unit.type=="logistic"){#rescale X.in in a uniform way, so that all input channels are within (0,1)=[offset,1-offset]
      #shift to [0,...]:
      X.in <- X.in - X.in.min
      #rescale uniformly to fit all rows within [0,1]:
      X.in <- X.in/(X.in.max-X.in.min)
      #rescale uniformly to fit all rows withing [offset,1-offset]:
      L <- 1-2*offset
      X.in <- X.in*L + offset
    }
    if (unit.type=="tanh"){#rescale X.in so that all input channels are within (-1,1)=[-1+offset,1-offset]
      #shift to [0,...]:
      X.in <- X.in - X.in.min
      #rescale uniformly to fit all rows within [0,2]:
      X.in <- X.in/(X.in.max-X.in.min)*2
      #rescale uniformly to fit all rows within [offset,2-offset], and then shift to [-1+offset,1-offset]:
      L <- 1-offset
      X.in <- X.in*L + offset - 1
    }
    
    #Return rescaled X.in, X.in.min, X.in.max, and unit.type, for rescaling back
    return(list("X.rescaled"=X.in,"X.min"=X.in.min,"X.max"=X.in.max,"unit.type"=unit.type,"offset"=offset))
  }
  
  rescale.back <- function(X.in,X.in.min,X.in.max,unit.type,offset){#revert the scaling done by rescale()
    if (unit.type=="logistic"){
      X.in <- (X.in-offset)/(1-2*offset)*(X.in.max-X.in.min) + X.in.min
    }
    if (unit.type=="tanh"){
      X.in <- (X.in-offset+1)/(1-offset)*(X.in.max-X.in.min)/2 + X.in.min
    }
    #Return rescaled back X.in
    return(list("X.rescaled"=X.in))
  }
  
  ### END of DEFINING INNER FUNCTIONS ###
  
  # Extract autoencoder parameters from the input:
  if (class(object)=="autoencoder"){
    W <- object$W
    b <- object$b
    nl <- object$nl
    sl <- object$sl
    N.hidden <- object$N.hidden
    unit.type <- object$unit.type
    rescale.flag <- object$rescaling$rescale.flag
    rescaling.min <- object$rescaling$rescaling.min
    rescaling.max <- object$rescaling$rescaling.max
    rescaling.offset <- object$rescaling$rescaling.offset
  } else stop("predict.autoencoder: the object class is '",class(object),"', must be 'autoencoder'.")
  
  if(!is.numeric(X.input)) stop("No data to predict for is supplied, stopping.")
  
  if (hidden.output==FALSE) {
    # rescale X.input using rescaling.min and rescaling.max used with the trained network 
    # (so that the records in X.input identical to those in training.matrix used to train the autoencoder, remain identical after rescaling to the corresponding ones in the rescaled training.matrix):
    if (rescale.flag){
      X.input.rescaled <- rescale(X.in=X.input,X.in.min=rescaling.min,X.in.max=rescaling.max,unit.type=unit.type,offset=rescaling.offset)$X.rescaled
      #Calculate X.output by feedforward pass on the network:
      X.output <- feedforward.pass.matrix(W,b,X.input.rescaled)$a[[nl]]  #calculate output matrix (rows correspond to outputs calculated from input examples)
      # rescale X.output back:
      X.output <- rescale.back(X.in=X.output,X.in.min=rescaling.min,X.in.max=rescaling.max,unit.type=unit.type,offset=rescaling.offset)$X.rescaled
    } else {
      #Calculate X.output by feedforward pass on the network:
      X.output <- feedforward.pass.matrix(W,b,X.input)$a[[nl]]  #calculate output matrix (rows correspond to outputs calculated from input examples)
    }
    mean.error <- mean(rowSums((X.output - X.input)^2)) 
    
  }
  if (hidden.output==TRUE) {
    # rescale X.input using rescaling.min and rescaling.max used with the trained network 
    # (so that the records in X.input identical to those in training.matrix used to train the autoencoder, remain identical after rescaling to the corresponding ones in the rescaled training.matrix):
    if (rescale.flag){
      X.input.rescaled <- rescale(X.in=X.input,X.in.min=rescaling.min,X.in.max=rescaling.max,unit.type=unit.type,offset=rescaling.offset)$X.rescaled
      X.output <- feedforward.pass.matrix(W,b,X.input.rescaled)$a[[nl-1]]  #calculate hidden layer output matrix (rows correspond to outputs calculated from rescaled input examples)
    } else X.output <- feedforward.pass.matrix(W,b,X.input)$a[[nl-1]]  #calculate hidden layer output matrix (rows correspond to outputs calculated from input examples)
    mean.error <- NULL
  }
  
  # Output X.output,hidden.output,mean.error:
  return(list("X.output"=X.output,"hidden.output"=hidden.output,"mean.error"=mean.error))
}


visualize.hidden.units <- function(object,Nx.patch,Ny.patch){
  #Visualize hidden units of the sparse autoencoder
  ### VISUALIZATION OF MAXIMUM ACTIVATION WEIGHTS FOR HIDDEN UNITS: ###
  # Extract autoencoder parameters from the input:
  if (class(object)=="autoencoder"){
    W <- object$W
    nl <- object$nl
    sl <- object$sl
  } else stop("predict.autoencoder: the object class is '",class(object),"', must be 'autoencoder'.")
  
  pixels <- list()
  op <- par(no.readonly = TRUE) # the whole list of settable par's.
  par(mfrow=c(floor(sqrt(sl[[nl-1]])),floor(sqrt(sl[[nl-1]]))), mar=c(0.2,0.2,0.2,0.2), oma=c(3,3,3,3))
  
  for (i in 1:sl[[nl-1]]){#loop over all hidden units
    denom <- sqrt(sum(W[[nl-2]][i,]^2))
    pixels[[i]] <- matrix(W[[nl-2]][i,]/denom,nrow=Ny.patch,ncol=Nx.patch)
    image(pixels[[i]],axes=F,col=gray((0:32)/32))
  } 
  par(op)  #restore plotting parameters
}