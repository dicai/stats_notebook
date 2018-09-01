
using Random
using LinearAlgebra
using PyPlot
using Seaborn
#PyPlot.svg(true)

function iterate_CG(xk, rk, pk, A)
    """
    Basic iteration of the conjugate gradient algorithm
    Parameters:
    xk: current iterate
    rk: current residual
    pk: current direction
    A: matrix of interest
    """
    
    # form products
    rkk = rk' * rk
    Apk = A * pk 
    
    # construct step size
    αk = rkk / (pk' * Apk)

    # take a step in current conjugate direction
    xk_new = xk + αk * pk

    # construct new residual
    rk_new = rk + αk * Apk

    # construct new linear combination
    betak_new = (rk_new' * rk_new) / rkk

    # generate new conjugate vector
    pk_new = -rk_new + betak_new * pk

    return xk_new, rk_new, pk_new

end

function run_conjugate_gradient(x0, A, b; max_iter = 2000)
    """
    Conjugate gradient algorithm

    Parameters:
    x0: initial point
    A: matrix of interest
    b: vector in linear system (Ax = b)
    max_iter: max number of iterations to run CG
    """

    # initial iteration
    xk = x0
    rk = A * xk - b
    pk = -rk
    
    err = sum(abs.(rk))
    errors = [err]
    D = pk

    #while sum(abs.(rk)) > 10e-6
    for i in 1:max_iter
        xk, rk, pk = iterate_CG(xk, rk, pk, A)
        
        D = hcat(D,pk)
        
        err = sum(abs.(rk))
        push!(errors, err)
        
        # print iteration
        i % 500 == 0 && println("Iteration $i: error: $err")

        # break if we reach desired error 
        if err < 10e-6
            println("Terminated in $i iterations")
            break
        end
    end
    
    return xk, errors, D
end

Random.seed!(123)
b = rand(5)

A = zeros(5,5)
A[1,1]=10; A[2,2]=10.1; A[3,3]=10.2; A[4,4]=2; A[5,5] = 1

x = A \ b

x0 = [0.3,0.1,0.1,0.1,0.1]
xsol, errors, D = run_conjugate_gradient(x0, A, b);
xsol

Seaborn.semilogy(0:length(errors)-1, errors)
xlabel("iteration"); ylabel("total absolute error")

# confirm these are 0
@assert D[:,1]' * A * D[:,2] < 10e-6
@assert D[:,1]' * A * D[:,3] < 10e-6
@assert D[:,1]' * A * D[:,4] < 10e-6

# confirm these aren't 0
@assert D[:,1]' * A * D[:,1] < 10e-6

# Construct a random symmetric positive definite matrix
Random.seed!(111)
A = rand(5,5); A = 0.5*(A+A'); A = A + 5*I;

eigvals(A) 

x2 = A \ b

x0 = [0.,0.,0.,0.,0.]
xsol, errors, D = run_conjugate_gradient(x0, A, b);
xsol

Seaborn.semilogy(0:length(errors)-1, errors)
xlabel("iteration"); ylabel("total absolute error")

N = 1000
Random.seed!(123)
b = rand(N)
A = rand(N,N); A = 0.5*(A+A'); A = A + 50*I;
println("The condition number is: ", cond(A))

# distribution of the eigenvalues
plt[:hist](eigvals(A), bins=500); xlim(0,100)

x0 = zeros(N)
xsol, errors, D = run_conjugate_gradient(x0, A, b);
xsol;

Seaborn.semilogy(0:length(errors)-1, errors)
xlabel("iteration"); ylabel("total absolute error")

D[:,1]' * A * D[:,2]

D[:,1]' * A * D[:,10]

D[:,1]' * A * D[:,33]

Random.seed!(123)

# Generate random s.p.d. matrix
A = rand(N,N); A = 0.5*(A+A'); A = A + 13*I;
println("The condition number is: ", cond(A))

# distribution of the eigenvalues
plt[:hist](eigvals(A), bins=500); xlim(0,100)

x0 = zeros(N)
xsol, errors, D = run_conjugate_gradient(x0, A, b; max_iter=5000);

Seaborn.semilogy(0:length(errors)-1, errors)
xlabel("iteration"); ylabel("total absolute error")

D[:,1]' * A * D[:,2]

D[:,1]' * A * D[:,7]

D[:,1]' * A * D[:,75]
