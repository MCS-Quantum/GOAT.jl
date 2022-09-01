using QuantumOptics, FFTW, Random


function make_fock_projector(basis,levels)
    P = Matrix(identityoperator(basis).data)
    lmul!(0,P)
    for level in levels
        P += Matrix(tensor(fockstate(basis,level),dagger(fockstate(basis,level))).data)
    end
    return P
end



function make_operator_basis(basis, c, ket_level, bra_level)
    return Matrix(tensor(fockstate(basis,ket_level),dagger(fockstate(basis,bra_level))).data)
end



"""
Returns the direct sum of two square matrices A, B of the same type.
"""
function direct_sum(A,B)
    T = typeof(A[1,1])
    dimA = size(A)
    dimB = size(B)
    C = zeros(T,(dimA[1]+dimB[1],dimA[2]+dimB[2]))
    C[1:dimA[1],1:dimA[2]] = A
    C[dimA[1]+1:end,dimA[2]+1:end] = B
    return C
end


"""
Returns the sparse direct sum of two square matrices A, B of the same type.
"""
function sparse_direct_sum(A,B)
    T = typeof(A[1,1])
    dimA = size(A)
    dimB = size(B)
    C = spzeros(T,dimA[1]+dimB[1],dimA[2]+dimB[2])
    C[1:dimA[1],1:dimA[2]] = A
    C[dimA[1]+1:end,dimA[2]+1:end] = B
    return C
end

function isunitary(A,dim;tol=1e-9)
    return abs(tr(adjoint(A)*A)-dim) <tol && abs(tr(A*adjoint(A)) - dim) <tol
end

function embed_square_matrix(U,M,U_index,M_index)
    for (i,i_) in zip(U_index,M_index)
        for (j,j_) in zip(U_index,M_index)
            M[i_,j_] = U[i,j]
        end
    end
    return M
end


"""
Operates in-place on a matrix X by a complex Givens operator U on the right: X=XU, specified by four parameters: j, k, θ, β.
"""
function givens_rmul!(X,j,k,θ,β)
    a = cos(θ)
    q = sin(θ)
    b = -q*cis(-β)
    c = q*cis(β)
    
    n = size(X,1)
    for i in 1:n
        x1 = X[i,j]
        x2 = X[i,k]
        X[i,j] = x1*a + x2*c
        X[i,k] = x1*b + x2*a
    end
end

"""
Generates an SU(n) unitary matrix via Givens rotations based on the parameters: αs,θs,βs. The function operates in-place on the input X.
"""
function Givens_SUn!(X,αs,θs,βs)
    n = size(X,1)
    @assert length(αs) == n-1
    @assert length(θs) == n*(n-1)/2
    @assert length(βs) == n*(n-1)/2
    
    lmul!(0.0,X)
    
    αsum = sum(αs)
    X[n,n] = cis(-αsum)
    for i in 1:n-1
        X[i,i] = cis(αs[i])
    end
    q = 1
    for k in 2:n
        for j in 1:k-1
            givens_rmul!(X,j,k,θs[q],βs[q])
            q+=1
        end
    end
end

"""
Generates a unitary matrix on SU(n)⊗SU(n) via Givens rotations based on the parameters: αs,θs,βs. The function operates in-place on the input bigX by doing the in-place Kronecker product. 
"""
function SUnSUn!(bigX, smallX, smallY, αs, θs, βs)
    lmul!(0.0,bigX)
    small_n = size(smallX, 1)
    αind = Int((small_n-1))
    θind = Int(small_n*(small_n-1)/2)
    βind = Int(small_n*(small_n-1)/2)
    
    big_n = size(bigX, 1)
    @assert big_n == small_n^2
    @assert 2*αind == 2*(small_n-1)
    @assert 2*θind == small_n*(small_n-1)
    @assert 2*βind == small_n*(small_n-1)
    
    first_αs = @view αs[1:αind]
    first_θs = @view θs[1:θind]
    first_βs = @view βs[1:βind]
    
    second_αs = @view αs[αind+1:end]
    second_θs = @view θs[θind+1:end]
    second_βs = @view βs[βind+1:end]
    
    # Apply the rotations for the first small operator
    Givens_SUn!(smallX,first_αs,first_θs,first_βs)
    # Apply the rotations for the second small operator
    Givens_SUn!(smallY,second_αs,second_θs,second_βs)
    
    @inbounds kron!(bigX, smallX, smallY)
    return nothing
end

"""
Embed the square matrix A into the larger matrix B at B_indices. Setting B to the identity elsewhere. 

Specifically, if i* = B_indices[i] and j* = B_indinces[j], then B[i*,j*] = A[i,j]. 
"""
function embed_A_into_B!(A,B,B_indices) 
    lmul!(0.0,B)
    d = size(B,1)
    for i in 1:d
        B[i,i] = 1.0
    end
            
    for (i,i_) in enumerate(B_indices)
        for (j,j_) in enumerate(B_indices)
            B[i_,j_] = A[i,j]
        end
    end
end

function create_initial_vector_U_∂U(num_nonzero; block_inds=nothing, linear_u_index_from_pair=nothing)
    @assert block_inds !== nothing
    @assert linear_u_index_from_pair !== nothing
    
    u0 = zeros(ComplexF64,num_nonzero*2)
    for block in block_inds
        for row in block
            i = linear_u_index_from_pair[row,row,1]
            u0[i] = 1
        end
    end
    return u0
end

function create_initial_vector_U(num_nonzero; block_inds=nothing, linear_u_index_from_pair=nothing)
    @assert block_inds !== nothing
    @assert linear_u_index_from_pair !== nothing
    u0 = zeros(ComplexF64,num_nonzero)
    for block in block_inds
        for row in block
            i = linear_u_index_from_pair[row,row,1]
            u0[i] = 1
        end
    end
    return u0
end

function unpack_u_∂u(u; block_inds=nothing, linear_u_index_from_pair=nothing)
    @assert block_inds !== nothing
    @assert linear_u_index_from_pair !== nothing
    row_inds = Int[]
    col_inds = Int[]
    vals_u = ComplexF64[]
    vals_∂u = ComplexF64[]
    
    for block in block_inds
        for row in block
            for col in block
                push!(row_inds,row)
                push!(col_inds,col)
                push!(vals_u,u[linear_u_index_from_pair[row,col, 1]])
                push!(vals_∂u,u[linear_u_index_from_pair[row,col, 2]])
            end
        end
    end
    u = sparse(row_inds,col_inds,vals_u)
    ∂u = sparse(row_inds,col_inds,vals_∂u)
    return u,∂u
end

function unpack_u(u; block_inds=nothing, linear_u_index_from_pair=nothing)
    @assert block_inds !== nothing
    @assert linear_u_index_from_pair !== nothing
    row_inds = Int[]
    col_inds = Int[]
    vals_u = ComplexF64[]
    
    for block in block_inds
        for row in block
            for col in block
                push!(row_inds,row)
                push!(col_inds,col)
                push!(vals_u,u[linear_u_index_from_pair[row,col, 1]])
            end
        end
    end
    u = sparse(row_inds,col_inds,vals_u)
    return u
end

"""
    unpack_us_∂us(us; block_inds=nothing, linear_u_index_from_pair=nothing)

TBW
"""
function unpack_us_∂us(us; block_inds=nothing, linear_u_index_from_pair=nothing)
    @assert block_inds !== nothing
    @assert linear_u_index_from_pair !== nothing
    row_inds = Int[]
    col_inds = Int[]
    vals_us = [ComplexF64[] for u in us]
    vals_∂us = [ComplexF64[] for u in us]
    
    for block in block_inds
        for row in block
            for col in block
                push!(row_inds,row)
                push!(col_inds,col)
                k1 = linear_u_index_from_pair[row,col, 1]
                k2 = linear_u_index_from_pair[row,col, 2]
                for i in 1:eachindex(us)
                    push!(vals_us[i],us[i][k1])
                    push!(vals_∂us[i],us[i][k2])
                end
            end
        end
    end
    us = [sparse(row_inds,col_inds,vals_u) for vals_u in vals_us]
    ∂us = [sparse(row_inds,col_inds,vals_∂u) for vals_∂u in vals_∂us]
    return us, ∂us
end

"""
    get_sinusoidal_coefficients_from_FFT(ts, s)

Compute the sinusoidal amplitude-phase coefficients and frequencies from FFT of the signal s with timeseries ts. 
"""
function get_sinusoidal_coefficients_from_FFT(ts, s)
    P = ts[end]
    dt = ts[2]-ts[1]
    fs = 1/dt
    N = size(ts,1)
    Fs = fft(s)[1:N÷2]
    freqs = 2*pi*fftfreq(length(s),fs)
    ak = (2/N)*real.(Fs)
    bk = (-2/N)*imag.(Fs)
    ak[1] = ak[1]/2
    phi_k = atan.(bk./ak)
    Ak = ak ./ cos.(phi_k)
    return Ak, -phi_k, freqs
end


"""
    truncated_inv_fft(t, Aks, phi_ks, freqs ; N=nothing)

Compute the signal at time t defined by the sinusoidal amplitude-phase coefficients and FFT frequencies truncated
to the N frequency components with largest magnitude.
"""
function truncated_inv_fft(t, Aks, phi_ks, freqs ; N=nothing)
    c = 0.0
    if N === nothing
        for (Ak, phi_k, f) in zip(Aks, phi_ks, freqs)
            c += Ak*cos(f*t+phi_k)
        end
    else
        I = sortperm(abs.(Aks), rev=true)[1:N]
        for i in I
            c+= Aks[i]*cos(freqs[i]*t+phi_ks[i])
        end
    end
    return c
end

function pink_noise(t; lf=1e-3, hf=20.0, n=50,alpha=1, seed=121314)
    freqs = range(lf,hf,length=n)
    rand_phases = rand(MersenneTwister(seed), Float64, size(freqs,1))
    st = 0.0
    for (f,p) in zip(freqs,rand_phases)
        st += sin(t*f*2*pi+2*pi*p)/(f^alpha)
    end
    return st
end