const ROSENBROCK_INV_CUTOFF = 7 # https://github.com/SciML/OrdinaryDiffEq.jl/pull/1539

function make_static_Wop(W, callinv = true)
    if callinv && (size(W, 1) <= ROSENBROCK_INV_CUTOFF)
        return inv(MatrixOperator(inv(W)))
    else
        return MatrixOperator(W)
    end
end

function calc_tderivative!(integrator, cache, dtd1, repeat_step)
    @inbounds begin
        @unpack t, dt, uprev, u, f, p = integrator
        @unpack du2, fsalfirst, dT, tf, linsolve_tmp = cache

        # Time derivative
        if !repeat_step # skip calculation if step is repeated
            if DiffEqBase.has_tgrad(f)
                f.tgrad(dT, uprev, p, t)
            else
                tf.uprev = uprev
                tf.p = p
                derivative!(dT, tf, t, du2, integrator, cache.grad_config)
            end
        end

        @.. broadcast=false linsolve_tmp=fsalfirst + dtd1 * dT
    end
end

function calc_tderivative(integrator, cache)
    @unpack t, dt, uprev, u, f, p = integrator

    # Time derivative
    if DiffEqBase.has_tgrad(f)
        dT = f.tgrad(uprev, p, t)
    else
        tf = cache.tf
        tf.u = uprev
        tf.p = p
        dT = derivative(tf, t, integrator)
    end
    dT
end

"""
    calc_J(integrator, cache, next_step::Bool = false)

Return a new Jacobian object.

If `integrator.f` has a custom Jacobian update function, then it will be called. Otherwise,
either automatic or finite differencing will be used depending on the `uf` object of the
cache. If `next_step`, then it will evaluate the Jacobian at the next step.
"""
function calc_J(integrator, cache, next_step::Bool = false)
    @unpack dt, t, uprev, f, p, alg = integrator
    if next_step
        t = t + dt
        uprev = integrator.u
    end

    if alg isa DAEAlgorithm
        if DiffEqBase.has_jac(f)
            J = f.jac(duprev, uprev, p, t) # G: this would be update_coefficients for OOP w/ scimlop prototype
        else
            @unpack uf = cache
            x = zero(uprev)
            J = jacobian(uf, x, integrator)
        end
    else
        if DiffEqBase.has_jac(f)
            J = f.jac(uprev, p, t)
        else
            @unpack uf = cache

            uf.f = nlsolve_f(f, alg)
            uf.p = p
            uf.t = t

            J = jacobian(uf, uprev, integrator)
        end

        integrator.stats.njacs += 1

        if alg isa CompositeAlgorithm
            integrator.eigen_est = constvalue(opnorm(J, Inf))
        end
    end

    J
end

"""
    calc_J!(J, integrator, cache, next_step::Bool = false) -> J

Update the Jacobian object `J`.

If `integrator.f` has a custom Jacobian update function, then it will be called. Otherwise,
either automatic or finite differencing will be used depending on the `cache`.
If `next_step`, then it will evaluate the Jacobian at the next step.
"""
function calc_J!(J, integrator, cache, next_step::Bool = false)
    @unpack dt, t, uprev, f, p, alg = integrator
    # this logic is already done in calc_W!, no need to do it again here.
    # instead, just encode all this behaviour below into the updating behaviour.
    if next_step
        t = t + dt
        uprev = integrator.u
    end

    if alg isa DAEAlgorithm
        if DiffEqBase.has_jac(f)
            duprev = integrator.duprev
            uf = cache.uf
            f.jac(J, duprev, uprev, p, uf.α * uf.invγdt, t) # what is second last arg? ah, for DAE?
        else
            @unpack du1, uf, jac_config = cache
            # using `dz` as temporary array
            x = cache.dz
            uf.t = t
            fill!(x, zero(eltype(x)))
            jacobian!(J, uf, x, du1, integrator, jac_config)
        end
    else
        if DiffEqBase.has_jac(f)
            f.jac(J, uprev, p, t)
        else
            @unpack du1, uf, jac_config = cache # G: what is uf here

            uf.f = nlsolve_f(f, alg)
            uf.t = t
            if !(p isa DiffEqBase.NullParameters) # G: ooh, cool, might want to use this elsewhere
                uf.p = p
            end

            jacobian!(J, uf, uprev, du1, integrator, jac_config)
        end
    end

    integrator.stats.njacs += 1

    if alg isa CompositeAlgorithm
        integrator.eigen_est = constvalue(opnorm(J, Inf))
    end

    return nothing
end

# TODO: move to SciMLBase
function SciMLOperators.update_coefficients!(J::UJacobianWrapper, u, p, t)
    J.p = p
    J.t = t
end

"""
    make_Wop(mass_matrix, dtgamma, J, uprev; transform=false, iip, concrete=false)

A linear operator that represents the W matrix of an ODEProblem, defined as

```math
W = MM - \\gamma J
```

or, if `transform=true`:

```math
W = \\frac{1}{\\gamma}MM - J
```

where `MM` is the mass matrix (a regular `AbstractMatrix` or a `UniformScaling`),
`γ` is a real number proportional to the time step, and `J` is the Jacobian
operator (must be a `AbstractSciMLOperator`). 

`concrete` can be set to `true` to force the operator to be a concrete matrix.
"""
function make_Wop(mass_matrix, dtgamma, J, uprev; transform = false, iip, concrete=false)
    gamma_op = ScalarOperator(dtgamma; update_func=(old_op, u, p, t; dtgamma) -> dtgamma, accepted_kwargs=(:dtgamma,))

    get_transform(dtgamma, transform) = transform ? inv(dtgamma) : one(dtgamma)
    transform_op = ScalarOperator(get_transform(dtgamma, transform); 
                                  update_func = (old_op, u, p, t; dtgamma, transform) -> get_transform(dtgamma, transform), 
                                  accepted_kwargs=(:dtgamma, :transform))

    # G: this logic needs to go *somewhere*...
    # _J = if J isa AbstractMatrix
    #     MatrixOperator(J) # G: does this neglect jac?
    # elseif J isa AbstractSciMLOperator
    #     J
    # else
    #     error("Jacobian J is of unexpected type $(typeof(J))")
    # end

    _W = -(mass_matrix - gamma_op * J) * transform_op 
    W = if isconvertible(J) || concrete
        ConcretizedOperator(_W) # TODO: handle any compplications in sparse case
    else
        cache_operator(_W, uprev)
    end 
    return W
end

"""
    islinearfunction(integrator) -> Tuple{Bool,Bool}

return the tuple `(is_linear_wrt_odealg, islinearodefunction)`.
"""
islinearfunction(integrator) = islinearfunction(integrator.f, integrator.alg)

"""
    islinearfunction(f, alg) -> Tuple{Bool,Bool}

return the tuple `(is_linear_wrt_odealg, islinearodefunction)`.
"""
function islinearfunction(f, alg)::Tuple{Bool, Bool}
    isode = f isa ODEFunction && islinear(f.f)
    islin = isode || (alg isa SplitAlgorithms && f isa SplitFunction && islinear(f.f1.f))
    return islin, isode
end

function do_newJW(integrator, alg, nlsolver, repeat_step)::NTuple{2, Bool}
    integrator.iter <= 1 && return true, true # at least one JW eval at the start
    repeat_step && return false, false
    islin, _ = islinearfunction(integrator)
    islin && return false, false # no further JW eval when it's linear
    !integrator.opts.adaptive && return true, true # Not adaptive will always refactorize
    errorfail = integrator.EEst > one(integrator.EEst)
    if alg isa DAEAlgorithm
        return true, true
    end
    # TODO: add `isJcurrent` support for Rosenbrock solvers
    if !isnewton(nlsolver)
        isfreshJ = !(integrator.alg isa CompositeAlgorithm) &&
                   (integrator.iter > 1 && errorfail && !integrator.u_modified)
        return !isfreshJ, true
    end
    isfirstcall(nlsolver) && return true, true
    isfs = isfirststage(nlsolver)
    isfreshJ = isJcurrent(nlsolver, integrator) && !integrator.u_modified
    iszero(nlsolver.fast_convergence_cutoff) && return isfs && !isfreshJ, isfs
    mm = integrator.f.mass_matrix
    is_varying_mm = !isconstant(mm)
    if isfreshJ
        jbad = false
        smallstepchange = true
    else
        W_iγdt = inv(nlsolver.cache.W_γdt)
        iγdt = inv(nlsolver.γ * integrator.dt)
        smallstepchange = abs(iγdt / W_iγdt - 1) <= get_new_W_γdt_cutoff(nlsolver)
        jbad = nlsolver.status === TryAgain && smallstepchange
    end
    wbad = (!smallstepchange) || (isfs && errorfail) || nlsolver.status === Divergence
    return jbad, (is_varying_mm || jbad || wbad)
end

@noinline _throwWJerror(W, J) = throw(DimensionMismatch("W: $(axes(W)), J: $(axes(J))"))
@noinline function _throwWMerror(W, mass_matrix)
    throw(DimensionMismatch("W: $(axes(W)), mass matrix: $(axes(mass_matrix))"))
end
@noinline function _throwJMerror(J, mass_matrix)
    throw(DimensionMismatch("J: $(axes(J)), mass matrix: $(axes(mass_matrix))"))
end

function jacobian2W!(W::AbstractMatrix, mass_matrix::MT, dtgamma::Number, J::AbstractMatrix,
    W_transform::Bool)::Nothing where {MT}
    # check size and dimension
    iijj = axes(W)
    @boundscheck (iijj == axes(J) && length(iijj) == 2) || _throwWJerror(W, J)
    mass_matrix isa UniformScaling ||
        @boundscheck axes(mass_matrix) == axes(W) || _throwWMerror(W, mass_matrix)
    @inbounds if W_transform
        invdtgamma = inv(dtgamma)
        if MT <: UniformScaling
            copyto!(W, J)
            idxs = diagind(W)
            λ = -mass_matrix.λ
            if ArrayInterface.fast_scalar_indexing(J) &&
               ArrayInterface.fast_scalar_indexing(W)
                @inbounds for i in 1:size(J, 1)
                    W[i, i] = muladd(λ, invdtgamma, J[i, i])
                end
            else
                @.. broadcast=false @view(W[idxs])=muladd(λ, invdtgamma, @view(J[idxs]))
            end
        else
            @.. broadcast=false W=muladd(-mass_matrix, invdtgamma, J)
        end
    else
        if MT <: UniformScaling
            λ = -mass_matrix.λ
            if W isa AbstractSparseMatrix && !(W isa SparseMatrixCSC)
                # This is specifically to catch the GPU sparse matrix cases
                # Which do not support diagonal indexing
                # https://github.com/JuliaGPU/CUDA.jl/issues/1395

                Wn = nonzeros(W)
                Jn = nonzeros(J)

                # I would hope to check this generically, but `CuSparseMatrixCSC` has `colPtr`
                # and `rowVal` while SparseMatrixCSC is colptr and rowval, and there is no
                # standard for checking sparsity patterns in general. So for now, write it for
                # the convention of CUDA.jl and handle the case of some other convention when
                # it comes up.

                @assert J.colPtr == W.colPtr
                @assert J.rowVal == W.rowVal

                @.. broadcast=false Wn=dtgamma * Jn
                W .= W + λ * I
            elseif W isa SparseMatrixCSC
                #=
                using LinearAlgebra, SparseArrays, FastBroadcast
                J = sparse(Diagonal(ones(4)))
                W = sparse(Diagonal(ones(4)));
                J[4,4] = 0
                gamma = 1.0
                W .= gamma .* J

                4×4 SparseMatrixCSC{Float64, Int64} with 3 stored entries:
                1.0   ⋅    ⋅   ⋅
                ⋅   1.0   ⋅    ⋅
                ⋅    ⋅   1.0   ⋅
                ⋅    ⋅    ⋅     ⋅

                Thus broadcast cannot be used.

                Instead, check the sparsity pattern is correct and directly broadcast the nzval
                =#
                @assert J.colptr == W.colptr
                @assert J.rowval == W.rowval
                @.. broadcast=false W.nzval=dtgamma * J.nzval
                idxs = diagind(W)
                @.. broadcast=false @view(W[idxs])=@view(W[idxs]) + λ
            else # Anything not a sparse matrix
                @.. broadcast=false W=dtgamma * J
                idxs = diagind(W)
                @.. broadcast=false @view(W[idxs])=@view(W[idxs]) + λ
            end
        else
            @.. broadcast=false W=muladd(dtgamma, J, -mass_matrix)
        end
    end
    return nothing
end

function jacobian2W!(W::Matrix, mass_matrix::MT, dtgamma::Number, J::Matrix,
    W_transform::Bool)::Nothing where {MT}
    # check size and dimension
    iijj = axes(W)
    @boundscheck (iijj == axes(J) && length(iijj) == 2) || _throwWJerror(W, J)
    mass_matrix isa UniformScaling ||
        @boundscheck axes(mass_matrix) == axes(W) || _throwWMerror(W, mass_matrix)
    @inbounds if W_transform
        invdtgamma = inv(dtgamma)
        if MT <: UniformScaling
            copyto!(W, J)
            idxs = diagind(W)
            λ = -mass_matrix.λ
            @inbounds for i in 1:size(J, 1)
                W[i, i] = muladd(λ, invdtgamma, J[i, i])
            end
        else
            @inbounds @simd ivdep for i in eachindex(W)
                W[i] = muladd(-mass_matrix[i], invdtgamma, J[i])
            end
        end
    else
        if MT <: UniformScaling
            idxs = diagind(W)
            @inbounds @simd ivdep for i in eachindex(W)
                W[i] = dtgamma * J[i]
            end
            λ = -mass_matrix.λ
            @inbounds for i in idxs
                W[i] = W[i] + λ
            end
        else
            @inbounds @simd ivdep for i in eachindex(W)
                W[i] = muladd(dtgamma, J[i], -mass_matrix[i])
            end
        end
    end
    return nothing
end

function jacobian2W(mass_matrix::MT, dtgamma::Number, J::AbstractMatrix,
    W_transform::Bool)::Nothing where {MT}
    # check size and dimension
    mass_matrix isa UniformScaling ||
        @boundscheck axes(mass_matrix) == axes(J) || _throwJMerror(J, mass_matrix)
    @inbounds if W_transform
        invdtgamma = inv(dtgamma)
        if MT <: UniformScaling
            λ = -mass_matrix.λ
            W = J + (λ * invdtgamma) * I
        else
            W = muladd(-mass_matrix, invdtgamma, J)
        end
    else
        if MT <: UniformScaling
            λ = -mass_matrix.λ
            W = dtgamma * J + λ * I
        else
            W = muladd(dtgamma, J, -mass_matrix)
        end
    end
    return W
end

function calc_W_J!(W, J, integrator, nlsolver::Union{Nothing, AbstractNLSolver}, cache, dtgamma,
    repeat_step, W_transform = false, newJW = nothing)
    @unpack t, dt, uprev, u, f, p = integrator
    lcache = nlsolver === nothing ? cache : nlsolver.cache
    next_step = is_always_new(nlsolver)
    if next_step
        t = t + integrator.dt
        uprev = integrator.u
    end

    @unpack J = lcache
    isdae = integrator.alg isa DAEAlgorithm
    alg = unwrap_alg(integrator, true)
    if !isdae
        mass_matrix = integrator.f.mass_matrix
    end
    is_compos = integrator.alg isa CompositeAlgorithm

    # handle Wfact
    if W_transform && DiffEqBase.has_Wfact_t(f)
        f.Wfact_t(W, u, p, dtgamma, t)
        isnewton(nlsolver) && set_W_γdt!(nlsolver, dtgamma)
        is_compos && (integrator.eigen_est = constvalue(opnorm(LowerTriangular(W), Inf)) +
                                inv(dtgamma)) # TODO: better estimate
        # It's equivalent with evaluating a new Jacobian, but not a new W,
        # because we won't call `lu!`, and the iteration matrix is fresh.
        return (true, false)
    elseif !W_transform && DiffEqBase.has_Wfact(f)
        f.Wfact(W, u, p, dtgamma, t)
        isnewton(nlsolver) && set_W_γdt!(nlsolver, dtgamma)
        if is_compos
            opn = opnorm(LowerTriangular(W), Inf)
            integrator.eigen_est = (constvalue(opn) + one(opn)) / dtgamma # TODO: better estimate
        end
        return (true, false)
    end

    # check if we need to update J or W
    if newJW === nothing
        new_jac, new_W = do_newJW(integrator, alg, nlsolver, repeat_step)
    else
        new_jac, new_W = newJW
    end

    if new_jac && isnewton(lcache)
        lcache.J_t = t
        if isdae
            lcache.uf.α = nlsolver.α
            lcache.uf.invγdt = inv(dtgamma)
            lcache.uf.tmp = nlsolver.tmp
        end
    end

    # calculate W
    @assert W isa AbstractSciMLOperator "W operator of unexpected type $(typeof(W)))"
    # only update W here if solver is not newton, since we will call `update_coefficients!` in NLNewton.
    isnewton(nlsolver) || update_coefficients!(W, uprev, p, t; dtgamma, transform=W_transform) 
    if J !== nothing && !(J isa AbstractSciMLOperator)
        islin, isode = islinearfunction(integrator)
        islin ? (J = isode ? f.f : f.f1.f) :
        (new_jac && (calc_J!(J, integrator, lcache, next_step)))
        new_W && !isdae &&
            jacobian2W!(W._concrete_form, mass_matrix, dtgamma, J, W_transform)
    end
    if isnewton(nlsolver)
        set_new_W!(nlsolver, new_W)
        if new_jac && isdae
            set_W_γdt!(nlsolver, nlsolver.α * inv(dtgamma))
        elseif new_W && !isdae
            set_W_γdt!(nlsolver, dtgamma)
        end
    end

    new_W && (integrator.stats.nw += 1)
    return new_jac, new_W
end

@noinline function calc_W(integrator, nlsolver, dtgamma, repeat_step, W_transform = false) # TODO: make W_transform a Val?
    @unpack t, uprev, p, f = integrator

    next_step = is_always_new(nlsolver)
    if next_step
        t = t + integrator.dt
        uprev = integrator.u
    end
    # Handle Rosenbrock has no nlsolver so passes cache directly
    cache = nlsolver isa OrdinaryDiffEqCache ? nlsolver : nlsolver.cache

    isdae = integrator.alg isa DAEAlgorithm
    if !isdae
        mass_matrix = integrator.f.mass_matrix
    end
    isarray = uprev isa AbstractArray
    # calculate W
    is_compos = integrator.alg isa CompositeAlgorithm
    islin, isode = islinearfunction(integrator)
    !isdae && update_coefficients!(mass_matrix, uprev, p, t)

    if islin
        J = isode ? f.f : f.f1.f # unwrap the Jacobian accordingly
        W = make_Wop(mass_matrix, dtgamma, J, uprev; transform = W_transform, iip = Val{false}())
    elseif DiffEqBase.has_jac(f)
        J = f.jac(uprev, p, t)
        # G: what is the difference between these two branches here? I guess previously, the latter was only for mutating,
        # but distinction should not exist any more.
        if typeof(J) <: StaticArray &&
           typeof(integrator.alg) <:
           Union{Rosenbrock23, Rodas4, Rodas4P, Rodas4P2, Rodas5, Rodas5P}
            W = W_transform ? J - mass_matrix * inv(dtgamma) :
                dtgamma * J - mass_matrix
        else
            # G: what's going on here? We're lazily making the W operator right here?
            # should we really use OOP update_coefficients?
            if !isa(J, AbstractSciMLOperator) && (!isnewton(nlsolver) ||
                nlsolver.cache.W.J isa AbstractSciMLOperator)
                J = MatrixOperator(J)
            end
            W = make_Wop(mass_matrix, dtgamma, J, uprev; transform = W_transform, iip = Val{false}())
        end
        integrator.stats.nw += 1
    else
        integrator.stats.nw += 1
        J = calc_J(integrator, cache, next_step)
        if isdae
            W = J
        else
            W_full = W_transform ? J - mass_matrix * inv(dtgamma) :
                     dtgamma * J - mass_matrix
            len = StaticArrayInterface.known_length(typeof(W_full))
            W = if W_full isa Number
                W_full
            elseif len !== nothing &&
                   typeof(integrator.alg) <:
                   Union{Rosenbrock23, Rodas4, Rodas4P, Rodas4P2, Rodas5, Rodas5P}
                make_static_Wop(W_full) # G: makes a new static W operator with the true dense W 
            else
                DiffEqBase.default_factorize(W_full) # what is this case doing? an LU...
            end
        end
    end
    (unwrap_alg(integrator, true) isa NewtonAlgorithm) &&
        (W = update_coefficients!(W, uprev, p, t)) # we will call `update_coefficients!` in NLNewton
    is_compos && (integrator.eigen_est = isarray ? constvalue(opnorm(J, Inf)) :
                            integrator.opts.internalnorm(J, t))
    return W
end

function calc_rosenbrock_differentiation!(integrator, cache, dtd1, dtgamma, repeat_step,
    W_transform)
    nlsolver = nothing
    # we need to skip calculating `J` and `W` when a step is repeated
    new_jac = new_W = false
    if !repeat_step
        new_jac, new_W = calc_W_J!(cache.W, cache.J, integrator, nlsolver, cache, dtgamma, repeat_step,
            W_transform)
    end
    # If the Jacobian is not updated, we won't have to update ∂/∂t either.
    calc_tderivative!(integrator, cache, dtd1, repeat_step || !new_jac)
    return new_W
end

# update W matrix (only used in Newton method)
function update_W!(integrator, cache, dtgamma, repeat_step, newJW = nothing)
    update_W!(cache.nlsolver, integrator, cache, dtgamma, repeat_step, newJW)
end

function update_W!(nlsolver::AbstractNLSolver,
    integrator::SciMLBase.DEIntegrator{<:Any, true}, cache, dtgamma,
    repeat_step::Bool, newJW = nothing)
    if isnewton(nlsolver)
        # todo: newton solve stuff needs some updating...
        calc_W_J!(get_W(nlsolver), integrator, nlsolver, cache, dtgamma, repeat_step, true,
            newJW)
    end
    nothing
end

# todo: newton solve stuff needs some updating...
function update_W!(nlsolver::AbstractNLSolver,
    integrator::SciMLBase.DEIntegrator{<:Any, false}, cache, dtgamma,
    repeat_step::Bool, newJW = nothing)
    if isnewton(nlsolver)
        isdae = integrator.alg isa DAEAlgorithm
        new_jac, new_W = true, true
        if isdae && new_jac
            lcache = nlsolver.cache
            lcache.uf.α = nlsolver.α
            lcache.uf.invγdt = inv(dtgamma)
            lcache.uf.tmp = @. nlsolver.tmp
            lcache.uf.uprev = @. integrator.uprev
        end
        nlsolver.cache.W = calc_W(integrator, nlsolver, dtgamma, repeat_step, true)
        #TODO: jacobian reuse for oop
        new_jac && (nlsolver.cache.J_t = integrator.t)
        set_new_W!(nlsolver, new_W)
        if new_jac && isdae
            set_W_γdt!(nlsolver, nlsolver.α * inv(dtgamma))
        elseif new_W && !isdae
            set_W_γdt!(nlsolver, dtgamma)
        end
    end
    nothing
end

function build_J_W(alg, u, uprev, p, t, dt, f::F, ::Type{uEltypeNoUnits},
    ::Val{IIP}) where {IIP, uEltypeNoUnits, F}
    # TODO - when making Jacobian a SciMLOperator, encode its updating behaviour within its update_func rather than separately/implicitly
    # (currently, if J is an AbstractMatrix, we will update it manually and J_op implicitly depends on it.)
    # TODO - make mass matrix a SciMLOperator so it can be updated with time. Default to IdentityOperator
    islin, isode = islinearfunction(f, alg)
    if f.jac_prototype isa AbstractSciMLOperator
        J = deepcopy(f.jac_prototype)
        W = make_Wop(f.mass_matrix, dt, J, u; iip=Val{IIP}())
    elseif IIP && f.jac_prototype !== nothing && concrete_jac(alg) === nothing &&
           (alg.linsolve === nothing ||
            alg.linsolve !== nothing &&
            LinearSolve.needs_concrete_A(alg.linsolve))
        # If linear solve is a factorization, then force concrete W 
        J = similar(f.jac_prototype) 
        J_op = MatrixOperator(J; update_func=jac) # TODO: should J also be set to this MatrixOperator?
        W = make_Wop(f.mass_matrix, dt, J_op, u; iip=Val{IIP}(), concrete=true)
    elseif (IIP && (concrete_jac(alg) === nothing || !concrete_jac(alg)) &&
            alg.linsolve !== nothing &&
            !LinearSolve.needs_concrete_A(alg.linsolve))
        # If the user has chosen GMRES but no sparse Jacobian, assume that the dense
        # Jacobian is a bad idea and create a fully matrix-free solver. This can
        # be overridden with concrete_jac.

        _f = islin ? (isode ? f.f : f.f1.f) : f
        J = JacVec(UJacobianWrapper(_f, t, p), copy(u), p, t;
            autodiff = alg_autodiff(alg), tag = OrdinaryDiffEqTag()) 
        W = make_Wop(f.mass_matrix, dt, J, u; iip=Val{IIP}())

    elseif alg.linsolve !== nothing && !LinearSolve.needs_concrete_A(alg.linsolve) ||
           concrete_jac(alg) !== nothing && concrete_jac(alg)
        # The linear solver does not need a concrete Jacobian, but the user has
        # asked for one. This will happen when the Jacobian is used in the preconditioner
        # Thus setup JacVec for use in the W operator, but keep J itself concrete,
        # using sparsity when possible.
        _f = islin ? (isode ? f.f : f.f1.f) : f
        J = if f.jac_prototype === nothing
            ArrayInterface.undefmatrix(u)
        else
            deepcopy(f.jac_prototype)
        end
        J_op = JacVec(UJacobianWrapper(_f, t, p), copy(u), p, t;
            autodiff = alg_autodiff(alg), tag = OrdinaryDiffEqTag())
        W = make_Wop(f.mass_matrix, dt, J_op, u; iip=Val{IIP}())

    elseif islin || (!IIP && DiffEqBase.has_jac(f))
        # The ODE function is either linear or OOP with a provided Jacobian.
        J = islin ? (isode ? f.f : f.f1.f) : f.jac(uprev, p, t) # unwrap the Jacobian accordingly
        J_op = if !isa(J, AbstractSciMLOperator)
            MatrixOperator(_J; update_func=f.jac)
        else
            J
        end
        W = make_Wop(f.mass_matrix, dt, J_op, u; iip=Val{IIP}())
    else
        # Make static placeholders for J and W 
        J = if f.jac_prototype === nothing
            ArrayInterface.undefmatrix(u)
        else
            deepcopy(f.jac_prototype)
        end
        isdae = alg isa DAEAlgorithm
        W = if isdae
            J
        elseif IIP
            similar(J)
        else
            len = StaticArrayInterface.known_length(typeof(J))
            # G: what is this branch? (len !== nothing means J is static)
            if len !== nothing &&
               typeof(alg) <:
               Union{Rosenbrock23, Rodas4, Rodas4P, Rodas4P2, Rodas5, Rodas5P}
                make_static_Wop(J, false) # G: so J is a placeholder here with the right structure?
            else
                ArrayInterface.lu_instance(J) # why this?
            end
        end
    end
    return J, W
end

build_uf(alg, nf, t, p, ::Val{true}) = UJacobianWrapper(nf, t, p)
build_uf(alg, nf, t, p, ::Val{false}) = UDerivativeWrapper(nf, t, p)

# TODO: any of this logic below still relevant for SciMLOp W?

# function LinearSolve.init_cacheval(alg::LinearSolve.DefaultLinearSolver, A::WOperator, b, u,
#     Pl, Pr,
#     maxiters::Int, abstol, reltol, verbose::Bool,
#     assumptions::OperatorAssumptions)
#     LinearSolve.init_cacheval(alg, A.J, b, u, Pl, Pr,
#         maxiters::Int, abstol, reltol, verbose::Bool,
#         assumptions::OperatorAssumptions)
# end

# for alg in InteractiveUtils.subtypes(OrdinaryDiffEq.LinearSolve.AbstractFactorization)
#     @eval function LinearSolve.init_cacheval(alg::$alg, A::WOperator, b, u, Pl, Pr,
#         maxiters::Int, abstol, reltol, verbose::Bool,
#         assumptions::OperatorAssumptions)
#         LinearSolve.init_cacheval(alg, A.J, b, u, Pl, Pr,
#             maxiters::Int, abstol, reltol, verbose::Bool,
#             assumptions::OperatorAssumptions)
#     end
# end
