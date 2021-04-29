"""

"""
struct StressRecovoryIntegrationValues{dims,dimp,T,Mp,Ms}
    σ   ::SymmetricTensor{2,dims,T,Ms}
    ∇₁σ ::SymmetricTensor{2,dims,T,Ms}
    ∇₂σ ::SymmetricTensor{2,dims,T,Ms}
    ∇₁₁σ::SymmetricTensor{2,dims,T,Ms}
    ∇₂₁σ::SymmetricTensor{2,dims,T,Ms}
    ∇₁₂σ::SymmetricTensor{2,dims,T,Ms}
    ∇₂₂σ::SymmetricTensor{2,dims,T,Ms}
    κ ::Tensor{2,dimp,T,Mp}
    a ::Vec{dimp,T}
    da::Tensor{2,dimp,T,Mp}
    λ ::Vec{dimp,T}
    ζ ::T
end

function StressRecovoryIntegrationValues{dim_s,dim_p}() where {dim_s,dim_p}
    T = Float64
    
    Ms = Tensors.n_components(Tensors.get_base(SymmetricTensor{2,dim_s,T}))
    Mp = Tensors.n_components(Tensors.get_base((Tensor{2,dim_p,T})))

    return StressRecovoryIntegrationValues{dim_s,dim_p,T,Mp,Ms}(
        zero(SymmetricTensor{2,dim_s,T}),
        zero(SymmetricTensor{2,dim_s,T}),
        zero(SymmetricTensor{2,dim_s,T}),
        zero(SymmetricTensor{2,dim_s,T}),
        zero(SymmetricTensor{2,dim_s,T}),
        zero(SymmetricTensor{2,dim_s,T}),
        zero(SymmetricTensor{2,dim_s,T}),
        zero(Tensor{2,dim_p,T}),
        zero(Vec{dim_p,T}),
        zero(Tensor{2,dim_p,T}),
        zero(Vec{dim_p,T}),
        zero(T))
end


function extract(a::StressRecovoryIntegrationValues) 

    ∇₁σ₁₁ = a.∇₁σ[1,1]
    ∇₂σ₁₂ = a.∇₂σ[1,2]
    ∇₂σ₂₂ = a.∇₂σ[2,2]
    ∇₁σ₁₂ = a.∇₁σ[1,2]
    σ₁₁   = a.σ[1,1]
    σ₂₂   = a.σ[2,2]
    σ₁₂   = a.σ[1,2]
    ∇₁₁σ₁₁ = a.∇₁₁σ[1,1]
    ∇₂₁σ₁₂ = a.∇₂₁σ[1,2]
    ∇₁σ₂₂ = a.∇₁σ[2,2]
    ∇₂σ₁₁ = a.∇₂σ[1,1]

    ∇₂₂σ₂₂ = a.∇₂₂σ[2,2]
    ∇₁₂σ₁₂ = a.∇₁₂σ[1,2]

    return  σ₁₁, σ₂₂, σ₁₂,
            ∇₁σ₁₁, ∇₂σ₁₂,
            ∇₂σ₂₂, ∇₁σ₁₂, 
            ∇₁₁σ₁₁, ∇₂₁σ₁₂,
            ∇₂₂σ₂₂, ∇₁₂σ₁₂,
            ∇₁σ₂₂, ∇₂σ₁₁,
            a.κ, a.a[1], a.a[2], a.da[1,2], a.da[2,1], a.λ[1], a.λ[2], a.ζ

end

function midpoints(v1::StressRecovoryIntegrationValues{dims,dimp,T,Mp,Ms}, v2::StressRecovoryIntegrationValues{dims,dimp,T,Mp,Ms}) where {dims,dimp,T,Mp,Ms}
    midvalues = Any[]
    for field in fieldnames(StressRecovoryIntegrationValues)
        push!(midvalues, 0.5*(getproperty(v1, field) + getproperty(v2, field)))
    end
    return StressRecovoryIntegrationValues{dims,dimp,T,Mp,Ms}(midvalues...)
end

function extrapolate(y1::StressRecovoryIntegrationValues{dims,dimp,T,Mp,Ms},y2::StressRecovoryIntegrationValues{dims,dimp,T,Mp,Ms},x) where {dims,dimp,T,Mp,Ms}
    
    x1 = y1.ζ
    x2 = y2.ζ
    
    midvalues = Any[]
    for field in fieldnames(StressRecovoryIntegrationValues)
        push!(midvalues, extrapolate(getproperty(y1, field), getproperty(y2, field), x1, x2, x))
    end
    return StressRecovoryIntegrationValues{dims,dimp,T,Mp,Ms}(midvalues...)
end


"""
RecoveredStresses
    Stores the recovered stresses att different zeta-coordinates
"""

struct RecoveredStresses{T}
    σᶻˣ::T
    σᶻʸ::T
    σᶻᶻ::T
    ζ::T
end

RecoveredStresses{T}() where T = RecoveredStresses{T}(0.0,0.0,0.0,0.0)

recovered_stress_interface_index(iint::Int, nqp_oop_per_layer::Int) = iint*(nqp_oop_per_layer+1) + nqp_oop_per_layer -1

"""
CacheStressRecoveryValues

"""

struct CacheStressRecoveryValues{dim_s,T}

    Xᴸ::Vector{Vec{dim_s,T}}
    uᴸ::Vector{Vec{dim_s,T}}

    #Vector of results of SR integration of each layer
    ∫σᶻˣ::Vector{T} 
    ∫σᶻʸ::Vector{T}
    ∫σᶻᶻ::Vector{T}
    ∫∇σᶻˣ::Vector{T}
    ∫∇σᶻʸ::Vector{T}

    #Needed for sigma_zz
    σᶻˣ_layer_bc::Base.RefValue{T}
    σᶻʸ_layer_bc::Base.RefValue{T}
    ∇σᶻˣ_layer_bc::Base.RefValue{T}
    ∇σᶻʸ_layer_bc::Base.RefValue{T}
end

function CacheStressRecoveryValues{dim_s,T}(nipᴸ_basefuncs::Int, nqp_oop_lay::Int, nlay::Int) where {dim_s,T}

    Xᴸ = zeros(Vec{dim_s,T}, nipᴸ_basefuncs)
    uᴸ = zeros(Vec{dim_s,T}, nipᴸ_basefuncs)

    ∫σᶻˣ = zeros(T, (nqp_oop_lay+1)*nlay) 
    ∫σᶻʸ = zeros(T, (nqp_oop_lay+1)*nlay)
    ∫σᶻᶻ = zeros(T, (nqp_oop_lay+1)*nlay)
    ∫∇σᶻˣ = zeros(T, (nqp_oop_lay+1)*nlay)
    ∫∇σᶻʸ = zeros(T, (nqp_oop_lay+1)*nlay)

    return CacheStressRecoveryValues(
        Xᴸ, uᴸ, 
        ∫σᶻˣ, ∫σᶻʸ, ∫σᶻᶻ, ∫∇σᶻˣ, ∫∇σᶻʸ,
        Ref(T(0.0)),Ref(T(0.0)),Ref(T(0.0)),Ref(T(0.0)))
end

"""
IGAShellStressRecovory 
    Main structure dealing with stress recovory
"""

struct IGAShellStressRecovory{dim_s,dim_p,T,Mp,Ms}

    #Lagrange basis
    ipᴸ::MyLagrange{dim_p,RefCube} #Not stabil
    cvᴸ::BasisValues{dim_p,T} #Not stabil
    qrᴸ::QuadratureRule{dim_p,RefCube,T}
    orderingᴸ::Vector{Int}

    #
    nqp_oop_lay::Int
    nlayers::Int
    ninterfaces::Int
    nqp_inp::Int
    thickness::T
    
    #
    bc_σᶻᶻ::Tuple{T,T}
    bc_σˣᶻ::Tuple{T,T}
    bc_σʸᶻ::Tuple{T,T}

    #Vector of the values stored needed for integration, κ, λ etc
    integration_values::Array{StressRecovoryIntegrationValues{3,2,T,Mp,Ms}, 3}
    cache::CacheStressRecoveryValues{dim_s,T}
    recovered_stresses::Matrix{RecoveredStresses{T}}
end

function IGAShellStressRecovory(data::IGAShellData{dim_p,dim_s,T}) where {dim_s,dim_p,T}

    nqp_inp = getnquadpoints_inplane(data)
    lagrange_order = data.nqp_inplane_order - 1

    ipᴸ = MyLagrange{dim_p,RefCube,lagrange_order}()
    qrᴸ = QuadratureRule{dim_p,RefCube}(1)
    cvᴸ = BasisValues(qrᴸ, ipᴸ)
    orderingᴸ = _lagrange_ordering(ipᴸ)

    nqp_oop_lay = getnquadpoints_ooplane_per_layer(data)
    cache = CacheStressRecoveryValues{dim_s,T}(
        getnbasefunctions(ipᴸ), 
        getnquadpoints_ooplane_per_layer(data),
        nlayers(data))

    integration_values = [StressRecovoryIntegrationValues{3,2}() for _ in 1:(nqp_oop_lay+2), _ in 1:nlayers(data),  _ in 1:getncells(data)]
    recovered_stresses = [RecoveredStresses{T}() for _ in 1:((nqp_oop_lay+1)*nlayers(data) + 1), _ in 1:getncells(data)] 
    return IGAShellStressRecovory(
        ipᴸ, cvᴸ, qrᴸ,orderingᴸ,
        getnquadpoints_ooplane_per_layer(data), nlayers(data), ninterfaces(data), nqp_inp, data.thickness,
        (0.0, 0.0),(0.0, 0.0),(0.0, 0.0),
        integration_values, cache, recovered_stresses)
end

function recover_cell_stresses(srdata::IGAShellStressRecovory, σ_states::Vector{<:SymmetricTensor}, celldata::NamedTuple, cv, cv_sr)

    for ilay in 1:celldata.nlayers

        calculate_integration_values_for_layer!(srdata, σ_states, cv, cv_sr, srdata.ipᴸ, srdata.cvᴸ, celldata, ilay)

        calculate_stress_recovory_integrals_for_layer!(srdata, ilay, celldata.ic)
    end

    calculate_recovered_stresses!(srdata, celldata.ic)
    #Used for sigma_zz, reset for next cell
    srdata.cache.∇σᶻˣ_layer_bc[] = 0.0
    srdata.cache.∇σᶻʸ_layer_bc[] = 0.0
    srdata.cache.σᶻˣ_layer_bc[]  = 0.0
    srdata.cache.σᶻʸ_layer_bc[]  = 0.0
end

function calculate_integration_values_for_layer!(srdata::IGAShellStressRecovory{dim_s,dim_p,T}, σ_states::Vector{<:SymmetricTensor}, cv_sr::ISL, cv::ISL, ipᴸ, cvᴸ, celldata, ilay::Int) where {ISL<:IGAShellValues,dim_s,dim_p,T} 

    Xᴸ = srdata.cache.Xᴸ; uᴸ = srdata.cache.uᴸ
    
    h = srdata.thickness
    nqp_oop_lay = srdata.nqp_oop_lay
    integration_values = @view srdata.integration_values[:,:,celldata.ic]

    for lqp in 1:nqp_oop_lay
        qp_sr = (ilay-1) * nqp_oop_lay + lqp
        
        w = get_oop_qp_weight(cv_sr, qp_sr)
        ξ = get_qp_coord(cv_sr, qp_sr)
        ζ = ξ[dim_s] 

        #Get coords of inplane quadrature points for interpolation of stresses
        range = collect(get_inplane_qp_range(nqp_oop_lay*srdata.nqp_inp, srdata.nqp_inp, ilay, lqp))
        range = range[srdata.orderingᴸ]
        for i in 1:length(range)
            Xᴸ[i] = spatial_coordinate(cv, range[i], celldata.Xᵇ)
            uᴸ[i] = function_value(cv, range[i], celldata.ue)
        end

        xᴸ = Xᴸ + uᴸ 

        _a, _da, _λ, _κ = calculate_stress_recovory_variables(ipᴸ, Xᴸ, h, reinterpret(T,uᴸ), Vec{dim_s,T}((ξ[1:dim_p]..., 0.0)))
        a, da, λ, κ = _store_as_tensors(_a, _da, _λ, _κ)

        # lambda and kappa are both wrong from the lagrange-interpolation
        # until problem is found, calculate from the following:
        #g = calculate_g(cv_sr, qp_sr, celldata.ue)
        #_a = [(sqrt(g[i]⋅g[i])) for i in 1:dim_p]
        #a = Vec{dim_p,T}(Tuple(_a))
        #da = zero(Tensor{2,dim_p,T})
            _λ = zeros(T,2)
            _λ[1] = 1+cv_sr.κᵐ[1][1,1]*ζ*h/2
            _λ[2] = (dim_s == 2) ? 1.0 : 1+cv_sr.κᵐ[1][2,2]*ζ*h/2
            _κ = zeros(T,2,2)
            _κ[1,1] = cv_sr.κᵐ[1][1,1]
            _κ[1,2] = (dim_s == 2) ? 0.0 : cv_sr.κᵐ[1][1,2]
            _κ[2,1] = (dim_s == 2) ? 0.0 : cv_sr.κᵐ[1][2,1]
            _κ[2,2] = (dim_s == 2) ? 0.0 : cv_sr.κᵐ[1][2,2]
            λ = Vec{2,T}(Tuple(_λ))
            κ = Tensor{2,2,T,4}(Tuple(_κ))
            p, V = eigen(κ, sortby = x -> -abs(x))
        
        #Calculate and extract gradients for stress recovory equations
        σ = function_value(cvᴸ, 1, σ_states, range)
        ∇σ = Ferrite.function_derivative(cvᴸ, 1, σ_states, range)
        ∇∇σ = function_second_derivative(cvᴸ, 1, σ_states, range)
        ∇₁σ, ∇₂σ, ∇₁₁σ, ∇₂₁σ, ∇₁₂σ, ∇₂₂σ  = _store_as_tensors(∇σ, ∇∇σ)
        
        integration_values[lqp+1,ilay] = StressRecovoryIntegrationValues(σ, ∇₁σ, ∇₂σ, ∇₁₁σ, ∇₂₁σ, ∇₁₂σ, ∇₂₂σ, κ, a, da, λ, ζ)
    end

    #Extrapolate to interface
    z_interface1 = (ilay-1)*(2/celldata.nlayers) - 1
    z_interface2 = (ilay-0)*(2/celldata.nlayers) - 1
    
    extrp1 = extrapolate(integration_values[2,ilay], integration_values[3,ilay], z_interface1)
    integration_values[1,ilay] = extrp1

    extrp2 = extrapolate(integration_values[nqp_oop_lay,ilay], integration_values[nqp_oop_lay+1,ilay], z_interface2)
    integration_values[nqp_oop_lay+2,ilay] = extrp2
end

function calculate_stress_recovory_integrals_for_layer!(srdata::IGAShellStressRecovory{dim_s,dim_p,T}, ilay::Int, ic::Int) where {dim_s,dim_p,T} 
    
    h = srdata.thickness
    integration_values = srdata.integration_values
    nqp_oop_lay = srdata.nqp_oop_lay
    Z = 0.0

    #Needed for sigma_zz
    _σᶻˣ = zeros(T, nqp_oop_lay+2); _σᶻˣ[1] = srdata.cache.σᶻˣ_layer_bc[]
    _σᶻʸ = zeros(T, nqp_oop_lay+2); _σᶻʸ[1] = srdata.cache.σᶻʸ_layer_bc[]
    _∇σᶻˣ = zeros(T, nqp_oop_lay+2);_∇σᶻˣ[1] = srdata.cache.∇σᶻˣ_layer_bc[]
    _∇σᶻʸ = zeros(T, nqp_oop_lay+2);_∇σᶻʸ[1] = srdata.cache.∇σᶻʸ_layer_bc[]

    for i in (1:nqp_oop_lay+1)# .+ (ilay-1)*getnquadpoints_ooplane_per_layer(igashell)
        
        icount = (ilay-1)*(nqp_oop_lay+1) + i #Counter for current integration interval

        ival = midpoints(integration_values[i,ilay,ic], integration_values[i+1,ilay,ic])
        
        σ₁₁, σ₂₂, σ₁₂,
        ∇₁σ₁₁, ∇₂σ₁₂,
        ∇₂σ₂₂, ∇₁σ₁₂, 
        ∇₁₁σ₁₁, ∇₂₁σ₁₂,
        ∇₂₂σ₂₂, ∇₁₂σ₁₂,
        ∇₁σ₂₂, ∇₂σ₁₁,
        κ, a₁, a₂, ∂₂a₁, ∂₁a₂, λ₁, λ₂, ζ =  extract(ival)

        dZ = (integration_values[i+1,ilay,ic].ζ - integration_values[i,ilay,ic].ζ)*h/2
        Z += dZ
        
        #SR-equations
        σᶻˣ = (λ₁*λ₂/a₁ * ∇₁σ₁₁ +
                λ₁*λ₁/a₂ * ∇₂σ₁₂ +
                λ₁*λ₁*∂₁a₂/a₁/a₂ * (σ₁₁ - σ₂₂) + 
               2λ₁*λ₂*∂₂a₁/a₁/a₂ * σ₁₂) * dZ

        σᶻʸ = (λ₁*λ₂/a₂ * ∇₂σ₂₂ +
                λ₂*λ₂/a₁ * ∇₁σ₁₂ +
                λ₂*λ₂*∂₂a₁/a₁/a₂ * (σ₂₂ - σ₁₁) + 
               2λ₁*λ₂*∂₁a₂/a₁/a₂ * σ₁₂) * dZ

        ∇σᶻˣ = (λ₁*λ₂/a₁ * ∇₁₁σ₁₁ +
                λ₁*λ₁/a₂ * ∇₂₁σ₁₂ +
                λ₁*λ₁*∂₁a₂/a₁/a₂ * (∇₁σ₁₁ - ∇₁σ₂₂) + 
               2λ₁*λ₂*∂₂a₁/a₁/a₂ * ∇₁σ₁₂) * dZ

        ∇σᶻʸ = (λ₁*λ₂/a₂ * ∇₂₂σ₂₂ +
                λ₂*λ₂/a₁ * ∇₁₂σ₁₂ +
                λ₂*λ₂*∂₂a₁/a₁/a₂ * (∇₂σ₂₂ - ∇₂σ₁₁) + 
               2λ₁*λ₂*∂₁a₂/a₁/a₂ * ∇₂σ₁₂) * dZ
    

        #
        srdata.cache.∫σᶻˣ[icount] = σᶻˣ
        srdata.cache.∫σᶻʸ[icount] = σᶻʸ
        srdata.cache.∫∇σᶻˣ[icount] = ∇σᶻˣ
        srdata.cache.∫∇σᶻʸ[icount] = ∇σᶻʸ
        
        #This is a bit wrong...
        # Thhe sigma_zz integration should be in anouther loop,
        # but this works for now since C=0.0 in most cases
        C = 0.0
        λ_tmp = integration_values[i+1,ilay,ic].λ
         
        σᶻˣ = (-sum(srdata.cache.∫σᶻˣ[icount])+C)/(λ_tmp[1]λ_tmp[1]λ_tmp[2])
        σᶻʸ = (-sum(srdata.cache.∫σᶻʸ[icount])+C)/(λ_tmp[1]λ_tmp[2]λ_tmp[2])
        ∇σᶻˣ = (-sum(srdata.cache.∫∇σᶻˣ[icount]))/(λ_tmp[1]λ_tmp[1]λ_tmp[2])
        ∇σᶻʸ = (-sum(srdata.cache.∫∇σᶻʸ[icount]))/(λ_tmp[1]λ_tmp[2]λ_tmp[2])

        _σᶻˣ[i+1] = σᶻˣ
        _σᶻʸ[i+1] = σᶻʸ
        _∇σᶻˣ[i+1] = ∇σᶻˣ
        _∇σᶻʸ[i+1] = ∇σᶻʸ

        #  
        ∇₁σᶻˣ_mid = (_∇σᶻˣ[i+1] + _∇σᶻˣ[i])/2
        ∇₂σᶻʸ_mid = (_∇σᶻʸ[i+1] + _∇σᶻʸ[i])/2
        σᶻˣ_mid = (_σᶻˣ[i+1] + _σᶻˣ[i])/2
        σᶻʸ_mid = (_σᶻʸ[i+1] + _σᶻʸ[i])/2
 

        #Note: fel lambda avänds här... midpointlambda skrivs över
        σᶻᶻ =  (λ₂/a₁ * ∇₁σᶻˣ_mid + 
                λ₁/a₂ * ∇₂σᶻʸ_mid +
                -κ[1,1]*λ₂*σ₁₁ + 
                -κ[2,2]*λ₁*σ₂₂ +
                λ₁*∂₁a₂/a₂/a₁ * σᶻˣ_mid +
                λ₂*∂₂a₁/a₂/a₁ * σᶻʸ_mid) * dZ
            
        srdata.cache.∫σᶻᶻ[icount] = σᶻᶻ
    end

    srdata.cache.∇σᶻˣ_layer_bc[] = _∇σᶻˣ[end]
    srdata.cache.∇σᶻʸ_layer_bc[] = _∇σᶻʸ[end]
    srdata.cache.σᶻˣ_layer_bc[]  = _σᶻˣ[end]
    srdata.cache.σᶻʸ_layer_bc[]  = _σᶻʸ[end]
end

function calculate_recovered_stresses!(srdata::IGAShellStressRecovory{dim_s,dim_p,T}, ic::Int) where {dim_s,dim_p,T}
    h = srdata.thickness
    
    λᴮ = srdata.integration_values[1,1,ic].λ
    λᵀ = srdata.integration_values[end,end,ic].λ

    C₂ᶻ = (1/h) *(srdata.bc_σᶻᶻ[1]*λᴮ[1]*λᴮ[2] - 
                 srdata.bc_σᶻᶻ[2]*λᵀ[1]*λᵀ[2] -
                 sum(srdata.cache.∫σᶻᶻ))
    C₁ᶻ = -srdata.bc_σᶻᶻ[1]*λᴮ[1]*λᴮ[2] + C₂ᶻ*h/2
    Cᶻˣ = 0.0
    Cᶻʸ = 0.0

    
    #Add first stress recovered point seperatly
    λ₁, λ₂ = srdata.integration_values[1,1,ic].λ
    ζ = srdata.integration_values[1,1,ic].ζ*h/2
    σᶻˣ = -(Cᶻˣ )/(λ₁*λ₁*λ₂)
    σᶻʸ = -( Cᶻʸ)/(λ₁*λ₂*λ₂) 
    σᶻᶻ = -(ζ*C₂ᶻ + C₁ᶻ)/(λ₁*λ₂)
    
    srdata.recovered_stresses[1,ic] = RecoveredStresses(σᶻˣ, σᶻʸ, σᶻᶻ, ζ)

    i=1
    for ilay in 1:srdata.nlayers
        for qp in 1:(srdata.nqp_oop_lay+1)            
            ∑∫σᶻˣ = 0.0
            ∑∫σᶻʸ = 0.0
            ∑∫σᶻᶻ = 0.0

            for j in 1:i
                ∑∫σᶻˣ += srdata.cache.∫σᶻˣ[j]
                ∑∫σᶻʸ += srdata.cache.∫σᶻʸ[j]
                ∑∫σᶻᶻ += srdata.cache.∫σᶻᶻ[j]  
            end

            λ₁, λ₂ = srdata.integration_values[qp+1,ilay,ic].λ
            ζ = srdata.integration_values[qp+1,ilay,ic].ζ*h/2

            σᶻˣ = -(∑∫σᶻˣ + Cᶻˣ )/(λ₁*λ₁*λ₂)
            σᶻʸ = -(∑∫σᶻʸ + Cᶻʸ)/(λ₁*λ₂*λ₂) 
            σᶻᶻ = -(∑∫σᶻᶻ + ζ*C₂ᶻ + C₁ᶻ)/(λ₁*λ₂)
            
            srdata.recovered_stresses[i+1,ic] = RecoveredStresses(σᶻˣ, σᶻʸ, σᶻᶻ, ζ)
            i+=1
        end
    end
end


function _store_as_tensors(_a::Vector{T}, _da::Vector{Vec{2,T}}, _λ::Vector{T}, _κ::Tensor{2,1}) where {T}
    
    a = Vec{2,T}((_a[1], 1.0))
    da = Tensor{2,2}((_da[1][1], _da[1][2], 0.0, 0.0))

    λ = Vec((_λ[1], 1.0))
    κ = Tensor{2,2}((_κ[1,1], 0.0, 0.0, 0.0))

    return a, da, λ, κ
end

function _store_as_tensors(_a::Vector{T}, _da::Vector{Vec{3,T}}, _λ::Vector{T}, _κ::Tensor{2,2}) where {T}
    a = Tensor{1,2,T}(Tuple(_a))
    da = Tensor{2,2,T}((_da[1][1], _da[1][2], _da[2][1], _da[2][2]))
    λ = Vec{2,T}(Tuple(_λ))

    return a, da, λ, _κ
end

function _store_as_tensors(∇σ::Vector, ∇∇σ::Matrix)
    if length(∇σ) == 1
        ∇₁σ, ∇₂σ, = (∇σ[1], zero(SymmetricTensor{2,3,Float64}))
        ∇₁₁σ, ∇₂₁σ, ∇₁₂σ, ∇₂₂σ = (∇∇σ[1,1], zero(SymmetricTensor{2,3,Float64}),
                                  zero(SymmetricTensor{2,3,Float64}),
                                  zero(SymmetricTensor{2,3,Float64}))
    elseif length(∇σ) == 2
        ∇₁σ, ∇₂σ, ∇₁₁σ, ∇₂₁σ, ∇₁₂σ, ∇₂₂σ = (∇σ..., ∇∇σ...)
    else
        error("Wrong size")
    end

    return ∇₁σ, ∇₂σ, ∇₁₁σ, ∇₂₁σ, ∇₁₂σ, ∇₂₂σ
end