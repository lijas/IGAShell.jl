
function _calculate_X₀(msip::Interpolation, coords::Vector{Vec{dim_s,T2}}, ξ::Vec{dim_s,T})where {dim_s,T,T2}
    dim_p = dim_s-1
    ξ2 = Vec((ξ[1:dim_p]...,))

    x = zero(Vec{dim_s,T})
    for i in 1:getnbasefunctions(msip)
        N = Ferrite.value(msip,i,ξ2)
        x += N * coords[i]
    end
    return x
end

function _calculate_E(msip::Interpolation, coords::Vector{Vec{dim_s,T2}}, ξ::Vec{dim_s,T}, α::Int)where {dim_s,T,T2}
    return gradient((ξ) -> _calculate_X₀(msip, coords, ξ), ξ)[:,α]
end

function _calculate_D(msip::Interpolation, coords::Vector{Vec{dim_s,T2}}, h::T2, ξ::Vec{dim_s,T}) where {dim_s,T,T2}
    
    E = [gradient((ξ) -> _calculate_X₀(msip, coords, ξ), ξ)[:,d] for d in 1:(dim_s-1)]

    D = cross(E...)
    D /= norm(D)
    D *= 0.5h
    return D
end

function _calculate_X(msip::Interpolation, coords::Vector{Vec{dim_s,T2}}, h::T2, ξ::Vec{dim_s,T})where {dim_s,T,T2}
    X₀ = _calculate_X₀(msip, coords, ξ)
    D = _calculate_D(msip, coords, h, ξ)
    ζ = ξ[dim_s]
    return X₀ + ζ*D
end

function _calculate_x(ip::Interpolation, coords::Vector{Vec{dim_s,T}}, h::T, ue::AbstractVector{T}, ξ::Vec{dim_s,T2}) where {dim_s,T,T2}
    X = _calculate_X(ip, coords, h, ξ)
    u = _calculate_u(ip, ue, ξ)
    return X + u
end

function _calculate_x(ip::Tuple{Interpolation, Vector{Interpolation}, IGA.BezierExtractionOperator}, coords::Vector{Vec{dim_s,T}}, h::T, ue::AbstractVector{T}, ξ::Vec{dim_s,T2}) where {dim_s,T,T2}
    X = _calculate_X(ip[1], coords, h, ξ)
    u = _calculate_u(ip, ue, ξ)
    return X + u
end

function _calculate_u(ip::Tuple{Interpolation, Vector{Interpolation}, IGA.BezierExtractionOperator}, ue::AbstractVector{T}, ξ::Vec{dim_s,T2}) where {dim_s,T,T2}
    
    msip = ip[1]
    opip = ip[2]
    Ce = ip[3]

    dim_p = dim_s-1

    ξ2 = Vec((ξ[1:dim_p]...,))
    ζ = Vec((ξ[dim_s]))

    N = T2[]
    B = [Ferrite.value(msip, i, ξ2) for i in 1:getnbasefunctions(msip)]

    for i in 1:getnbasefunctions(msip)
        S = sum(Ce[i] .* B)
        for j in 1:getnbasefunctions(opip[i])
            H = Ferrite.value(opip[i], j, ζ)
            push!(N, S*H)
        end
    end

    count = 1
    u = zero(Vec{dim_s,T2})
    for _N in N
        for d in 1:dim_s
            basevec = eᵢ(Vec{dim_s, Float64}, d)
            u += _N*basevec* ue[count]
            count +=1
        end
    end

    return u
end

function _calculate_u(ip::Interpolation, ue::AbstractVector{T}, ξ::Vec{dim_s,T2}) where {dim_s,T,T2}
    
    u = zero(Vec{dim_s,T2})
    ξ2 = Vec(ntuple(i->ξ[i], dim_s-1))

    count = 1
    for i in 1:getnbasefunctions(ip)
        N = Ferrite.value(ip, i, ξ2)
        for d in 1:dim_s
            u += N * ue[count] * eᵢ(Vec{dim_s, Float64}, d)
            count += 1
        end
    end

    return u
end

function _calculate_G(ip, X::Vector{Vec{dim_s,T2}}, h::T2, ξ::Vec{dim_s,T}, α::Int) where {dim_s,T,T2}
    return gradient((ξ) -> _calculate_X(ip, X, h, ξ), ξ)[:,α]
end

function _calculate_g(ip, X::Vector{Vec{dim_s,T2}}, h::T2, ue::AbstractVector{T2}, ξ::Vec{dim_s,T}, α::Int) where {dim_s,T,T2}

    #G = _calculate_G(msip, coords, h, ξ, α)
    #du = gradient((ξ) ->_calculate_u(msip, opip, Ce, ue, ξ), ξ)
    #g1 = G + du[:,α]
    g = gradient((ξ) ->_calculate_x(ip, X, h, ue, ξ), ξ)[:,α]

    return g
end


function calculate_a(ip, X::Vector{Vec{dim_s,T}}, h::T, ue::AbstractVector{T}, ξ::Vec{dim_s,T2}, α::Int) where {dim_s, T,T2}
    g = _calculate_g(ip, X, h, ue, ξ, α)
    a = sqrt(g⋅g)
    return a
end

function calculate_stress_recovory_variables(ip, X::Vector{Vec{dim_s,T}}, h::T, ue::AbstractVector{T}, ξ::Vec{dim_s,T2}) where {dim_s, T,T2}

    dim_p = dim_s-1

    da = [gradient((ξ)->calculate_a(ip, X, h, ue, ξ, i), ξ) for i in 1:dim_p]
    #da = Tensor{2,dim_p,T}((da[1][1], da[1][2], da[2][1], da[2][2]))

    a = [calculate_a(ip, X, h, ue, ξ, i) for i in 1:dim_p]
    #a = Tensor{1,dim_p,T}(Tuple(a))
    
    g = [_calculate_g(ip, X, h, ue, ξ, d) for d in 1:dim_p]
    ∇g₃ = [gradient((ξ)->_calculate_g(ip, X, h, ue, ξ, dim_s), ξ)[:, d] for d in 1:dim_p]
    
    E = [ _calculate_E(ip, X, ξ, d) for d in 1:dim_p]
    Dₐ = [gradient((ξ)->_calculate_D(ip, X, h, ξ), ξ)[:,d] for d in 1:dim_p]

    FI = Tensor{2,dim_p,T}((α,β)-> E[α]⋅E[β])
    FII = Tensor{2,dim_p,T}((α,β)-> E[α] ⋅ (Dₐ[β] * 2/h))

    #FI = Tensor{2,dim_p,T}((α,β)-> g[α]⋅g[β])
    #FII = Tensor{2,dim_p,T}((α,β)-> g[α] ⋅ (∇g₃[β] ))
    κ = inv(FI)⋅FII

    λ = [1+κ[i,i]*ξ[dim_s]*h/2 for i in 1:dim_p]
    #λ = Vec{dim_p,T}(Tuple(λ))
    
    return a, da, λ, κ
end

function shellgradient(f::F, v::V) where {F, V <: Union{SecondOrderTensor, Vec, Number}}
    v_dual = Tensors._load(v, Tensors.Tag(f, V))
    res = f(v_dual)
    return Tensors._extract_gradient(res, v)
end

function shellhessian(f::F, v::V) where {F, V <: Union{SecondOrderTensor, Vec, Number}}
    gradf = y -> shellgradient(f, y)
    return gradient(gradf, v)
end

function Tensors._extract_gradient(v::SymmetricTensor{2, 3, <: Tensors.Dual}, ::Vec{2})
    p1, p2, p3 = Tensors.partials(v[1,1]), Tensors.partials(v[2,1]), Tensors.partials(v[3,1])
    p4, p5, p6 = Tensors.partials(v[2,2]), Tensors.partials(v[3,2]), Tensors.partials(v[3,3])

    v1 = SymmetricTensor{2, 3}((p1[1], p2[1], p3[1], p4[1], p5[1], p6[1]))
    v2 = SymmetricTensor{2, 3}((p1[2], p2[2], p3[2], p4[2], p5[2], p6[2]))
    return v1,v2
end

function Tensors._extract_gradient((v1,v2)::NTuple{2, SymmetricTensor{2, 3, <: Tensors.Dual}}, V::Vec{2})
    v11,v12 = Tensors._extract_gradient(v1, V)
    v21,v22 = Tensors._extract_gradient(v2, V)
    return (v11, v12, v21, v22)
end

function Tensors._extract_gradient(v::Vec{3, <: Tensors.Dual}, ::Vec{2})
    p1, p2, p3 = Tensors.partials(v[1]), Tensors.partials(v[2]), Tensors.partials(v[3])
    v1 = Vec{3}((p1[1],p2[1],p3[1]))
    v2 = Vec{3}((p1[2],p2[2],p3[2]))
    return v1,v2
end

function Tensors._extract_gradient((v1,v2)::Tuple{Vec{3, <: Tensors.Dual}, Vec{3, <: Tensors.Dual}}, V::Vec{3})
    v11,v12 = Tensors._extract_gradient(v1, V)
    v21,v22 = Tensors._extract_gradient(v2, V)
    return (v11, v12, v21, v22)
end

function eval_stress((mip, oip, C), x::Vector{Vec{dim,T2}}, h::Float64, Rmat, material, uvec::Vector{Vec{3,Float64}}, ξ::Vec{2,T}, ζ) where {dim,T,T2}
    eval_stress((mip, oip, C), (x,ones(Float64,length(x))), h, Rmat, material, uvec, ξ, ζ)
end

function eval_stress((mip, oip, C), (x,w)::NurbsCoords, h::Float64, Rmat, material, uvec::Vector{Vec{3,Float64}}, ξ::Vec{2,T}, ζ) where T

    @assert(getnbasefunctions(mip) == length(oip))

    R(ξ) = begin
        B = Ferrite.value(mip,ξ)
        N = C * B
        W = sum(N.*w)
        N.*w./W
    end

    X(ξ) = sum(R(ξ).*x) 
    E(ξ) = shellgradient(ξ -> X(ξ), ξ)
    D(ξ) = begin 
        E1, E2 = E(ξ)
        cross(E1,E2)/norm(cross(E1,E2))
    end

    Dₐ = shellgradient(ξ -> D(ξ), ξ)
    E1, E2 = E(ξ)

    G = zeros(Vec{3,T}, 3)
    G[1] = E1 + ζ*h/2*Dₐ[1]
    G[2] = E2 + ζ*h/2*Dₐ[2]
    G[3] = h/2*D(ξ)

    Gᵢⱼ = SymmetricTensor{2,3}((i,j)->G[i]⋅G[j])
    Gⁱʲ = inv(Gᵢⱼ)
    Gᴵ = similar(G)
    for i in 1:3
        Gᴵ[i] = zero(Vec{3,T})
        for j in 1:3
            Gᴵ[i] += Gⁱʲ[i,j]*G[j]
        end
    end
    
    _u(ξζ::Vec{3,T}) where T = begin
        ξ = ξζ[1:2] |> Tuple |> Vec{2}
        ζ = ξζ[3] |> Tuple |> Vec{1}
        _R = R(ξ)
        I = 0
        u = zero(Vec{3,T})
        for i in 1:length(_R)
            B = Ferrite.value(oip[i], ζ)
            for j in 1:length(B)
                I += 1
                u += _R[i] * B[j] * uvec[I]
            end
        end
        return u
    end

    dudξ = gradient(x->_u(x), Vec{3,T}( (ξ[1],ξ[2],T(ζ)) ))

    g = similar(G)
    for d in 1:3
        g[d] = G[d] + dudξ[:,d]
    end

    F = zero(Tensor{2,3,T})
    for i in 1:3
        F += g[i]⊗Gᴵ[i]
    end

    ε = symmetric(F) - symmetric(one(F))
    _̂ε = symmetric(Rmat' ⋅ ɛ ⋅ Rmat)
    _,σ = Five._constitutive_driver(material, _̂ε)
    return σ
end