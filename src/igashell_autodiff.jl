
function _calculate_X₀(msip::Interpolation, coords::Vector{Vec{dim_s,T2}}, ξ::Vec{dim_s,T})where {dim_s,T,T2}
    dim_p = dim_s-1
    ξ2 = Vec((ξ[1:dim_p]...,))

    x = zero(Vec{dim_s,T})
    for i in 1:getnbasefunctions(msip)
        N = JuAFEM.value(msip,i,ξ2)
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
    B = [JuAFEM.value(msip, i, ξ2) for i in 1:getnbasefunctions(msip)]

    for i in 1:getnbasefunctions(msip)
        S = sum(Ce[i] .* B)
        for j in 1:getnbasefunctions(opip[i])
            H = JuAFEM.value(opip[i], j, ζ)
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
        N = JuAFEM.value(ip, i, ξ2)
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