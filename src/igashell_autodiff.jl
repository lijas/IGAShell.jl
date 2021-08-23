
function _calculate_X₀(msip::Interpolation, coords::Vector{Vec{dim_s,Float64}}, ξ::Vec{dim_s,T})::Vec{dim_s,T} where {dim_s,T}
    dim_p = dim_s-1
    ξ2 = Vec{dim_p,T}(i->ξ[i])

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

function _calculate_D(msip::Interpolation, coords::Vector{Vec{2,T2}}, h::T2, ξ::Vec{2,T}) where {T,T2}
    
    E = gradient((ξ) -> _calculate_X₀(msip, coords, ξ), ξ)

    D = cross(E...)
    D /= norm(D)
    D *= 0.5h
    return D
end

function _calculate_D(msip::Interpolation, coords::Vector{Vec{3,T2}}, h::T2, ξ::Vec{3,T}) where {T,T2}
    
    _E = gradient((ξ) -> _calculate_X₀(msip, coords, ξ), ξ)
    E1 = _E[:,1]
    E2 = _E[:,2]

    D = Tensors.cross(E1,E2)
    D /= norm(D)
    D *= 0.5h
    return D
end

function _calculate_X(msip::Interpolation, coords::Vector{Vec{dim_s,T2}}, h::T2, ξ::Vec{dim_s,T})::Vec{dim_s,T} where {dim_s,T,T2}
    X₀ = _calculate_X₀(msip, coords, ξ)
    D = _calculate_D(msip, coords, h, ξ)
    ζ = ξ[dim_s]
    return X₀ + ζ*D
end

function _calculate_x(ip::Interpolation, coords::Vector{Vec{dim_s,T}}, h::T, ue::AbstractVector{T}, ξ::Vec{dim_s,T2})::Vec{dim_s,T2} where {dim_s,T,T2}
    X = _calculate_X(ip, coords, h, ξ)
    u = _calculate_u(ip, ue, ξ)
    return X + u
end

function _calculate_x(ip::Tuple{Interpolation, Vector{Interpolation}, IGA.BezierExtractionOperator}, coords::Vector{Vec{dim_s,T}}, h::T, ue::AbstractVector{T}, ξ::Vec{dim_s,T2})::Vec{dim_s,T2} where {dim_s,T,T2}
    X = _calculate_X(ip[1], coords, h, ξ)
    u = _calculate_u(ip, ue, ξ)
    return X + u
end

function _calculate_u(ip::Tuple{Interpolation, Vector{Interpolation}, IGA.BezierExtractionOperator}, ue::AbstractVector{T}, ξ::Vec{dim_s,T2})::Vec{dim_s,T2} where {dim_s,T,T2}
    
    msip = ip[1]
    opip = ip[2]
    Ce = ip[3]

    dim_p = dim_s-1

    ξ2 = Vec{dim_p,T2}(i->ξ[i])
    ζ = Vec{1,T2}((ξ[3],))

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

function _calculate_u(ip::Interpolation, ue::AbstractVector{T}, ξ::Vec{dim_s,T2})::Vec{dim_s,T2} where {dim_s,T,T2}
    
    u = zero(Vec{dim_s,T2})
    ξ2 = Vec{dim_s-1,T2}(i->ξ[i])

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


function calculate_a(ip, X::Vector{Vec{dim_s,T}}, h::T, ue::AbstractVector{T}, ξ::Vec{dim_s,T2}, α::Int)::T2 where {dim_s, T,T2}
    g = _calculate_g(ip, X, h, ue, ξ, α)
    a = sqrt(g⋅g)
    return a
end

function calculate_stress_recovory_variables!(a, da, ip, X::Vector{Vec{dim_s,T}}, h::T, ue::AbstractVector{T}, ξ::Vec{dim_s,T2}) where {dim_s, T,T2}

    dim_p = dim_s-1

    for i in 1:dim_p
        da[i] = gradient((ξ)->calculate_a(ip, X, h, ue, ξ, i), ξ)   
    end

    for i in 1:dim_p
        a[i] = calculate_a(ip, X, h, ue, ξ, i)
    end
    #=
    for i in 1:dim_p
        g[i] = _calculate_g(ip, X, h, ue, ξ, d)
        #∇g₃ = gradient((ξ)->_calculate_g(ip, X, h, ue, ξ, dim_s), ξ)[:, i]
        E[i] = _calculate_E(ip, X, ξ, i)
        Dₐ[i] = gradient((ξ)->_calculate_D(ip, X, h, ξ), ξ)[:,i]
    end
    
    FI = Tensor{2,dim_p,T,dim_p^2}((α,β)-> E[α]⋅E[β])
    FII = Tensor{2,dim_p,T,dim_p^2}((α,β)-> E[α] ⋅ (Dₐ[β] * 2/h))

    κ = inv(FI)⋅FII

    for i in 1:dim_p
        λ[i] = 1+κ[i,i]*ξ[dim_s]*h/2    
    end
    =#
    return a, da#, λ, κ
end