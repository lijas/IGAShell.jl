
struct BasisValues{dim,T,M}
    N::Matrix{T}
    dNdξ::Matrix{Vec{dim,T}}
    d²Ndξ²::Matrix{Tensor{2,dim,T,M}}
end
JuAFEM.getnquadpoints(b::BasisValues) = size(b.N,2)
JuAFEM.getnbasefunctions(b::BasisValues) = size(b.N,1)

function BasisValues{dim,T}() where {dim,T}
    d²Ndξ² = Matrix{Tensor{2,dim,T}}(undef,0,0)
    M = Tensors.n_components(Tensors.get_base(eltype(d²Ndξ²)))
    return BasisValues{dim,T,M}(Matrix{T}(undef,0,0), Matrix{Vec{dim,T}}(undef,0,0), d²Ndξ²)
end

function BasisValues(quad_rule::QuadratureRule{dim,RefCube,T}, ip::Interpolation{dim}) where {dim,T}
    nqp = length(quad_rule.weights)
    N    = fill(zero(T)      * T(NaN), getnbasefunctions(ip), nqp)
    dNdξ = fill(zero(Vec{dim,T}) * T(NaN), getnbasefunctions(ip), nqp)
    d²Ndξ² = fill(zero(Tensor{2,dim,T}) * T(NaN), getnbasefunctions(ip), nqp)
    for (qp, ξ) in enumerate(quad_rule.points)
        for i in 1:getnbasefunctions(ip)
            d²Ndξ²[i, qp], dNdξ[i, qp], N[i, qp] = hessian(ξ -> JuAFEM.value(ip, i, ξ), ξ, :all)
        end
    end
    M = Tensors.n_components(Tensors.get_base(eltype(d²Ndξ²)))
    return BasisValues{dim,T,M}(N,dNdξ,d²Ndξ²)
end

function JuAFEM.function_value(fe_v::BasisValues{dim_p}, q_point::Int, u::AbstractVector{T2}, dof_order::AbstractVector{Int} = collect(1:length(u))) where {dim_p,T2}
    n_base_funcs = getnbasefunctions(fe_v)
    @assert length(dof_order) == n_base_funcs
    @boundscheck checkbounds(u, dof_order)
    val = zero(T2)
    @inbounds for (i, j) in enumerate(dof_order)
        val += fe_v.N[i,q_point] * u[j]
    end
    return val 
end

function JuAFEM.function_derivative(fe_v::BasisValues{dim_p}, q_point::Int, u::AbstractVector{T2}, dof_order::AbstractVector{Int} = collect(1:length(u))) where {dim_p,T2}
    n_base_funcs = getnbasefunctions(fe_v)
    @assert length(dof_order) == n_base_funcs
    @boundscheck checkbounds(u, dof_order)
    dudξ = zeros(T2,dim_p)
    @inbounds for (i, j) in enumerate(dof_order)
        for d in 1:dim_p
            dudξ[d] += fe_v.dNdξ[i,q_point][d] * u[j]
        end
    end
    return dudξ
end


function function_second_derivative(fe_v::BasisValues{dim_p}, q_point::Int, u::AbstractVector{T2}, dof_order::AbstractVector{Int} = collect(1:length(u))) where {dim_p,T2}
    n_base_funcs = getnbasefunctions(fe_v)
    @assert length(dof_order) == n_base_funcs
    @boundscheck checkbounds(u, dof_order)
    d²udξ² = zeros(T2,dim_p,dim_p)
    @inbounds for (i, j) in enumerate(dof_order)
        for d1 in 1:dim_p, d2 in 1:dim_p
            d²udξ²[d1,d2] += fe_v.d²Ndξ²[i,q_point][d1,d2] * u[j]
        end
    end
    return d²udξ²
end


"""

"""
struct Triad{dim,T}
    triad::MVector{dim,Vec{dim,T}}
end

function Base.zero(::Type{Triad{dim,T}})  where {dim,T}
    return Triad{dim,T}(zeros(MVector{dim,Vec{dim,T}}))
end

function Base.getindex(t::Triad{dim,T}, n::Int)  where {dim,T}
    return t.triad[n]
end

function Base.setindex!(t::Triad{dim,T}, v::Vec{dim,T}, n::Int) where {dim,T}
    t.triad[n] = v
end

Base.iterate(I::Triad{dim}, state::Int=1) where {dim} = (state==dim+1) ?  nothing : (I[state], state+1)

"""

"""

struct IGAShellValues{dim_s,dim_p,T<:Real,M,M2} 
    
    #
    inplane_values_nurbs::BasisValues{dim_p,T,M}
    inplane_values_bezier::BasisValues{dim_p,T,M}
    
    #
    H::Vector{BasisValues{1,T,1}}

    detJdA::Vector{T}
    detJdV::Vector{T}

    G  ::Vector{  Triad{dim_s,T}   }  #iqp, dim
    Gᴵ ::Vector{  Triad{dim_s,T}   }  #iqp, dim
    Eₐ ::Vector{  Triad{dim_s,T}   }  #iqp, dim
    Dₐ ::Vector{  Triad{dim_s,T}   }  #iqp, dim x dim
    κ  ::Vector{ Tensor{2,dim_p,T,M}}
    κᵐ ::Vector{ Tensor{2,dim_p,T,M}}
    R  ::Vector{  Tensor{2,dim_s,T,M2}}
    
    N::Matrix{  Vec{dim_s,T}  }
    dNdξ::Matrix{  Tensor{2,dim_s,T,M2}  }
    #d²Udξ²::Matrix{  Array{T,3}  } #Tensor{2,dim_s,T,M2}
    #_N::Matrix{  Vec{dim_s,T}  } #Bezier values
    #_dNdξ::Matrix{  Tensor{2,dim_s,T,M2}  } #Bezier values
    #_d²Udξ²::Matrix{  Array{T,3}  } 

    nbasisfunctions::Base.RefValue{Int}
    current_bezier_operator::Base.RefValue{IGA.BezierExtractionOperator{T}}

    qr::QuadratureRule{dim_s,RefCube,T}
    iqr::QuadratureRule{dim_p,RefCube,T}
    oqr::QuadratureRule{1,RefCube,T}
    
    thickness::T

    inp_ip::Interpolation
    oop_ip::Vector{Interpolation}
end

getnquadpoints_ooplane(cv::IGAShellValues) = return length(cv.oqr.weights)
getnquadpoints_inplane(cv::IGAShellValues) = return length(cv.iqr.weights)
JuAFEM.getnquadpoints(cv::IGAShellValues) = return getnquadpoints_ooplane(cv)*getnquadpoints_inplane(cv)

getnbasefunctions_inplane(cv::IGAShellValues) = return size(cv.inplane_values_bezier.N, 1)
getnbasefunctions_ooplane(cv::IGAShellValues, i::Int) = return getnbasefunctions(cv.H[i])
JuAFEM.getnbasefunctions(cv::IGAShellValues) = return cv.nbasisfunctions[]

JuAFEM.getdetJdV(cv::IGAShellValues, qp::Int) = cv.detJdV[qp]

function getdetJdA(cv::IGAShellValues, qp::Int, idx::EdgeIndex)
    _,edgeid = idx
    edgeid==1 && return norm(cross(cv.G[qp][1], cv.G[qp][3]))*cv.qr.weights[qp]
    edgeid==2 && return norm(cross(cv.G[qp][2], cv.G[qp][3]))*cv.qr.weights[qp]
    edgeid==3 && return norm(cross(cv.G[qp][3], cv.G[qp][1]))*cv.qr.weights[qp]
    edgeid==4 && return norm(cross(cv.G[qp][3], cv.G[qp][2]))*cv.qr.weights[qp]
    error("Edge not found")
end

function getdetJdA(cv::IGAShellValues{3}, qp::Int, idx::EdgeInterfaceIndex)
    _,edgeid,face = idx
    edgeid==1 && return norm(cv.G[qp][1])*cv.qr.weights[qp]
    edgeid==2 && return norm(cv.G[qp][2])*cv.qr.weights[qp]
    edgeid==3 && return norm(cv.G[qp][1])*cv.qr.weights[qp]
    edgeid==4 && return norm(cv.G[qp][2])*cv.qr.weights[qp]
    error("Interface not found")
end

function getdetJdA(cv::IGAShellValues{2}, qp::Int, idx::VertexIndex)
    return norm(cv.G[qp][2])*cv.qr.weights[qp]
end

function getdetJdA(cv::IGAShellValues, qp::Int, idx::VertexInterfaceIndex)
    return 1.0
end

#getdetJdA(cv::IGAShellValues, qp::Int) = cv.detJdA[qp]
function getdetJdA(cv::IGAShellValues, qp::Int, idx::FaceIndex)
    _,faceid = idx
    faceid==1 && return norm(cross(cv.G[qp][1], cv.G[qp][2]))*cv.qr.weights[qp]
    faceid==2 && return norm(cross(cv.G[qp][1], cv.G[qp][2]))*cv.qr.weights[qp]
    error("Face not found")
end

function getdetJdA(cv::IGAShellValues{2,1}, qp::Int, idx::FaceIndex)
    _,faceid = idx
    faceid==1 && return norm(cv.G[qp][1]) * cv.qr.weights[qp]
    faceid==2 && return norm(cv.G[qp][1]) * cv.qr.weights[qp]
    error("Face not found")
end

function set_quadraturerule!(cv::IGAShellValues{dim_s,dim_p,T}, qr::QuadratureRule{1,RefCube}) where {dim_s,dim_p,T}
    @assert length(qr.points) == length(cv.oqr.points)
    cv.oqr.points .= qr.points
    cv.oqr.weights .= qr.weights
    
    qp = 0
    for oqp in 1:length(cv.oqr.points)
        for iqp in 1:length(cv.iqr.points)
            qp+=1
            local p
            if dim_s == 3; p = (cv.iqr.points[iqp][1], cv.iqr.points[iqp][2], cv.oqr.points[oqp][1]);
            elseif dim_s==2; p = (cv.iqr.points[iqp][1], cv.oqr.points[oqp][1]);
            end
            cv.qr.points[qp] = Vec{dim_s,T}(p)
            cv.qr.weights[qp] = cv.iqr.weights[iqp]*cv.oqr.weights[oqp]
        end
    end
end

function IGA.set_bezier_operator!(cv::IGAShellValues{dim_s,dim_p,T}, C::IGA.BezierExtractionOperator{T}) where {dim_s,dim_p,T}
    cv.current_bezier_operator[] = C
    return nothing
end

function set_inp_basefunctions!(cv::IGAShellValues{dim_s,dim_p,T}, inplane_basisvalues::BasisValues{dim_p,T,M}) where {dim_s,dim_p,T,M}
    
    @assert getnbasefunctions(inplane_basisvalues) == getnbasefunctions_inplane(cv)
    @assert getnquadpoints(inplane_basisvalues) == getnquadpoints_inplane(cv)

    cv.N .= inplane_basisvalues.N
    cv.dNdξ .= inplane_basisvalues.dNdξ
    cv.dN²dξ² .= inplane_basisvalues.d²Ndξ²

end

function set_oop_basefunctions!(cv::IGAShellValues{dim_s,dim_p,T}, ooplane_basisvalues::BasisValues{1,T,1}) where {dim_s,dim_p,T}
    set_oop_basefunctions!(cv, [ooplane_basisvalues for i in 1:getnbasefunctions_inplane(cv)])
end

function set_oop_basefunctions!(cv::IGAShellValues{dim_s,dim_p,T}, ooplane_basisvalues::Vector{BasisValues{1,T,1}}) where {dim_s,dim_p,T}
    
    @assert length(ooplane_basisvalues) == getnbasefunctions_inplane(cv)
    nbasefunctions_ooplane =sum(getnbasefunctions.(ooplane_basisvalues))::Int 
    _nquadpoints_ooplane = getnquadpoints(ooplane_basisvalues[1]) 

    @assert _nquadpoints_ooplane == getnquadpoints_ooplane(cv)
    
    for i in 1:length(ooplane_basisvalues)
        cv.H[i] = ooplane_basisvalues[i];
    end

    #_build_basefunctions!(cv)

end

function _inplane_nurbs_bezier_extraction(cv::IGAShellValues{dim_s,dim_p,T}, C::IGA.BezierExtractionOperator{T}) where {dim_s,dim_p,T}
    dBdξ   = cv.inplane_values_bezier.dNdξ
    B      = cv.inplane_values_bezier.N
    
    for iq in 1:getnquadpoints_inplane(cv)
        for ib in 1:getngeombasefunctions_inplane(cv)
            
            cv.inplane_values_nurbs.N[ib, iq] = zero(eltype(cv.inplane_values_nurbs.N))
            cv.inplane_values_nurbs.dNdξ[ib, iq] = zero(eltype(cv.inplane_values_nurbs.dNdξ))

            C_ib = C[ib]
            
            for (i, nz_ind) in enumerate(C_ib.nzind)                
                val = C_ib.nzval[i]
                cv.inplane_values_nurbs.N[ib, iq]    += val*   B[nz_ind, iq]
                cv.inplane_values_nurbs.dNdξ[ib, iq] += val*dBdξ[nz_ind, iq]
            end
        end
    end
end


function _build_shape_values!(cv::IGAShellValues{dim_s,dim_p,T}) where {dim_s,dim_p,T}
    
    #
    U_temp = 0.0
    dUdξ_temp = fill(0.0, dim_s)
    #d²Udξ²_temp = fill(0.0, dim_s,dim_s)

    #
    B_comp = fill(0.0, dim_s)
    dB_comp = fill(0.0, dim_s, dim_s)

    qp = 0
    basefunc_count = 0
    for oqp in 1:getnquadpoints_ooplane(cv)
        for iqp in 1:getnquadpoints_inplane(cv)
            qp +=1
            basefunc_count = 0
            for i in 1:getnbasefunctions_inplane(cv)
                ooplane_basis = cv.H[i]

                #Out of plane
                H = ooplane_basis.N
                dHdζ = ooplane_basis.dNdξ

                #Inplane
                S = cv.inplane_values_nurbs.N[i,iqp]
                dSdξ = cv.inplane_values_nurbs.dNdξ[i,iqp]

                for j in 1:getnbasefunctions(ooplane_basis)    
                    Hj = H[j,oqp]
                    dHj = dHdζ[j,oqp][1]

                    U_temp = S * Hj

                    for d1 in 1:dim_p
                        dUdξ_temp[d1] = dSdξ[d1] * Hj
                    end
                    dUdξ_temp[dim_s] = S * dHj

                    for comp in 1:dim_s
                        basefunc_count += 1

                        #Nurbs
                        fill!(B_comp, 0.0)
                        @inbounds B_comp[comp] = U_temp
                        @inbounds cv.N[basefunc_count, qp] = Vec{dim_s,T}(NTuple{dim_s,T}(B_comp))

                        fill!(dB_comp, 0.0)
                        @inbounds dB_comp[comp, :] = dUdξ_temp
                        @inbounds cv.dNdξ[basefunc_count, qp] = Tensor{2,dim_s,T,dim_s^2}(NTuple{dim_s^2,T}(dB_comp))

                    end
                end
            end
        end
    end
    cv.nbasisfunctions[] = basefunc_count

end

function _reinit_layer!(cv::IGAShellValues{dim_s,dim_p,T}, qp_indx) where {dim_s,dim_p,T}

    qp, iqp, oqp = qp_indx
    
    Eₐ = cv.Eₐ[iqp]
    Dₐ = cv.Dₐ[iqp]
    D = cv.Eₐ[iqp][dim_s]

    G = cv.G[qp]
    Gᴵ = cv.Gᴵ[qp]

    ζ = cv.oqr.points[oqp]

    #Covarient matrix
    for d in 1:dim_p
        G[d] = Eₐ[d] + ζ[1]*Dₐ[d]
    end
    G[dim_s] = D

    detJ = dim_s == 3 ? norm((cross(G[1], G[2]))) : norm(cross(G[1]))
    cv.detJdA[qp] = detJ*cv.qr.weights[qp]
    cv.detJdV[qp] = detJ*cv.iqr.weights[iqp]*cv.oqr.weights[oqp]*0.5*cv.thickness
    
    Gⁱʲ = inv(SymmetricTensor{2,dim_s,T}((i,j)-> G[i]⋅G[j]))

    #Contravarient matrix
    for i in 1:dim_s
        Gᴵ[i] = zero(Vec{dim_s,T})
        for j in 1:dim_s
            Gᴵ[i] += Gⁱʲ[i,j]*G[j]
        end
    end
    
    FI = Tensor{2,dim_p,T}((α,β)-> G[α]⋅G[β])
    FII = Tensor{2,dim_p,T}((α,β)-> G[α] ⋅ (cv.Dₐ[iqp][β] * 2/cv.thickness))
    cv.κ[qp] = inv(FI)⋅FII

    cv.R[qp] = calculate_R(G...)
end

function calculate_R(g1::Vec{3,T},g2::Vec{3,T}, ::Vec{3,T}) where {T}
    e3 = cross(g1, g2)
    e3 /= norm(e3)

    e1 = g1/norm(g1)
    e2 = cross(e3,e1) #order?

    _R = hcat(e1, e2, e3)
    return Tensor{2,3,T}(_R)
end

function calculate_R(g1::Vec{2,T}, ::Vec{2,T}) where {T}
    e2 = Vec{2,T}((-g1[2], g1[1]))
    e2 /= norm(e2)
    e1 = g1/norm(g1)
    return Tensor{2,2,T,2^2}((e1[1], e1[2], e2[1], e2[2]))
end

function _reinit_midsurface!(cv::IGAShellValues{dim_s,dim_p,T}, iqp::Int, coords) where {dim_s,dim_p,T}

    Eₐₐ = zeros(Vec{dim_s,T}, dim_p, dim_p)
    Eₐ = cv.Eₐ[iqp]

    for d1 in 1:dim_p
        Eₐ[d1] = shape_parent_derivative(cv, iqp, coords, d1)
        for d2 in 1:dim_p
            Eₐₐ[d1,d2] = shape_parent_second_derivative(cv, iqp, coords, (d1,d2))
        end
    end

    #The derivatives of the director-vector must be treated differently in 2d/3d
    if dim_s == 3
        a = cross(Eₐ[1], Eₐ[2])
        da1 = cross(Eₐₐ[1,1], Eₐ[2]) + cross(Eₐ[1], Eₐₐ[2,1])
        da2 = cross(Eₐₐ[1,2], Eₐ[2]) + cross(Eₐ[1], Eₐₐ[2,2])
        c1 = a/norm(a) ⋅ da1
        c2 = a/norm(a) ⋅ da2
        Eₐ[dim_s]  = 0.5cv.thickness*a/norm(a)
        cv.Dₐ[iqp][1] = 0.5cv.thickness*(da1*norm(a) - a*c1)/(a⋅a) 
        cv.Dₐ[iqp][2] = 0.5cv.thickness*(da2*norm(a) - a*c2)/(a⋅a) 
    elseif dim_s == 2
        scew = Tensor{2,dim_s,T}((0.0, 1.0, -1.0, 0.0))
        a = scew ⋅ cv.Eₐ[iqp][1]
        da1 = scew ⋅ Eₐₐ[1,1]
        Eₐ[dim_s] = 0.5cv.thickness * (a/norm(a))
        c1 = a/norm(a) ⋅ da1
        cv.Dₐ[iqp][1] = 0.5cv.thickness * (da1*norm(a) - a*c1)/(a⋅a)
    end

    FI = Tensor{2,dim_p,T}((α,β)-> Eₐ[α]⋅Eₐ[β])
    FII = Tensor{2,dim_p,T}((α,β)-> Eₐ[α] ⋅ (cv.Dₐ[iqp][β] * 2/cv.thickness))
    cv.κᵐ[iqp] = inv(FI)⋅FII

end

function reinit_midsurface!(cv::IGAShellValues, coords::Vector{Vec{dim_s,T}}) where {dim_s,T}

    qp = 0
    for iqp in 1:getnquadpoints_inplane(cv)
        _reinit_midsurface!(cv, iqp, coords)
    end
    
    _inplane_nurbs_bezier_extraction(cv, cv.current_bezier_operator[])
    
    for oqp in 1:getnquadpoints_ooplane(cv)
        for iqp in 1:getnquadpoints_inplane(cv)
            qp+=1
            _reinit_layer!(cv, (qp,iqp,oqp))
        end
    end

    _build_shape_values!(cv)#, ilay)
    
end

function calculate_g(cv::IGAShellValues{dim_s,dim_p,T}, qp, ue) where {dim_s,dim_p,T}

    g = zeros(Vec{dim_s,T}, dim_s)
    for d in 1:dim_s
        g[d] = cv.G[qp][d] + function_parent_derivative(cv, qp, ue, d)
    end
    return g
end

function calculate_F!(F::Tensor{2}, δF::Vector{Tensor{2}}, g::Vector{Vec{dim_s}}, cv::IGAShellValues{dim_s,dim_p,T}, qp, ue) where {dim_s,dim_p,T}
    for i in 1:dim_s
        F += g[i]⊗cv.Gᴵ[qp][i]
        for j in 1:ndofs
            #Extract the i:th derivative wrt to parent coords \xi, \eta, \zeta
            δg = basis_parent_derivative(cv, qp, j, i)

            δF[j] += δg⊗cv.Gᴵ[qp][i]
        end
    end
end

function get_qp_coord(cv, qp)
    return cv.qr.points[qp]
end


get_qp_weight(cv::IGAShellValues, qp::Int) = cv.qr.weights[qp]
get_oop_qp_weight(cv::IGAShellValues, oqp::Int) = cv.oqr.weights[oqp]
get_iop_qp_weight(cv::IGAShellValues, iqp::Int) = cv.iqr.weights[iqp]

function IGAShellValues(thickness::T, qr_inplane::QuadratureRule{dim_p}, qr_ooplane::QuadratureRule{1}, mid_ip::Interpolation, sizehint::Int = 10, oop_ip::Vector{<:Interpolation}=Interpolation[]) where {dim_p,T}
    
    n_oop_qp = length(getweights(qr_ooplane))
    n_inp_qp = length(getweights(qr_inplane))
    nqp = n_inp_qp*n_oop_qp

    n_midplane_basefuncs = getnbasefunctions(mid_ip)
    dim_s = dim_p+1
    @assert JuAFEM.getdim(mid_ip) == dim_p

    # Function interpolation
    inplane_values_bezier = BasisValues(qr_inplane, mid_ip)
    
    H = [BasisValues{1,T}() for _ in 1:n_midplane_basefuncs]

    G  = [zero(Triad{dim_s,T}) for _ in 1:nqp]
    Gᴵ = [zero(Triad{dim_s,T}) for _ in 1:nqp]
    R =  [zero(Tensor{2,dim_s,T}) for _ in 1:nqp]
    κ =  [zero(Tensor{2,dim_p,T}) for _ in 1:nqp]
    κᵐ = [zero(Tensor{2,dim_p,T}) for _ in 1:n_inp_qp]
    Eₐ = [zero(Triad{dim_s,T}) for _ in 1:n_inp_qp]
    Dₐ = [zero(Triad{dim_s,T}) for _ in 1:n_inp_qp]

    max_nbasefunctions = n_midplane_basefuncs*dim_s * sizehint #hardcoded
    U = fill(zero(Tensor{1,dim_s,T}) * T(NaN), max_nbasefunctions, nqp)
    dUdξ = fill(zero(Tensor{2,dim_s,T}) * T(NaN), max_nbasefunctions, nqp)
    d²Udξ² = fill(zeros(dim_s,dim_s,dim_s) * T(NaN), max_nbasefunctions, nqp) 

    detJdV = fill(T(NaN), nqp)
    detJdA = fill(T(NaN), nqp)

    MM1 = Tensors.n_components(Tensors.get_base(eltype(κ)))
    MM2 = Tensors.n_components(Tensors.get_base(eltype(dUdξ)))

    #combine the two quadrature rules
    points = Vec{dim_s,T}[]
    weights = T[]
    for oqp in 1:n_oop_qp
        for iqp in 1:n_inp_qp
            _p = [qr_inplane.points[iqp]..., qr_ooplane.points[oqp]...]
            _w = qr_inplane.weights[iqp]*qr_ooplane.weights[oqp]
            push!(points, Vec{dim_s,T}((_p...,)))
            push!(weights, _w)
        end
    end
    qr = QuadratureRule{dim_s,RefCube,T}(weights, points)

    #Initalize bezier operator as NaN
    bezier_operator = IGA.bezier_extraction_to_vector(sparse(Diagonal(fill(NaN, n_midplane_basefuncs))))

    return IGAShellValues{dim_s,dim_p,T,MM1,MM2}(inplane_values_bezier, deepcopy(inplane_values_bezier), H, detJdV, detJdA, G, Gᴵ, Eₐ, Dₐ, κ, κᵐ, R, U, dUdξ, Ref(max_nbasefunctions), Ref(bezier_operator), qr, 
                                                 deepcopy(qr_inplane), deepcopy(qr_ooplane), thickness, mid_ip, oop_ip)
end

function function_parent_derivative(cv::IGAShellValues{dim_s,dim_p,T}, qp::Int, ue::AbstractVector{T}, Θ::Int, active_dofs::AbstractVector{Int} = 1:length(ue)) where {dim_s,dim_p,T}
    n_base_funcs = getnbasefunctions(cv)
    grad = zero(Vec{dim_s,T})
    @assert(length(ue) == length(active_dofs))
    @inbounds for (i,j) in enumerate(active_dofs)
        grad += cv.dNdξ[j,qp][:,Θ] * ue[i]
    end
    return grad
end

function JuAFEM.function_value(cv::IGAShellValues{dim_s,dim_p,T}, qp::Int, ue::AbstractVector{T}, active_dofs::AbstractVector{Int} = 1:length(ue)) where {dim_s,dim_p,T}
    val = zero(Vec{dim_s,T})
    @assert(length(ue) == length(active_dofs))
    @inbounds for (i,j) in enumerate(active_dofs)
        val += cv.N[i,qp] * ue[i]
    end
    return val
end

function JuAFEM.shape_value(cv::IGAShellValues{dim_s,dim_p,T}, qp::Int, ue::AbstractVector{Vec{dim_s,T}}, active_dofs::AbstractVector{Int} = 1:length(ue)) where {dim_s,dim_p,T}
    val = zero(Vec{dim_s,T})
    @assert(length(ue) == length(active_dofs))
    @inbounds for (i,j) in enumerate(active_dofs)
        val += cv.inplane_values_bezier.N[j,qp] * ue[i]
    end
    return val
end

function shape_parent_derivative(cv::IGAShellValues{dim_s,dim_p,T}, qp::Int, ue::AbstractVector{Vec{dim_s,T}}, Θ::Int, active_dofs::AbstractVector{Int} = 1:length(ue)) where {dim_s,dim_p,T}
    grad = zero(Vec{dim_s,T})
    @assert(length(ue) == length(active_dofs))
    @inbounds for (i,j) in enumerate(active_dofs)
        grad += cv.inplane_values_bezier.dNdξ[j,qp][Θ] * ue[i]
    end
    return grad
end

function shape_parent_second_derivative(cv::IGAShellValues{dim_s,dim_p,T}, qp::Int, ue::AbstractVector{Vec{dim_s,T}}, Θ::Tuple{Int,Int}, active_dofs::AbstractVector{Int} = 1:length(ue)) where {dim_s,dim_p,T}
    grad = zero(Vec{dim_s,T})
    @assert(length(ue) == length(active_dofs))
    @inbounds for (i,j) in enumerate(active_dofs)
        grad += cv.inplane_values_bezier.dN²dξ²[j,qp][Θ[1],Θ[2]] * ue[i]
    end
    return grad
end

function JuAFEM.spatial_coordinate(cv::IGAShellValues{dim_s,dim_p,T}, qp::Int, x::AbstractVector{Vec{dim_s,T}}) where {dim_s,dim_p,T}
    i2s = CartesianIndices((getnquadpoints_inplane(cv), getnquadpoints_ooplane(cv)))
    iqp, oqp = Tuple(i2s[qp])
    D = cv.Eₐ[iqp][dim_s]

    Xᴹ = shape_value(cv, iqp, x)
    return Xᴹ + cv.oqr.points[oqp][1]*D
end

function basis_parent_derivative(cv::IGAShellValues{dim_s,dim_p,T}, qp::Int, i::Int, Θ::Int) where {dim_s,dim_p,T}
    return cv.dNdξ[i,qp][:,Θ]
end

function basis_value(cv::IGAShellValues{dim_s,dim_p,T}, qp::Int, i::Int) where {dim_s,dim_p,T}
    return cv.N[i,qp]
end