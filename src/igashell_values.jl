
"""
    BasisValues
Stores the values N and derivatives dNdξ for a interpolation given a quadrature rule
"""
struct BasisValues{dim,T,M}
    N::Matrix{T}
    dNdξ::Matrix{Vec{dim,T}}
    d²Ndξ²::Matrix{Tensor{2,dim,T,M}}
end
JuAFEM.getnquadpoints(b::BasisValues) = size(b.N,2)
JuAFEM.getnbasefunctions(b::BasisValues) = size(b.N,1)

function BasisValues{dim,T}(nqp::Int = 0, nb::Int = 0) where {dim,T}
    d²Ndξ² = zeros(Tensor{2,dim,T},nb,nqp) .* NaN
    M = Tensors.n_components(Tensors.get_base(eltype(d²Ndξ²)))
    return BasisValues{dim,T,M}(zeros(T,nb,nqp) * NaN ,zeros(Vec{dim,T},nb,nqp) * NaN, d²Ndξ²) 
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
    OOPBasisValues - Short hand for Vector{BasisValues{1,T,1}}
For each layer, given a quadrature rule, stores the non-zero shape functions of a BSpline interpolation
"""
const OOPBasisValues{T} = Vector{BasisValues{1,T,1}}

function OOPBasisValues(oop_qr::LayerQuadratureRule{1,T}, oop_ip::IGA.BSplineBasis{1,T,order}) where {T,order}
    
    # Evaluate the out-of-plane interpolation at out-of-plane quadrature points,
    # but also filter out basis values which are zero
    r = order[1]
    nlay = nlayers(oop_qr)
    basis = BasisValues{1,T}[]
    for ilay in 1:nlay
        qr = oop_qr.qrs[ilay]
        nqp_layer = length(qr.weights)
        b = BasisValues{1,T}(nqp_layer,r+1)
        for qp in 1:nqp_layer
            ζ = qr.points[qp]
            index = IGA.findspan(oop_ip, ζ[1])
            for i in 0:r
                _dHdζ, _H = gradient(ζ -> JuAFEM.value(oop_ip, index+i-r, ζ), ζ, :all)
                
                b.N[i+1, qp] = _H
                b.dNdξ[i+1, qp] = _dHdζ
            end
        end 
        push!(basis, b)
    end   

    return basis
end


"""
    Triad
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




struct IGAShellValues{dim_s,dim_p,T<:Real,M,M2,M3} 

    #Inplane basefunctions, S(ξ, η)
    inplane_values_nurbs::BasisValues{dim_p,T,M}
    inplane_values_bezier::BasisValues{dim_p,T,M}

    #The values of the out-of-plane function values, H(ζ),
    # stored in a matrix (controlpoint × layer).
    ooplane_values::Matrix{ BasisValues{1,T,1} }  

    detJdA::Vector{T}
    detJdV::Vector{T}

    G  ::Vector{  Triad{dim_s,T}   }  #iqp, dim
    Gᴵ ::Vector{  Triad{dim_s,T}   }  #iqp, dim
    Eₐ ::Vector{  Triad{dim_s,T}   }  #iqp, dim
    Dₐ ::Vector{  Triad{dim_s,T}   }  #iqp, dim x dim
    κ  ::Vector{ Tensor{2,dim_p,T,M}}
    κᵐ ::Vector{ Tensor{2,dim_p,T,M}}
    R  ::Vector{  Tensor{2,dim_s,T,M2}}
    
    N::Matrix{ Vec{dim_s,T} }
    dNdξ::Matrix{  Tensor{2,dim_s,T,M3} }
    _N::Matrix{  Vec{dim_s,T}  } 
    _dNdξ::Matrix{  Tensor{2,dim_s,T,M2}  } 

    ngeombasisfunctions_per_layer::Int
    oop_order::Int
    nbasisfunctions::Base.RefValue{Int}
    thickness::T
    current_layer::Base.RefValue{Int}
    current_bezier_operator::Base.RefValue{IGA.BezierExtractionOperator{T}}

    #The quadrature rule inplane
    iqr::QuadratureRule{dim_p,RefCube,T}
    #The quadrature rule out-of-plane, in each layer
    oqr::LayerQuadratureRule{1,T}
    qr::LayerQuadratureRule{dim_s,T}
    
    inp_ip::Interpolation
    oop_ip::Vector{Interpolation}
end

function IGAShellValues(thickness::T, qr_inplane::QuadratureRule{dim_p}, qr_ooplane::LayerQuadratureRule{1}, mid_ip::Interpolation, oop_ip::IGA.BSplineBasis{1,T,order}) where {dim_p,T,order}
    
    r = order[1]
    nlay = nlayers(qr_ooplane)
    nqp_oop = getnweights(qr_ooplane)
    nqp_inp = length(getweights(qr_inplane))
    nqp_oop_per_layer = nqp_oop ÷ nlay == 0 ? 1 : nqp_oop ÷ nlay #Special case for quadrature rules on the top/bottom face 
    nqp_per_layer = nqp_inp*nqp_oop_per_layer
    dim_s = dim_p+1
    
    n_midplane_basefuncs = getnbasefunctions(mid_ip)
    n_ooplane_basefuncs_per_layer = r+1
    ngeombasefunctions_per_layer = n_ooplane_basefuncs_per_layer * n_midplane_basefuncs
    nbasefunctions_per_layer = ngeombasefunctions_per_layer * dim_s

    @assert JuAFEM.getdim(mid_ip) == dim_p
    
    # Inplane shape values
    inplane_values_bezier = BasisValues(qr_inplane, mid_ip)
    
    #Out of plane shape values
    ooplane_values = [BasisValues{1,T}() for i in 1:n_midplane_basefuncs, j in 1:nlay]
    for ib in 1:n_midplane_basefuncs
        basis = OOPBasisValues(qr_ooplane, oop_ip)
        for ilay in 1:nlay
            nqp_layer = length(qr_ooplane.qrs[ilay].weights)
            b = BasisValues{1,T}(nqp_layer, r+1)

            b.N .= basis[ilay].N
            b.dNdξ .= basis[ilay].dNdξ

            ooplane_values[ib,ilay] = b
        end
    end
    
    G  = [zero(Triad{dim_s,T}) for _ in 1:nqp_per_layer]
    Gᴵ = [zero(Triad{dim_s,T}) for _ in 1:nqp_per_layer]
    R =  [zero(Tensor{2,dim_s,T}) for _ in 1:nqp_per_layer]
    κ =  [zero(Tensor{2,dim_p,T}) for _ in 1:nqp_per_layer]
    κᵐ = [zero(Tensor{2,dim_p,T}) for _ in 1:nqp_inp]
    Eₐ = [zero(Triad{dim_s,T}) for _ in 1:nqp_inp]
    Dₐ = [zero(Triad{dim_s,T}) for _ in 1:nqp_inp]
    
    U =    fill(zero(Tensor{1,dim_s,T}) * T(NaN), nbasefunctions_per_layer, nqp_per_layer)
    dUdξ = fill(zero(Tensor{2,dim_s,T}) * T(NaN), nbasefunctions_per_layer, nqp_per_layer)

    detJdV = fill(T(NaN), nqp_per_layer)
    detJdA = fill(T(NaN), nqp_per_layer)

    MM1 = Tensors.n_components(Tensors.get_base(eltype(κ)))
    MM2 = Tensors.n_components(Tensors.get_base(eltype(R)))
    MM3 = Tensors.n_components(Tensors.get_base(eltype(dUdξ)))

    #combine the two quadrature rules
    layer_qrs = combine_qrs(qr_inplane, qr_ooplane)

    #Initalize bezier operator as NaN
    bezier_operator = IGA.bezier_extraction_to_vector(sparse(Diagonal(fill(NaN, n_midplane_basefuncs))))

    return IGAShellValues{dim_s,dim_p,T,MM1,MM2,MM3}(deepcopy(inplane_values_bezier), inplane_values_bezier, ooplane_values,
                                                        detJdV, detJdA, 
                                                        G, Gᴵ, Eₐ, Dₐ, κ, κᵐ, R, 
                                                        U, dUdξ, similar(U), similar(dUdξ), 
                                                        ngeombasefunctions_per_layer, r, Ref(0), thickness, Ref(0), Ref(bezier_operator), 
                                                        deepcopy(qr_inplane), deepcopy(qr_ooplane), layer_qrs, 
                                                        mid_ip, [oop_ip for _ in 1:n_midplane_basefuncs])
end

getnquadpoints_ooplane_per_layer(cv::IGAShellValues, ilay::Int = get_current_layer(cv)) = return length(cv.oqr[ilay].weights)
getnquadpoints_inplane(cv::IGAShellValues) = return length(cv.iqr.weights)
getnquadpoints_per_layer(cv::IGAShellValues, ilay::Int = get_current_layer(cv)) = return getnquadpoints_ooplane_per_layer(cv, ilay)*getnquadpoints_inplane(cv)

getngeombasefunctions_inplane(cv::IGAShellValues) = return size(cv.inplane_values_bezier.N, 1)
getngeombasefunctions_ooplane_per_layer(cv::IGAShellValues) = cv.oop_order + 1
getngeombasefunctions_per_layer(cv::IGAShellValues) = return cv.ngeombasisfunctions_per_layer
getnbasefunctions_per_layer(cv::IGAShellValues{dim_s}) where dim_s = return cv.ngeombasisfunctions_per_layer * dim_s

get_current_layer(cv::IGAShellValues)::Int = (cv.current_layer[])

get_current_layer_weight(cv::IGAShellValues, qp::Int) = cv.qr[get_current_layer(cv)].weights[qp]
JuAFEM.getdetJdV(cv::IGAShellValues, qp::Int) = cv.detJdV[qp]

function getdetJdA(cv::IGAShellValues, qp::Int, idx::EdgeIndex)
    getidx(idx)==1 && return norm(cross(cv.G[qp][1], cv.G[qp][3])) * get_current_layer_weight(cv, qp)
    getidx(idx)==2 && return norm(cross(cv.G[qp][2], cv.G[qp][3])) * get_current_layer_weight(cv, qp)
    getidx(idx)==3 && return norm(cross(cv.G[qp][3], cv.G[qp][1])) * get_current_layer_weight(cv, qp)
    getidx(idx)==4 && return norm(cross(cv.G[qp][3], cv.G[qp][2])) * get_current_layer_weight(cv, qp)
    error("Edge not found")
end

function getdetJdA(cv::IGAShellValues{3}, qp::Int, idx::EdgeInterfaceIndex)
    getidx(idx)==1 && return norm(cv.G[qp][1]) * get_current_layer_weight(cv, qp)
    getidx(idx)==2 && return norm(cv.G[qp][2]) * get_current_layer_weight(cv, qp)
    getidx(idx)==3 && return norm(cv.G[qp][1]) * get_current_layer_weight(cv, qp)
    getidx(idx)==4 && return norm(cv.G[qp][2]) * get_current_layer_weight(cv, qp)
    error("Interface not found")
end

function getdetJdA(cv::IGAShellValues{2}, qp::Int, idx::VertexIndex)
    return norm(cv.G[qp][2]) * get_current_layer_weight(cv, qp)
end

function getdetJdA(cv::IGAShellValues, qp::Int, idx::VertexInterfaceIndex)
    return 1.0
end

function getdetJdA(cv::IGAShellValues, qp::Int, idx::FaceIndex)
    getidx(idx)==1 && return norm(cross(cv.G[qp][1], cv.G[qp][2])) * get_current_layer_weight(cv, qp)
    getidx(idx)==2 && return norm(cross(cv.G[qp][1], cv.G[qp][2])) * get_current_layer_weight(cv, qp)
    error("Face not found")
end

function getdetJdA(cv::IGAShellValues{2,1}, qp::Int, idx::FaceIndex)
    getidx(idx)==1 && return norm(cv.G[qp][1]) * get_current_layer_weight(cv, qp)
    getidx(idx)==2 && return norm(cv.G[qp][1]) * get_current_layer_weight(cv, qp)
    error("Face not found")
end

function set_quadraturerule!(cv::IGAShellValues{dim_s,dim_p,T}, oqr::LayerQuadratureRule{1}) where {dim_s,dim_p,T}

    for ilay in 1:nlayers(oqr)
        cv.oqr.qrs[ilay] = deepcopy(oqr[ilay])
    end
    
    _qr = combine_qrs(cv.iqr, cv.oqr)
    for ilay in 1:nlayers(oqr)
        cv.qr.qrs[ilay] = _qr[ilay]
    end
end

function IGA.set_bezier_operator!(cv::IGAShellValues{dim_s,dim_p,T}, C::IGA.BezierExtractionOperator{T}) where {dim_s,dim_p,T}
    cv.current_bezier_operator[] = C
    return nothing
end

function set_inplane_basefunctions!(cv::IGAShellValues{dim_s,dim_p,T}, inplane_basisvalues::BasisValues{dim_p,T,M}) where {dim_s,dim_p,T,M}
    
    @assert getnbasefunctions(inplane_basisvalues) == getngeombasefunctions_inplane(cv)
    @assert getnquadpoints(inplane_basisvalues) == getnquadpoints_inplane(cv)

    cv.inplane_values_bezier.N .= inplane_basisvalues.N
    cv.inplane_values_bezier.dNdξ .= inplane_basisvalues.dNdξ
    #cv.inplane_values_bezier.dN²dξ² .= inplane_basisvalues.d²Ndξ²

end

#=function set_ooplane_basefunctions!(cv::IGAShellValues{dim_s,dim_p,T}, ooplane_basisvalues::OOPBasisValues{T}) where {dim_s,dim_p,T}
    set_oop_basefunctions!(cv, [ooplane_basisvalues for i in 1:getngeombasefunctions_inplane(cv)])
end=#

function set_ooplane_basefunctions!(cv::IGAShellValues{dim_s,dim_p,T}, ooplane_basisvalues::Vector{OOPBasisValues{T}}) where {dim_s,dim_p,T}
    
    @assert length(ooplane_basisvalues) == getngeombasefunctions_inplane(cv)
    
    for i in 1:length(ooplane_basisvalues)
        layer_values = ooplane_basisvalues[i]
        for ilay in 1:length(ooplane_basisvalues[i])
            cv.ooplane_values[i,ilay] = layer_values[ilay];
        end
    end

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

function _build_shape_values!(cv::IGAShellValues{dim_s,dim_p,T}, ilay::Int) where {dim_s,dim_p,T}
    
    #
    N_temp = 0.0
    dNdξ_temp = fill(0.0, dim_s)

    #
    B_comp = fill(0.0, dim_s)
    dB_comp = fill(0.0, dim_s, dim_s)

    qp = 0
    basefunc_count = 0
    for oqp in 1:getnquadpoints_ooplane_per_layer(cv, ilay)
        for iqp in 1:getnquadpoints_inplane(cv)
            qp +=1
            basefunc_count = 0
            for i in 1:getngeombasefunctions_inplane(cv)

                #Out of plane
                H = cv.ooplane_values[i,ilay].N
                dHdζ = cv.ooplane_values[i,ilay].dNdξ

                #Inplane
                S = cv.inplane_values_nurbs.N[i,iqp]
                dSdξ = cv.inplane_values_nurbs.dNdξ[i,iqp]
                
                for j in 1:getngeombasefunctions_ooplane_per_layer(cv)    

                    #Shape value
                    N_temp = S * H[j,oqp]

                    #Shape derivative
                    for d1 in 1:dim_p
                        dNdξ_temp[d1] = dSdξ[d1] * H[j,oqp]
                    end
                    dNdξ_temp[dim_s] = S * dHdζ[j,oqp][1]
                    
                    for comp in 1:dim_s
                        basefunc_count += 1
                        
                        #Nurbs
                        fill!(B_comp, 0.0)
                        @inbounds B_comp[comp] = N_temp
                        @inbounds cv.N[basefunc_count, qp] = Vec{dim_s,T}(NTuple{dim_s,T}(B_comp))

                        fill!(dB_comp, 0.0)
                        @inbounds dB_comp[comp, :] = dNdξ_temp
                        @inbounds cv.dNdξ[basefunc_count, qp] = Tensor{2,dim_s,T,dim_s^2}(NTuple{dim_s^2,T}(dB_comp))

                    end
                end
            end
        end
    end
end

function _reinit_layers!(cv::IGAShellValues{dim_s,dim_p,T}, qp_indx, ilay) where {dim_s,dim_p,T}

    qp, iqp, oqp = qp_indx
    
    Eₐ = cv.Eₐ[iqp]
    Dₐ = cv.Dₐ[iqp]
    D = cv.Eₐ[iqp][dim_s]

    G = cv.G[qp]
    Gᴵ = cv.Gᴵ[qp]

    ζ = cv.oqr[ilay].points[oqp]

    #Covarient matrix
    for d in 1:dim_p
        G[d] = Eₐ[d] + ζ[1]*Dₐ[d] * 0.5*cv.thickness
    end
    G[dim_s] = D * 0.5 * cv.thickness

    detJ = dim_s == 3 ? norm((cross(G[1], G[2]))) : norm(G[1])
    cv.detJdA[qp] = detJ * cv.iqr.weights[iqp]
    cv.detJdV[qp] = detJ * cv.qr[ilay].weights[qp] * 0.5 * cv.thickness
    
    Gⁱʲ = inv(SymmetricTensor{2,dim_s,T}((i,j)-> G[i]⋅G[j]))

    #Contravarient matrix
    for i in 1:dim_s
        Gᴵ[i] = zero(Vec{dim_s,T})
        for j in 1:dim_s
            Gᴵ[i] += Gⁱʲ[i,j]*G[j]
        end
    end
    
    FI = Tensor{2,dim_p,T}((α,β)-> G[α]⋅G[β])
    FII = Tensor{2,dim_p,T}((α,β)-> G[α] ⋅ (cv.Dₐ[iqp][β]))
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
        Eₐ[dim_s]  = a/norm(a) #0.5cv.thickness*
        cv.Dₐ[iqp][1] = (da1*norm(a) - a*c1)/(a⋅a) #0.5cv.thickness*
        cv.Dₐ[iqp][2] = (da2*norm(a) - a*c2)/(a⋅a) #0.5cv.thickness*
    elseif dim_s == 2
        scew = Tensor{2,dim_s,T}((0.0, 1.0, -1.0, 0.0))
        a = scew ⋅ cv.Eₐ[iqp][1]
        da1 = scew ⋅ Eₐₐ[1,1]
        Eₐ[dim_s] = (a/norm(a))
        c1 = a/norm(a) ⋅ da1 #0.5cv.thickness*
        cv.Dₐ[iqp][1] =  (da1*norm(a) - a*c1)/(a⋅a) #0.5cv.thickness*
    end

    FI = Tensor{2,dim_p,T}((α,β)-> Eₐ[α]⋅Eₐ[β])
    FII = Tensor{2,dim_p,T}((α,β)-> Eₐ[α] ⋅ (cv.Dₐ[iqp][β]))
    cv.κᵐ[iqp] = inv(FI)⋅FII

end

function reinit_midsurface!(cv::IGAShellValues, coords::Vector{Vec{dim_s,T}}) where {dim_s,T}

    for iqp in 1:getnquadpoints_inplane(cv)
        _reinit_midsurface!(cv, iqp, coords)
    end
   
    _inplane_nurbs_bezier_extraction(cv, cv.current_bezier_operator[])
    return nothing
end

function reinit_layer!(cv::IGAShellValues, ilay::Int)
    cv.current_layer[] = ilay

    qp_layer = 0
    for oqp in 1:getnquadpoints_ooplane_per_layer(cv, ilay)
        for iqp in 1:getnquadpoints_inplane(cv)
            qp_layer += 1
            _reinit_layers!(cv, (qp_layer, iqp, oqp), ilay)
        end
    end
    _build_shape_values!(cv, ilay)
    return nothing
end

get_oop_qp_weight(cv::IGAShellValues, oqp::Int) = cv.oqr.weights[oqp]
get_iop_qp_weight(cv::IGAShellValues, iqp::Int) = cv.iqr.weights[iqp]



function function_parent_derivative(cv::IGAShellValues{dim_s,dim_p,T}, qp::Int, ue::AbstractVector{T}, Θ::Int) where {dim_s,dim_p,T}
    grad = zero(Vec{dim_s,T})
    nbasefuncs = getnbasefunctions_per_layer(cv)
    @assert(length(ue) == nbasefuncs)
    @inbounds for i in 1:nbasefuncs
        grad += cv.dNdξ[i,qp][:,Θ] * ue[i]
    end
    return grad
end

function JuAFEM.function_value(cv::IGAShellValues{dim_s,dim_p,T}, qp::Int, ue::AbstractVector{T}) where {dim_s,dim_p,T}
    val = zero(Vec{dim_s,T})
    nbasefuncs = getnbasefunctions_per_layer(cv)
    @assert(length(ue) == nbasefuncs)
    @inbounds for i in 1:nbasefuncs
        val += cv.N[i,qp] * ue[i]
    end
    return val
end

function JuAFEM.shape_value(cv::IGAShellValues{dim_s,dim_p,T}, qp::Int, ue::AbstractVector{Vec{dim_s,T}}) where {dim_s,dim_p,T}
    val = zero(Vec{dim_s,T})
    nbasefuncs = getnbasefunctions(cv.inplane_values_bezier)
    @assert(length(ue) == nbasefuncs)
    @inbounds for i in 1:nbasefuncs
        val += cv.inplane_values_bezier.N[i,qp] * ue[i]
    end
    return val
end

function shape_parent_derivative(cv::IGAShellValues{dim_s,dim_p,T}, qp::Int, ue::AbstractVector{Vec{dim_s,T}}, Θ::Int) where {dim_s,dim_p,T}
    grad = zero(Vec{dim_s,T})
    nbasefuncs = getnbasefunctions(cv.inplane_values_bezier)
    @assert(length(ue) == nbasefuncs)
    @inbounds for i in 1:nbasefuncs
        grad += cv.inplane_values_bezier.dNdξ[i,qp][Θ] * ue[i]
    end
    return grad
end

function shape_parent_second_derivative(cv::IGAShellValues{dim_s,dim_p,T}, qp::Int, ue::AbstractVector{Vec{dim_s,T}}, Θ::Tuple{Int,Int}) where {dim_s,dim_p,T}
    grad = zero(Vec{dim_s,T})
    nbasefuncs = getnbasefunctions(cv.inplane_values_bezier)
    @assert(length(ue) == nbasefuncs)
    @inbounds for i in 1:nbasefuncs
        grad += cv.inplane_values_bezier.d²Ndξ²[i,qp][Θ[1],Θ[2]] * ue[i]
    end
    return grad
end

function JuAFEM.spatial_coordinate(cv::IGAShellValues{dim_s,dim_p,T}, qp::Int, x::AbstractVector{Vec{dim_s,T}}) where {dim_s,dim_p,T}
    ilay = cv.current_layer[][]
    i2s = CartesianIndices((getnquadpoints_inplane(cv), getnquadpoints_ooplane_per_layer(cv, ilay)))
    iqp, oqp = Tuple(i2s[qp])
    D = cv.Eₐ[iqp][dim_s]

    z = cv.oqr.qrs[ilay].points[oqp][1] * cv.thickness/2 

    Xᴹ = shape_value(cv, iqp, x)
    return Xᴹ + z*D
end

function basis_parent_derivative(cv::IGAShellValues{dim_s,dim_p,T}, qp::Int, i::Int, Θ::Int) where {dim_s,dim_p,T}
    return cv.dNdξ[i,qp][:,Θ]
end

function basis_value(cv::IGAShellValues{dim_s,dim_p,T}, qp::Int, i::Int) where {dim_s,dim_p,T}
    return cv.N[i,qp]
end