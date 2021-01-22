
"""
CachedOOPBasisValues
"""
struct CachedOOPBasisValues{dim_p,T,M}
    basis_values_lumped::OOPBasisValues{T}
    basis_values_layered::OOPBasisValues{T}
    basis_values_discont::OOPBasisValues{T}
    basis_values_weak_discont::Dict{Int, OOPBasisValues{T}} 
    basis_values_strong_discont::Dict{Int, OOPBasisValues{T}} 

    basis_cohseive_lumped::Tuple{ OOPBasisValues{T},  OOPBasisValues{T}} #1 = bottomface, 2 = topface
    basis_cohesive_layered::Tuple{ OOPBasisValues{T},  OOPBasisValues{T}}
    basis_cohesive_discont::Tuple{ OOPBasisValues{T},  OOPBasisValues{T}}
    basis_cohesive_weak_discont::Dict{Int, Tuple{ OOPBasisValues{T},  OOPBasisValues{T}}}
    basis_cohesive_strong_discont::Dict{Int, Tuple{ OOPBasisValues{T},  OOPBasisValues{T}}}

    basis_values_lumped_face::Tuple{ OOPBasisValues{T},  OOPBasisValues{T}} #1 = bottomface, 2 = topface
    basis_values_layered_face::Tuple{ OOPBasisValues{T},  OOPBasisValues{T}}
    basis_values_discont_face::Tuple{ OOPBasisValues{T},  OOPBasisValues{T}}
    basis_values_weak_discont_face::Dict{Int, Tuple{ OOPBasisValues{T},  OOPBasisValues{T}}}
    basis_values_strong_discont_face::Dict{Int, Tuple{ OOPBasisValues{T},  OOPBasisValues{T}}}

    #For side boudary conditions, note that it is the inplane basefunctions values that are cached
    basis_values_sideface::Array{BasisValues{dim_p,T,M}, 1} #1 = front, 2 = right, 3 = back, 4 = left

    #For vertex boudary conditions, note that it is the inplane basefunctions values that are cached
    basis_values_vertex::Array{BasisValues{dim_p,T,M}, 1} #1 lower left, #2 lower right, #3 upper right,  #4 upper left

    #
    active_layer_dofs_lumped::Vector{Vector{Int}}
    active_layer_dofs_layered::Vector{Vector{Int}}
    active_layer_dofs_discont::Vector{Vector{Int}}

    active_interface_dofs_lumped::Vector{Vector{Int}}
    active_interface_dofs_layered::Vector{Vector{Int}}
    active_interface_dofs_discont::Vector{Vector{Int}}

    active_local_interface_dofs_lumped::Vector{Vector{Int}}
    active_local_interface_dofs_layered::Vector{Vector{Int}}
    active_local_interface_dofs_discont::Vector{Vector{Int}}
end

function CachedOOPBasisValues(qr_cell_oop::LayerQuadratureRule{1,T}, 
                              qr_cohesive_oop::AbstractVector{<:LayerQuadratureRule}, 
                              qr_face::AbstractVector{<:LayerQuadratureRule}, 
                              qr_sides::AbstractVector{<:QuadratureRule},
                              qr_vertices::AbstractVector{<:QuadratureRule}, 
                              ip_inplane, ip_lumped, ip_layered, ip_discont,
                              ninterfaces::Int, ooplane_order::Int, dim_s::Int) where {T}

    nlayers = ninterfaces+1
    nbasefunctions_inplane = getnbasefunctions(ip_inplane)
    dim_p = dim_s-1

    #Cell integration quadrules
    a = OOPBasisValues(qr_cell_oop, ip_lumped) 
    b = OOPBasisValues(qr_cell_oop, ip_layered) 
    c = OOPBasisValues(qr_cell_oop, ip_discont) 
    d1 = Dict{Int, BasisValues{1,T}}()
    d2 = Dict{Int, BasisValues{1,T}}()

    #Cohesive zones
    e = (OOPBasisValues(qr_cohesive_oop[1], ip_lumped), OOPBasisValues(qr_cohesive_oop[2], ip_lumped))
    f = (OOPBasisValues(qr_cohesive_oop[1], ip_layered), OOPBasisValues(qr_cohesive_oop[2], ip_layered) )
    g = (OOPBasisValues(qr_cohesive_oop[1], ip_discont), OOPBasisValues(qr_cohesive_oop[2], ip_discont) )
    h1 = Dict{Int, Tuple{ OOPBasisValues{T},  OOPBasisValues{T}}}()
    h2 = Dict{Int, Tuple{ OOPBasisValues{T},  OOPBasisValues{T}}}()


    #Top/bottom face qruadrules
    k = (OOPBasisValues(qr_face[1], ip_lumped),  OOPBasisValues(qr_face[2], ip_lumped))
    l = (OOPBasisValues(qr_face[1], ip_layered), OOPBasisValues(qr_face[2], ip_layered))
    m = (OOPBasisValues(qr_face[1], ip_discont), OOPBasisValues(qr_face[2], ip_discont))
    n1 = Dict{Int, Tuple{ OOPBasisValues{T},  OOPBasisValues{T}}}()
    n2 = Dict{Int, Tuple{ OOPBasisValues{T},  OOPBasisValues{T}}}()

    #Side quadrules
    basis_values_sideface = Array{BasisValues{dim_p,T}, 1}()
    for i in 1:length(qr_sides)
        push!(basis_values_sideface, BasisValues(qr_sides[i], ip_inplane))
    end

    #Vertex quadrules
    basis_values_vertex = Array{BasisValues{dim_p,T}, 1}()
    for i in 1:length(qr_vertices)
        push!(basis_values_vertex, BasisValues(qr_vertices[i], ip_inplane))
    end
    
    M = Tensors.n_components(Tensors.get_base(eltype(basis_values_sideface[1].d²Ndξ²)))

    #Active basis values
    active_dofs_lumped = generate_active_layer_dofs(nlayers, ooplane_order, dim_s, [LUMPED for _ in 1:nbasefunctions_inplane])
    active_dofs_layered = generate_active_layer_dofs(nlayers, ooplane_order, dim_s, [LAYERED for _ in 1:nbasefunctions_inplane])
    active_dofs_discont = generate_active_layer_dofs(nlayers, ooplane_order, dim_s, [FULLY_DISCONTINIUOS for _ in 1:nbasefunctions_inplane])

    active_interface_dofs_lumped, local_interface_dofs_lumped = generate_active_layer_dofs(ninterfaces, ooplane_order, dim_s, [LUMPED for _ in 1:nbasefunctions_inplane])
    active_interface_dofs_layered, local_interface_dofs_lumped = generate_active_layer_dofs(ninterfaces, ooplane_order, dim_s, [LAYERED for _ in 1:nbasefunctions_inplane])
    active_interface_dofs_discont, local_interface_dofs_lumped = generate_active_layer_dofs(ninterfaces, ooplane_order, dim_s, [FULLY_DISCONTINIUOS for _ in 1:nbasefunctions_inplane])

    return CachedOOPBasisValues{dim_p,T,M}(a,b,c,d1,d2,
                                     e,f,g,h1,h2,
                                     k,l,m,n1,n2, 
                                     basis_values_sideface, basis_values_vertex,
                                     active_dofs_lumped, active_dofs_layered, active_dofs_discont,
                                     active_interface_dofs_lumped, active_interface_dofs_layered, active_interface_dofs_discont,
                                     local_interface_dofs_lumped, local_interface_dofs_lumped, local_interface_dofs_lumped)
end

struct IGAShellIntegrationData{dim_p,dim_s,T,ISV<:IGAShellValues,M}

    cell_values_mixed::ISV
    cell_values_lumped::ISV
    cell_values_layered::ISV
    cell_values_discont::ISV

    cell_values_sr::ISV #For stress recovory

    cell_values_cohesive_top::ISV
    cell_values_cohesive_bot::ISV

    cell_values_face::ISV
    cell_values_side::ISV
    cell_values_interface::ISV
    cell_values_vertices::ISV

    oqr::LayerQuadratureRule{1,T}
    iqr::QuadratureRule{dim_p,RefCube,T}
    qr_inp_cohesvie::QuadratureRule{dim_p,RefCube,T}
    qr_faces::Array{LayerQuadratureRule{1,T}, 1} #TODO: make tuple
    qr_cohesive::Array{LayerQuadratureRule{1,T}, 1}
    qr_sides::Array{QuadratureRule{dim_p,RefCube,T}, 1}
    qr_vertices::Array{QuadratureRule{dim_p,RefCube,T}, 1}

    cache_values::CachedOOPBasisValues{dim_p,T,M}

    extraction_operators::Vector{IGA.BezierExtractionOperator{T}}
    oop_order::Int
end

function cached_cell_basisvalues(intdata::IGAShellIntegrationData, i::CELLSTATE)
    ninterfaces = length(intdata.oqr.qrs) - 1
    is_lumped(i) && return intdata.cache_values.basis_values_lumped
    is_layered(i) && return intdata.cache_values.basis_values_layered
    is_fully_discontiniuos(i) && return intdata.cache_values.basis_values_discont
    is_weak_discontiniuos(i) && return get_or_create_discontinious_basisvalues!(intdata.oop_order, ninterfaces, i, intdata.cache_values.basis_values_weak_discont, intdata.oqr)
    is_strong_discontiniuos(i) && return get_or_create_discontinious_basisvalues!(intdata.oop_order, ninterfaces, i, intdata.cache_values.basis_values_strong_discont, intdata.oqr)
    error("wrong state")
end

function cached_cohesive_basisvalues(intdata::IGAShellIntegrationData, i::CELLSTATE)
    ninterfaces = length(intdata.oqr.qrs) - 1
    is_lumped(i) && return intdata.cache_values.basis_cohseive_lumped
    is_layered(i) && return intdata.cache_values.basis_cohesive_layered
    is_fully_discontiniuos(i) && return intdata.cache_values.basis_cohesive_discont
    is_weak_discontiniuos(i) && return get_or_create_discontinious_basisvalues!(intdata.oop_order, ninterfaces, i, intdata.cache_values.basis_cohesive_weak_discont, intdata.qr_cohesive)
    is_strong_discontiniuos(i) && return get_or_create_discontinious_basisvalues!(intdata.oop_order, ninterfaces, i, intdata.cache_values.basis_cohesive_strong_discont, intdata.qr_cohesive)
    error("wrong state")
end

function cached_face_basisvalues(intdata::IGAShellIntegrationData, i::CELLSTATE, face::Int)::OOPBasisValues{Float64}
    ninterfaces = length(intdata.oqr.qrs) - 1
    is_lumped(i) && return  intdata.cache_values.basis_values_lumped_face[face]
    is_layered(i) && return    intdata.cache_values.basis_values_layered_face[face]
    is_fully_discontiniuos(i) && return    intdata.cache_values.basis_values_discont_face[face]
    is_weak_discontiniuos(i) && return get_or_create_discontinious_basisvalues!(intdata.oop_order, ninterfaces, i, intdata.cache_values.basis_values_weak_discont_face, intdata.qr_faces)[face]
    is_strong_discontiniuos(i) && return get_or_create_discontinious_basisvalues!(intdata.oop_order, ninterfaces, i, intdata.cache_values.basis_values_strong_discont_face, intdata.qr_faces)[face]
    error("wrong state")
end

function cached_side_basisvalues(intdata::IGAShellIntegrationData, side::Int)
    return intdata.cache_values.basis_values_sideface[side]
end

function cached_vertex_basisvalues(intdata::IGAShellIntegrationData, vertex::Int)
    return intdata.cache_values.basis_values_vertex[vertex]
end

function get_or_create_discontinious_basisvalues!(order::Int, ninterfaces::Int, cellstate::CELLSTATE, dict::Dict, qrs)
    return get!(dict, cellstate.state2) do
        return create_discontinious_basisvalues(order, ninterfaces, cellstate, qrs)
    end
end

function create_discontinious_basisvalues(order::Int, ninterfaces::Int, cellstate::CELLSTATE, qrs::QuadratureRule)
    nlayers = ninterfaces+1
    knot_discont = generate_knot_vector(cellstate, order, ninterfaces)
    ip_discont = IGA.BSplineBasis(knot_discont, order)
    return OOPBasisValues(qrs, ip_discont)
end

function create_discontinious_basisvalues(order::Int, ninterfaces::Int, cellstate::CELLSTATE, qrs::AbstractVector{<:QuadratureRule})
    nlayers = ninterfaces+1
    knot_discont = generate_knot_vector(cellstate, order, ninterfaces)
    ip_discont = IGA.BSplineBasis(knot_discont, order)
    return (OOPBasisValues(qrs[1], ip_discont), OOPBasisValues(qrs[2], ip_discont))
end 

get_extraction_operator(intdata::IGAShellIntegrationData, cellid::Int) = return intdata.extraction_operators[cellid]

get_oop_quadraturerule(intdata::IGAShellIntegrationData) = intdata.oqr
get_inp_quadraturerule(intdata::IGAShellIntegrationData) = intdata.iqr
get_face_qr(intdata::IGAShellIntegrationData, face::Int) = intdata.qr_faces[face]
get_vertex_qr(intdata::IGAShellIntegrationData, vertex::Int) = intdata.qr_vertices[vertex]

function IGAShellIntegrationData(data::IGAShellData{dim_p,dim_s,T}, C::Vector{IGA.BezierExtractionOperator{T}}) where {dim_p,dim_s,T}
    
    order = data.orders[dim_s]
    nlayers = data.nlayers
    mid_ip = getmidsurface_ip(data)

    #Quadratre rules
    iqr = QuadratureRule{dim_p,RefCube}(data.nqp_inplane_order)
    iqr_cohesive = QuadratureRule{dim_p,RefCube}(data.nqp_interface_order)
    oqr = LayerQuadratureRule(data.zcoords,  data.nqp_ooplane_per_layer)
    
    oqr_face = generate_face_oop_qrs(nlayers)
    oqr_cohesive = generate_cohesive_oop_qrs(data.zcoords)
    iqr_sides =  JuAFEM.create_face_quad_rule(QuadratureRule{dim_p-1,RefCube}(5), Lagrange{dim_p,RefCube,1}()) #HARDCODED
    iqr_vertices =  JuAFEM.create_face_quad_rule(QuadratureRule{0,RefCube}(0), Lagrange{dim_p,RefCube,1}()) #HARDCODED

    #Create knot vectors for layers, lumped and dinscont
    knot_lumped  = generate_knot_vector(order, nlayers-1, 0)
    knot_layered = generate_knot_vector(order, nlayers-1, order)
    knot_discont = generate_knot_vector(order, nlayers-1, order+1)

    #Create basis values for layered, lumped, discont
    ip_lumped = IGA.BSplineBasis(knot_lumped,order)
    ip_layered = IGA.BSplineBasis(knot_layered, order)
    ip_discont = IGA.BSplineBasis(knot_discont, order)
    
    cache = CachedOOPBasisValues(oqr, oqr_cohesive, oqr_face, iqr_sides, iqr_vertices, 
                                 mid_ip, ip_lumped, ip_layered, ip_discont, 
                                 ninterfaces(data), order, dim_s)

    #
    cell_values = IGAShellValues(data.thickness, iqr, oqr,  mid_ip, ip_lumped)
    face_values = IGAShellValues(data.thickness, iqr, oqr_face[1],  mid_ip, ip_lumped)
    side_values = IGAShellValues(data.thickness, iqr_sides[1], oqr,  mid_ip, ip_lumped)
    interface_values = IGAShellValues(data.thickness, iqr_sides[1], oqr_face[1],  mid_ip, ip_lumped)
    vertices_values = IGAShellValues(data.thickness, iqr_vertices[1], oqr_face[1],  mid_ip, ip_lumped)
    
    #
    cell_values_lumped  = IGAShellValues(data.thickness, iqr, oqr, mid_ip, ip_lumped)
    cell_values_layered = IGAShellValues(data.thickness, iqr, oqr, mid_ip, ip_layered)
    cell_values_discont = IGAShellValues(data.thickness, iqr, oqr, mid_ip, ip_discont)
    
    #
    cell_values_sr = IGAShellValues(data.thickness, QuadratureRule{dim_p,RefCube}(1), oqr, mid_ip, ip_lumped )

    #
    cell_values_cohesive_bot = IGAShellValues(data.thickness, iqr_cohesive, oqr_cohesive[1], mid_ip, ip_lumped)
    cell_values_cohesive_top = IGAShellValues(data.thickness, iqr_cohesive, oqr_cohesive[2], mid_ip, ip_lumped)

    M = Tensors.n_components(Tensors.get_base(eltype(cache.basis_values_sideface[1].d²Ndξ²)))

    return IGAShellIntegrationData{dim_p,dim_s,T,typeof(cell_values),M}(
        cell_values, cell_values_lumped, cell_values_layered, cell_values_discont, 
        cell_values_sr,
        cell_values_cohesive_top, cell_values_cohesive_bot,
        face_values, side_values, interface_values, vertices_values,
        oqr, iqr, iqr_cohesive, oqr_face, oqr_cohesive, iqr_sides, iqr_vertices,
        cache,
        C, order)
end

get_inplane_quadraturerule(intdata::IGAShellIntegrationData) = intdata.iqr